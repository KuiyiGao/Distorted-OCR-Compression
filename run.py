import os
import torch
import collections
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments

from render.data import CUADQADataset
from render.attention import ImbalancedMRCTrainer, extract_full_saliency, apply_global_reweighting
from render.utils import stitch_and_smooth_saliency
from render.render import render_tsvr_image, visualize_single_attention

# ── Q3: Single configuration dictionary ─────────────────────────────────────
# All tunable knobs live here.  Everything else is derived.
CFG = {
    "model_name":        "nlpaueb/legal-bert-base-uncased",
    "max_length":        512,    # tokens per chunk
    "learning_rate":     2e-4,
    "num_train_epochs":  1,
    "n_train_docs":      None,   # None = use all docs; int = first N docs only
    "global_alpha":      0.3,    # weight of global retroactive attention (Q2)
    "output_dir":        "cuad_tsvr_results",
}

# Derived — not tunable separately
_STRIDE       = CFG["max_length"] // 4          # 25 % overlap, no free parameter
_BATCH_SIZE   = 16                               # hardware constant, not a model knob
_FROZEN_FRAC  = 10 / 12                          # freeze bottom 10 of 12 layers


def main():
    print("=== Starting Q/A LegalBERT Full-Contract TSVR Pipeline ===")
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps"  if torch.backends.mps.is_available() else "cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"], use_fast=True)

    print(f"Loading dataset (n_train_docs={CFG['n_train_docs']})...")
    dataset = CUADQADataset(
        "render/data/CUADv1.json", tokenizer,
        max_length=CFG["max_length"],
        n_docs=CFG["n_train_docs"],
    )

    train_size = int(0.8 * len(dataset))
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )

    print(f"Loading model: {CFG['model_name']}")
    model = AutoModelForTokenClassification.from_pretrained(
        CFG["model_name"], num_labels=2
    ).to(device)

    # Freeze bottom _FROZEN_FRAC of encoder layers
    n_layers      = model.config.num_hidden_layers          # 12 for BERT-base
    freeze_up_to  = int(n_layers * _FROZEN_FRAC)            # 10
    for name, param in model.bert.named_parameters():
        if not any(name.startswith(f"encoder.layer.{i}") for i in range(freeze_up_to, n_layers)):
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir              = "./tsvr_checkpoints",
        evaluation_strategy     = "epoch",
        save_strategy           = "epoch",
        learning_rate           = CFG["learning_rate"],
        per_device_train_batch_size = _BATCH_SIZE,
        per_device_eval_batch_size  = _BATCH_SIZE,
        num_train_epochs        = CFG["num_train_epochs"],
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        greater_is_better       = False,
        report_to               = "none",
    )

    trainer = ImbalancedMRCTrainer(
        model        = model,
        args         = training_args,
        train_dataset= train_dataset,
        eval_dataset = eval_dataset,
    )

    print("Training...")
    trainer.train()

    # ── Inference ────────────────────────────────────────────────────────────
    qa_grouped = collections.defaultdict(list)
    for sample in dataset.data:
        qa_grouped[sample['qa_id']].append(sample)

    # Pick one QA pair that has at least one positive label
    target_qas = [
        (qa_id, chunks)
        for qa_id, chunks in qa_grouped.items()
        if any(1 in c['labels'] for c in chunks)
    ][:1]

    os.makedirs(CFG["output_dir"], exist_ok=True)

    for qa_id, chunks in target_qas:
        question     = chunks[0]["question"]
        full_context = chunks[0]["context"]

        # ── Phase 1: one forward pass per chunk ──────────────────────────────
        # Collect local saliency, CLS hidden states, and token hidden states.
        # Q1: saliency is already CLS-attention × class-prob (geometric mean).
        phase1 = []
        cls_states = []
        for chunk in chunks:
            input_ids      = torch.tensor(chunk["input_ids"],      dtype=torch.long)
            attention_mask = torch.tensor(chunk["attention_mask"],  dtype=torch.long)

            saliency, cls_hidden, token_hidden = extract_full_saliency(
                trainer.model, input_ids, attention_mask, device
            )
            phase1.append({
                "saliency":     saliency,
                "token_hidden": token_hidden,
                "offsets":      chunk["offsets"],
                "sequence_ids": chunk["sequence_ids"],
            })
            cls_states.append(cls_hidden)

        # ── Phase 2: retroactive global attention — no extra forward passes ──
        # Q2: pool all CLS states into one global memory vector, then re-weight
        # every chunk's local saliency so earlier chunks gain global context.
        global_memory = torch.stack(cls_states).mean(0)   # [hidden_size]

        chunks_extracted = []
        for entry in phase1:
            reweighted = apply_global_reweighting(
                entry["saliency"],
                entry["token_hidden"],
                global_memory,
                alpha=CFG["global_alpha"],
            )
            chunks_extracted.append({
                "probs":        reweighted,
                "offsets":      entry["offsets"],
                "sequence_ids": entry["sequence_ids"],
            })

        print(f"Stitching {len(chunks)} overlapping windows...")
        word_weights = stitch_and_smooth_saliency(full_context, chunks_extracted)

        if not word_weights:
            continue

        visualize_single_attention(
            word_weights,
            os.path.join(CFG["output_dir"], "attention.png"),
        )
        tsvr_img = render_tsvr_image(word_weights)
        tsvr_img.save(os.path.join(CFG["output_dir"], "tsvr.png"))

        with open(os.path.join(CFG["output_dir"], "meta.txt"), "w") as f:
            f.write(f"Question: {question}\nQA ID: {qa_id}\n")
            f.write(f"Chunks: {len(chunks)}\n")
            high_att = [w for w, p in word_weights if p > 0.8]
            f.write(f"High-saliency terms: {high_att}\n")

    print(f"Done. Results saved to '{CFG['output_dir']}'.")


if __name__ == "__main__":
    main()
