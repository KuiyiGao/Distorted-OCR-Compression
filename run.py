import os
import re
import numpy as np
import torch
import collections
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from render.data import prepare_cuad_mrc_data
from render.utils import (
    stitch_word_signal, composite_saliency, assign_tiers,
)
from render.render import render_img_tiered
from render.attention import qa_token_saliency, question_token_relevance


_CUAD_CLAUSE_RE = re.compile(r'related to ["\u2018\u201c]([^"\u2019\u201d\']+)["\u2019\u201d\']',
                             re.IGNORECASE)


def extract_clause_type(question):
    """Extract the clause-type substring (e.g. 'Governing Law') and its
    char offsets within `question` from the CUAD template boilerplate."""
    m = _CUAD_CLAUSE_RE.search(question)
    if not m:
        return None, None
    return m.group(1), (m.start(1), m.end(1))


def _forward_chunk(model, chunk, device):
    """Single forward pass on one chunk. Returns (hidden_states, qa_saliency)."""
    ids = torch.tensor(chunk["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
    mask = torch.tensor(chunk["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(ids, attention_mask=mask, output_hidden_states=True)
    H = out.hidden_states[-1]                       # [1, T, d]
    qa_sal = qa_token_saliency(out.start_logits[0], out.end_logits[0],
                               chunk["sequence_ids"])
    return H, qa_sal


def _focused_question_pooled(H, chunk, clause_char_span):
    """Pool hidden states over the clause-type tokens only."""
    seq_ids = chunk["sequence_ids"]
    offsets = chunk["offsets"]

    focused_idx = []
    if clause_char_span is not None:
        cs, ce = clause_char_span
        for i, (s, off) in enumerate(zip(seq_ids, offsets)):
            if s == 0 and off is not None:
                a, b = off
                if a < ce and b > cs:
                    focused_idx.append(i)

    if not focused_idx:
        focused_idx = [i for i, s in enumerate(seq_ids) if s == 0] or list(range(8))

    return H[0, focused_idx].mean(dim=0)            # [d]


def main():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[init] device={device}")

    json_path = "render/data/CUADv1.json"
    model_name = "akdeniz27/roberta-base-cuad"

    print(f"[load] {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device).eval()

    print("[data] preparing CUAD chunks for ALL clause-type questions...")
    dataset = prepare_cuad_mrc_data(json_path, tokenizer, target_docs=1)

    qa_grouped = collections.defaultdict(list)
    for s in dataset:
        qa_grouped[s['qa_id']].append(s)

    # All QAs share the same document context
    full_context = list(qa_grouped.values())[0][0]["context"]
    print(f"[data] {len(qa_grouped)} clause-type questions, context length={len(full_context)} chars")

    # ----------------------------------------------------------------
    # Phase 1: For EACH of the 41 clause-type questions, run the QA
    # model on its chunks, stitch S and kappa to word-level.
    # ----------------------------------------------------------------
    all_S_words = []
    all_K_words = []
    clause_types_used = []
    words = None                                    # same for every question

    for qi, (qa_id, chunks) in enumerate(qa_grouped.items()):
        question = chunks[0]["question"]
        clause_type, clause_span = extract_clause_type(question)
        clause_types_used.append(clause_type or qa_id[:30])

        # Get the focused pooled question vector from chunk 0's hidden states
        H0, _ = _forward_chunk(model, chunks[0], device)
        q_pooled = _focused_question_pooled(H0, chunks[0], clause_span)

        qa_chunks_data  = []
        qrel_chunks_data = []

        for chunk in chunks:
            H, qa_sal = _forward_chunk(model, chunk, device)
            qrel = question_token_relevance(H, q_pooled, chunk["sequence_ids"])

            qa_chunks_data.append({
                'probs': qa_sal.detach().cpu().tolist(),
                'offsets': chunk['offsets'],
                'sequence_ids': chunk['sequence_ids'],
            })
            qrel_chunks_data.append({
                'probs': qrel.detach().cpu().tolist(),
                'offsets': chunk['offsets'],
                'sequence_ids': chunk['sequence_ids'],
            })

        w, S_word = stitch_word_signal(full_context, qa_chunks_data,
                                       key='probs', reducer='max')
        _, K_word = stitch_word_signal(full_context, qrel_chunks_data,
                                       key='probs', reducer='max')
        if words is None:
            words = w

        all_S_words.append(S_word)
        all_K_words.append(K_word)
        print(f"[qa {qi+1:2d}/{len(qa_grouped)}] {clause_type or qa_id[:30]:35s} "
              f"| S_max={S_word.max():.3f} | chunks={len(chunks)}")

    # ----------------------------------------------------------------
    # Phase 2: Max-pool S and kappa across all 41 questions.
    # If ANY question thinks a region is an answer, it's important.
    # ----------------------------------------------------------------
    S_all = np.stack(all_S_words, axis=0)           # [41, N_words]
    K_all = np.stack(all_K_words, axis=0)
    S_agg = S_all.max(axis=0)                       # [N_words]
    K_agg = K_all.max(axis=0)

    print(f"[aggregate] S_agg range=[{S_agg.min():.4f}, {S_agg.max():.4f}] | "
          f"K_agg range=[{K_agg.min():.4f}, {K_agg.max():.4f}]")

    # ----------------------------------------------------------------
    # Phase 3: Composite saliency (uses updated defaults from utils.py:
    # alpha=0.6, beta=0.9, delta=0.8, gamma=0.3).
    # ----------------------------------------------------------------
    phi = composite_saliency(words, S_agg, K_agg)
    tiers = assign_tiers(phi, primary_pct=98.0, secondary_pct=85.0)

    n_pri = int((tiers == 2).sum())
    n_sec = int((tiers == 1).sum())
    n_ter = len(tiers) - n_pri - n_sec
    print(f"[tiers] primary={n_pri} | secondary={n_sec} | tertiary={n_ter}")

    word_weights = list(zip(words, phi.tolist()))

    # ----------------------------------------------------------------
    # Phase 4: Render with a meaningful header (not the CUAD template).
    # ----------------------------------------------------------------
    out_dir = "cuad_distorted_rendering"
    os.makedirs(out_dir, exist_ok=True)
    header = (f"Document-wide legal saliency "
              f"(aggregated across {len(qa_grouped)} CUAD clause-type questions)")
    print(f"[render] {len(word_weights)} words")
    img = render_img_tiered(word_weights, tiers, question_text=header)
    img.save(os.path.join(out_dir, "distorted_rendering.png"))

    with open(os.path.join(out_dir, "meta.txt"), "w") as f:
        f.write(f"Mode: all-clause aggregation\n")
        f.write(f"Clause types queried: {clause_types_used}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Tiers: primary={n_pri} secondary={n_sec} tertiary={n_ter}\n")
        primary_words = [w for w, t in zip(words, tiers) if t == 2]
        f.write(f"Primary tokens: {primary_words}\n\n")

        # Per-question S_max so we can see which clauses the model found
        f.write("Per-clause-type S_max (descending):\n")
        ranked = sorted(zip(clause_types_used,
                            [float(s.max()) for s in all_S_words]),
                        key=lambda x: x[1], reverse=True)
        for ct, smax in ranked:
            f.write(f"  {smax:.4f}  {ct}\n")

    print(f"[done] output -> {out_dir}/")


if __name__ == "__main__":
    main()
