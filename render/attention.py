import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class ImbalancedMRCTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        class_weights = torch.tensor([1.0, 30.0]).to(model.device)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)

        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def extract_full_saliency(model, input_ids, attention_mask, device):
    """
    Single forward pass returning three signals per chunk:

      1. saliency  (np.ndarray [seq_len]):
         Geometric mean of the final-layer CLS→token attention (PDF Eq. 2) and
         the token-level class-1 probability.  This is the "attention in the CLS
         layer" asked for in Q1.

      2. cls_hidden (torch.Tensor [hidden_size]):
         CLS hidden state from the last encoder layer.  Collected across chunks
         and pooled into a global memory vector (Q2).

      3. token_hidden (torch.Tensor [seq_len, hidden_size]):
         All token hidden states from the last encoder layer.  Used in
         apply_global_reweighting() to compute cosine similarity against the
         pooled global memory, giving previously-visited chunks retroactive
         global context (Q2).
    """
    model.eval()
    with torch.no_grad():
        inputs = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        mask   = attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask

        outputs = model(
            inputs.to(device),
            mask.to(device),
            output_attentions=True,    # expose per-layer attention weights
            output_hidden_states=True, # expose per-layer token hidden states
        )

        # ── Q1: CLS-layer attention ──────────────────────────────────────────
        # outputs.attentions[-1] shape: [batch, num_heads, seq_len, seq_len]
        # Row 0 of the query dimension = CLS token attending to every position.
        # Average over heads to collapse the multi-head dimension.
        cls_attn = outputs.attentions[-1][0, :, 0, :].mean(0).cpu()   # [seq_len]

        # Token classification probabilities for the answer class (class 1)
        class_probs = F.softmax(outputs.logits[0], dim=-1)[:, 1].cpu() # [seq_len]

        # Geometric mean: both signals must agree for high saliency
        saliency = (cls_attn * class_probs).sqrt().numpy()

        # ── Q2: hidden states for global memory ─────────────────────────────
        # outputs.hidden_states[-1] shape: [batch, seq_len, hidden_size]
        last_hidden  = outputs.hidden_states[-1][0].cpu()  # [seq_len, hidden]
        cls_hidden   = last_hidden[0]                      # [hidden]

    return saliency, cls_hidden, last_hidden


def apply_global_reweighting(local_saliency, token_hidden, global_memory, alpha=0.3):
    """
    Q2 — Retroactive global attention for previously visited chunks.

    After all chunks have been forward-passed once, their CLS hidden states are
    pooled into `global_memory` (caller's responsibility, e.g. torch.stack(...).mean(0)).
    This function blends each chunk's local saliency with a global-context weight
    derived from cosine similarity of every token against that pooled memory,
    so earlier chunks benefit from information seen only in later chunks.

    Args:
        local_saliency  np.ndarray [seq_len]          output of extract_full_saliency
        token_hidden    torch.Tensor [seq_len, hidden] output of extract_full_saliency
        global_memory   torch.Tensor [hidden]          mean of all cls_hidden tensors
        alpha           float                          weight of global signal (default 0.3)

    Returns:
        np.ndarray [seq_len]  re-weighted saliency
    """
    # Cosine similarity of each token against the global document memory
    sim = F.cosine_similarity(token_hidden, global_memory.unsqueeze(0), dim=-1)  # [seq_len]
    global_weight = F.softmax(sim, dim=0).numpy()

    return (1.0 - alpha) * local_saliency + alpha * global_weight
