"""MemSlot attention over a frozen RoBERTa backbone.

Why this module exists
----------------------
The original `run.py` pipeline computed saliency by fine-tuning LegalBERT as
a **token classifier conditioned on the question**. That leaks the question
into the rendering, which biases everything that follows (OCR, prune,
summary). To keep the evaluation honest we need a saliency signal that is a
function of the contract alone.

Design
------
* Backbone: frozen ``roberta-base``. It is pretrained on generic text, so
  the CUAD test split cannot leak through it.
* MemSlot: ``K`` learnable query vectors attend over the contract tokens.
  Unsupervised training on CUAD *train-split contracts only* with two
  losses:
    1. reconstruction - the mixture of slot vectors must approximate each
       contextual hidden state (slot attention objective);
    2. diversity - slots should span orthogonal directions.
  No question, no answer, no clause label is ever consumed.
* Saliency: for each token, ``s_i = max_k softmax_k(M H_i^T / sqrt d)``.
  Gaussian smoothed, min-max scaled to [0, 1] to feed `render_tsvr_image`.

Math
----
Let ``H ∈ R^{L×d}`` be the frozen token embeddings, ``M ∈ R^{K×d}`` the
trainable slots.

    A_{k,i} = softmax_i( (M Q)(H W_k)^T / sqrt d )             (slot -> token)
    S_k     = sum_i A_{k,i} V(H_i)                              (slot state)
    H_hat_i = sum_k softmax_k( Q'(H_i) K'(S)^T / sqrt d ) V'(S)  (reconstruct)

Losses::

    L_rec  = mean ||H - H_hat||_2^2
    L_div  = ||M_norm M_norm^T - I||_F^2
    L      = L_rec + λ · L_div
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- module

class MemSlotAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_slots: int = 32, d_proj: int = 256):
        super().__init__()
        self.n_slots = n_slots
        self.slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_proj, bias=False)
        self.k_proj = nn.Linear(d_model, d_proj, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # reconstruction path: token queries over slots
        self.rq_proj = nn.Linear(d_model, d_proj, bias=False)
        self.rk_proj = nn.Linear(d_model, d_proj, bias=False)
        self.rv_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_proj ** -0.5

    def forward(self, H: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # H: (B, L, d)
        Q = self.q_proj(self.slots)                            # (K, p)
        K = self.k_proj(H)                                     # (B, L, p)
        V = self.v_proj(H)                                     # (B, L, d)
        attn_logits = torch.einsum("kp,blp->bkl", Q, K) * self.scale
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask[:, None, :], -1e9)
        attn = attn_logits.softmax(dim=-1)                     # (B, K, L): distribute each slot over tokens
        S = torch.einsum("bkl,bld->bkd", attn, V)              # (B, K, d)

        # Reconstruct each token from the slot states
        rq = self.rq_proj(H)                                   # (B, L, p)
        rk = self.rk_proj(S)                                   # (B, K, p)
        rv = self.rv_proj(S)                                   # (B, K, d)
        recon_logits = torch.einsum("blp,bkp->blk", rq, rk) * self.scale
        recon = recon_logits.softmax(dim=-1)                   # (B, L, K)
        H_hat = torch.einsum("blk,bkd->bld", recon, rv)        # (B, L, d)

        return H_hat, attn, S

    def saliency(self, H: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-token saliency in [0, 1] from the slot->token attention weights."""
        _, attn, _ = self.forward(H, mask)
        s = attn.max(dim=1).values                             # (B, L)
        if mask is not None:
            s = s.masked_fill(~mask, 0.0)
            # ignore padding when computing the per-sample min
            s_for_min = s.masked_fill(~mask, float("inf"))
            s_min = s_for_min.min(dim=-1, keepdim=True).values
            s_min = torch.where(torch.isinf(s_min), torch.zeros_like(s_min), s_min)
        else:
            s_min = s.min(dim=-1, keepdim=True).values
        s_max = s.max(dim=-1, keepdim=True).values
        return ((s - s_min) / (s_max - s_min + 1e-9)).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- trainer

@dataclass
class MemSlotConfig:
    n_slots: int = 32
    lr: float = 1e-3
    epochs: int = 2
    max_length: int = 512
    stride: int = 256
    lambda_div: float = 0.1
    batch_size: int = 8


class MemSlotSaliency:
    """End-to-end wrapper: frozen RoBERTa + trainable MemSlot.

    The class never sees questions or gold answers. It only reads raw
    contract strings from the CUAD *train split* for training; at inference
    it produces `(word, weight)` lists consumable by ``render/render.py``.
    """

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        config: MemSlotConfig = MemSlotConfig(),
        device: str = "cuda",
    ):
        from transformers import AutoModel, AutoTokenizer

        self.cfg = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=True, add_prefix_space=True)
        self.backbone = AutoModel.from_pretrained(backbone_name).to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        d_model = self.backbone.config.hidden_size
        self.memslot = MemSlotAttention(d_model=d_model, n_slots=config.n_slots).to(device)

    # ------------------------------------------------------------------ training

    def _embed(self, texts: list[str]) -> list[dict]:
        """Slice each context into overlapping windows; return encoder inputs."""
        records = []
        for text in texts:
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.cfg.max_length,
                stride=self.cfg.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
            for i in range(len(enc["input_ids"])):
                records.append({
                    "input_ids": torch.tensor(enc["input_ids"][i]),
                    "attention_mask": torch.tensor(enc["attention_mask"][i]),
                })
        return records

    def train_on_contracts(self, contexts: Iterable[str], verbose: bool = True):
        contexts = list(contexts)
        records = self._embed(contexts)
        if verbose:
            print(f"[MemSlot] training on {len(records)} windows from {len(contexts)} contracts")
        opt = torch.optim.AdamW(self.memslot.parameters(), lr=self.cfg.lr)

        self.memslot.train()
        for epoch in range(self.cfg.epochs):
            perm = torch.randperm(len(records))
            running = 0.0
            n_batches = 0
            for i in range(0, len(perm), self.cfg.batch_size):
                idx = perm[i : i + self.cfg.batch_size]
                ids = torch.stack([records[j]["input_ids"] for j in idx]).to(self.device)
                mask = torch.stack([records[j]["attention_mask"] for j in idx]).to(self.device).bool()
                with torch.no_grad():
                    H = self.backbone(input_ids=ids, attention_mask=mask.long()).last_hidden_state
                H_hat, attn, S = self.memslot(H, mask=mask)
                loss_rec = ((H - H_hat) ** 2 * mask.unsqueeze(-1)).sum() / (mask.sum() * H.shape[-1]).clamp(min=1)
                M_norm = F.normalize(self.memslot.slots, dim=-1)
                gram = M_norm @ M_norm.T
                loss_div = ((gram - torch.eye(self.cfg.n_slots, device=gram.device)) ** 2).mean()
                loss = loss_rec + self.cfg.lambda_div * loss_div
                opt.zero_grad(); loss.backward(); opt.step()
                running += loss.item(); n_batches += 1
            if verbose:
                print(f"  epoch {epoch + 1}/{self.cfg.epochs}  loss={running / max(n_batches,1):.4f}")
        self.memslot.eval()

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def word_weights(self, context: str, smooth_sigma: float = 2.0) -> list[tuple[str, float]]:
        """Question-agnostic word-level saliency for `render_tsvr_image`.

        Windows the text the same way as training, fuses overlapping window
        scores via max, then maps back to whitespace-delimited words via
        byte offsets, finally Gaussian smooths along word position.
        """
        import numpy as np

        enc = self.tokenizer(
            context,
            truncation=True,
            max_length=self.cfg.max_length,
            stride=self.cfg.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # per-char saliency so overlapping windows merge naturally
        char_score = np.zeros(len(context), dtype=np.float32)
        char_hit = np.zeros(len(context), dtype=np.bool_)
        for w in range(len(enc["input_ids"])):
            ids = torch.tensor(enc["input_ids"][w]).unsqueeze(0).to(self.device)
            mask = torch.tensor(enc["attention_mask"][w]).unsqueeze(0).to(self.device).bool()
            H = self.backbone(input_ids=ids, attention_mask=mask.long()).last_hidden_state
            s = self.memslot.saliency(H, mask=mask).squeeze(0).cpu().numpy()  # (L,)
            for tok_idx, (a, b) in enumerate(enc["offset_mapping"][w]):
                if b <= a:
                    continue
                char_score[a:b] = np.maximum(char_score[a:b], s[tok_idx])
                char_hit[a:b] = True

        # aggregate to words (whitespace split) via max-pool over their char range
        pairs: list[tuple[str, float]] = []
        i = 0
        while i < len(context):
            if context[i].isspace():
                i += 1
                continue
            j = i
            while j < len(context) and not context[j].isspace():
                j += 1
            word = context[i:j]
            score = float(char_score[i:j].max()) if char_hit[i:j].any() else 0.0
            pairs.append((word, score))
            i = j

        # Gaussian smoothing along word position
        if pairs and smooth_sigma > 0:
            scores = np.array([s for _, s in pairs], dtype=np.float32)
            radius = max(1, int(3 * smooth_sigma))
            kx = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (kx / smooth_sigma) ** 2); kernel /= kernel.sum()
            smoothed = np.convolve(scores, kernel, mode="same")
            # rescale so the visual layer still uses the full [0,1] range
            mn, mx = smoothed.min(), smoothed.max()
            smoothed = (smoothed - mn) / (mx - mn + 1e-9)
            pairs = [(w, float(s)) for (w, _), s in zip(pairs, smoothed)]

        return pairs

    # ------------------------------------------------------------------ persistence
    def save(self, path: str):
        torch.save({"state_dict": self.memslot.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.memslot.load_state_dict(ckpt["state_dict"])
