import torch
import torch.nn as nn
import torch.nn.functional as F

class MemSlotModel(nn.Module):
    def __init__(self, d_model=768, num_slots=32, lambda_cos=0.1, tau_init=0.5):
        super().__init__()
        self.C = num_slots
        self.d = d_model
        self.lambda_cos = lambda_cos
        self.eps = 1e-4

        self.theta = nn.Parameter(torch.full((num_slots, 1), float(tau_init)))

        self.W_vr = nn.Linear(d_model, d_model, bias=False)
        self.W_sr = nn.Linear(d_model, d_model)
        self.W_vz = nn.Linear(d_model, d_model, bias=False)
        self.W_sz = nn.Linear(d_model, d_model)
        self.W_v  = nn.Linear(d_model, d_model, bias=False)
        self.W_s  = nn.Linear(d_model, d_model)

        self.register_buffer('S_0', torch.randn(1, num_slots, d_model) * 0.01)

    def init_slots_from_question(self, question_hidden, noise=0.15):
        # Mix question token embeddings with noise so slots stay diverse
        Nq = question_hidden.shape[0]
        if Nq >= self.C:
            slots = question_hidden[:self.C].clone()
        else:
            reps = (self.C + Nq - 1) // Nq
            slots = question_hidden.repeat(reps, 1)[:self.C].clone()
        scale = slots.norm(dim=1, keepdim=True).mean()
        slots = slots + torch.randn_like(slots) * float(noise) * scale
        self.S_0 = slots.unsqueeze(0).detach()

    def forward_attention(self, S, H):
        S_n = F.normalize(S, dim=2)
        H_n = F.normalize(H, dim=2)
        tau = F.softplus(self.theta) + self.eps
        A = F.softmax(torch.bmm(S_n, H_n.transpose(1, 2)) / tau, dim=2)
        V = torch.bmm(A, H)
        return A, V

    def gru_update(self, S, V):
        B, C, d = S.shape
        s = S.reshape(B * C, d)
        v = V.reshape(B * C, d)
        R = torch.sigmoid(self.W_vr(v) + self.W_sr(s))
        Z = torch.sigmoid(self.W_vz(v) + self.W_sz(s))
        S_cand = torch.tanh(self.W_v(v) + self.W_s(R * s))
        S_new = (1 - Z) * s + Z * S_cand
        return S_new.reshape(B, C, d)

    def cosine_penalty(self, S):
        S_n = F.normalize(S, dim=2)
        G = torch.bmm(S_n, S_n.transpose(1, 2))
        mask = 1.0 - torch.eye(self.C, device=S.device).unsqueeze(0)
        return ((G * mask) ** 2).sum() / (S.shape[0] * self.C * (self.C - 1))

    def forward(self, hidden_states_list):
        B = hidden_states_list[0].shape[0]
        S = self.S_0.expand(B, -1, -1).to(hidden_states_list[0].device)

        attn_maps = []
        total_cos = 0.0
        for H_t in hidden_states_list:
            A, V = self.forward_attention(S, H_t)
            S = self.gru_update(S, V)
            total_cos += self.cosine_penalty(S)
            # Per-token: max across slots, min-max scaled to [0,1] for clean reweighting
            w = A.max(dim=1)[0]
            mn = w.min(dim=1, keepdim=True)[0]
            mx = w.max(dim=1, keepdim=True)[0]
            w = (w - mn) / (mx - mn + self.eps)
            attn_maps.append(w)

        return attn_maps, total_cos / len(hidden_states_list)


def question_token_relevance(hidden_states, question_pooled, sequence_ids):
    """
    kappa(t) — cosine similarity between each context token's contextual
    embedding and the pooled question embedding. Pooled question embedding
    is the L2-normalised mean of question-token hidden states.

    Returns: 1-D tensor of length T (sequence length). Non-context positions
    are zeroed out so they cannot leak into the stitched word signal.
    """
    H = hidden_states[0] if hidden_states.dim() == 3 else hidden_states  # [T, d]
    q = F.normalize(question_pooled, dim=-1)                              # [d]
    Hn = F.normalize(H, dim=-1)                                           # [T, d]
    sim = torch.clamp(Hn @ q, min=0.0)                                    # [T] in [0,1]
    seq = torch.tensor(sequence_ids, device=H.device)
    sim = sim * (seq == 1).float()
    return sim


def qa_token_saliency(start_logits, end_logits, sequence_ids, top_k=15, max_span=40):
    # Primary saliency from the CUAD-fine-tuned QA head: aggregate top-k spans
    # into per-token coverage scores. Boundary signal added on top.
    #
    # IMPORTANT: do NOT min-max normalise per chunk. The raw probability scale
    # is what distinguishes "this chunk contains the answer" (peak ~0.3-0.9)
    # from "no answer here" (peak ~1e-4). Per-chunk normalisation re-saturates
    # every chunk to 1.0 and destroys global ranking when stitching.
    seq = torch.tensor(sequence_ids, device=start_logits.device)
    ctx = (seq == 1)
    sl = start_logits.masked_fill(~ctx, -1e9)
    el = end_logits.masked_fill(~ctx, -1e9)

    sp = F.softmax(sl, dim=-1)
    ep = F.softmax(el, dim=-1)
    n = sp.shape[0]
    k = min(top_k, int(ctx.sum().item()) or 1)

    sk = torch.topk(sp, k)
    ek = torch.topk(ep, k)

    sal = torch.zeros(n, device=sp.device)
    for si, sv in zip(sk.indices.tolist(), sk.values.tolist()):
        for ei, ev in zip(ek.indices.tolist(), ek.values.tolist()):
            if si <= ei and (ei - si) < max_span:
                sal[si:ei + 1] += sv * ev

    # Scale boundary term to the same magnitude as the span-coverage sum so it
    # *augments* (does not dominate) the in-span signal.
    span_max = sal.max().clamp_min(1e-8)
    boundary = 0.5 * span_max * (sp + ep) / (sp + ep).max().clamp_min(1e-8)
    sal = sal + boundary
    sal = sal.masked_fill(~ctx, 0.0)
    return sal
