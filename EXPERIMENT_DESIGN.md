# Multi-Page A4 Experiment — Design Document

Companion to [`notebooks/DeepSeek_OCR_CUAD_Multipage_Colab.ipynb`](notebooks/DeepSeek_OCR_CUAD_Multipage_Colab.ipynb).

## 1. Research question

> At fixed 5× / 10× compression, does **DeepSeek-OCR on a MemSlot-saliency-driven
> rendering** preserve more CUAD QA signal than (a) direct text compression
> (prune / summary / random / lead) and (b) OCR applied to the text *output* of
> those same baselines?

The ablation in (b) is what isolates MemSlot. If `memslot-ocr` beats `prune`
but *ties* `prune-ocr` (same OCR pass on a different text source), the win is
coming from the OCR step, not MemSlot. If `memslot-ocr` beats `prune-ocr`, the
saliency step is doing real work.

## 2. Data

- **Source:** CUADv1 (Atticus Project). 510 commercial contracts, 41 clause
  types, ~10 k QAs, gold spans + `is_impossible` flags.
- **Doc-level split:** `random.seed(0)` shuffle, first 8 docs → test. No QA
  from a test doc is ever seen during MemSlot training.
- **Training subset:** 50 % of the remaining 502 docs (~245). MemSlot only
  consumes the *first 120* of those (≈ 2 epochs fit under 3 min on A100).
- **Test sampling:** per test doc, 2 answerable + 2 unanswerable QAs
  (`K_ANS=2, K_UNANS=2`). 8 × 4 = **32 QAs** spanning 8 unique contexts.
  The 50/50 answerable/unanswerable split is deliberate — CUAD's native
  ~15 % answerable rate lets a "always abstain" policy score EM=0.85 and
  masks real compression damage (the abstention floor failure mode from the
  previous notebook).

## 3. Compression protocol

One tokenizer (`roberta-base`) measures every method, so 5×/10× means the
same thing across branches.

For each test context with `N` original tokens and target ratio `R ∈ {5, 10}`:

| Branch | What it feeds the QA reader | Selection rule |
|---|---|---|
| `full` | original context | — (upper bound) |
| `prune-R` | top-`N/R` tokens (word-level) | MemSlot trained saliency, reading-order preserved |
| `selective-R` | top-`N/R` tokens by RoBERTa-MLM self-information | Selective-Context / LLMLingua-lite (Li 2023, Jiang 2023) |
| `summary-R` | API-summary ≤ `N/R` tokens | `deepseek-chat` → `qwen-plus` fallback |
| `random-R` | random sample of `N/R` text tokens | uniform, seeded by `(doc_hash, R)` |
| `lead-R` | first `N/R` tokens | — (truncation baseline) |
| `memslot-ocr-trained-R` | OCR decode of MemSlot-rendered A4 pages | trained slots → top-`B_words` → A4 pages → OCR |
| `memslot-ocr-untrained-R` | same, but MemSlot **never trained** | random-init slots (ablates training objective) |
| `summary-ocr-R` | OCR decode of summary text rendered on A4 | uniform font, no saliency |
| `prune-ocr-R` | OCR decode of pruned text rendered on A4 | uniform font, no saliency |

Total labels: `1 + 2 × 9 = 19` → about **608 QA reader calls** per run.

### 3.1 Why render-then-crop (not chunk-then-render)

DeepSeek-OCR resizes any input to a ~1024² patch grid. The legacy
`render_tsvr_image` produced a `1024 × ~30 000 px` scroll for an 11 k-token
contract; the resize crushed it to illegible noise and OCR decoded nearly
nothing — which is why the previous 5X10X notebook reported zero F1 on every
OCR branch.

A naive fix would be: split the kept words into page-sized chunks of ~350
words and render each chunk on its own A4 canvas with a font auto-shrunk to
fit. The problem is that every chunk then binary-searches its own font size,
so page 2's emphasis words can end up visually smaller than page 1's
background words. MemSlot's document-wide saliency signal is destroyed.

We do the opposite — **render once, then crop** — using the existing
`run.py` rendering pipeline (`render_img_tiered` + `assign_tiers` from
`render/`). Concretely:

1. MemSlot sees the whole passage (`word_weights(context)` processes the
   full context in one pass).
2. The top-`B_words = 0.75 × B_tokens` words (weight-ranked, reading-order
   preserved) are tiered by `assign_tiers(phi, primary_pct=98,
   secondary_pct=85)`: ~2 % primary (topic colour, s_max = 42 pt), ~13 %
   secondary (dark grey, interpolated s_mid = 28 pt), the rest tertiary
   (light grey, s_min = 16 pt). Same treatment the repo's distorted-
   rendering artifact uses.
3. `render_img_tiered` emits a 1024-wide tall image; the whole passage is
   on a single global font scale so relative emphasis is consistent.
4. The tall image is sliced vertically into 1024 × 1448 px A4 tiles. The
   last tile is padded with white to keep dimensions uniform. Each tile is
   its own OCR call; decoded text is concatenated in order.

This respects DeepSeek-OCR's ~1024² patch-grid constraint while preserving
every visual salience cue MemSlot computed.

The **text→render→OCR ablation** (`summary-ocr`, `prune-ocr`) uses plain
`render_img` with `s_min = s_max = 22` — uniform font size, no per-word
emphasis. Everything downstream of the text-selection step is pixel-
identical to the MemSlot branch, so any performance gap is attributable to
what was selected, not to the OCR pipeline.

### 3.2 Why the text→render→OCR ablation

`memslot-ocr-R` bundles two effects: (i) MemSlot decides *what* to render,
(ii) the OCR pipeline maps rendered text → vision tokens → decoded text.
`summary-ocr` and `prune-ocr` keep step (ii) identical but replace step (i)
with the text output of a different compressor. Any residual gap between
`memslot-ocr` and `{summary,prune}-ocr` at the same `R` is attributable to
MemSlot's saliency — not to OCR being a "magic compressor".

### 3.3 Why Selective-Context / LLMLingua-lite

`prune-R` uses MemSlot saliency — a *supervised-style* signal learned via
reconstruction on CUAD train contracts. To keep the comparison honest against
current text-compression literature, we also include a **zero-shot,
information-theoretic** baseline in the spirit of Selective-Context
(Li et al., EMNLP 2023) and LLMLingua (Jiang et al., EMNLP 2023):

1. Slide a 512-token window over the context (stride 256) through frozen
   `roberta-base` MLM.
2. For every token, take `-log p(token | bidirectional context)` — the
   self-information under the LM.
3. Collapse subword scores to word scores by max-pooling (robust to
   tokenization boundaries), Gaussian-smooth (σ=2 words) to suppress
   single-word spikes, min-max rescale to `[0,1]`.
4. Feed the resulting `(word, score)` list into the same `SaliencyPruner`
   that MemSlot uses, so `ratio_text` is computed identically.

This gives us a modern, training-free compression baseline that is
*architecturally comparable* to MemSlot (same tokenizer family, same
downstream pruner) but uses a completely different signal (token likelihood
vs. learned slot attention). If `prune-R` beats `selective-R`, MemSlot's
learned saliency is adding value beyond raw LM surprisal.

### 3.4 Why the no-training MemSlot variant

`MemSlotSaliency.word_weights()` only relies on the slot→token softmax. With
random-init slots it still produces a saliency signal (driven by RoBERTa's
frozen hidden states and the random slot projections). If training on CUAD
train contracts meaningfully improves selection, `memslot-ocr-trained` will
beat `memslot-ocr-untrained`. If not, the unsupervised reconstruction
objective is not buying us much over a pure RoBERTa-attention baseline —
useful to know.

## 4. QA reader

All branches feed the same reader, `qwen-long` (Alibaba DashScope, ~10 M
token context), `temperature=0`, `response_format=json_object`. A JSON reply
of `{"answer": ..., "confidence": 1-5}` is parsed; confidence is normalized
to `[0, 1]` for AUPR.

### 4.1 Why `qwen-long`

- Handles the `full` baseline (up to 36 k tokens in CUAD) without silent
  truncation, which broke `qwen-plus` (32 k cap) in the previous notebook.
- Same endpoint / key / prompt as `qwen-plus` so the compressed branches
  don't get a disadvantageous reader.
- Free-tier DashScope quota is enough for ~600 calls; no rate-limit hacks.

A preflight probe runs the reader against the *longest* test context before
the main loop; if the prediction is a verbatim prefix of the input, we abort
(the tell-tale sign of silent truncation).

### 4.2 Visual-vs-text token legitimacy

The QA reader is text-only. Visual tokens only matter upstream (as the
compression unit inside DeepSeek-OCR). For end-to-end QA accuracy we
compare methods on **decoded text tokens** — the same quantity each method
actually sends to the reader. The `ratio_vis` column (vision cost of the
OCR branches) is reported separately as metadata, not used for ranking.

## 5. Metrics

| Metric | Domain | What it catches |
|---|---|---|
| `success_rate` | all runs | API-health sanity; < 0.9 invalidates the row |
| `EM_ans` / `F1_ans` | answerable ∩ successful | span-level fidelity |
| `abstain_acc` | unanswerable ∩ successful | correct refusal when evidence is gone |
| `AUPR` | all successful | ranking quality via self-reported confidence (CUAD-official) |
| `P@80R` | all successful | operating-point precision at 80 % recall (CUAD-official) |
| `ratio_text` | all | actual text-token compression (`N_orig / n_compressed`) |
| `ratio_vis` | OCR branches | `N_orig / Σ vision_tokens` over all pages |

Failures are counted in `n_total - n_ok`, **never** folded into abstention.

## 6. Hardware

- **Target:** Colab A100-40G or H100-80G. Both run DeepSeek-OCR in bf16 with
  the SDPA attention kernel, which dispatches to FlashAttention-2 at runtime
  on Ampere/Ada/Hopper.
- **Training batch size** auto-scales: 24 if total VRAM > 30 GB, else 8.
- **MemSlot training** is the only gradient work; the OCR model is eval-only.
- A10G / L4 also works (smaller batches, ~2× slower). T4 will not run the
  OCR model in bf16.

## 7. Known limitations

1. **Discrete OCR modes.** DeepSeek-OCR exposes only `{tiny:64, small:100,
   base:256, large:400, gundam:795}` vision tokens per call. We use `base`
   for every page; total vision cost is a multiple of 256, so achieved
   ratios cluster rather than hitting 5.00× / 10.00× exactly.
2. **Summary length is a ceiling, not a floor.** The summarizer is told to
   produce ≤ `B` tokens but can return fewer; achieved `ratio_text` on the
   `summary-R` branches is often higher than `R`. Honest, not a bug — the
   absolute fidelity number is still comparable at the achieved ratio.
3. **32-QA sample.** Tight enough to run in < 30 min but wide confidence
   intervals on per-label metrics; treat ranking direction as the signal,
   not absolute gaps. Push `N_TEST_DOCS` to 20+ for publication-grade error
   bars.
4. **Reader ≠ OCR decoder.** The QA reader is an external text API. OCR's
   native advantage (a multimodal decoder that could skip the text round-trip)
   is not captured here; we are measuring QA fidelity *after* OCR → text
   conversion only.
5. **Untrained MemSlot is not "random"** in the strong sense — frozen
   RoBERTa still produces structured hidden states, which the random slot
   projections will preferentially re-weight. It is a soft ablation, not a
   uniform-random baseline (that role is filled by `random-R`).

## 8. Reproducibility

- `random.seed(0)` → document shuffle (identical to the legacy notebook).
- `rng = random.Random(1)` → QA sampling per doc.
- `hash((h, R)) & 0xffffffff` seed → `random-R` word selection.
- MemSlot checkpoint is saved to `runs/memslot_trained.pt`; rerun skips
  training if present.
- Rendered A4 pages are saved to `runs/renders/` for visual debugging.
- Every run row is written to `runs/cuad_multipage_results.csv` including
  the raw `prediction`, `gold`, `confidence`, `status`, and `error_msg`.

## 9. Artifacts produced

```
runs/
├── memslot_trained.pt           MemSlot checkpoint (reused across sessions)
├── cuad_multipage_results.csv   per-(method × ratio × QA) row-level results
├── multipage_compare.png        aggregate F1 / abstention / success / ratio bars
├── multipage_heatmap.png        per-QA F1 heatmap (answerable ∩ successful)
└── renders/                     every A4 page fed to OCR
    ├── {hash}_ms-tr_r5_p00.png
    ├── {hash}_ms-tr_r5_p01.png
    └── …
```
