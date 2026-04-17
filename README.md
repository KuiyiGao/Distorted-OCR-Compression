# Distorted-OCR-Compression

Compress long CUAD legal contracts through [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) by first rendering each contract as a **question-agnostic saliency-distorted image** (large/red = important), then letting DeepSeek-OCR's vision encoder pack the page into 64–400 vision tokens. The downstream reader answers CUAD questions from those tokens, and we compare against two text-side baselines (pruning / summarization) on the same reader, scored with CUAD-official metrics.

**Leakage-free by construction.** Saliency is produced by a MemSlot attention module on top of frozen RoBERTa, trained *only* on train-split contract text — no questions, no answers, no clause labels. Evaluation is on held-out test contracts. See [docs/METHODOLOGY.md §2](docs/METHODOLOGY.md).

## Layout
| Path | What it is |
|---|---|
| [run.py](run.py) | Legacy LegalBERT-TSVR + distorted-render pipeline. **Biased** (uses the question during saliency); kept for reference only. |
| [render/](render/) | TSVR rendering primitives (reused by the new pipeline). |
| [deepseek_pipeline/](deepseek_pipeline/) | MemSlot saliency, DeepSeek-OCR wrapper, pruning & summarization baselines, CUAD-style QA scorer. |
| [notebooks/DeepSeek_OCR_CUAD_Colab.ipynb](notebooks/DeepSeek_OCR_CUAD_Colab.ipynb) | End-to-end notebook for Colab A100. |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Bilingual (EN / 中文) math + implementation details. |

## Compression metric (see [docs/METHODOLOGY.md §2](docs/METHODOLOGY.md))
The proposed *memory* metric is only legitimate when every arm feeds the **same** downstream decoder; otherwise $D_{\text{vis}} \neq D_{\text{text}}$ and you are comparing unlike hidden spaces. We enforce a shared decoder (DeepSeek-3B-MoE or the DeepSeek-chat API), at which point memory ratio collapses to the **token-count ratio**

$$\text{ratio} = N_{\text{original text tokens}} / N_{\text{compressed tokens}}.$$

The notebook asserts this equality explicitly.

## Hardware
M1 **cannot** run DeepSeek-OCR — it needs CUDA, FlashAttention-2, bfloat16. Use the Colab notebook on A100/L4. Saliency + rendering (stages 1–2) still run fine on M1 if you want to prepare PNGs locally.

## API keys (reserved)
Set exactly one before running the baseline / QA-eval cells:
- `DEEPSEEK_API_KEY` — preferred; matches DeepSeek-OCR's decoder family.
- `DASHSCOPE_API_KEY` — Qwen fallback.

Read in `deepseek_pipeline/baselines.py` and `deepseek_pipeline/qa_eval.py` via `os.environ`; you never edit source code.

## Baselines (no training required)
- **Saliency pruning** (`SaliencyPruner`): keep top-$k$% words by MemSlot weight.
- **API summarization** (`ApiSummarizer`): compress via chat API to a target token budget matching DeepSeek-OCR's vision-token sizes. The question is **not** passed in, keeping the arm question-agnostic.

## Evaluation
CUAD-official metrics via [`cuad_evaluate`](deepseek_pipeline/cuad_metrics.py): EM, F1, **AUPR**, **Precision @ 80 % Recall**. Confidence is read from the QA reader's JSON reply.

## Running
```bash
# Full unbiased pipeline (Colab A100): MemSlot train → render → OCR/prune/summary → CUAD eval
#   open notebooks/DeepSeek_OCR_CUAD_Colab.ipynb
```
