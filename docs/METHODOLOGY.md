# Methodology / 方法论

Bilingual — EN / 中文 per subsection.

---

## 1. Problem statement · 问题陈述

**EN.** CUAD contracts are long (avg. ~10k tokens). Our research question: *Can DeepSeek-OCR's vision channel, fed a saliency-distorted rendering, pack a contract into fewer context slots than text-side pruning or summarization while preserving QA quality?* A valid answer requires the saliency signal to be **unbiased** — it must not depend on the evaluation question.

**中文.** CUAD 合同平均约 10k token。问题：将**按显著性畸变渲染**的合同喂给 DeepSeek-OCR，能否比文本侧剪枝 / 摘要用更少的上下文槽位、并保持问答质量？要让结论成立，显著性信号必须**与评测问题无关**。

---

## 2. Unbiased saliency via MemSlot attention · 基于 MemSlot 的无偏显著性

### 2.1 What was wrong before · 旧方案的偏置

**EN.** The original `run.py` trained LegalBERT as a token classifier over `(question, context)` pairs and used the learned class-1 probability as saliency. The question leaked into the render, which biased every downstream arm (OCR, prune, summary). Comparisons across arms are therefore not apples-to-apples.

**中文.** 旧流程以 `(question, context)` 对训练 LegalBERT token 分类头，用正类概率作显著性；问题文本泄漏到渲染中，下游三路（OCR / 剪枝 / 摘要）全部受偏置影响，失去可比性。

### 2.2 New pipeline · 新流程

**EN.** Frozen `roberta-base` + trainable MemSlot cross-attention:

1. **Split CUAD at the contract level** into train / test. Test contracts are untouched throughout saliency training.
2. **Train the MemSlot on *train-split contexts only***, no questions, answers or clause labels consumed.
3. At inference, saliency is a function of the contract alone; every arm (OCR / prune / summary) starts from the same unbiased rendering.

**中文.** 冻结 `roberta-base`，仅训练 MemSlot 交叉注意力：先按合同拆分 train / test；只用 train 集合同做无监督训练；推理阶段的显著性仅是合同的函数，三路下游共享同一份无偏渲染。

### 2.3 Math · 数学形式

Let $H \in \mathbb{R}^{L\times d}$ be frozen RoBERTa hidden states and $M \in \mathbb{R}^{K\times d}$ the $K=32$ learnable slot vectors.

$$A_{k,i} = \operatorname{softmax}_i\!\Big(\tfrac{(W_q M)_k \cdot (W_k H)_i^\top}{\sqrt{d_p}}\Big),\quad S_k = \sum_i A_{k,i}\, W_v H_i.$$

Per-token reconstruction:

$$\hat H_i = \sum_k \operatorname{softmax}_k\!\Big(\tfrac{(W_{q'}H)_i\cdot (W_{k'}S)_k^\top}{\sqrt{d_p}}\Big) W_{v'}S_k.$$

Losses (unsupervised, no QA signal):

$$\mathcal{L} = \underbrace{\tfrac{1}{Ld}\lVert H-\hat H\rVert_F^{2}}_{\text{reconstruction}} \;+\; \lambda\, \underbrace{\big\lVert \tilde M\tilde M^{\top} - I_K\big\rVert_F^{2}}_{\text{slot diversity}},\quad \tilde M = \operatorname{row-norm}(M).$$

Saliency:

$$s_i = \max_k A_{k,i},\qquad w = \operatorname{GaussianSmooth}_{\sigma=2}(s)\in[0,1]^W.$$

---

## 3. Compression-rate metric · 压缩率指标

### 3.1 User's proposal

Memory ratio $\frac{N_{\text{orig}}D_{\text{orig}}b}{N_{\text{vis}}D_{\text{vis}}b}$.

### 3.2 Fix (recap from prior round)

**EN.** Different $D$ across models mixes units. We route every arm through **one shared decoder** (DeepSeek-3B-MoE inside DeepSeek-OCR for the vision path, DeepSeek-chat API for text arms — same tokenizer family). Under a shared decoder the memory ratio collapses to the token-count ratio; the notebook asserts this.

$$\text{ratio} = \frac{N_{\text{original text tokens}}}{N_{\text{compressed tokens}}}.$$

**中文.** 不同模型 $D$ 不同则量纲混乱。令所有路径共享同一下游解码器（视觉路用 DeepSeek-OCR 内置 3B-MoE，文本路用同家族的 DeepSeek-chat），此时显存比即 token 数比，notebook 中有 `np.allclose` 断言。

---

## 4. CUAD-official metrics · CUAD 官方指标

**EN.** Following `github.com/TheAtticusProject/cuad`, we report:

* **EM / F1** — SQuAD-style, per answerable item.
* **AUPR** — predictions sorted by confidence; true positive iff Jaccard $\ge 0.5$ against any gold span. Integrated by the trapezoidal rule over (recall, precision).
* **Precision @ 80 % Recall** — precision once the confidence threshold admits 80 % of gold-answerable items.

Confidence is read from the reader's JSON output (1–5 self-report, normalised to [0, 1]). If unavailable, AUPR collapses to the binary answered / not-answered signal and we flag that in `CUADScore.threshold_note`.

**中文.** 参考 CUAD 官方评测：EM / F1 照旧；AUPR 按置信度排序，Jaccard ≥ 0.5 记作 TP，梯形积分；P@80%R 取召回率达到 0.8 时的最大精度。置信度由阅读器 JSON 返回的 1–5 自评分归一化得到，如缺省则降级为二元信号并在结果里标注。

---

## 5. Pipeline stages · 流水线

| Stage | Input | Output | Consumes QA? |
|---|---|---|---|
| Split | CUADv1.json | train docs / test docs | no |
| MemSlot train | train contexts | `memslot.pt` | **no** |
| Saliency | test context | word weights $w$ | **no** |
| Distorted render | $w$ | PNG | **no** |
| OCR compress | PNG | markdown + vision-token count | no |
| Prune baseline | $w$ + context | top-$k$% text | **no** |
| Summary baseline | context | target-length summary (no question given) | **no** |
| QA read | compressed context + test question | answer + confidence | yes (eval only) |
| CUAD score | predictions, golds | EM / F1 / AUPR / P@80%R | n/a |

The **only** stage that ever sees a question is the final reader. Every compression arm is question-agnostic.

---

## 6. Hardware · 硬件

| | M1/M2 MacBook | Colab T4 | Colab A100 / L4 |
|---|---|---|---|
| CUDA | ❌ | ✔ | ✔ |
| FlashAttention-2 | ❌ | ❌ (Turing) | ✔ |
| bfloat16 | emulated | ❌ | ✔ |
| DeepSeek-OCR | **cannot run** | **cannot run** | **runs** |

MemSlot training is small and CPU-feasible in a pinch, but practical runtime is Colab A100.

**中文.** M1 上无法运行 DeepSeek-OCR（需 CUDA + FlashAttention2 + bf16）。MemSlot 训练本身很轻，可先在本地训好、再到 Colab 跑推理，或全流程在 Colab 完成。

---

## 7. API-key placeholders · API Key

| env var | consumer |
|---|---|
| `DEEPSEEK_API_KEY` | `ApiSummarizer`, `ApiQAReader` with provider=`deepseek` |
| `DASHSCOPE_API_KEY` | same two classes with provider=`qwen` |

Exactly one required. DeepSeek is preferred (same tokenizer family as DeepSeek-OCR decoder).

---

## 8. Known limitations · 已知限制

* MemSlot is unsupervised; slot semantics are discovered, not labelled. The notebook plots three top-activating slots so you can sanity-check qualitatively.
* AUPR uses a self-reported confidence; a reader that is dishonest about certainty will distort the curve. Swap in a local 7B reader with real logprobs for the paper submission if available.
* Vision-token counts come from DeepSeek-OCR's spec table (static per mode). Hook `model.vision_model.forward` to measure per-image if you need exactness.
* Default sweep in the notebook (2 OCR modes × 2 prune keeps × 2 summary targets × 30 QAs) keeps API usage low. Scale up for the final paper.
