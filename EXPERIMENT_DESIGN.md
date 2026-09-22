# Multipage OCR experiment plan

This is a proposed comparison associated with `notebooks/DeepSeek_OCR_CUAD_Multipage_Colab.ipynb`. It is not evidence of a completed end-to-end evaluation. The notebooks retain their original outputs and environment assumptions; [current pipeline contracts](docs/METHODOLOGY.md) take precedence over older descriptions.

## Question and branches

Does saliency-guided text selection and rendering preserve useful contract information after OCR, at a constrained text budget?

The planned branches include full text, saliency pruning, API summarization, lead/random selection, and OCR of selected or summarized text. Trained and randomly initialized MemSlot variants are intended as a control for learning the slot representation. Frozen RoBERTa features still contain structure in the untrained variant, so it is not equivalent to uniform random selection.

Each OCR branch renders pages, decodes them with DeepSeek-OCR, concatenates decoded text, and supplies that text to a separate QA reader. OCR's internal visual tokens are not passed directly to that reader.

## Controls needed for a comparison

- Split at the contract level and preserve document IDs, sampled question IDs, and random seeds. Train saliency only on training-contract text.
- Use the same QA model, version, prompt, and response handling for each branch. Keep questions out of compression and saliency selection.
- Report actual decoded-text length under one counting tokenizer, along with word budgets, page counts, and estimated OCR visual tokens. A target ratio is not an achieved ratio or a measured memory reduction.
- Separate effects of word selection, font emphasis, page boundaries, and OCR errors. Shared OCR settings alone do not make different rendered inputs a controlled comparison.
- Record failed calls separately from valid empty answers. Preserve attempt counts, outputs, and costs before reporting aggregate quality.
- Record exact dependencies, model revisions, checkpoint identities, and data hashes. Python's built-in string hash is not a portable random seed.

## Evaluation and unfinished work

The package provides local EM/F1 and precision–recall diagnostics, not the official CUAD evaluator. The QA reader's self-reported confidence is not a calibrated probability. Saved notebook figures are qualitative artifacts and do not establish an advantage over the text baselines.

The complete comparison still requires a verified environment, held-out split, full output records, achieved-length comparisons, and uncertainty assessed at the contract level. Model compatibility, provider context limits, prices, and quotas must be checked before an authorized run. No training, OCR inference, or paid API experiment was performed during repository maintenance.
