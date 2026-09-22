# Pipeline and evaluation status

## Data flow

Contract text → frozen RoBERTa features → learned MemSlot attention → word selection and rendering → DeepSeek-OCR decoded text → a separate text QA reader.

MemSlot uses reconstruction and slot-diversity losses without question or answer labels. The intended data split is by contract. Question-agnostic saliency alone does not establish absence of split leakage; a completed run needs its exact split, data, checkpoint, and output records. The earlier `run.py` uses question-conditioned saliency and is a different experimental setting.

The text branches prune words or request a summary before the same QA-reader stage. Word budgets, tokenizer counts, rendered-page counts, OCR visual-token estimates, and decoded-text token counts are different quantities. They must be recorded separately. A common tokenizer is a counting convention, not a shared decoder or a measured memory equivalence.

## Local span-QA diagnostics

`deepseek_pipeline/cuad_metrics.py` retains the `cuad_evaluate(predictions, golds, confidences=None, jaccard_threshold=0.5)` API for notebook callers. It is not the official CUAD evaluator.

- Each item contains a prediction and a list of acceptable nonempty gold spans. An empty gold list means unanswerable; a blank prediction is an abstention.
- EM and token F1 are averaged over all items. An abstention on an unanswerable item receives one for each.
- A nonblank prediction is a true positive when a gold span exists and normalized word-set Jaccard reaches the threshold. Other attempted answers, including answers to unanswerable items, remain in the precision denominator.
- Recall divides by all answerable items, including incorrect answers and abstentions. Each complete confidence tie group defines one threshold; missing confidence gives one tied operating point.
- Area uses trapezoidal integration from `(recall=0, precision=1)` to the observed operating points, with no extension beyond achieved recall. `precision_at_80_recall` is the maximum observed precision at recall ≥ 0.8, or zero if unreachable.
- With no answerable items, precision–recall statistics are undefined (`NaN`). An empty batch also has undefined EM/F1.

Reader confidence is self-reported and is not a calibrated probability. These local definitions and the tie convention can differ from official benchmark aggregation.

## Maintenance scope

The offline tests verify metric edge cases, deterministic word pruning, and OCR wrapper contracts using fixtures. Gaussian smoothing now uses a centered full-convolution slice to keep the correct positions and output length when the word sequence is shorter than the kernel. The wrapper no longer rewrites downloaded model files or installs placeholder Transformers classes. A missing decoded-text return raises an error rather than silently becoming the string `None`.

`n_vision_tokens` is a mode-based estimate from `metrics.py`; dynamic crops are not instrumented. No end-to-end run, OCR compatibility matrix, GPU execution, QA gain, or compression result was established by this maintenance work. Archived notebooks and their saved outputs have not been rerun.
