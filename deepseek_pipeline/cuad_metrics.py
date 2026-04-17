"""CUAD-style evaluation metrics.

The official CUAD evaluator (``github.com/TheAtticusProject/cuad``) scores a
predicted span against every gold annotation using Jaccard similarity, then
summarizes the whole test set with:

* **AUPR** – area under the precision-recall curve (predictions sorted by
  the reader's confidence, true positive iff Jaccard ≥ τ).
* **Precision @ 80 % Recall** – the precision attained once a confidence
  threshold admits 80 % of the gold-answerable questions.
* Plain SQuAD **EM / F1** on answerable items.

We preserve that structure here.  Because an API reader rarely exposes token
log-probabilities, we accept an explicit ``confidence`` per prediction (the
notebook elicits a 1-5 self-report from the reader and normalises it to
[0, 1]). If confidence is absent, AUPR collapses to mean precision at the
single operating point and we flag that in the output.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


# ----- helpers -------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def jaccard(pred: str, gold: str) -> float:
    p, g = set(_normalize(pred).split()), set(_normalize(gold).split())
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    return len(p & g) / len(p | g)


def squad_f1(pred: str, gold: str) -> float:
    pt, gt = _normalize(pred).split(), _normalize(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if not same:
        return 0.0
    prec = same / len(pt); rec = same / len(gt)
    return 2 * prec * rec / (prec + rec)


# ----- core evaluator ------------------------------------------------------

@dataclass
class CUADScore:
    em: float
    f1: float
    aupr: float
    precision_at_80_recall: float
    n: int
    threshold_note: str


def cuad_evaluate(
    predictions: list[str],
    golds: list[list[str]],
    confidences: Optional[list[float]] = None,
    jaccard_threshold: float = 0.5,
) -> CUADScore:
    """Evaluate one bucket (e.g. one compression method).

    * ``predictions``  – model answers (empty string means "no answer").
    * ``golds``        – list of gold spans per question; empty list means
                         the question is truly unanswerable.
    * ``confidences``  – optional [0, 1] self-reported confidence per item.

    A prediction is a **true positive** iff Jaccard(pred, some-gold) ≥ τ on
    an answerable item.  An empty prediction on an answerable item is a
    false negative.  An empty gold list with a non-empty prediction is a
    false positive.
    """
    assert len(predictions) == len(golds)
    n = len(predictions)
    if confidences is None:
        confidences = [1.0 if p.strip() else 0.0 for p in predictions]
        note = "confidence unavailable; AUPR computed with binary signal"
    else:
        note = "AUPR computed from reader self-reported confidence"
        assert len(confidences) == n

    # per-item score
    ems, f1s, tps = [], [], []
    is_answerable = []
    for pred, gold_list in zip(predictions, golds):
        answerable = bool(gold_list)
        is_answerable.append(answerable)
        if not answerable:
            ems.append(float(not pred.strip()))
            f1s.append(float(not pred.strip()))
            tps.append(not pred.strip())
            continue
        em_i = max(float(_normalize(pred) == _normalize(g)) for g in gold_list)
        f1_i = max(squad_f1(pred, g) for g in gold_list)
        tp_i = max(jaccard(pred, g) for g in gold_list) >= jaccard_threshold
        ems.append(em_i); f1s.append(f1_i); tps.append(bool(tp_i))

    em_mean = float(np.mean(ems))
    f1_mean = float(np.mean(f1s))

    # PR curve over answerable items, sorted by descending confidence
    order = np.argsort(-np.asarray(confidences))
    tp_sorted = np.asarray(tps)[order]
    answ_sorted = np.asarray(is_answerable)[order]

    n_gold = int(sum(is_answerable))
    if n_gold == 0:
        return CUADScore(em_mean, f1_mean, float("nan"), float("nan"), n, "no answerable items")

    tp_cum = np.cumsum(tp_sorted & answ_sorted)
    pred_cum = np.cumsum(answ_sorted.astype(int))        # only count answerable predictions
    # guard against empty rank prefixes
    precision = np.where(pred_cum > 0, tp_cum / np.maximum(pred_cum, 1), 1.0)
    recall = tp_cum / n_gold

    # AUPR via trapezoidal integration on (recall, precision)
    order_r = np.argsort(recall)
    aupr = float(np.trapz(precision[order_r], recall[order_r]))

    # interpolated precision @ 80% recall
    if recall.max() < 0.8:
        p_at_80 = 0.0
    else:
        p_at_80 = float(precision[recall >= 0.8].max())

    return CUADScore(em=em_mean, f1=f1_mean, aupr=aupr,
                     precision_at_80_recall=p_at_80, n=n, threshold_note=note)
