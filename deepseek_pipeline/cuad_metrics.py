from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Optional
import numpy as np


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
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
    same = sum((Counter(pt) & Counter(gt)).values())
    if not same:
        return 0.0
    precision, recall = same / len(pt), same / len(gt)
    return 2 * precision * recall / (precision + recall)


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
    n = len(predictions)
    if len(golds) != n:
        raise ValueError("predictions and golds must have equal lengths")
    if not np.isfinite(jaccard_threshold) or not 0 < jaccard_threshold <= 1:
        raise ValueError("jaccard_threshold must be in (0, 1]")
    if any(not isinstance(p, str) for p in predictions):
        raise ValueError("each prediction must be a string")
    if any(not isinstance(gs, (list, tuple)) or
           any(not isinstance(g, str) or not g.strip() for g in gs) for gs in golds):
        raise ValueError("each gold entry must be a list of nonempty spans, or []")
    if confidences is None:
        confidence = np.ones(n)
        note = "local diagnostic; confidence unavailable; one tied operating point"
    else:
        confidence = np.asarray(confidences, dtype=float)
        if confidence.shape != (n,) or not np.isfinite(confidence).all():
            raise ValueError("confidences must be a finite one-dimensional value per item")
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("confidences must be in [0, 1]")
        note = "local diagnostic; supplied confidence; ties grouped by threshold"

    if not n:
        return CUADScore(float("nan"), float("nan"), float("nan"),
                         float("nan"), 0, note + "; empty batch")
    ems, f1s, true_positives = [], [], []
    attempts = np.asarray([bool(p.strip()) for p in predictions])
    n_gold = sum(bool(gs) for gs in golds)
    for pred, spans in zip(predictions, golds):
        if not spans:
            ems.append(float(not pred.strip()))
            f1s.append(float(not pred.strip()))
            true_positives.append(False)
        else:
            ems.append(max(float(_normalize(pred) == _normalize(g)) for g in spans))
            f1s.append(max(squad_f1(pred, g) for g in spans))
            true_positives.append(bool(pred.strip()) and
                                  max(jaccard(pred, g) for g in spans) >= jaccard_threshold)
    em, f1 = float(np.mean(ems)), float(np.mean(f1s))
    if n_gold == 0:
        return CUADScore(em, f1, float("nan"), float("nan"), n,
                         note + "; no answerable items")
    if not attempts.any():
        return CUADScore(em, f1, 0.0, 0.0, n, note + "; all predictions abstained")

    order = np.argsort(-confidence[attempts], kind="stable")
    sorted_confidence = confidence[attempts][order]
    sorted_tp = np.asarray(true_positives, dtype=int)[attempts][order]

    ends = np.r_[np.flatnonzero(np.diff(sorted_confidence) != 0), len(order) - 1]
    tp = np.cumsum(sorted_tp)[ends]
    precision = tp / (ends + 1)
    recall = tp / n_gold
    curve_precision = np.r_[1.0, precision]
    curve_recall = np.r_[0.0, recall]

    aupr = float(np.sum(np.diff(curve_recall) *
                        (curve_precision[:-1] + curve_precision[1:]) / 2))
    p80 = float(precision[recall >= 0.8].max()) if np.any(recall >= 0.8) else 0.0
    return CUADScore(em, f1, aupr, p80, n, note)
