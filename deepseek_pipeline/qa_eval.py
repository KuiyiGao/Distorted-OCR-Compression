"""CUAD QA evaluator backed by a chat-completion API.

We use the same reader for every compression branch so that the only
variable between branches is the compressed context.

>>> ### RESERVE API KEY: set DEEPSEEK_API_KEY or DASHSCOPE_API_KEY before use.
"""
from __future__ import annotations

import os
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass
class QAPrediction:
    question: str
    prediction: str
    gold_answers: list[str]
    em: float
    f1: float
    confidence: float = 0.5


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def squad_em_f1(pred: str, golds: Iterable[str]) -> tuple[float, float]:
    """SQuAD-style EM and F1; CUAD golds can be an empty list (unanswerable)."""
    golds = list(golds)
    if not golds:
        return (float(pred.strip() == ""), float(pred.strip() == ""))

    em = max(float(_normalize(pred) == _normalize(g)) for g in golds)

    def _f1(p: str, g: str) -> float:
        pt, gt = _normalize(p).split(), _normalize(g).split()
        if not pt or not gt:
            return float(pt == gt)
        common = Counter(pt) & Counter(gt)
        same = sum(common.values())
        if same == 0:
            return 0.0
        prec = same / len(pt)
        rec = same / len(gt)
        return 2 * prec * rec / (prec + rec)

    f1 = max(_f1(pred, g) for g in golds)
    return em, f1


class ApiQAReader:
    DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
    QWEN_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self, provider: str = "deepseek", model_name: str | None = None):
        self.provider = provider
        if provider == "deepseek":
            self.api_key_env = "DEEPSEEK_API_KEY"
            self.endpoint = self.DEEPSEEK_ENDPOINT
            self.model_name = model_name or "deepseek-chat"
        elif provider in ("qwen", "qwen-long"):
            self.api_key_env = "DASHSCOPE_API_KEY"
            self.endpoint = self.QWEN_ENDPOINT
            # qwen-long supports ~10M-token inputs; qwen-plus caps at ~32k.
            self.model_name = model_name or ("qwen-long" if provider == "qwen-long" else "qwen-plus")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def answer(self, context: str, question: str) -> tuple[str, float]:
        """Return ``(answer, confidence)``; confidence is read from the
        model's own JSON reply and normalised to [0, 1]. Used for the CUAD
        AUPR / precision@recall computation.
        """
        import json as _json
        import requests

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} not set.")

        sys_msg = (
            "You are a legal-contract QA reader. Given a (possibly compressed) contract excerpt and a "
            "question, reply with a JSON object of exactly two fields: "
            '{"answer": <shortest exact-span answer or "" if unanswerable>, '
            '"confidence": <integer 1-5 where 5 is certain>}. No commentary.'
        )
        user_msg = f"Contract:\n{context}\n\nQuestion: {question}"

        resp = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        try:
            parsed = _json.loads(raw)
            ans = str(parsed.get("answer", "")).strip()
            conf = float(parsed.get("confidence", 3)) / 5.0
        except Exception:
            # Fall back to raw text with a neutral confidence, but keep the
            # value readable rather than dumping the raw JSON string.
            m = re.search(r'"answer"\s*:\s*"([^"]*)"', raw)
            ans = (m.group(1) if m else raw).strip()
            conf = 0.5
        return ans, max(0.0, min(1.0, conf))

    def score(self, context: str, question: str, gold_answers: list[str]) -> QAPrediction:
        pred, conf = self.answer(context, question)
        em, f1 = squad_em_f1(pred, gold_answers)
        out = QAPrediction(question=question, prediction=pred, gold_answers=gold_answers, em=em, f1=f1)
        out.confidence = conf  # type: ignore[attr-defined]
        return out
