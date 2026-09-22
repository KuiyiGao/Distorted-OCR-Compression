from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class CompressedText:
    text: str
    n_tokens: int
    method: str


class SaliencyPruner:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compress(self, word_weights: list[tuple[str, float]], keep_ratio: float) -> CompressedText:
        if not math.isfinite(keep_ratio) or not 0 <= keep_ratio <= 1:
            raise ValueError("keep_ratio must be finite and in [0, 1]")
        if any(not isinstance(word, str) or not word.strip() for word, _ in word_weights):
            raise ValueError("words must be nonempty strings")
        weights = [float(weight) for _, weight in word_weights]
        if not all(math.isfinite(weight) for weight in weights):
            raise ValueError("saliency weights must be finite")
        n_keep = int(len(word_weights) * keep_ratio)
        if not n_keep:
            return CompressedText(text="", n_tokens=0, method=f"prune@{keep_ratio:.2f}")
        ranking = sorted(range(len(word_weights)), key=lambda index: (-weights[index], index))
        kept = [word_weights[index][0] for index in sorted(ranking[:n_keep])]
        text = " ".join(kept)
        n_tok = len(self.tokenizer.encode(text, add_special_tokens=False))
        return CompressedText(text=text, n_tokens=n_tok, method=f"prune@{keep_ratio:.2f}")


class ApiSummarizer:
    DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
    QWEN_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self, tokenizer, provider: str = "deepseek"):
        self.tokenizer = tokenizer
        self.provider = provider
        if provider == "deepseek":
            self.api_key_env = "DEEPSEEK_API_KEY"
            self.endpoint = self.DEEPSEEK_ENDPOINT
            self.model_name = "deepseek-chat"
        elif provider == "qwen":
            self.api_key_env = "DASHSCOPE_API_KEY"
            self.endpoint = self.QWEN_ENDPOINT
            self.model_name = "qwen-plus"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def compress(self, context: str, target_tokens: int, question: Optional[str] = None) -> CompressedText:
        import requests

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} not set. Please export the key before running the summarizer."
            )

        guidance = (
            f"You are compressing a legal contract for a downstream QA reader. "
            f"Produce a faithful summary of at most {target_tokens} tokens. "
            f"Preserve named parties, dates, dollar amounts, defined terms, and clause labels verbatim."
        )
        if question:
            guidance += f" The reader will be asked: {question!r}. Emphasize clauses relevant to that question but do not answer it."

        resp = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": guidance},
                    {"role": "user", "content": context},
                ],
                "temperature": 0.0,
                "max_tokens": int(target_tokens * 1.25),
            },
            timeout=120,
        )
        resp.raise_for_status()
        summary = resp.json()["choices"][0]["message"]["content"]
        n_tok = len(self.tokenizer.encode(summary, add_special_tokens=False))
        return CompressedText(text=summary, n_tokens=n_tok, method=f"summary@{target_tokens}")
