"""Baselines: (1) saliency-based token pruning, (2) API-based summarization.

Both produce a compressed *text* string with a known token count, so the
downstream QA reader (qa_eval.ApiQAReader) can score them head-to-head
against the DeepSeek-OCR vision path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class CompressedText:
    text: str
    n_tokens: int
    method: str


class SaliencyPruner:
    """Keep the top-k fraction of words ranked by TSVR saliency score.

    Input: list of (word, weight) tuples produced by the existing
    render.utils.stitch_and_smooth_saliency pipeline.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compress(self, word_weights: list[tuple[str, float]], keep_ratio: float) -> CompressedText:
        if not word_weights:
            return CompressedText(text="", n_tokens=0, method=f"prune@{keep_ratio}")
        n_keep = max(1, int(len(word_weights) * keep_ratio))
        # Keep positional order so the reader sees a coherent(ish) excerpt.
        threshold = sorted((w for _, w in word_weights), reverse=True)[n_keep - 1]
        kept = [w for (w, s) in word_weights if s >= threshold]
        text = " ".join(kept)
        n_tok = len(self.tokenizer.encode(text, add_special_tokens=False))
        return CompressedText(text=text, n_tokens=n_tok, method=f"prune@{keep_ratio:.2f}")


class ApiSummarizer:
    """Summarize long context via a chat-completion API.

    Supported providers (set ``provider`` and put the key in env):

    * ``deepseek``  – needs ``DEEPSEEK_API_KEY``. Same model family as the
      DeepSeek-OCR decoder, giving the fairest baseline.
    * ``qwen``      – needs ``DASHSCOPE_API_KEY``.

    >>> ### RESERVE API KEY: set DEEPSEEK_API_KEY or DASHSCOPE_API_KEY
    """

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
