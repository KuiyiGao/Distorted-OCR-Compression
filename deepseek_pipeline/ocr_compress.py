"""Thin wrapper around DeepSeek-OCR.

Hardware note: the upstream model requires CUDA + FlashAttention-2 +
bfloat16. It will *not* run on Apple-Silicon MPS or CPU. Target: Colab
A100-40G (also works on A10G / L4 with reduced batch).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .metrics import RESOLUTION_MODES, vision_token_count_for_mode


@dataclass
class OCRCompressionResult:
    decoded_text: str
    n_vision_tokens: int
    mode: str
    image_path: str


class DeepSeekOCRCompressor:
    """Loads deepseek-ai/DeepSeek-OCR and compresses an image into text.

    Vision tokens are the *compression unit* — we count them from the
    resolution mode (static per DeepSeek's spec table). We keep the decoded
    text so the downstream QA reader can consume it directly.
    """

    DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-OCR",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self._torch = torch
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model = self.model.eval().to(device).to(torch_dtype)
        self.device = device

    def compress(
        self,
        image_path: str,
        mode: str = "base",
        prompt: Optional[str] = None,
        output_dir: str = "./ocr_out",
    ) -> OCRCompressionResult:
        os.makedirs(output_dir, exist_ok=True)
        cfg = RESOLUTION_MODES[mode.lower()]
        prompt = prompt or self.DEFAULT_PROMPT

        decoded = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_dir,
            base_size=cfg["base_size"],
            image_size=cfg["image_size"],
            crop_mode=cfg["crop_mode"],
            save_results=False,
            test_compress=False,
        )
        if isinstance(decoded, (list, tuple)):
            decoded = decoded[0] if decoded else ""

        n_vis = vision_token_count_for_mode(mode)
        return OCRCompressionResult(
            decoded_text=str(decoded),
            n_vision_tokens=n_vis,
            mode=mode,
            image_path=image_path,
        )
