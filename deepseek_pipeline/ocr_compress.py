"""Thin wrapper around DeepSeek-OCR.

Hardware note: the upstream model requires CUDA + bfloat16. It will
*not* run on Apple-Silicon MPS or CPU. Target: Colab A100-40G (also
works on A10G / L4 with reduced batch).

Attention backend: defaults to PyTorch's built-in SDPA, which already
dispatches to FlashAttention-2 kernels on Ampere/Ada/Hopper GPUs in
bf16/fp16 — so no separate `flash-attn` wheel needs to be compiled.
Pass ``attn_implementation="flash_attention_2"`` to force the external
package (only useful if you have a matching prebuilt wheel) or
``"eager"`` to fall back to the pure-PyTorch path.
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
        attn_implementation: str = "sdpa",
    ):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self._torch = torch
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Try the requested backend, then fall back to widely-available ones so
        # users on bleeding-edge torch versions (no prebuilt flash-attn wheel)
        # are not blocked by an hour-long source build.
        backend_chain = []
        for b in (attn_implementation, "sdpa", "eager"):
            if b not in backend_chain:
                backend_chain.append(b)
        last_err: Optional[Exception] = None
        for backend in backend_chain:
            try:
                self.model = AutoModel.from_pretrained(
                    model_id,
                    _attn_implementation=backend,
                    trust_remote_code=True,
                    use_safetensors=True,
                )
                self.attn_implementation = backend
                break
            except (ImportError, ValueError) as e:  # flash_attn missing, etc.
                last_err = e
                continue
        else:
            raise RuntimeError(
                f"Could not load DeepSeek-OCR with any of {backend_chain}: {last_err}"
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
            eval_mode=True,   # REQUIRED: upstream infer() only returns decoded
                              # text when eval_mode=True, otherwise it prints
                              # to stdout and returns None.
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
