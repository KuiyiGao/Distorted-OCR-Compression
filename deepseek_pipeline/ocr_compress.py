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
    DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-OCR",
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        revision: Optional[str] = None,
    ):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self._torch = torch
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, revision=revision)
        self.model = AutoModel.from_pretrained(
            model_id, _attn_implementation=attn_implementation,
            trust_remote_code=True, use_safetensors=True,
            torch_dtype=torch_dtype, device_map={"": device}, revision=revision)
        self.model.eval()
        self.device = device
        self.attn_implementation = attn_implementation

    def compress(
        self,
        image_path: str,
        mode: str = "base",
        prompt: Optional[str] = None,
        output_dir: str = "./ocr_out",
    ) -> OCRCompressionResult:
        os.makedirs(output_dir, exist_ok=True)
        cfg = RESOLUTION_MODES[mode.lower()]
        decoded = self.model.infer(
            self.tokenizer, prompt=prompt or self.DEFAULT_PROMPT,
            image_file=image_path, output_path=output_dir,
            base_size=cfg["base_size"], image_size=cfg["image_size"],
            crop_mode=cfg["crop_mode"], save_results=False,
            test_compress=False, eval_mode=True)
        if isinstance(decoded, (list, tuple)):
            decoded = decoded[0] if decoded else ""
        if not isinstance(decoded, str):
            raise RuntimeError("DeepSeek-OCR did not return decoded text")
        return OCRCompressionResult(decoded, vision_token_count_for_mode(mode), mode, image_path)
