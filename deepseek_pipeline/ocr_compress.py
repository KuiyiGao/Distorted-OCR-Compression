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
import sys
from dataclasses import dataclass
from typing import Optional

from .metrics import RESOLUTION_MODES, vision_token_count_for_mode


# ── Symbols removed in transformers ≥ 4.46 that DeepSeek-OCR's remote code imports ──
_COMPAT_FALLBACKS = {
    "LlamaFlashAttention2": (
        "    from transformers.models.llama.modeling_llama import LlamaAttention as LlamaFlashAttention2"
    ),
    "LlamaSdpaAttention": (
        "    from transformers.models.llama.modeling_llama import LlamaAttention as LlamaSdpaAttention"
    ),
    "is_torch_fx_available":    "    is_torch_fx_available = lambda *a, **kw: False",
    "is_torchdynamo_compiling": "    is_torchdynamo_compiling = lambda *a, **kw: False",
    "is_torch_fx_proxy":        "    is_torch_fx_proxy = lambda *a, **kw: False",
}

# Preamble injected at the top of the cached remote-code file
_COMPAT_PREAMBLE = """\
# ===== auto-compat patch for transformers >= 4.46 =====
try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention)
except (ImportError, AttributeError):
    from transformers.models.llama.modeling_llama import LlamaAttention
    LlamaFlashAttention2 = LlamaAttention
    LlamaSdpaAttention = LlamaAttention
try:
    from transformers.utils.import_utils import (
        is_torch_fx_available, is_torchdynamo_compiling, is_torch_fx_proxy)
except (ImportError, AttributeError):
    is_torch_fx_available = lambda *a, **kw: False
    is_torchdynamo_compiling = lambda *a, **kw: False
    is_torch_fx_proxy = lambda *a, **kw: False
# ===== end auto-compat patch =====
"""


def _patch_hf_remote_code() -> None:
    """Patch modeling_deepseekocr.py in the HF modules cache for transformers >= 4.46.

    Strategy:
      1. Prepend a safe-preamble that defines all needed names with try/except fallbacks.
      2. Wrap every import block from transformers that contains a known-removed name
         in ``try: ... except (ImportError, AttributeError): pass`` so the original
         imports can succeed on older transformers and fall back gracefully on newer ones.
      3. Delete stale .pyc files and evict the module from sys.modules so the next
         AutoModel.from_pretrained re-imports from the patched source.
    """
    import pathlib, re

    _BAD = frozenset(_COMPAT_FALLBACKS.keys())

    cache = pathlib.Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    for target in cache.glob("**/modeling_deepseekocr.py"):
        try:
            src = target.read_text("utf-8")
        except OSError:
            continue
        if "auto-compat patch" in src:
            continue  # already patched

        lines = src.splitlines(keepends=True)
        out: list[str] = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            # Detect start of a 'from transformers...' import statement
            if re.match(r"\s*from\s+transformers", ln) and "import" in ln:
                block = [ln]
                j = i + 1
                # Collect multi-line paren import block
                if "(" in ln and ")" not in ln:
                    while j < len(lines):
                        block.append(lines[j])
                        if ")" in lines[j]:
                            j += 1
                            break
                        j += 1
                block_src = "".join(block)
                bad_here = [n for n in _BAD if re.search(r"\b" + re.escape(n) + r"\b", block_src)]
                if bad_here:
                    # Wrap the block: preamble already defines fallbacks, so except: pass is safe
                    indented = "".join("    " + bl for bl in block)
                    out.append(
                        f"try:\n{indented}except (ImportError, AttributeError):\n    pass\n"
                    )
                else:
                    out.extend(block)
                i = j
            else:
                out.append(ln)
                i += 1

        target.write_text(_COMPAT_PREAMBLE + "".join(out), "utf-8")

    # Delete stale pyc files so Python recompiles from patched source
    for pyc in cache.glob("**/__pycache__/modeling_deepseekocr*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass

    # Evict cached remote-code module so next import uses the patched file
    for k in list(sys.modules):
        if "transformers_modules" in k or (
            "deepseek" in k.lower() and "deepseek_pipeline" not in k
        ):
            del sys.modules[k]


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

        # ── In-memory patches: set all removed symbols on the live module objects
        # BEFORE from_pretrained executes the remote modeling code.
        # modeling_deepseekocr.py runs with the same sys.modules as this process,
        # so patching transformers.utils.import_utils here propagates into it.
        import transformers.models.llama.modeling_llama as _llama
        import transformers.utils as _tu
        import transformers.utils.import_utils as _triu

        _false = lambda *a, **kw: False  # noqa: E731

        for _sym, _stub in (
            ("LlamaFlashAttention2", _llama.LlamaAttention),
            ("LlamaSdpaAttention",   _llama.LlamaAttention),
        ):
            if not hasattr(_llama, _sym):
                setattr(_llama, _sym, _stub)

        for _sym in ("is_torch_fx_available", "is_torchdynamo_compiling", "is_torch_fx_proxy"):
            for _mod in (_tu, _triu):
                if not hasattr(_mod, _sym):
                    setattr(_mod, _sym, _false)
        # ─────────────────────────────────────────────────────────────────────

        self._torch = torch
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

        # Download tokenizer (also triggers download of all remote .py files)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Patch the now-downloaded modeling file on disk, then evict stale imports
        _patch_hf_remote_code()

        backend_chain: list[str] = []
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
            except (ImportError, ValueError) as e:
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
                              # text when eval_mode=True, otherwise returns None.
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
