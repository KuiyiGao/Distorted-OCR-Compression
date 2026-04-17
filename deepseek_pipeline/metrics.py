"""Compression metrics for DeepSeek-OCR vs text-token baselines.

Key insight on the user's proposed "vision-token memory vs text-token memory"
metric: memory = N_tokens * D_hidden * bytes_per_elem. Comparing across
models with different D_hidden hides the real quantity we care about — how
many context slots the downstream reader has to spend. The legitimate
comparison requires a **shared downstream decoder** so D_hidden cancels.
In DeepSeek-OCR the decoder is DeepSeek3B-MoE; vision tokens and text
tokens both enter that same embedding space, so:

    memory_ratio = (N_vis * D) * dtype / (N_text * D) * dtype = N_vis / N_text

Therefore when the decoder is fixed, token-count ratio and memory ratio are
numerically identical. We report both for transparency.
"""
from __future__ import annotations

RESOLUTION_MODES: dict[str, dict] = {
    "tiny": {"image_size": 512, "vision_tokens": 64, "base_size": 512, "crop_mode": False},
    "small": {"image_size": 640, "vision_tokens": 100, "base_size": 640, "crop_mode": False},
    "base": {"image_size": 1024, "vision_tokens": 256, "base_size": 1024, "crop_mode": False},
    "large": {"image_size": 1280, "vision_tokens": 400, "base_size": 1280, "crop_mode": False},
    "gundam": {"image_size": 1024, "vision_tokens": 795, "base_size": 1024, "crop_mode": True},
}


def vision_token_count_for_mode(mode: str, n_crops: int = 1) -> int:
    """Static vision-token count for a resolution mode.

    For 'gundam' the effective count is n_crops * 100 + 256 for the global
    view; pass n_crops from the actual crop planner (1 if unknown).
    """
    mode = mode.lower()
    if mode == "gundam":
        return n_crops * 100 + 256
    return RESOLUTION_MODES[mode]["vision_tokens"]


def compression_ratio_tokens(n_original_tokens: int, n_compressed_tokens: int) -> float:
    """How many original text tokens are packed into one compressed token."""
    return n_original_tokens / max(n_compressed_tokens, 1)


def compression_ratio_memory(
    n_original: int,
    n_compressed: int,
    d_original: int = 4096,
    d_compressed: int = 4096,
    dtype_bytes: int = 2,
) -> dict:
    """Memory-based compression ratio.

    When d_original == d_compressed (the decoder is shared), this reduces to
    the token-count ratio. We still return both raw byte figures so users
    can see the units.
    """
    bytes_original = n_original * d_original * dtype_bytes
    bytes_compressed = n_compressed * d_compressed * dtype_bytes
    return {
        "bytes_original": bytes_original,
        "bytes_compressed": bytes_compressed,
        "ratio": bytes_original / max(bytes_compressed, 1),
    }
