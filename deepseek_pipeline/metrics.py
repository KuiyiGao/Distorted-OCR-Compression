from __future__ import annotations

RESOLUTION_MODES: dict[str, dict] = {
    "tiny": {"image_size": 512, "vision_tokens": 64, "base_size": 512, "crop_mode": False},
    "small": {"image_size": 640, "vision_tokens": 100, "base_size": 640, "crop_mode": False},
    "base": {"image_size": 1024, "vision_tokens": 256, "base_size": 1024, "crop_mode": False},
    "large": {"image_size": 1280, "vision_tokens": 400, "base_size": 1280, "crop_mode": False},
    "gundam": {"image_size": 1024, "vision_tokens": 795, "base_size": 1024, "crop_mode": True},
}


def vision_token_count_for_mode(mode: str, n_crops: int = 1) -> int:
    mode = mode.lower()
    if mode == "gundam":
        return n_crops * 100 + 256
    return RESOLUTION_MODES[mode]["vision_tokens"]


def compression_ratio_tokens(n_original_tokens: int, n_compressed_tokens: int) -> float:
    return n_original_tokens / max(n_compressed_tokens, 1)


def compression_ratio_memory(
    n_original: int,
    n_compressed: int,
    d_original: int = 4096,
    d_compressed: int = 4096,
    dtype_bytes: int = 2,
) -> dict:
    bytes_original = n_original * d_original * dtype_bytes
    bytes_compressed = n_compressed * d_compressed * dtype_bytes
    return {
        "bytes_original": bytes_original,
        "bytes_compressed": bytes_compressed,
        "ratio": bytes_original / max(bytes_compressed, 1),
    }
