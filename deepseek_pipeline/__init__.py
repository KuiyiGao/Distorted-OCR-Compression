from importlib import import_module

_EXPORTS = {
    "metrics": ["vision_token_count_for_mode", "compression_ratio_tokens",
                "compression_ratio_memory", "RESOLUTION_MODES"],
    "ocr_compress": ["DeepSeekOCRCompressor"],
    "baselines": ["SaliencyPruner", "ApiSummarizer"],
    "qa_eval": ["ApiQAReader", "squad_em_f1"],
    "memslot": ["MemSlotAttention", "MemSlotSaliency", "MemSlotConfig"],
    "cuad_metrics": ["cuad_evaluate", "CUADScore", "jaccard", "squad_f1"],
}
_MODULES = {name: module for module, names in _EXPORTS.items() for name in names}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
