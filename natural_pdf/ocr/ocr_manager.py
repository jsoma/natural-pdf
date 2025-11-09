"""Backward compatibility shim for legacy OCR manager imports."""

from .ocr_provider import (
    ENGINE_REGISTRY,
    OCRRunResult,
    cleanup_engine,
    infer_engine_from_options,
    list_available_engines,
    normalize_ocr_options,
    register_ocr_engines,
    resolve_ocr_device,
    resolve_ocr_engine_name,
    resolve_ocr_languages,
    resolve_ocr_min_confidence,
    run_ocr_apply,
    run_ocr_engine,
    run_ocr_extract,
)

__all__ = [
    "ENGINE_REGISTRY",
    "OCRRunResult",
    "cleanup_engine",
    "infer_engine_from_options",
    "list_available_engines",
    "normalize_ocr_options",
    "register_ocr_engines",
    "resolve_ocr_device",
    "resolve_ocr_engine_name",
    "resolve_ocr_languages",
    "resolve_ocr_min_confidence",
    "run_ocr_apply",
    "run_ocr_engine",
    "run_ocr_extract",
]
