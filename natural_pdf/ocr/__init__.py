"""Public OCR helpers for natural-pdf."""

import logging

logger = logging.getLogger("natural_pdf.ocr")

from natural_pdf.engine_registry import register_ocr_engine

from .engine import OCREngine
from .ocr_manager import (
    cleanup_engine,
    infer_engine_from_options,
    normalize_ocr_options,
    resolve_ocr_device,
    resolve_ocr_engine_name,
    resolve_ocr_languages,
    resolve_ocr_min_confidence,
)
from .ocr_options import (
    BaseOCROptions,
    ChandraOCROptions,
    DoctrOCROptions,
    EasyOCROptions,
    OCROptions,
    PaddleOCROptions,
    PaddleOCRVLOptions,
    RapidOCROptions,
    SuryaOCROptions,
)
from .unified_dispatch import list_engines as _list_engine_entries

__all__ = [
    "OCREngine",
    "OCROptions",
    "BaseOCROptions",
    "ChandraOCROptions",
    "DoctrOCROptions",
    "EasyOCROptions",
    "PaddleOCRVLOptions",
    "PaddleOCROptions",
    "RapidOCROptions",
    "SuryaOCROptions",
    "register_ocr_engine",
    "cleanup_engine",
    "normalize_ocr_options",
    "infer_engine_from_options",
    "resolve_ocr_engine_name",
    "resolve_ocr_languages",
    "resolve_ocr_min_confidence",
    "resolve_ocr_device",
    "list_registered_engines",
]


def list_registered_engines() -> tuple[str, ...]:
    """Return registered OCR engine names."""

    return tuple(sorted(_list_engine_entries().keys()))
