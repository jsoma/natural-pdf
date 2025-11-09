"""Public OCR helpers for natural-pdf."""

import logging

logger = logging.getLogger("natural_pdf.ocr")

from natural_pdf.engine_registry import register_ocr_engine

from .engine import OCREngine
from .ocr_factory import OCRFactory
from .ocr_manager import ENGINE_REGISTRY, OCRRunResult, cleanup_engine, infer_engine_from_options
from .ocr_manager import list_available_engines as list_registered_engines
from .ocr_manager import (
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
from .ocr_options import (
    BaseOCROptions,
    EasyOCROptions,
    OCROptions,
    PaddleOCROptions,
    SuryaOCROptions,
)

__all__ = [
    "OCREngine",
    "OCROptions",
    "BaseOCROptions",
    "EasyOCROptions",
    "PaddleOCROptions",
    "SuryaOCROptions",
    "OCRFactory",
    "ENGINE_REGISTRY",
    "register_ocr_engine",
    "register_ocr_engines",
    "OCRRunResult",
    "run_ocr_apply",
    "run_ocr_engine",
    "run_ocr_extract",
    "cleanup_engine",
    "normalize_ocr_options",
    "infer_engine_from_options",
    "resolve_ocr_engine_name",
    "resolve_ocr_languages",
    "resolve_ocr_min_confidence",
    "resolve_ocr_device",
    "list_registered_engines",
    "get_engine",
    "list_available_engines",
]


def get_engine(engine_name=None, **kwargs):
    """
    Get OCR engine by name with graceful handling of missing dependencies.

    Args:
        engine_name: Name of the engine to use ('easyocr', 'paddle', 'surya', 'doctr')
                     If None, the best available engine is used
        **kwargs: Additional arguments to pass to the engine constructor

    Returns:
        OCREngine instance

    Raises:
        ImportError: If the requested engine's dependencies aren't installed
        ValueError: If the engine_name is unknown
    """
    logger.debug(f"Initializing OCR engine: {engine_name or 'best available'}")

    try:
        if engine_name is None or engine_name == "default":
            # Use the factory to get the best available engine
            engine = OCRFactory.get_recommended_engine(**kwargs)
            logger.info(f"Using recommended OCR engine: {engine.__class__.__name__}")
            return engine

        # Use the factory to create a specific engine
        normalized_name = engine_name.lower()
        if normalized_name in ["easyocr", "paddle", "surya", "doctr"]:
            return OCRFactory.create_engine(normalized_name, **kwargs)
        else:
            raise ValueError(f"Unknown OCR engine: {engine_name}")

    except ImportError as e:
        logger.error(f"OCR engine dependency error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error initializing OCR engine: {e}")
        raise


def list_available_engines():
    """
    List all available OCR engines.

    Returns:
        Dict[str, bool]: Dictionary mapping engine names to availability status
    """
    return OCRFactory.list_available_engines()
