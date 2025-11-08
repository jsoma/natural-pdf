from typing import TYPE_CHECKING

from .base import FinetuneExporter

if TYPE_CHECKING:
    from .paddleocr import PaddleOCRRecognitionExporter


# Lazy import for PaddleOCRRecognitionExporter to avoid heavy paddle dependencies at module level
def _get_paddleocr_exporter():
    """Lazy import for PaddleOCRRecognitionExporter."""
    from .paddleocr import PaddleOCRRecognitionExporter

    return PaddleOCRRecognitionExporter


# Make PaddleOCRRecognitionExporter available through attribute access
def __getattr__(name):
    if name == "PaddleOCRRecognitionExporter":
        return _get_paddleocr_exporter()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["FinetuneExporter", "PaddleOCRRecognitionExporter"]
