"""Public helpers for guide detection engines."""

from natural_pdf.engine_registry import register_guides_engine

from .guides_provider import GuidesDetectionResult, register_guides_engines, run_guides_detect

__all__ = [
    "GuidesDetectionResult",
    "register_guides_engine",
    "register_guides_engines",
    "run_guides_detect",
]
