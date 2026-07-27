"""Guides: guide detection engines plus the Guides grid/table framework."""

from natural_pdf.engine_registry import register_guides_engine

from .base import Guides, GuidesList
from .guides_provider import GuidesDetectionResult, register_guides_engines, run_guides_detect
from .ocr import (
    GuideCells,
    GuideColumns,
    GuideOCRPlan,
    GuideOCRPlanningOptions,
    GuideOCRResult,
    GuideRows,
)

__all__ = [
    "GuideCells",
    "GuideColumns",
    "GuideOCRPlan",
    "GuideOCRPlanningOptions",
    "GuideOCRResult",
    "GuideRows",
    "Guides",
    "GuidesDetectionResult",
    "GuidesList",
    "register_guides_engine",
    "register_guides_engines",
    "run_guides_detect",
]
