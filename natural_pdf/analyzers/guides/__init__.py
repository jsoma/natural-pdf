"""Compatibility shim — the canonical package is :mod:`natural_pdf.guides`.

Kept permanently so existing ``natural_pdf.analyzers.guides`` imports continue
to work. Submodules (``base``, ``ocr``, ...) are thin stubs that alias the
canonical modules in ``sys.modules``.
"""

from natural_pdf.guides import (
    GuideCells,
    GuideColumns,
    GuideOCRPlan,
    GuideOCRPlanningOptions,
    GuideOCRResult,
    GuideRows,
    Guides,
    GuidesList,
)

__all__ = [
    "GuideCells",
    "GuideColumns",
    "GuideOCRPlan",
    "GuideOCRPlanningOptions",
    "GuideOCRResult",
    "GuideRows",
    "Guides",
    "GuidesList",
]
