"""
Analyzers for natural_pdf.
"""

from typing import TYPE_CHECKING

# Import these directly as they don't depend on Region
from natural_pdf.analyzers.guides import Guides

# ShapeDetectionMixin is lazy-loaded via __getattr__ below to avoid
# pulling in scipy/sklearn at import time.
from natural_pdf.analyzers.text_options import TextStyleOptions
from natural_pdf.analyzers.text_structure import TextStyleAnalyzer

if TYPE_CHECKING:
    from natural_pdf.analyzers.layout.layout_analyzer import LayoutAnalyzer
    from natural_pdf.analyzers.layout.layout_options import LayoutOptions
    from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin


# Lazy imports to avoid circular dependencies and heavy deps (scipy/sklearn)
def __getattr__(name):
    if name == "LayoutAnalyzer":
        from natural_pdf.analyzers.layout.layout_analyzer import LayoutAnalyzer

        return LayoutAnalyzer
    elif name == "LayoutOptions":
        from natural_pdf.analyzers.layout.layout_options import LayoutOptions

        return LayoutOptions
    elif name == "ShapeDetectionMixin":
        from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin

        return ShapeDetectionMixin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LayoutAnalyzer",
    "LayoutOptions",
    "ShapeDetectionMixin",
    "TextStyleOptions",
    "TextStyleAnalyzer",
    "Guides",
]
