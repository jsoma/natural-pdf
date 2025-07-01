"""
Analyzers for natural_pdf.
"""

# Import these directly as they don't depend on Region
from natural_pdf.analyzers.guides import Guides
from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin
from natural_pdf.analyzers.text_options import TextStyleOptions
from natural_pdf.analyzers.text_structure import TextStyleAnalyzer


# Lazy imports to avoid circular dependencies
# These will be imported when actually accessed
def __getattr__(name):
    if name == "LayoutAnalyzer":
        from natural_pdf.analyzers.layout.layout_analyzer import LayoutAnalyzer

        return LayoutAnalyzer
    elif name == "LayoutManager":
        from natural_pdf.analyzers.layout.layout_manager import LayoutManager

        return LayoutManager
    elif name == "LayoutOptions":
        from natural_pdf.analyzers.layout.layout_options import LayoutOptions

        return LayoutOptions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LayoutAnalyzer",
    "LayoutManager",
    "LayoutOptions",
    "ShapeDetectionMixin",
    "TextStyleOptions",
    "TextStyleAnalyzer",
    "Guides",
]
