"""
Rectangle element class for natural-pdf.
"""

from typing import TYPE_CHECKING, Any, Dict, Tuple

from natural_pdf.elements.base import Element
from natural_pdf.text.contracts import ExtractedText, TextLayoutOptions
from natural_pdf.text.facades import SpatialTextMixin
from natural_pdf.text.pipeline import extract_spatial_text

if TYPE_CHECKING:
    from natural_pdf.core.page import Page


class RectangleElement(SpatialTextMixin, Element):
    """
    Represents a rectangle element in a PDF.

    This class is a wrapper around pdfplumber's rectangle objects,
    providing additional functionality for analysis and extraction.
    """

    def __init__(self, obj: Dict[str, Any], page: "Page"):
        """
        Initialize a rectangle element.

        Args:
            obj: The underlying pdfplumber object
            page: The parent Page object
        """
        super().__init__(obj, page)

    @property
    def type(self) -> str:
        """Element type."""
        return "rect"

    @property
    def fill(self) -> Tuple:
        """Get the fill color of the rectangle (RGB tuple)."""
        from natural_pdf.utils.color_utils import normalize_pdf_color

        return normalize_pdf_color(self._obj.get("non_stroking_color"))

    @property
    def stroke(self) -> Tuple:
        """Get the stroke color of the rectangle (RGB tuple)."""
        from natural_pdf.utils.color_utils import normalize_pdf_color

        return normalize_pdf_color(self._obj.get("stroking_color"))

    @property
    def stroke_width(self) -> float:
        """Get the stroke width of the rectangle."""
        return self._obj.get("linewidth", 0)

    @property
    def is_horizontal(self) -> bool:
        """Check if this is a horizontal line based on coordinates."""
        # Calculate absolute difference in coordinates
        dx = abs(self.x1 - self.x0)
        dy = abs(self.top - self.bottom)

        # Define a tolerance for near-horizontal lines (e.g., 1 point)
        tolerance = 1.0

        # Horizontal if y-change is within tolerance and x-change is significant
        return dy <= tolerance and dx > tolerance

    @property
    def is_vertical(self) -> bool:
        """Check if this is a vertical line based on coordinates."""
        # Calculate absolute difference in coordinates
        dx = abs(self.x1 - self.x0)
        dy = abs(self.top - self.bottom)

        # Define a tolerance for near-vertical lines (e.g., 1 point)
        tolerance = 1.0

        # Vertical if x-change is within tolerance and y-change is significant
        return dx <= tolerance and dy > tolerance

    @property
    def orientation(self) -> str:
        """Get the orientation of the line ('horizontal', 'vertical', or 'diagonal')."""
        if self.is_horizontal:
            return "horizontal"
        elif self.is_vertical:
            return "vertical"
        return "diagonal"

    @property
    def text(self) -> str:
        """Get text content inside this rectangle (delegates to extract_text())."""
        return self.extract_text() or ""

    def _extract_spatial_text_result(
        self,
        *,
        layout: bool | TextLayoutOptions,
        apply_exclusions: bool,
    ) -> ExtractedText:
        """Acquire rectangle text through Region geometry with this source identity."""

        from natural_pdf.elements.region import Region

        region = Region(self.page, self.bbox)
        return extract_spatial_text(
            region._spatial_text_input(apply_exclusions=apply_exclusions, source=self),
            layout=layout,
        )

    def __repr__(self) -> str:
        """String representation of the rectangle element."""
        return f"<RectangleElement fill={self.fill} stroke={self.stroke} bbox={self.bbox}>"
