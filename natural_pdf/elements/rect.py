"""
Rectangle element class for natural-pdf.
"""

from typing import TYPE_CHECKING, Any, Dict, Tuple

from natural_pdf.elements.base import Element

if TYPE_CHECKING:
    from natural_pdf.core.page import Page


class RectangleElement(Element):
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

    def extract_text(
        self,
        preserve_whitespace: bool = True,
        apply_exclusions: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Extract text from inside this rectangle.

        Args:
            preserve_whitespace: Whether to keep blank characters (default: True)
            apply_exclusions: Whether to apply exclusion regions (default: True)
            **kwargs: Additional extraction parameters

        Returns:
            Extracted text as string
        """
        # Backward compatibility alias
        if "use_exclusions" in kwargs:
            import warnings

            warnings.warn(
                "use_exclusions is deprecated, use apply_exclusions instead",
                DeprecationWarning,
                stacklevel=2,
            )
            apply_exclusions = kwargs.pop("use_exclusions")
        # Use the region to extract text
        from natural_pdf.elements.region import Region

        region = Region(self.page, self.bbox)
        return region.extract_text(
            preserve_whitespace=preserve_whitespace,
            apply_exclusions=apply_exclusions,
            **kwargs,
        )

    def __repr__(self) -> str:
        """String representation of the rectangle element."""
        return f"<RectangleElement fill={self.fill} stroke={self.stroke} bbox={self.bbox}>"
