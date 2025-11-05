"""
Line element class for natural-pdf.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from natural_pdf.elements.base import Element

if TYPE_CHECKING:
    from natural_pdf.core.page import Page


class LineElement(Element):
    """
    Represents a line element in a PDF.

    This class is a wrapper around pdfplumber's line objects,
    providing additional functionality for analysis and extraction.
    """

    def __init__(self, obj: Dict[str, Any], page: "Page"):
        """
        Initialize a line element.

        Args:
            obj: The underlying pdfplumber object
            page: The parent Page object
        """
        super().__init__(obj, page)

    @property
    def source(self) -> Optional[str]:
        """Get the source of this line element (e.g., 'pdf', 'detected')."""
        return self._obj.get("source")

    @property
    def type(self) -> str:
        """Element type."""
        return "line"

    @property
    def color(self) -> Tuple:
        """Get the line color (RGB tuple)."""
        # PDFs often use non-RGB values, so we handle different formats
        color = self._obj.get("stroking_color", (0, 0, 0))

        # If it's a single value, treat as grayscale
        if isinstance(color, (int, float)):
            return (color, color, color)

        # If it's a tuple of 3 values, treat as RGB
        if isinstance(color, tuple) and len(color) == 3:
            return color

        # If it's a tuple of 4 values, treat as CMYK and convert to approximate RGB
        if isinstance(color, tuple) and len(color) == 4:
            c, m, y, k = color
            r = 1 - min(1, c + k)
            g = 1 - min(1, m + k)
            b = 1 - min(1, y + k)
            return (r, g, b)

        # Default to black
        return (0, 0, 0)

    @property
    def width(self) -> float:
        """Get the line thickness (extracted from PDF properties)."""
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
        else:
            return "diagonal"

    def extract_text(
        self,
        preserve_whitespace: bool = True,
        use_exclusions: bool = True,
        **kwargs,
    ) -> str:
        """
        Lines don't have text, so this returns an empty string.

        Args:
            preserve_whitespace: Unused, kept for API compatibility with Element.
            use_exclusions: Unused, kept for API compatibility with Element.
            **kwargs: Additional extraction parameters (ignored).

        Returns:
            Empty string
        """
        # Backward compatibility: honour legacy keyword names if provided.
        if "keep_blank_chars" in kwargs:
            preserve_whitespace = kwargs.pop("keep_blank_chars")  # noqa: F841
        if "apply_exclusions" in kwargs:
            use_exclusions = kwargs.pop("apply_exclusions")  # noqa: F841
        return ""

    def __repr__(self) -> str:
        """String representation of the line element."""
        return f"<LineElement type={self.orientation} width={self.width:.1f} bbox={self.bbox}>"
