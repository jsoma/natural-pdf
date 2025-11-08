from typing import TYPE_CHECKING, Any, Dict, Tuple

from natural_pdf.elements.base import Element

if TYPE_CHECKING:
    from natural_pdf.core.page import Page


class ImageElement(Element):
    """Represents a raster XObject (embedded image) on a PDF page."""

    def __init__(self, obj: Dict[str, Any], page: "Page"):
        super().__init__(obj, page)

    # ------------------------------------------------------------------
    # Simple attribute proxies
    # ------------------------------------------------------------------
    @property
    def type(self) -> str:  # noqa: D401 – short description already given
        return "image"

    @property
    def width(self) -> float:  # override just to use dict value directly
        return float(self._obj.get("width", 0))

    @property
    def height(self) -> float:
        return float(self._obj.get("height", 0))

    @property
    def srcsize(self) -> Tuple[float, float]:
        """Original pixel dimensions of the embedded image (width, height)."""
        value = self._obj.get("srcsize")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            width_raw, height_raw = value
            width = float(width_raw) if width_raw is not None else 0.0
            height = float(height_raw) if height_raw is not None else 0.0
            return (width, height)
        return (0.0, 0.0)

    @property
    def colorspace(self):  # raw pdfminer data
        return self._obj.get("colorspace")

    # No text extraction for images
    def extract_text(self, *args, **kwargs) -> str:  # noqa: D401 – consistent signature
        return ""

    def __repr__(self):
        return f"<ImageElement bbox={self.bbox} srcsize={self.srcsize}>"
