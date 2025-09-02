import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

from pdfplumber.utils.geometry import get_bbox_overlap, merge_bboxes, objects_to_bbox

# New Imports
from pdfplumber.utils.text import TEXTMAP_KWARGS, WORD_EXTRACTOR_KWARGS, chars_to_textmap
from tqdm.auto import tqdm

from natural_pdf.analyzers.checkbox.mixin import CheckboxDetectionMixin
from natural_pdf.analyzers.layout.pdfplumber_table_finder import find_text_based_tables

# --- Shape Detection Mixin --- #
from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin
from natural_pdf.classification.manager import ClassificationManager  # Keep for type hint

# --- Classification Imports --- #
from natural_pdf.classification.mixin import ClassificationMixin

# Add Visualizable import
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.describe.mixin import DescribeMixin
from natural_pdf.elements.base import DirectionalMixin
from natural_pdf.elements.text import TextElement  # ADDED IMPORT
from natural_pdf.extraction.mixin import ExtractionMixin  # Import extraction mixin
from natural_pdf.ocr.utils import _apply_ocr_correction_to_elements  # Import utility
from natural_pdf.selectors.parser import parse_selector, selector_to_filter_func

# ------------------------------------------------------------------
# Table utilities
# ------------------------------------------------------------------
from natural_pdf.tables import TableResult
from natural_pdf.text_mixin import TextMixin
from natural_pdf.utils.locks import pdf_render_lock  # Import the lock

# Import new utils
from natural_pdf.utils.text_extraction import filter_chars_spatially, generate_text_layout
from natural_pdf.vision.mixin import VisualSearchMixin

# Import viewer widget support
from natural_pdf.widgets.viewer import _IPYWIDGETS_AVAILABLE, InteractiveViewerWidget

# --- End Classification Imports --- #


# --- End Shape Detection Mixin --- #

if TYPE_CHECKING:
    # --- NEW: Add Image type hint for classification --- #
    from PIL.Image import Image

    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element  # Added for type hint
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.text import TextElement

# Import OCRManager conditionally to avoid circular imports
try:
    from natural_pdf.ocr import OCRManager
except ImportError:
    # OCRManager will be imported directly in methods that use it
    pass

logger = logging.getLogger(__name__)


class RegionContext:
    """Context manager for constraining directional operations to a region."""

    def __init__(self, region: "Region"):
        """Initialize the context manager with a region.

        Args:
            region: The Region to use as a constraint for directional operations
        """
        self.region = region
        self.previous_within = None

    def __enter__(self):
        """Enter the context, setting the global directional_within option."""
        import natural_pdf

        self.previous_within = natural_pdf.options.layout.directional_within
        natural_pdf.options.layout.directional_within = self.region
        return self.region

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, restoring the previous directional_within option."""
        import natural_pdf

        natural_pdf.options.layout.directional_within = self.previous_within
        return False  # Don't suppress exceptions


class Region(
    TextMixin,
    DirectionalMixin,
    ClassificationMixin,
    ExtractionMixin,
    ShapeDetectionMixin,
    CheckboxDetectionMixin,
    DescribeMixin,
    VisualSearchMixin,
    Visualizable,
):
    """Represents a rectangular region on a page.

    Regions are fundamental building blocks in natural-pdf that define rectangular
    areas of a page for analysis, extraction, and navigation. They can be created
    manually or automatically through spatial navigation methods like .below(), .above(),
    .left(), and .right() from elements or other regions.

    Regions integrate multiple analysis capabilities through mixins and provide:
    - Element filtering and collection within the region boundary
    - OCR processing for the region area
    - Table detection and extraction
    - AI-powered classification and structured data extraction
    - Visual rendering and debugging capabilities
    - Text extraction with spatial awareness

    The Region class supports both rectangular and polygonal boundaries, making it
    suitable for complex document layouts and irregular shapes detected by layout
    analysis algorithms.

    Attributes:
        page: Reference to the parent Page object.
        bbox: Bounding box tuple (x0, top, x1, bottom) in PDF coordinates.
        x0: Left x-coordinate.
        top: Top y-coordinate (minimum y).
        x1: Right x-coordinate.
        bottom: Bottom y-coordinate (maximum y).
        width: Region width (x1 - x0).
        height: Region height (bottom - top).
        polygon: List of coordinate points for non-rectangular regions.
        label: Optional descriptive label for the region.
        metadata: Dictionary for storing analysis results and custom data.

    Example:
        Creating regions:
        ```python
        pdf = npdf.PDF("document.pdf")
        page = pdf.pages[0]

        # Manual region creation
        header_region = page.region(0, 0, page.width, 100)

        # Spatial navigation from elements
        summary_text = page.find('text:contains("Summary")')
        content_region = summary_text.below(until='text[size>12]:bold')

        # Extract content from region
        tables = content_region.extract_table()
        text = content_region.get_text()
        ```

        Advanced usage:
        ```python
        # OCR processing
        region.apply_ocr(engine='easyocr', resolution=300)

        # AI-powered extraction
        data = region.extract_structured_data(MySchema)

        # Visual debugging
        region.show(highlights=['tables', 'text'])
        ```
    """

    def __init__(
        self,
        page: "Page",
        bbox: Tuple[float, float, float, float],
        polygon: List[Tuple[float, float]] = None,
        parent=None,
        label: Optional[str] = None,
    ):
        """Initialize a region.

        Creates a Region object that represents a rectangular or polygonal area on a page.
        Regions are used for spatial navigation, content extraction, and analysis operations.

        Args:
            page: Parent Page object that contains this region and provides access
                to document elements and analysis capabilities.
            bbox: Bounding box coordinates as (x0, top, x1, bottom) tuple in PDF
                coordinate system (points, with origin at bottom-left).
            polygon: Optional list of coordinate points [(x1,y1), (x2,y2), ...] for
                non-rectangular regions. If provided, the region will use polygon-based
                intersection calculations instead of simple rectangle overlap.
            parent: Optional parent region for hierarchical document structure.
                Useful for maintaining tree-like relationships between regions.
            label: Optional descriptive label for the region, useful for debugging
                and identification in complex workflows.

        Example:
            ```python
            pdf = npdf.PDF("document.pdf")
            page = pdf.pages[0]

            # Rectangular region
            header = Region(page, (0, 0, page.width, 100), label="header")

            # Polygonal region (from layout detection)
            table_polygon = [(50, 100), (300, 100), (300, 400), (50, 400)]
            table_region = Region(page, (50, 100, 300, 400),
                                polygon=table_polygon, label="table")
            ```

        Note:
            Regions are typically created through page methods like page.region() or
            spatial navigation methods like element.below(). Direct instantiation is
            used mainly for advanced workflows or layout analysis integration.
        """
        self._page = page
        self._bbox = bbox
        self._polygon = polygon

        self.metadata: Dict[str, Any] = {}
        # Analysis results live under self.metadata['analysis'] via property

        # Standard attributes for all elements
        self.object_type = "region"  # For selector compatibility

        # Layout detection attributes
        self.region_type = None
        self.normalized_type = None
        self.confidence = None
        self.model = None

        # Region management attributes
        self.name = None
        self.label = label
        self.source = None  # Will be set by creation methods

        # Hierarchy support for nested document structure
        self.parent_region = parent
        self.child_regions = []
        self.text_content = None  # Direct text content (e.g., from Docling)
        self.associated_text_elements = []  # Native text elements that overlap with this region

    def _get_render_specs(
        self,
        mode: Literal["show", "render"] = "show",
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        highlights: Optional[Union[List[Dict[str, Any]], bool]] = None,
        crop: Union[
            bool, int, str, "Region", Literal["wide"]
        ] = True,  # Default to True for regions
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> List[RenderSpec]:
        """Get render specifications for this region.

        Args:
            mode: Rendering mode - 'show' includes highlights, 'render' is clean
            color: Color for highlighting this region in show mode
            highlights: Additional highlight groups to show, or False to disable all highlights
            crop: Cropping mode:
                - False: No cropping
                - True: Crop to region bounds (default for regions)
                - int: Padding in pixels around region
                - 'wide': Full page width, cropped vertically to region
                - Region: Crop to the bounds of another region
            crop_bbox: Explicit crop bounds (overrides region bounds)
            **kwargs: Additional parameters

        Returns:
            List containing a single RenderSpec for this region's page
        """
        from typing import Literal

        spec = RenderSpec(page=self.page)

        # Handle cropping
        if crop_bbox:
            spec.crop_bbox = crop_bbox
        elif crop:
            x0, y0, x1, y1 = self.bbox

            if crop is True:
                # Crop to region bounds
                spec.crop_bbox = self.bbox
            elif isinstance(crop, (int, float)):
                # Add padding around region
                padding = float(crop)
                spec.crop_bbox = (
                    max(0, x0 - padding),
                    max(0, y0 - padding),
                    min(self.page.width, x1 + padding),
                    min(self.page.height, y1 + padding),
                )
            elif crop == "wide":
                # Full page width, cropped vertically to region
                spec.crop_bbox = (0, y0, self.page.width, y1)
            elif hasattr(crop, "bbox"):
                # Crop to another region's bounds
                spec.crop_bbox = crop.bbox

        # Add highlights in show mode (unless explicitly disabled with highlights=False)
        if mode == "show" and highlights is not False:
            # Only highlight this region if:
            # 1. We're not cropping, OR
            # 2. We're cropping but color was explicitly specified, OR
            # 3. We're cropping to another region (not tight crop)
            if not crop or color is not None or (crop and not isinstance(crop, bool)):
                spec.add_highlight(
                    bbox=self.bbox,
                    polygon=self.polygon if self.has_polygon else None,
                    color=color or "blue",
                    label=self.label or self.name or "Region",
                )

            # Add additional highlight groups if provided (and highlights is a list)
            if highlights and isinstance(highlights, list):
                for group in highlights:
                    elements = group.get("elements", [])
                    group_color = group.get("color", color)
                    group_label = group.get("label")

                    for elem in elements:
                        spec.add_highlight(element=elem, color=group_color, label=group_label)

        return [spec]

    def _direction(
        self,
        direction: str,
        size: Optional[float] = None,
        cross_size: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "Region":
        """
        Region-specific wrapper around :py:meth:`DirectionalMixin._direction`.

        It performs any pre-processing required by *Region* (none currently),
        delegates the core geometry work to the mix-in implementation via
        ``super()``, then attaches region-level metadata before returning the
        new :class:`Region` instance.
        """

        # Delegate to the shared implementation on DirectionalMixin
        region = super()._direction(
            direction=direction,
            size=size,
            cross_size=cross_size,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

        # Post-process: make sure callers can trace lineage and flags
        region.source_element = self
        region.includes_source = include_source

        return region

    def above(
        self,
        height: Optional[float] = None,
        width: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        **kwargs,
    ) -> "Region":
        """
        Select region above this region.

        Args:
            height: Height of the region above, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_source: Whether to include this region in the result (default: False)
            until: Optional selector string to specify an upper boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            offset: Pixel offset when excluding source/endpoint (default: None, uses natural_pdf.options.layout.directional_offset)
            **kwargs: Additional parameters

        Returns:
            Region object representing the area above
        """
        # Use global default if offset not provided
        if offset is None:
            import natural_pdf

            offset = natural_pdf.options.layout.directional_offset

        return self._direction(
            direction="above",
            size=height,
            cross_size=width,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=offset,
            **kwargs,
        )

    def below(
        self,
        height: Optional[float] = None,
        width: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        **kwargs,
    ) -> "Region":
        """
        Select region below this region.

        Args:
            height: Height of the region below, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_source: Whether to include this region in the result (default: False)
            until: Optional selector string to specify a lower boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            offset: Pixel offset when excluding source/endpoint (default: None, uses natural_pdf.options.layout.directional_offset)
            **kwargs: Additional parameters

        Returns:
            Region object representing the area below
        """
        # Use global default if offset not provided
        if offset is None:
            import natural_pdf

            offset = natural_pdf.options.layout.directional_offset

        return self._direction(
            direction="below",
            size=height,
            cross_size=width,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=offset,
            **kwargs,
        )

    def left(
        self,
        width: Optional[float] = None,
        height: str = "element",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        **kwargs,
    ) -> "Region":
        """
        Select region to the left of this region.

        Args:
            width: Width of the region to the left, in points
            height: Height mode - "full" for full page height or "element" for element height
            include_source: Whether to include this region in the result (default: False)
            until: Optional selector string to specify a left boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            offset: Pixel offset when excluding source/endpoint (default: None, uses natural_pdf.options.layout.directional_offset)
            **kwargs: Additional parameters

        Returns:
            Region object representing the area to the left
        """
        # Use global default if offset not provided
        if offset is None:
            import natural_pdf

            offset = natural_pdf.options.layout.directional_offset

        return self._direction(
            direction="left",
            size=width,
            cross_size=height,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=offset,
            **kwargs,
        )

    def right(
        self,
        width: Optional[float] = None,
        height: str = "element",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        **kwargs,
    ) -> "Region":
        """
        Select region to the right of this region.

        Args:
            width: Width of the region to the right, in points
            height: Height mode - "full" for full page height or "element" for element height
            include_source: Whether to include this region in the result (default: False)
            until: Optional selector string to specify a right boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            offset: Pixel offset when excluding source/endpoint (default: None, uses natural_pdf.options.layout.directional_offset)
            **kwargs: Additional parameters

        Returns:
            Region object representing the area to the right
        """
        # Use global default if offset not provided
        if offset is None:
            import natural_pdf

            offset = natural_pdf.options.layout.directional_offset

        return self._direction(
            direction="right",
            size=width,
            cross_size=height,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=offset,
            **kwargs,
        )

    @property
    def type(self) -> str:
        """Element type."""
        # Return the specific type if detected (e.g., from layout analysis)
        # or 'region' as a default.
        return self.region_type or "region"  # Prioritize specific region_type if set

    @property
    def page(self) -> "Page":
        """Get the parent page."""
        return self._page

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Get the bounding box as (x0, top, x1, bottom)."""
        return self._bbox

    @property
    def x0(self) -> float:
        """Get the left coordinate."""
        return self._bbox[0]

    @property
    def top(self) -> float:
        """Get the top coordinate."""
        return self._bbox[1]

    @property
    def x1(self) -> float:
        """Get the right coordinate."""
        return self._bbox[2]

    @property
    def bottom(self) -> float:
        """Get the bottom coordinate."""
        return self._bbox[3]

    @property
    def width(self) -> float:
        """Get the width of the region."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Get the height of the region."""
        return self.bottom - self.top

    @property
    def has_polygon(self) -> bool:
        """Check if this region has polygon coordinates."""
        return self._polygon is not None and len(self._polygon) >= 3

    @property
    def polygon(self) -> List[Tuple[float, float]]:
        """Get polygon coordinates if available, otherwise return rectangle corners."""
        if self._polygon:
            return self._polygon
        else:
            # Create rectangle corners from bbox as fallback
            return [
                (self.x0, self.top),  # top-left
                (self.x1, self.top),  # top-right
                (self.x1, self.bottom),  # bottom-right
                (self.x0, self.bottom),  # bottom-left
            ]

    @property
    def origin(self) -> Optional[Union["Element", "Region"]]:
        """The element/region that created this region (if it was created via directional method)."""
        return getattr(self, "source_element", None)

    @property
    def endpoint(self) -> Optional["Element"]:
        """The element where this region stopped (if created with 'until' parameter)."""
        return getattr(self, "boundary_element", None)

    def _is_point_in_polygon(self, x: float, y: float) -> bool:
        """
        Check if a point is inside the polygon using ray casting algorithm.

        Args:
            x: X coordinate of the point
            y: Y coordinate of the point

        Returns:
            bool: True if the point is inside the polygon
        """
        if not self.has_polygon:
            return (self.x0 <= x <= self.x1) and (self.top <= y <= self.bottom)

        # Ray casting algorithm
        inside = False
        j = len(self.polygon) - 1

        for i in range(len(self.polygon)):
            if ((self.polygon[i][1] > y) != (self.polygon[j][1] > y)) and (
                x
                < (self.polygon[j][0] - self.polygon[i][0])
                * (y - self.polygon[i][1])
                / (self.polygon[j][1] - self.polygon[i][1])
                + self.polygon[i][0]
            ):
                inside = not inside
            j = i

        return inside

    def is_point_inside(self, x: float, y: float) -> bool:
        """
        Check if a point is inside this region using ray casting algorithm for polygons.

        Args:
            x: X coordinate of the point
            y: Y coordinate of the point

        Returns:
            bool: True if the point is inside the region
        """
        if not self.has_polygon:
            return (self.x0 <= x <= self.x1) and (self.top <= y <= self.bottom)

        # Ray casting algorithm
        inside = False
        j = len(self.polygon) - 1

        for i in range(len(self.polygon)):
            if ((self.polygon[i][1] > y) != (self.polygon[j][1] > y)) and (
                x
                < (self.polygon[j][0] - self.polygon[i][0])
                * (y - self.polygon[i][1])
                / (self.polygon[j][1] - self.polygon[i][1])
                + self.polygon[i][0]
            ):
                inside = not inside
            j = i

        return inside

    def is_element_center_inside(self, element: "Element") -> bool:
        """
        Check if the center point of an element's bounding box is inside this region.

        Args:
            element: Element to check

        Returns:
            True if the element's center point is inside the region, False otherwise.
        """
        # Check if element is on the same page
        if not hasattr(element, "page") or element.page != self._page:
            return False

        # Ensure element has necessary attributes
        if not all(hasattr(element, attr) for attr in ["x0", "x1", "top", "bottom"]):
            logger.warning(
                f"Element {element} lacks bounding box attributes. Cannot check center point."
            )
            return False  # Cannot determine position

        # Calculate center point
        center_x = (element.x0 + element.x1) / 2
        center_y = (element.top + element.bottom) / 2

        # Use the existing is_point_inside check
        return self.is_point_inside(center_x, center_y)

    def _is_element_in_region(self, element: "Element", use_boundary_tolerance=True) -> bool:
        """
        Check if an element intersects or is contained within this region.

        Args:
            element: Element to check
            use_boundary_tolerance: Whether to apply a small tolerance for boundary elements

        Returns:
            True if the element is in the region, False otherwise
        """
        # Use centralized spatial utility for consistency
        from natural_pdf.utils.spatial import is_element_in_region

        return is_element_in_region(element, self, strategy="center", check_page=True)

    def contains(self, element: "Element") -> bool:
        """
        Check if this region completely contains an element.

        Args:
            element: Element to check

        Returns:
            True if the element is completely contained within the region, False otherwise
        """
        # Check if element is on the same page
        if not hasattr(element, "page") or element.page != self._page:
            return False

        # Ensure element has necessary attributes
        if not all(hasattr(element, attr) for attr in ["x0", "x1", "top", "bottom"]):
            return False  # Cannot determine position

        # For rectangular regions, check if element's bbox is fully inside region's bbox
        if not self.has_polygon:
            return (
                self.x0 <= element.x0
                and element.x1 <= self.x1
                and self.top <= element.top
                and element.bottom <= self.bottom
            )

        # For polygon regions, check if all corners of the element are inside the polygon
        element_corners = [
            (element.x0, element.top),  # top-left
            (element.x1, element.top),  # top-right
            (element.x1, element.bottom),  # bottom-right
            (element.x0, element.bottom),  # bottom-left
        ]

        return all(self.is_point_inside(x, y) for x, y in element_corners)

    def intersects(self, element: "Element") -> bool:
        """
        Check if this region intersects with an element (any overlap).

        Args:
            element: Element to check

        Returns:
            True if the element overlaps with the region at all, False otherwise
        """
        # Check if element is on the same page
        if not hasattr(element, "page") or element.page != self._page:
            return False

        # Ensure element has necessary attributes
        if not all(hasattr(element, attr) for attr in ["x0", "x1", "top", "bottom"]):
            return False  # Cannot determine position

        # For rectangular regions, check for bbox overlap
        if not self.has_polygon:
            return (
                self.x0 < element.x1
                and self.x1 > element.x0
                and self.top < element.bottom
                and self.bottom > element.top
            )

        # For polygon regions, check if any corner of the element is inside the polygon
        element_corners = [
            (element.x0, element.top),  # top-left
            (element.x1, element.top),  # top-right
            (element.x1, element.bottom),  # bottom-right
            (element.x0, element.bottom),  # bottom-left
        ]

        # First check if any element corner is inside the polygon
        if any(self.is_point_inside(x, y) for x, y in element_corners):
            return True

        # Also check if any polygon corner is inside the element's rectangle
        for x, y in self.polygon:
            if element.x0 <= x <= element.x1 and element.top <= y <= element.bottom:
                return True

        # Also check if any polygon edge intersects with any rectangle edge
        # This is a simplification - for complex cases, we'd need a full polygon-rectangle
        # intersection algorithm

        # For now, return True if bounding boxes overlap (approximation for polygon-rectangle case)
        return (
            self.x0 < element.x1
            and self.x1 > element.x0
            and self.top < element.bottom
            and self.bottom > element.top
        )

    def exclude(self):
        """
        Exclude this region from text extraction and other operations.

        This excludes everything within the region's bounds.
        """
        self.page.add_exclusion(self, method="region")

    def highlight(
        self,
        label: Optional[str] = None,
        color: Optional[Union[Tuple, str]] = None,
        use_color_cycling: bool = False,
        annotate: Optional[List[str]] = None,
        existing: str = "append",
    ) -> "Region":
        """
        Highlight this region on the page.

        Args:
            label: Optional label for the highlight
            color: Color tuple/string for the highlight, or None to use automatic color
            use_color_cycling: Force color cycling even with no label (default: False)
            annotate: List of attribute names to display on the highlight (e.g., ['confidence', 'type'])
            existing: How to handle existing highlights ('append' or 'replace').

        Returns:
            Self for method chaining
        """
        # Access the highlighter service correctly
        highlighter = self.page._highlighter

        # Prepare common arguments
        highlight_args = {
            "page_index": self.page.index,
            "color": color,
            "label": label,
            "use_color_cycling": use_color_cycling,
            "element": self,  # Pass the region itself so attributes can be accessed
            "annotate": annotate,
            "existing": existing,
        }

        # Call the appropriate service method
        if self.has_polygon:
            highlight_args["polygon"] = self.polygon
            highlighter.add_polygon(**highlight_args)
        else:
            highlight_args["bbox"] = self.bbox
            highlighter.add(**highlight_args)

        return self

    def save(
        self,
        filename: str,
        resolution: Optional[float] = None,
        labels: bool = True,
        legend_position: str = "right",
    ) -> "Region":
        """
        Save the page with this region highlighted to an image file.

        Args:
            filename: Path to save the image to
            resolution: Resolution in DPI for rendering (default: uses global options, fallback to 144 DPI)
            labels: Whether to include a legend for labels
            legend_position: Position of the legend

        Returns:
            Self for method chaining
        """
        # Apply global options as defaults
        import natural_pdf

        if resolution is None:
            if natural_pdf.options.image.resolution is not None:
                resolution = natural_pdf.options.image.resolution
            else:
                resolution = 144  # Default resolution when none specified

        # Highlight this region if not already highlighted
        self.highlight()

        # Save the highlighted image
        self._page.save_image(
            filename, resolution=resolution, labels=labels, legend_position=legend_position
        )
        return self

    def save_image(
        self,
        filename: str,
        resolution: Optional[float] = None,
        crop: bool = False,
        include_highlights: bool = True,
        **kwargs,
    ) -> "Region":
        """
        Save an image of just this region to a file.

        Args:
            filename: Path to save the image to
            resolution: Resolution in DPI for rendering (default: uses global options, fallback to 144 DPI)
            crop: If True, only crop the region without highlighting its boundaries
            include_highlights: Whether to include existing highlights (default: True)
            **kwargs: Additional parameters for rendering

        Returns:
            Self for method chaining
        """
        # Apply global options as defaults
        import natural_pdf

        if resolution is None:
            if natural_pdf.options.image.resolution is not None:
                resolution = natural_pdf.options.image.resolution
            else:
                resolution = 144  # Default resolution when none specified

        # Use export() to save the image
        if include_highlights:
            # With highlights, use export() which includes them
            self.export(
                path=filename,
                resolution=resolution,
                crop=crop,
                **kwargs,
            )
        else:
            # Without highlights, use render() and save manually
            image = self.render(resolution=resolution, crop=crop, **kwargs)
            if image:
                image.save(filename)
            else:
                logger.error(f"Failed to render region image for saving to {filename}")

        return self

    def trim(
        self,
        padding: int = 1,
        threshold: float = 0.95,
        resolution: Optional[float] = None,
        pre_shrink: float = 0.5,
    ) -> "Region":
        """
        Trim visual whitespace from the edges of this region.

        Similar to Python's string .strip() method, but for visual whitespace in the region image.
        Uses pixel analysis to detect rows/columns that are predominantly whitespace.

        Args:
            padding: Number of pixels to keep as padding after trimming (default: 1)
            threshold: Threshold for considering a row/column as whitespace (0.0-1.0, default: 0.95)
                      Higher values mean more strict whitespace detection.
                      E.g., 0.95 means if 95% of pixels in a row/column are white, consider it whitespace.
            resolution: Resolution for image rendering in DPI (default: uses global options, fallback to 144 DPI)
            pre_shrink: Amount to shrink region before trimming, then expand back after (default: 0.5)
                       This helps avoid detecting box borders/slivers as content.

        Returns
        ------

        New Region with visual whitespace trimmed from all edges

        Examples
        --------

        ```python
        # Basic trimming with 1 pixel padding and 0.5px pre-shrink
        trimmed = region.trim()

        # More aggressive trimming with no padding and no pre-shrink
        tight = region.trim(padding=0, threshold=0.9, pre_shrink=0)

        # Conservative trimming with more padding
        loose = region.trim(padding=3, threshold=0.98)
        ```
        """
        # Apply global options as defaults
        import natural_pdf

        if resolution is None:
            if natural_pdf.options.image.resolution is not None:
                resolution = natural_pdf.options.image.resolution
            else:
                resolution = 144  # Default resolution when none specified

        # Pre-shrink the region to avoid box slivers
        work_region = (
            self.expand(left=-pre_shrink, right=-pre_shrink, top=-pre_shrink, bottom=-pre_shrink)
            if pre_shrink > 0
            else self
        )

        # Get the region image
        # Use render() for clean image without highlights, with cropping
        image = work_region.render(resolution=resolution, crop=True)

        if image is None:
            logger.warning(
                f"Region {self.bbox}: Could not generate image for trimming. Returning original region."
            )
            return self

        # Convert to grayscale for easier analysis
        import numpy as np

        # Convert PIL image to numpy array
        img_array = np.array(image.convert("L"))  # Convert to grayscale
        height, width = img_array.shape

        if height == 0 or width == 0:
            logger.warning(
                f"Region {self.bbox}: Image has zero dimensions. Returning original region."
            )
            return self

        # Normalize pixel values to 0-1 range (255 = white = 1.0, 0 = black = 0.0)
        normalized = img_array.astype(np.float32) / 255.0

        # Find content boundaries by analyzing row and column averages

        # Analyze rows (horizontal strips) to find top and bottom boundaries
        row_averages = np.mean(normalized, axis=1)  # Average each row
        content_rows = row_averages < threshold  # True where there's content (not whitespace)

        # Find first and last rows with content
        content_row_indices = np.where(content_rows)[0]
        if len(content_row_indices) == 0:
            # No content found, return a minimal region at the center
            logger.warning(
                f"Region {self.bbox}: No content detected during trimming. Returning center point."
            )
            center_x = (self.x0 + self.x1) / 2
            center_y = (self.top + self.bottom) / 2
            return Region(self.page, (center_x, center_y, center_x, center_y))

        top_content_row = max(0, content_row_indices[0] - padding)
        bottom_content_row = min(height - 1, content_row_indices[-1] + padding)

        # Analyze columns (vertical strips) to find left and right boundaries
        col_averages = np.mean(normalized, axis=0)  # Average each column
        content_cols = col_averages < threshold  # True where there's content

        content_col_indices = np.where(content_cols)[0]
        if len(content_col_indices) == 0:
            # No content found in columns either
            logger.warning(
                f"Region {self.bbox}: No column content detected during trimming. Returning center point."
            )
            center_x = (self.x0 + self.x1) / 2
            center_y = (self.top + self.bottom) / 2
            return Region(self.page, (center_x, center_y, center_x, center_y))

        left_content_col = max(0, content_col_indices[0] - padding)
        right_content_col = min(width - 1, content_col_indices[-1] + padding)

        # Convert trimmed pixel coordinates back to PDF coordinates
        scale_factor = resolution / 72.0  # Scale factor used in render()

        # Calculate new PDF coordinates and ensure they are Python floats
        trimmed_x0 = float(work_region.x0 + (left_content_col / scale_factor))
        trimmed_top = float(work_region.top + (top_content_row / scale_factor))
        trimmed_x1 = float(
            work_region.x0 + ((right_content_col + 1) / scale_factor)
        )  # +1 because we want inclusive right edge
        trimmed_bottom = float(
            work_region.top + ((bottom_content_row + 1) / scale_factor)
        )  # +1 because we want inclusive bottom edge

        # Ensure the trimmed region doesn't exceed the work region boundaries
        final_x0 = max(work_region.x0, trimmed_x0)
        final_top = max(work_region.top, trimmed_top)
        final_x1 = min(work_region.x1, trimmed_x1)
        final_bottom = min(work_region.bottom, trimmed_bottom)

        # Ensure valid coordinates (width > 0, height > 0)
        if final_x1 <= final_x0 or final_bottom <= final_top:
            logger.warning(
                f"Region {self.bbox}: Trimming resulted in invalid dimensions. Returning original region."
            )
            return self

        # Create the trimmed region
        trimmed_region = Region(self.page, (final_x0, final_top, final_x1, final_bottom))

        # Expand back by the pre_shrink amount to restore original positioning
        if pre_shrink > 0:
            trimmed_region = trimmed_region.expand(
                left=pre_shrink, right=pre_shrink, top=pre_shrink, bottom=pre_shrink
            )

        # Copy relevant metadata
        trimmed_region.region_type = self.region_type
        trimmed_region.normalized_type = self.normalized_type
        trimmed_region.confidence = self.confidence
        trimmed_region.model = self.model
        trimmed_region.name = self.name
        trimmed_region.label = self.label
        trimmed_region.source = "trimmed"  # Indicate this is a derived region
        trimmed_region.parent_region = self

        logger.debug(
            f"Region {self.bbox}: Trimmed to {trimmed_region.bbox} (padding={padding}, threshold={threshold}, pre_shrink={pre_shrink})"
        )
        return trimmed_region

    def clip(
        self,
        obj: Optional[Any] = None,
        left: Optional[float] = None,
        top: Optional[float] = None,
        right: Optional[float] = None,
        bottom: Optional[float] = None,
    ) -> "Region":
        """
        Clip this region to specific bounds, either from another object with bbox or explicit coordinates.

        The clipped region will be constrained to not exceed the specified boundaries.
        You can provide either an object with bounding box properties, specific coordinates, or both.
        When both are provided, explicit coordinates take precedence.

        Args:
            obj: Optional object with bbox properties (Region, Element, TextElement, etc.)
            left: Optional left boundary (x0) to clip to
            top: Optional top boundary to clip to
            right: Optional right boundary (x1) to clip to
            bottom: Optional bottom boundary to clip to

        Returns:
            New Region with bounds clipped to the specified constraints

        Examples:
            # Clip to another region's bounds
            clipped = region.clip(container_region)

            # Clip to any element's bounds
            clipped = region.clip(text_element)

            # Clip to specific coordinates
            clipped = region.clip(left=100, right=400)

            # Mix object bounds with specific overrides
            clipped = region.clip(obj=container, bottom=page.height/2)
        """
        from natural_pdf.elements.base import extract_bbox

        # Start with current region bounds
        clip_x0 = self.x0
        clip_top = self.top
        clip_x1 = self.x1
        clip_bottom = self.bottom

        # Apply object constraints if provided
        if obj is not None:
            obj_bbox = extract_bbox(obj)
            if obj_bbox is not None:
                obj_x0, obj_top, obj_x1, obj_bottom = obj_bbox
                # Constrain to the intersection with the provided object
                clip_x0 = max(clip_x0, obj_x0)
                clip_top = max(clip_top, obj_top)
                clip_x1 = min(clip_x1, obj_x1)
                clip_bottom = min(clip_bottom, obj_bottom)
            else:
                logger.warning(
                    f"Region {self.bbox}: Cannot extract bbox from clipping object {type(obj)}. "
                    "Object must have bbox property or x0/top/x1/bottom attributes."
                )

        # Apply explicit coordinate constraints (these take precedence)
        if left is not None:
            clip_x0 = max(clip_x0, left)
        if top is not None:
            clip_top = max(clip_top, top)
        if right is not None:
            clip_x1 = min(clip_x1, right)
        if bottom is not None:
            clip_bottom = min(clip_bottom, bottom)

        # Ensure valid coordinates
        if clip_x1 <= clip_x0 or clip_bottom <= clip_top:
            logger.warning(
                f"Region {self.bbox}: Clipping resulted in invalid dimensions "
                f"({clip_x0}, {clip_top}, {clip_x1}, {clip_bottom}). Returning minimal region."
            )
            # Return a minimal region at the clip area's top-left
            return Region(self.page, (clip_x0, clip_top, clip_x0, clip_top))

        # Create the clipped region
        clipped_region = Region(self.page, (clip_x0, clip_top, clip_x1, clip_bottom))

        # Copy relevant metadata
        clipped_region.region_type = self.region_type
        clipped_region.normalized_type = self.normalized_type
        clipped_region.confidence = self.confidence
        clipped_region.model = self.model
        clipped_region.name = self.name
        clipped_region.label = self.label
        clipped_region.source = "clipped"  # Indicate this is a derived region
        clipped_region.parent_region = self

        logger.debug(
            f"Region {self.bbox}: Clipped to {clipped_region.bbox} "
            f"(constraints: obj={type(obj).__name__ if obj else None}, "
            f"left={left}, top={top}, right={right}, bottom={bottom})"
        )
        return clipped_region

    def region(
        self,
        left: float = None,
        top: float = None,
        right: float = None,
        bottom: float = None,
        width: Union[str, float, None] = None,
        height: Optional[float] = None,
        relative: bool = False,
    ) -> "Region":
        """
        Create a sub-region within this region using the same API as Page.region().

        By default, coordinates are absolute (relative to the page), matching Page.region().
        Set relative=True to use coordinates relative to this region's top-left corner.

        Args:
            left: Left x-coordinate (absolute by default, or relative to region if relative=True)
            top: Top y-coordinate (absolute by default, or relative to region if relative=True)
            right: Right x-coordinate (absolute by default, or relative to region if relative=True)
            bottom: Bottom y-coordinate (absolute by default, or relative to region if relative=True)
            width: Width definition (same as Page.region())
            height: Height of the region (same as Page.region())
            relative: If True, coordinates are relative to this region's top-left (0,0).
                     If False (default), coordinates are absolute page coordinates.

        Returns:
            Region object for the specified coordinates, clipped to this region's bounds

        Examples:
            # Absolute coordinates (default) - same as page.region()
            sub = region.region(left=100, top=200, width=50, height=30)

            # Relative to region's top-left
            sub = region.region(left=10, top=10, width=50, height=30, relative=True)

            # Mix relative positioning with this region's bounds
            sub = region.region(left=region.x0 + 10, width=50, height=30)
        """
        # If relative coordinates requested, convert to absolute
        if relative:
            if left is not None:
                left = self.x0 + left
            if top is not None:
                top = self.top + top
            if right is not None:
                right = self.x0 + right
            if bottom is not None:
                bottom = self.top + bottom

            # For numeric width/height with relative coords, we need to handle the calculation
            # in the context of absolute positioning

        # Use the parent page's region method to create the region with all its logic
        new_region = self.page.region(
            left=left, top=top, right=right, bottom=bottom, width=width, height=height
        )

        # Clip the new region to this region's bounds
        return new_region.clip(self)

    def get_elements(
        self, selector: Optional[str] = None, apply_exclusions=True, **kwargs
    ) -> List["Element"]:
        """
        Get all elements within this region.

        Args:
            selector: Optional selector to filter elements
            apply_exclusions: Whether to apply exclusion regions
            **kwargs: Additional parameters for element filtering

        Returns:
            List of elements in the region
        """
        if selector:
            # Find elements on the page matching the selector
            page_elements = self.page.find_all(
                selector, apply_exclusions=apply_exclusions, **kwargs
            )
            # Filter those elements to only include ones within this region
            elements = [e for e in page_elements if self._is_element_in_region(e)]
        else:
            # Get all elements from the page
            page_elements = self.page.get_elements(apply_exclusions=apply_exclusions)
            # Filter to elements in this region
            elements = [e for e in page_elements if self._is_element_in_region(e)]

        # Apply boundary exclusions if this is a section with boundary settings
        if hasattr(self, "_boundary_exclusions") and self._boundary_exclusions != "both":
            excluded_ids = set()

            if self._boundary_exclusions == "none":
                # Exclude both start and end elements
                if hasattr(self, "start_element") and self.start_element:
                    excluded_ids.add(id(self.start_element))
                if hasattr(self, "end_element") and self.end_element:
                    excluded_ids.add(id(self.end_element))
            elif self._boundary_exclusions == "start":
                # Exclude only end element
                if hasattr(self, "end_element") and self.end_element:
                    excluded_ids.add(id(self.end_element))
            elif self._boundary_exclusions == "end":
                # Exclude only start element
                if hasattr(self, "start_element") and self.start_element:
                    excluded_ids.add(id(self.start_element))

            if excluded_ids:
                elements = [e for e in elements if id(e) not in excluded_ids]

        return elements

    def extract_text(
        self,
        granularity: str = "chars",
        apply_exclusions: bool = True,
        debug: bool = False,
        *,
        overlap: str = "center",
        newlines: Union[bool, str] = True,
        content_filter=None,
        **kwargs,
    ) -> str:
        """
        Extract text from this region, respecting page exclusions and using pdfplumber's
        layout engine (chars_to_textmap).

        Args:
            granularity: Level of text extraction - 'chars' (default) or 'words'.
                - 'chars': Character-by-character extraction (current behavior)
                - 'words': Word-level extraction with configurable overlap
            apply_exclusions: Whether to apply exclusion regions defined on the parent page.
            debug: Enable verbose debugging output for filtering steps.
            overlap: How to determine if words overlap with the region (only used when granularity='words'):
                - 'center': Word center point must be inside (default)
                - 'full': Word must be fully inside the region
                - 'partial': Any overlap includes the word
            newlines: Whether to strip newline characters from the extracted text.
            content_filter: Optional content filter to exclude specific text patterns. Can be:
                - A regex pattern string (characters matching the pattern are EXCLUDED)
                - A callable that takes text and returns True to KEEP the character
                - A list of regex patterns (characters matching ANY pattern are EXCLUDED)
            **kwargs: Additional layout parameters passed directly to pdfplumber's
                      `chars_to_textmap` function (e.g., layout, x_density, y_density).
                      See Page.extract_text docstring for more.

        Returns:
            Extracted text as string, potentially with layout-based spacing.
        """
        # Validate granularity parameter
        if granularity not in ("chars", "words"):
            raise ValueError(f"granularity must be 'chars' or 'words', got '{granularity}'")

        # Allow 'debug_exclusions' for backward compatibility
        debug = kwargs.get("debug", debug or kwargs.get("debug_exclusions", False))
        logger.debug(
            f"Region {self.bbox}: extract_text called with granularity='{granularity}', overlap='{overlap}', kwargs: {kwargs}"
        )

        # Handle word-level extraction
        if granularity == "words":
            # Use find_all to get words with proper overlap and exclusion handling
            word_elements = self.find_all(
                "text", overlap=overlap, apply_exclusions=apply_exclusions
            )

            # Join the text from all matching words
            text_parts = []
            for word in word_elements:
                word_text = word.extract_text()
                if word_text:  # Skip empty strings
                    text_parts.append(word_text)

            result = " ".join(text_parts)

            # Apply newlines processing if requested
            if newlines is False:
                result = result.replace("\n", " ").replace("\r", " ")
            elif isinstance(newlines, str):
                result = result.replace("\n", newlines).replace("\r", newlines)

            return result

        # Original character-level extraction logic follows...
        # 1. Get Word Elements potentially within this region (initial broad phase)
        # Optimization: Could use spatial query if page elements were indexed
        page_words = self.page.words  # Get all words from the page

        # 2. Gather all character dicts from words potentially in region
        # We filter precisely in filter_chars_spatially
        all_char_dicts = []
        for word in page_words:
            # Quick bbox check to avoid processing words clearly outside
            if get_bbox_overlap(self.bbox, word.bbox) is not None:
                all_char_dicts.extend(getattr(word, "_char_dicts", []))

        if not all_char_dicts:
            logger.debug(f"Region {self.bbox}: No character dicts found overlapping region bbox.")
            return ""

        # 3. Get Relevant Exclusions (overlapping this region)
        apply_exclusions_flag = kwargs.get("apply_exclusions", apply_exclusions)
        exclusion_regions = []
        if apply_exclusions_flag:
            # Always call _get_exclusion_regions to get both page and PDF level exclusions
            all_page_exclusions = self._page._get_exclusion_regions(
                include_callable=True, debug=debug
            )
            overlapping_exclusions = []
            for excl in all_page_exclusions:
                if get_bbox_overlap(self.bbox, excl.bbox) is not None:
                    overlapping_exclusions.append(excl)
            exclusion_regions = overlapping_exclusions
            if debug:
                logger.debug(
                    f"Region {self.bbox}: Found {len(all_page_exclusions)} total exclusions, "
                    f"{len(exclusion_regions)} overlapping this region."
                )
        elif debug:
            logger.debug(f"Region {self.bbox}: Not applying exclusions (apply_exclusions=False).")

        # Add boundary element exclusions if this is a section with boundary settings
        if hasattr(self, "_boundary_exclusions") and self._boundary_exclusions != "both":
            boundary_exclusions = []

            if self._boundary_exclusions == "none":
                # Exclude both start and end elements
                if hasattr(self, "start_element") and self.start_element:
                    boundary_exclusions.append(self.start_element)
                if hasattr(self, "end_element") and self.end_element:
                    boundary_exclusions.append(self.end_element)
            elif self._boundary_exclusions == "start":
                # Exclude only end element
                if hasattr(self, "end_element") and self.end_element:
                    boundary_exclusions.append(self.end_element)
            elif self._boundary_exclusions == "end":
                # Exclude only start element
                if hasattr(self, "start_element") and self.start_element:
                    boundary_exclusions.append(self.start_element)

            # Add boundary elements as exclusion regions
            for elem in boundary_exclusions:
                if hasattr(elem, "bbox"):
                    exclusion_regions.append(elem)
                    if debug:
                        logger.debug(
                            f"Adding boundary exclusion: {elem.extract_text().strip()} at {elem.bbox}"
                        )

        # 4. Spatially Filter Characters using Utility
        # Pass self as the target_region for precise polygon checks etc.
        filtered_chars = filter_chars_spatially(
            char_dicts=all_char_dicts,
            exclusion_regions=exclusion_regions,
            target_region=self,  # Pass self!
            debug=debug,
        )

        # 5. Generate Text Layout using Utility
        # Add content_filter to kwargs if provided
        final_kwargs = kwargs.copy()
        if content_filter is not None:
            final_kwargs["content_filter"] = content_filter

        result = generate_text_layout(
            char_dicts=filtered_chars,
            layout_context_bbox=self.bbox,  # Use region's bbox for context
            user_kwargs=final_kwargs,  # Pass kwargs including content_filter
        )

        # Flexible newline handling (same logic as TextElement)
        if isinstance(newlines, bool):
            if newlines is False:
                replacement = " "
            else:
                replacement = None
        else:
            replacement = str(newlines)

        if replacement is not None:
            result = result.replace("\n", replacement).replace("\r", replacement)

        logger.debug(f"Region {self.bbox}: extract_text finished, result length: {len(result)}.")
        return result

    def extract_table(
        self,
        method: Optional[str] = None,  # Make method optional
        table_settings: Optional[dict] = None,  # Use Optional
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,  # Use Optional
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[["Region"], Optional[str]]] = None,
        # --- NEW: Add tqdm control option --- #
        show_progress: bool = False,  # Controls progress bar for text method
        content_filter: Optional[
            Union[str, Callable[[str], bool], List[str]]
        ] = None,  # NEW: Content filtering
        apply_exclusions: bool = True,  # Whether to apply exclusion regions during extraction
        verticals: Optional[List] = None,  # Explicit vertical lines
        horizontals: Optional[List] = None,  # Explicit horizontal lines
    ) -> TableResult:  # Return type allows Optional[str] for cells
        """
        Extract a table from this region.

        Args:
            method: Method to use: 'tatr', 'pdfplumber', 'text', 'stream', 'lattice', or None (auto-detect).
                    'stream' is an alias for 'pdfplumber' with text-based strategies (equivalent to
                    setting `vertical_strategy` and `horizontal_strategy` to 'text').
                    'lattice' is an alias for 'pdfplumber' with line-based strategies (equivalent to
                    setting `vertical_strategy` and `horizontal_strategy` to 'lines').
            table_settings: Settings for pdfplumber table extraction (used with 'pdfplumber', 'stream', or 'lattice' methods).
            use_ocr: Whether to use OCR for text extraction (currently only applicable with 'tatr' method).
            ocr_config: OCR configuration parameters.
            text_options: Dictionary of options for the 'text' method, corresponding to arguments
                          of analyze_text_table_structure (e.g., snap_tolerance, expand_bbox).
            cell_extraction_func: Optional callable function that takes a cell Region object
                                  and returns its string content. Overrides default text extraction
                                  for the 'text' method.
            show_progress: If True, display a progress bar during cell text extraction for the 'text' method.
            content_filter: Optional content filter to apply during cell text extraction. Can be:
                - A regex pattern string (characters matching the pattern are EXCLUDED)
                - A callable that takes text and returns True to KEEP the character
                - A list of regex patterns (characters matching ANY pattern are EXCLUDED)
                Works with all extraction methods by filtering cell content.
            apply_exclusions: Whether to apply exclusion regions during text extraction (default: True).
                When True, text within excluded regions (e.g., headers/footers) will not be extracted.
            verticals: Optional list of explicit vertical lines for table extraction. When provided,
                       automatically sets vertical_strategy='explicit' and explicit_vertical_lines.
            horizontals: Optional list of explicit horizontal lines for table extraction. When provided,
                         automatically sets horizontal_strategy='explicit' and explicit_horizontal_lines.

        Returns:
            Table data as a list of rows, where each row is a list of cell values (str or None).
        """
        # Default settings if none provided
        if table_settings is None:
            table_settings = {}
        if text_options is None:
            text_options = {}  # Initialize empty dict

        # Handle explicit vertical and horizontal lines
        if verticals is not None:
            table_settings["vertical_strategy"] = "explicit"
            table_settings["explicit_vertical_lines"] = verticals
        if horizontals is not None:
            table_settings["horizontal_strategy"] = "explicit"
            table_settings["explicit_horizontal_lines"] = horizontals

        # Auto-detect method if not specified
        if method is None:
            # If this is a TATR-detected region, use TATR method
            if hasattr(self, "model") and self.model == "tatr" and self.region_type == "table":
                effective_method = "tatr"
            else:
                # Try lattice first, then fall back to stream if no meaningful results
                logger.debug(f"Region {self.bbox}: Auto-detecting table extraction method...")

                # --- NEW: Prefer already-created table_cell regions if they exist --- #
                try:
                    cell_regions_in_table = [
                        c
                        for c in self.page.find_all(
                            "region[type=table_cell]", apply_exclusions=False
                        )
                        if self.intersects(c)
                    ]
                except Exception as _cells_err:
                    cell_regions_in_table = []  # Fallback silently

                if cell_regions_in_table:
                    logger.debug(
                        f"Region {self.bbox}: Found {len(cell_regions_in_table)} pre-computed table_cell regions – using 'cells' method."
                    )
                    return TableResult(
                        self._extract_table_from_cells(
                            cell_regions_in_table,
                            content_filter=content_filter,
                            apply_exclusions=apply_exclusions,
                        )
                    )

                # --------------------------------------------------------------- #

                try:
                    logger.debug(f"Region {self.bbox}: Trying 'lattice' method first...")
                    lattice_result = self.extract_table(
                        "lattice", table_settings=table_settings.copy()
                    )

                    # Check if lattice found meaningful content
                    if (
                        lattice_result
                        and len(lattice_result) > 0
                        and any(
                            any(cell and cell.strip() for cell in row if cell)
                            for row in lattice_result
                        )
                    ):
                        logger.debug(
                            f"Region {self.bbox}: 'lattice' method found table with {len(lattice_result)} rows"
                        )
                        return lattice_result
                    else:
                        logger.debug(
                            f"Region {self.bbox}: 'lattice' method found no meaningful content"
                        )
                except Exception as e:
                    logger.debug(f"Region {self.bbox}: 'lattice' method failed: {e}")

                # Fall back to stream
                logger.debug(f"Region {self.bbox}: Falling back to 'stream' method...")
                return self.extract_table("stream", table_settings=table_settings.copy())
        else:
            effective_method = method

        # Handle method aliases for pdfplumber
        if effective_method == "stream":
            logger.debug("Using 'stream' method alias for 'pdfplumber' with text-based strategies.")
            effective_method = "pdfplumber"
            # Set default text strategies if not already provided by the user
            table_settings.setdefault("vertical_strategy", "text")
            table_settings.setdefault("horizontal_strategy", "text")
        elif effective_method == "lattice":
            logger.debug(
                "Using 'lattice' method alias for 'pdfplumber' with line-based strategies."
            )
            effective_method = "pdfplumber"
            # Set default line strategies if not already provided by the user
            table_settings.setdefault("vertical_strategy", "lines")
            table_settings.setdefault("horizontal_strategy", "lines")

        # -------------------------------------------------------------
        # Auto-inject tolerances when text-based strategies are requested.
        # This must happen AFTER alias handling (so strategies are final)
        # and BEFORE we delegate to _extract_table_* helpers.
        # -------------------------------------------------------------
        if "text" in (
            table_settings.get("vertical_strategy"),
            table_settings.get("horizontal_strategy"),
        ):
            page_cfg = getattr(self.page, "_config", {})
            # Ensure text_* tolerances passed to pdfplumber
            if "text_x_tolerance" not in table_settings and "x_tolerance" not in table_settings:
                if page_cfg.get("x_tolerance") is not None:
                    table_settings["text_x_tolerance"] = page_cfg["x_tolerance"]
            if "text_y_tolerance" not in table_settings and "y_tolerance" not in table_settings:
                if page_cfg.get("y_tolerance") is not None:
                    table_settings["text_y_tolerance"] = page_cfg["y_tolerance"]

            # Snap / join tolerances (~ line spacing)
            if "snap_tolerance" not in table_settings and "snap_x_tolerance" not in table_settings:
                snap = max(1, round((page_cfg.get("y_tolerance", 1)) * 0.9))
                table_settings["snap_tolerance"] = snap
            if "join_tolerance" not in table_settings and "join_x_tolerance" not in table_settings:
                table_settings["join_tolerance"] = table_settings["snap_tolerance"]

        logger.debug(f"Region {self.bbox}: Extracting table using method '{effective_method}'")

        # For stream method with text-based edge detection and explicit vertical lines,
        # adjust guides to ensure they fall within text bounds for proper intersection
        if (
            effective_method == "pdfplumber"
            and table_settings.get("horizontal_strategy") == "text"
            and table_settings.get("vertical_strategy") == "explicit"
            and "explicit_vertical_lines" in table_settings
        ):

            text_elements = self.find_all("text", apply_exclusions=apply_exclusions)
            if text_elements:
                text_bounds = text_elements.merge().bbox
                text_left = text_bounds[0]
                text_right = text_bounds[2]

                # Adjust vertical guides to fall within text bounds
                original_verticals = table_settings["explicit_vertical_lines"]
                adjusted_verticals = []

                for v in original_verticals:
                    if v < text_left:
                        # Guide is left of text bounds, clip to text start
                        adjusted_verticals.append(text_left)
                        logger.debug(
                            f"Region {self.bbox}: Adjusted left guide from {v:.1f} to {text_left:.1f}"
                        )
                    elif v > text_right:
                        # Guide is right of text bounds, clip to text end
                        adjusted_verticals.append(text_right)
                        logger.debug(
                            f"Region {self.bbox}: Adjusted right guide from {v:.1f} to {text_right:.1f}"
                        )
                    else:
                        # Guide is within text bounds, keep as is
                        adjusted_verticals.append(v)

                # Update table settings with adjusted guides
                table_settings["explicit_vertical_lines"] = adjusted_verticals
                logger.debug(
                    f"Region {self.bbox}: Adjusted {len(original_verticals)} guides for stream extraction. "
                    f"Text bounds: {text_left:.1f}-{text_right:.1f}"
                )

        # Use the selected method
        if effective_method == "tatr":
            table_rows = self._extract_table_tatr(
                use_ocr=use_ocr,
                ocr_config=ocr_config,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
            )
        elif effective_method == "text":
            current_text_options = text_options.copy()
            current_text_options["cell_extraction_func"] = cell_extraction_func
            current_text_options["show_progress"] = show_progress
            current_text_options["content_filter"] = content_filter
            current_text_options["apply_exclusions"] = apply_exclusions
            table_rows = self._extract_table_text(**current_text_options)
        elif effective_method == "pdfplumber":
            table_rows = self._extract_table_plumber(
                table_settings, content_filter=content_filter, apply_exclusions=apply_exclusions
            )
        else:
            raise ValueError(
                f"Unknown table extraction method: '{method}'. Choose from 'tatr', 'pdfplumber', 'text', 'stream', 'lattice'."
            )

        return TableResult(table_rows)

    def extract_tables(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
    ) -> List[List[List[str]]]:
        """
        Extract all tables from this region using pdfplumber-based methods.

        Note: Only 'pdfplumber', 'stream', and 'lattice' methods are supported for extract_tables.
        'tatr' and 'text' methods are designed for single table extraction only.

        Args:
            method: Method to use: 'pdfplumber', 'stream', 'lattice', or None (auto-detect).
                    'stream' uses text-based strategies, 'lattice' uses line-based strategies.
            table_settings: Settings for pdfplumber table extraction.

        Returns:
            List of tables, where each table is a list of rows, and each row is a list of cell values.
        """
        if table_settings is None:
            table_settings = {}

        # Auto-detect method if not specified (try lattice first, then stream)
        if method is None:
            logger.debug(f"Region {self.bbox}: Auto-detecting tables extraction method...")

            # Try lattice first
            try:
                lattice_settings = table_settings.copy()
                lattice_settings.setdefault("vertical_strategy", "lines")
                lattice_settings.setdefault("horizontal_strategy", "lines")

                logger.debug(f"Region {self.bbox}: Trying 'lattice' method first for tables...")
                lattice_result = self._extract_tables_plumber(lattice_settings)

                # Check if lattice found meaningful tables
                if (
                    lattice_result
                    and len(lattice_result) > 0
                    and any(
                        any(
                            any(cell and cell.strip() for cell in row if cell)
                            for row in table
                            if table
                        )
                        for table in lattice_result
                    )
                ):
                    logger.debug(
                        f"Region {self.bbox}: 'lattice' method found {len(lattice_result)} tables"
                    )
                    return lattice_result
                else:
                    logger.debug(f"Region {self.bbox}: 'lattice' method found no meaningful tables")

            except Exception as e:
                logger.debug(f"Region {self.bbox}: 'lattice' method failed: {e}")

            # Fall back to stream
            logger.debug(f"Region {self.bbox}: Falling back to 'stream' method for tables...")
            stream_settings = table_settings.copy()
            stream_settings.setdefault("vertical_strategy", "text")
            stream_settings.setdefault("horizontal_strategy", "text")

            return self._extract_tables_plumber(stream_settings)

        effective_method = method

        # Handle method aliases
        if effective_method == "stream":
            logger.debug("Using 'stream' method alias for 'pdfplumber' with text-based strategies.")
            effective_method = "pdfplumber"
            table_settings.setdefault("vertical_strategy", "text")
            table_settings.setdefault("horizontal_strategy", "text")
        elif effective_method == "lattice":
            logger.debug(
                "Using 'lattice' method alias for 'pdfplumber' with line-based strategies."
            )
            effective_method = "pdfplumber"
            table_settings.setdefault("vertical_strategy", "lines")
            table_settings.setdefault("horizontal_strategy", "lines")

        # Use the selected method
        if effective_method == "pdfplumber":
            return self._extract_tables_plumber(table_settings)
        else:
            raise ValueError(
                f"Unknown tables extraction method: '{method}'. Choose from 'pdfplumber', 'stream', 'lattice'."
            )

    def _extract_tables_plumber(self, table_settings: dict) -> List[List[List[str]]]:
        """
        Extract all tables using pdfplumber's table extraction.

        Args:
            table_settings: Settings for pdfplumber table extraction

        Returns:
            List of tables, where each table is a list of rows, and each row is a list of cell values
        """
        # Inject global PDF-level text tolerances if not explicitly present
        pdf_cfg = getattr(self.page, "_config", getattr(self.page._parent, "_config", {}))
        _uses_text = "text" in (
            table_settings.get("vertical_strategy"),
            table_settings.get("horizontal_strategy"),
        )
        if (
            _uses_text
            and "text_x_tolerance" not in table_settings
            and "x_tolerance" not in table_settings
        ):
            x_tol = pdf_cfg.get("x_tolerance")
            if x_tol is not None:
                table_settings.setdefault("text_x_tolerance", x_tol)
        if (
            _uses_text
            and "text_y_tolerance" not in table_settings
            and "y_tolerance" not in table_settings
        ):
            y_tol = pdf_cfg.get("y_tolerance")
            if y_tol is not None:
                table_settings.setdefault("text_y_tolerance", y_tol)

        if (
            _uses_text
            and "snap_tolerance" not in table_settings
            and "snap_x_tolerance" not in table_settings
        ):
            snap = max(1, round((pdf_cfg.get("y_tolerance", 1)) * 0.9))
            table_settings.setdefault("snap_tolerance", snap)
        if (
            _uses_text
            and "join_tolerance" not in table_settings
            and "join_x_tolerance" not in table_settings
        ):
            join = table_settings.get("snap_tolerance", 1)
            table_settings.setdefault("join_tolerance", join)
            table_settings.setdefault("join_x_tolerance", join)
            table_settings.setdefault("join_y_tolerance", join)

        # -------------------------------------------------------------
        # Apply char-level exclusion filtering, if any exclusions are
        # defined on the parent Page.  We create a lightweight
        # pdfplumber.Page copy whose .chars list omits characters that
        # fall inside any exclusion Region.  Other object types are
        # left untouched for now ("chars-only" strategy).
        # -------------------------------------------------------------
        base_plumber_page = self.page._page

        if getattr(self.page, "_exclusions", None):
            # Resolve exclusion Regions (callables already evaluated)
            exclusion_regions = self.page._get_exclusion_regions(include_callable=True)

            def _keep_char(obj):
                """Return True if pdfplumber obj should be kept."""
                if obj.get("object_type") != "char":
                    # Keep non-char objects unchanged – lattice grids etc.
                    return True

                # Compute character centre point
                cx = (obj["x0"] + obj["x1"]) / 2.0
                cy = (obj["top"] + obj["bottom"]) / 2.0

                # Reject if the centre lies inside ANY exclusion Region
                for reg in exclusion_regions:
                    if reg.x0 <= cx <= reg.x1 and reg.top <= cy <= reg.bottom:
                        return False
                return True

            try:
                filtered_page = base_plumber_page.filter(_keep_char)
            except Exception as _filter_err:
                # Fallback – if filtering fails, log and proceed unfiltered
                logger.warning(
                    f"Region {self.bbox}: Failed to filter pdfplumber chars for exclusions: {_filter_err}"
                )
                filtered_page = base_plumber_page
        else:
            filtered_page = base_plumber_page

        # Ensure bbox is within pdfplumber page bounds
        page_bbox = filtered_page.bbox
        clipped_bbox = (
            max(self.bbox[0], page_bbox[0]),  # x0
            max(self.bbox[1], page_bbox[1]),  # y0
            min(self.bbox[2], page_bbox[2]),  # x1
            min(self.bbox[3], page_bbox[3]),  # y1
        )

        # Only crop if the clipped bbox is valid (has positive width and height)
        if clipped_bbox[2] > clipped_bbox[0] and clipped_bbox[3] > clipped_bbox[1]:
            cropped = filtered_page.crop(clipped_bbox)
        else:
            # If the region is completely outside the page bounds, return empty list
            return []

        # Extract all tables from the cropped area
        tables = cropped.extract_tables(table_settings)

        # Apply RTL text processing to all tables
        if tables:
            processed_tables = []
            for table in tables:
                processed_table = []
                for row in table:
                    processed_row = []
                    for cell in row:
                        if cell is not None:
                            # Apply RTL text processing to each cell
                            rtl_processed_cell = self._apply_rtl_processing_to_text(cell)
                            processed_row.append(rtl_processed_cell)
                        else:
                            processed_row.append(cell)
                    processed_table.append(processed_row)
                processed_tables.append(processed_table)
            return processed_tables

        # Return empty list if no tables found
        return []

    def _extract_table_plumber(
        self, table_settings: dict, content_filter=None, apply_exclusions=True
    ) -> List[List[str]]:
        """
        Extract table using pdfplumber's table extraction.
        This method extracts the largest table within the region.

        Args:
            table_settings: Settings for pdfplumber table extraction
            content_filter: Optional content filter to apply to cell values

        Returns:
            Table data as a list of rows, where each row is a list of cell values
        """
        # Inject global PDF-level text tolerances if not explicitly present
        pdf_cfg = getattr(self.page, "_config", getattr(self.page._parent, "_config", {}))
        _uses_text = "text" in (
            table_settings.get("vertical_strategy"),
            table_settings.get("horizontal_strategy"),
        )
        if (
            _uses_text
            and "text_x_tolerance" not in table_settings
            and "x_tolerance" not in table_settings
        ):
            x_tol = pdf_cfg.get("x_tolerance")
            if x_tol is not None:
                table_settings.setdefault("text_x_tolerance", x_tol)
        if (
            _uses_text
            and "text_y_tolerance" not in table_settings
            and "y_tolerance" not in table_settings
        ):
            y_tol = pdf_cfg.get("y_tolerance")
            if y_tol is not None:
                table_settings.setdefault("text_y_tolerance", y_tol)

        # -------------------------------------------------------------
        # Apply char-level exclusion filtering (chars only) just like in
        # _extract_tables_plumber so header/footer text does not appear
        # in extracted tables.
        # -------------------------------------------------------------
        base_plumber_page = self.page._page

        if apply_exclusions and getattr(self.page, "_exclusions", None):
            exclusion_regions = self.page._get_exclusion_regions(include_callable=True)

            def _keep_char(obj):
                if obj.get("object_type") != "char":
                    return True
                cx = (obj["x0"] + obj["x1"]) / 2.0
                cy = (obj["top"] + obj["bottom"]) / 2.0
                for reg in exclusion_regions:
                    if reg.x0 <= cx <= reg.x1 and reg.top <= cy <= reg.bottom:
                        return False
                return True

            try:
                filtered_page = base_plumber_page.filter(_keep_char)
            except Exception as _filter_err:
                logger.warning(
                    f"Region {self.bbox}: Failed to filter pdfplumber chars for exclusions (single table): {_filter_err}"
                )
                filtered_page = base_plumber_page
        else:
            filtered_page = base_plumber_page

        # Now crop the (possibly filtered) page to the region bbox
        # Ensure bbox is within pdfplumber page bounds
        page_bbox = filtered_page.bbox
        clipped_bbox = (
            max(self.bbox[0], page_bbox[0]),  # x0
            max(self.bbox[1], page_bbox[1]),  # y0
            min(self.bbox[2], page_bbox[2]),  # x1
            min(self.bbox[3], page_bbox[3]),  # y1
        )

        # Only crop if the clipped bbox is valid (has positive width and height)
        if clipped_bbox[2] > clipped_bbox[0] and clipped_bbox[3] > clipped_bbox[1]:
            cropped = filtered_page.crop(clipped_bbox)
        else:
            # If the region is completely outside the page bounds, return empty table
            return []

        # Extract the single largest table from the cropped area
        table = cropped.extract_table(table_settings)

        # Return the table or an empty list if none found
        if table:
            # Apply RTL text processing and content filtering if provided
            processed_table = []
            for row in table:
                processed_row = []
                for cell in row:
                    if cell is not None:
                        # Apply RTL text processing first
                        rtl_processed_cell = self._apply_rtl_processing_to_text(cell)

                        # Then apply content filter if provided
                        if content_filter is not None:
                            filtered_cell = self._apply_content_filter_to_text(
                                rtl_processed_cell, content_filter
                            )
                            processed_row.append(filtered_cell)
                        else:
                            processed_row.append(rtl_processed_cell)
                    else:
                        processed_row.append(cell)
                processed_table.append(processed_row)
            return processed_table
        return []

    def _extract_table_tatr(
        self, use_ocr=False, ocr_config=None, content_filter=None, apply_exclusions=True
    ) -> List[List[str]]:
        """
        Extract table using TATR structure detection.

        Args:
            use_ocr: Whether to apply OCR to each cell for better text extraction
            ocr_config: Optional OCR configuration parameters
            content_filter: Optional content filter to apply to cell values

        Returns:
            Table data as a list of rows, where each row is a list of cell values
        """
        # Find all rows and headers in this table
        rows = self.page.find_all(f"region[type=table-row][model=tatr]")
        headers = self.page.find_all(f"region[type=table-column-header][model=tatr]")
        columns = self.page.find_all(f"region[type=table-column][model=tatr]")

        # Filter to only include rows/headers/columns that overlap with this table region
        def is_in_table(region):
            # Check for overlap - simplifying to center point for now
            region_center_x = (region.x0 + region.x1) / 2
            region_center_y = (region.top + region.bottom) / 2
            return (
                self.x0 <= region_center_x <= self.x1 and self.top <= region_center_y <= self.bottom
            )

        rows = [row for row in rows if is_in_table(row)]
        headers = [header for header in headers if is_in_table(header)]
        columns = [column for column in columns if is_in_table(column)]

        # Sort rows by vertical position (top to bottom)
        rows.sort(key=lambda r: r.top)

        # Sort columns by horizontal position (left to right)
        columns.sort(key=lambda c: c.x0)

        # Create table data structure
        table_data = []

        # Prepare OCR config if needed
        if use_ocr:
            # Default OCR config focuses on small text with low confidence
            default_ocr_config = {
                "enabled": True,
                "min_confidence": 0.1,  # Lower than default to catch more text
                "detection_params": {
                    "text_threshold": 0.1,  # Lower threshold for low-contrast text
                    "link_threshold": 0.1,  # Lower threshold for connecting text components
                },
            }

            # Merge with provided config if any
            if ocr_config:
                if isinstance(ocr_config, dict):
                    # Update default config with provided values
                    for key, value in ocr_config.items():
                        if (
                            isinstance(value, dict)
                            and key in default_ocr_config
                            and isinstance(default_ocr_config[key], dict)
                        ):
                            # Merge nested dicts
                            default_ocr_config[key].update(value)
                        else:
                            # Replace value
                            default_ocr_config[key] = value
                else:
                    # Not a dict, use as is
                    default_ocr_config = ocr_config

            # Use the merged config
            ocr_config = default_ocr_config

        # Add header row if headers were detected
        if headers:
            header_texts = []
            for header in headers:
                if use_ocr:
                    # Try OCR for better text extraction
                    ocr_elements = header.apply_ocr(**ocr_config)
                    if ocr_elements:
                        ocr_text = " ".join(e.text for e in ocr_elements).strip()
                        if ocr_text:
                            header_texts.append(ocr_text)
                            continue

                # Fallback to normal extraction
                header_text = header.extract_text(apply_exclusions=apply_exclusions).strip()
                if content_filter is not None:
                    header_text = self._apply_content_filter_to_text(header_text, content_filter)
                header_texts.append(header_text)
            table_data.append(header_texts)

        # Process rows
        for row in rows:
            row_cells = []

            # If we have columns, use them to extract cells
            if columns:
                for column in columns:
                    # Create a cell region at the intersection of row and column
                    cell_bbox = (column.x0, row.top, column.x1, row.bottom)

                    # Create a region for this cell
                    from natural_pdf.elements.region import (  # Import here to avoid circular imports
                        Region,
                    )

                    cell_region = Region(self.page, cell_bbox)

                    # Extract text from the cell
                    if use_ocr:
                        # Apply OCR to the cell
                        ocr_elements = cell_region.apply_ocr(**ocr_config)
                        if ocr_elements:
                            # Get text from OCR elements
                            ocr_text = " ".join(e.text for e in ocr_elements).strip()
                            if ocr_text:
                                row_cells.append(ocr_text)
                                continue

                    # Fallback to normal extraction
                    cell_text = cell_region.extract_text(apply_exclusions=apply_exclusions).strip()
                    if content_filter is not None:
                        cell_text = self._apply_content_filter_to_text(cell_text, content_filter)
                    row_cells.append(cell_text)
            else:
                # No column information, just extract the whole row text
                if use_ocr:
                    # Try OCR on the whole row
                    ocr_elements = row.apply_ocr(**ocr_config)
                    if ocr_elements:
                        ocr_text = " ".join(e.text for e in ocr_elements).strip()
                        if ocr_text:
                            row_cells.append(ocr_text)
                            continue

                # Fallback to normal extraction
                row_text = row.extract_text(apply_exclusions=apply_exclusions).strip()
                if content_filter is not None:
                    row_text = self._apply_content_filter_to_text(row_text, content_filter)
                row_cells.append(row_text)

            table_data.append(row_cells)

        return table_data

    def _extract_table_text(self, **text_options) -> List[List[Optional[str]]]:
        """
        Extracts table content based on text alignment analysis.

        Args:
            **text_options: Options passed to analyze_text_table_structure,
                          plus optional 'cell_extraction_func', 'coordinate_grouping_tolerance',
                          'show_progress', and 'content_filter'.

        Returns:
            Table data as list of lists of strings (or None for empty cells).
        """
        cell_extraction_func = text_options.pop("cell_extraction_func", None)
        # --- Get show_progress option --- #
        show_progress = text_options.pop("show_progress", False)
        # --- Get content_filter option --- #
        content_filter = text_options.pop("content_filter", None)
        # --- Get apply_exclusions option --- #
        apply_exclusions = text_options.pop("apply_exclusions", True)

        # Analyze structure first (or use cached results)
        if "text_table_structure" in self.analyses:
            analysis_results = self.analyses["text_table_structure"]
            logger.debug("Using cached text table structure analysis results.")
        else:
            analysis_results = self.analyze_text_table_structure(**text_options)

        if analysis_results is None or not analysis_results.get("cells"):
            logger.warning(f"Region {self.bbox}: No cells found using 'text' method.")
            return []

        cell_dicts = analysis_results["cells"]

        # --- Grid Reconstruction Logic --- #
        if not cell_dicts:
            return []

        # 1. Get unique sorted top and left coordinates (cell boundaries)
        coord_tolerance = text_options.get("coordinate_grouping_tolerance", 1)
        tops = sorted(
            list(set(round(c["top"] / coord_tolerance) * coord_tolerance for c in cell_dicts))
        )
        lefts = sorted(
            list(set(round(c["left"] / coord_tolerance) * coord_tolerance for c in cell_dicts))
        )

        # Refine boundaries (cluster_coords helper remains the same)
        def cluster_coords(coords):
            if not coords:
                return []
            clustered = []
            current_cluster = [coords[0]]
            for c in coords[1:]:
                if abs(c - current_cluster[-1]) <= coord_tolerance:
                    current_cluster.append(c)
                else:
                    clustered.append(min(current_cluster))
                    current_cluster = [c]
            clustered.append(min(current_cluster))
            return clustered

        unique_tops = cluster_coords(tops)
        unique_lefts = cluster_coords(lefts)

        # Determine iterable for tqdm
        cell_iterator = cell_dicts
        if show_progress:
            # Only wrap if progress should be shown
            cell_iterator = tqdm(
                cell_dicts,
                desc=f"Extracting text from {len(cell_dicts)} cells (text method)",
                unit="cell",
                leave=False,  # Optional: Keep bar after completion
            )
        # --- End tqdm Setup --- #

        # 2. Create a lookup map for cell text: {(rounded_top, rounded_left): cell_text}
        cell_text_map = {}
        # --- Use the potentially wrapped iterator --- #
        for cell_data in cell_iterator:
            try:
                cell_region = self.page.region(**cell_data)
                cell_value = None  # Initialize
                if callable(cell_extraction_func):
                    try:
                        cell_value = cell_extraction_func(cell_region)
                        if not isinstance(cell_value, (str, type(None))):
                            logger.warning(
                                f"Custom cell_extraction_func returned non-string/None type ({type(cell_value)}) for cell {cell_data}. Treating as None."
                            )
                            cell_value = None
                    except Exception as func_err:
                        logger.error(
                            f"Error executing custom cell_extraction_func for cell {cell_data}: {func_err}",
                            exc_info=True,
                        )
                        cell_value = None
                else:
                    cell_value = cell_region.extract_text(
                        layout=False,
                        apply_exclusions=apply_exclusions,
                        content_filter=content_filter,
                    ).strip()

                rounded_top = round(cell_data["top"] / coord_tolerance) * coord_tolerance
                rounded_left = round(cell_data["left"] / coord_tolerance) * coord_tolerance
                cell_text_map[(rounded_top, rounded_left)] = cell_value
            except Exception as e:
                logger.warning(f"Could not process cell {cell_data} for text extraction: {e}")

        # 3. Build the final list-of-lists table (loop remains the same)
        final_table = []
        for row_top in unique_tops:
            row_data = []
            for col_left in unique_lefts:
                best_match_key = None
                min_dist_sq = float("inf")
                for map_top, map_left in cell_text_map.keys():
                    if (
                        abs(map_top - row_top) <= coord_tolerance
                        and abs(map_left - col_left) <= coord_tolerance
                    ):
                        dist_sq = (map_top - row_top) ** 2 + (map_left - col_left) ** 2
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            best_match_key = (map_top, map_left)
                cell_value = cell_text_map.get(best_match_key)
                row_data.append(cell_value)
            final_table.append(row_data)

        return final_table

    # --- END MODIFIED METHOD --- #

    @overload
    def find(
        self,
        *,
        text: str,
        overlap: str = "full",
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> Optional["Element"]: ...

    @overload
    def find(
        self,
        selector: str,
        *,
        overlap: str = "full",
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> Optional["Element"]: ...

    def find(
        self,
        selector: Optional[str] = None,  # Now optional
        *,
        text: Optional[str] = None,  # New text parameter
        overlap: str = "full",  # How elements overlap with the region
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> Optional["Element"]:
        """
        Find the first element in this region matching the selector OR text content.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for (equivalent to 'text:contains(...)').
            overlap: How to determine if elements overlap with the region: 'full' (fully inside),
                     'partial' (any overlap), or 'center' (center point inside).
                     (default: "full")
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            **kwargs: Additional parameters for element filtering.

        Returns:
            First matching element or None.
        """
        # Delegate validation and selector construction to find_all
        elements = self.find_all(
            selector=selector,
            text=text,
            overlap=overlap,
            apply_exclusions=apply_exclusions,
            regex=regex,
            case=case,
            **kwargs,
        )
        return elements.first if elements else None

    @overload
    def find_all(
        self,
        *,
        text: str,
        overlap: str = "full",
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> "ElementCollection": ...

    @overload
    def find_all(
        self,
        selector: str,
        *,
        overlap: str = "full",
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> "ElementCollection": ...

    def find_all(
        self,
        selector: Optional[str] = None,  # Now optional
        *,
        text: Optional[str] = None,  # New text parameter
        overlap: str = "full",  # How elements overlap with the region
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> "ElementCollection":
        """
        Find all elements in this region matching the selector OR text content.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for (equivalent to 'text:contains(...)').
            overlap: How to determine if elements overlap with the region: 'full' (fully inside),
                     'partial' (any overlap), or 'center' (center point inside).
                     (default: "full")
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            **kwargs: Additional parameters for element filtering.

        Returns:
            ElementCollection with matching elements.
        """
        from natural_pdf.elements.element_collection import ElementCollection

        if selector is not None and text is not None:
            raise ValueError("Provide either 'selector' or 'text', not both.")
        if selector is None and text is None:
            raise ValueError("Provide either 'selector' or 'text'.")

        # Validate overlap parameter
        if overlap not in ["full", "partial", "center"]:
            raise ValueError(
                f"Invalid overlap value: {overlap}. Must be 'full', 'partial', or 'center'"
            )

        # Construct selector if 'text' is provided
        effective_selector = ""
        if text is not None:
            escaped_text = text.replace('"', '\\"').replace("'", "\\'")
            effective_selector = f'text:contains("{escaped_text}")'
            logger.debug(
                f"Using text shortcut: find_all(text='{text}') -> find_all('{effective_selector}')"
            )
        elif selector is not None:
            effective_selector = selector
        else:
            raise ValueError("Internal error: No selector or text provided.")

        # Normal case: Region is on a single page
        try:
            # Parse the final selector string
            selector_obj = parse_selector(effective_selector)

            # Get all potentially relevant elements from the page
            # Let the page handle its exclusion logic if needed
            potential_elements = self.page.find_all(
                selector=effective_selector,
                apply_exclusions=apply_exclusions,
                regex=regex,
                case=case,
                **kwargs,
            )

            # Filter these elements based on the specified containment method
            region_bbox = self.bbox
            matching_elements = []

            if overlap == "full":  # Fully inside (strict)
                matching_elements = [
                    el
                    for el in potential_elements
                    if el.x0 >= region_bbox[0]
                    and el.top >= region_bbox[1]
                    and el.x1 <= region_bbox[2]
                    and el.bottom <= region_bbox[3]
                ]
            elif overlap == "partial":  # Any overlap
                matching_elements = [el for el in potential_elements if self.intersects(el)]
            elif overlap == "center":  # Center point inside
                matching_elements = [
                    el for el in potential_elements if self.is_element_center_inside(el)
                ]

            return ElementCollection(matching_elements)

        except Exception as e:
            logger.error(f"Error during find_all in region: {e}", exc_info=True)
            return ElementCollection([])

    def apply_ocr(self, replace=True, **ocr_params) -> "Region":
        """
        Apply OCR to this region and return the created text elements.

        This method supports two modes:
        1. **Built-in OCR Engines** (default) – identical to previous behaviour. Pass typical
           parameters like ``engine='easyocr'`` or ``languages=['en']`` and the method will
           route the request through :class:`OCRManager`.
        2. **Custom OCR Function** – pass a *callable* under the keyword ``function`` (or
           ``ocr_function``). The callable will receive *this* Region instance and should
           return the extracted text (``str``) or ``None``.  Internally the call is
           delegated to :pymeth:`apply_custom_ocr` so the same logic (replacement, element
           creation, etc.) is re-used.

        Examples
        ---------
        ```python
        def llm_ocr(region):
            image = region.render(resolution=300, crop=True)
            return my_llm_client.ocr(image)
        region.apply_ocr(function=llm_ocr)
        ```

        Args:
            replace: Whether to remove existing OCR elements first (default ``True``).
            **ocr_params: Parameters for the built-in OCR manager *or* the special
                          ``function``/``ocr_function`` keyword to trigger custom mode.

        Returns
        -------
            Self – for chaining.
        """
        # --- Custom OCR function path --------------------------------------------------
        custom_func = ocr_params.pop("function", None) or ocr_params.pop("ocr_function", None)
        if callable(custom_func):
            # Delegate to the specialised helper while preserving key kwargs
            return self.apply_custom_ocr(
                ocr_function=custom_func,
                source_label=ocr_params.pop("source_label", "custom-ocr"),
                replace=replace,
                confidence=ocr_params.pop("confidence", None),
                add_to_page=ocr_params.pop("add_to_page", True),
            )

        # --- Original built-in OCR engine path (unchanged except docstring) ------------
        # Ensure OCRManager is available
        if not hasattr(self.page._parent, "_ocr_manager") or self.page._parent._ocr_manager is None:
            logger.error("OCRManager not available on parent PDF. Cannot apply OCR to region.")
            return self

        # If replace is True, find and remove existing OCR elements in this region
        if replace:
            logger.info(
                f"Region {self.bbox}: Removing existing OCR elements before applying new OCR."
            )

            # --- Robust removal: iterate through all OCR elements on the page and
            #     remove those that overlap this region. This avoids reliance on
            #     identity‐based look-ups that can break if the ElementManager
            #     rebuilt its internal lists.

            removed_count = 0

            # Helper to remove a single element safely
            def _safe_remove(elem):
                nonlocal removed_count
                success = False
                if hasattr(elem, "page") and hasattr(elem.page, "_element_mgr"):
                    etype = getattr(elem, "object_type", "word")
                    if etype == "word":
                        etype_key = "words"
                    elif etype == "char":
                        etype_key = "chars"
                    else:
                        etype_key = etype + "s" if not etype.endswith("s") else etype
                    try:
                        success = elem.page._element_mgr.remove_element(elem, etype_key)
                    except Exception:
                        success = False
                if success:
                    removed_count += 1

            # Remove OCR WORD elements overlapping region
            for word in list(self.page._element_mgr.words):
                if getattr(word, "source", None) == "ocr" and self.intersects(word):
                    _safe_remove(word)

            # Remove OCR CHAR dicts overlapping region
            for char in list(self.page._element_mgr.chars):
                # char can be dict or TextElement; normalise
                char_src = (
                    char.get("source") if isinstance(char, dict) else getattr(char, "source", None)
                )
                if char_src == "ocr":
                    # Rough bbox for dicts
                    if isinstance(char, dict):
                        cx0, ctop, cx1, cbottom = (
                            char.get("x0", 0),
                            char.get("top", 0),
                            char.get("x1", 0),
                            char.get("bottom", 0),
                        )
                    else:
                        cx0, ctop, cx1, cbottom = char.x0, char.top, char.x1, char.bottom
                    # Quick overlap check
                    if not (
                        cx1 < self.x0 or cx0 > self.x1 or cbottom < self.top or ctop > self.bottom
                    ):
                        _safe_remove(char)

            logger.info(
                f"Region {self.bbox}: Removed {removed_count} existing OCR elements (words & chars) before re-applying OCR."
            )

        ocr_mgr = self.page._parent._ocr_manager

        # Determine rendering resolution from parameters
        final_resolution = ocr_params.get("resolution")
        if final_resolution is None and hasattr(self.page, "_parent") and self.page._parent:
            final_resolution = getattr(self.page._parent, "_config", {}).get("resolution", 150)
        elif final_resolution is None:
            final_resolution = 150
        logger.debug(
            f"Region {self.bbox}: Applying OCR with resolution {final_resolution} DPI and params: {ocr_params}"
        )

        # Render the page region to an image using the determined resolution
        try:
            # Use render() for clean image without highlights, with cropping
            region_image = self.render(resolution=final_resolution, crop=True)
            if not region_image:
                logger.error("Failed to render region to image for OCR.")
                return self
            logger.debug(f"Region rendered to image size: {region_image.size}")
        except Exception as e:
            logger.error(f"Error rendering region to image for OCR: {e}", exc_info=True)
            return self

        # Prepare args for the OCR Manager
        manager_args = {
            "images": region_image,
            "engine": ocr_params.get("engine"),
            "languages": ocr_params.get("languages"),
            "min_confidence": ocr_params.get("min_confidence"),
            "device": ocr_params.get("device"),
            "options": ocr_params.get("options"),
            "detect_only": ocr_params.get("detect_only"),
        }
        manager_args = {k: v for k, v in manager_args.items() if v is not None}

        # Run OCR on this region's image using the manager
        results = ocr_mgr.apply_ocr(**manager_args)
        if not isinstance(results, list):
            logger.error(
                f"OCRManager returned unexpected type for single region image: {type(results)}"
            )
            return self
        logger.debug(f"Region OCR processing returned {len(results)} results.")

        # Convert results to TextElements
        scale_x = self.width / region_image.width if region_image.width > 0 else 1.0
        scale_y = self.height / region_image.height if region_image.height > 0 else 1.0
        logger.debug(f"Region OCR scaling factors (PDF/Img): x={scale_x:.2f}, y={scale_y:.2f}")
        created_elements = []
        for result in results:
            try:
                img_x0, img_top, img_x1, img_bottom = map(float, result["bbox"])
                pdf_height = (img_bottom - img_top) * scale_y
                page_x0 = self.x0 + (img_x0 * scale_x)
                page_top = self.top + (img_top * scale_y)
                page_x1 = self.x0 + (img_x1 * scale_x)
                page_bottom = self.top + (img_bottom * scale_y)
                raw_conf = result.get("confidence")
                # Convert confidence to float unless it is None/invalid
                try:
                    confidence_val = float(raw_conf) if raw_conf is not None else None
                except (TypeError, ValueError):
                    confidence_val = None

                text_val = result.get("text")  # May legitimately be None in detect_only mode

                element_data = {
                    "text": text_val,
                    "x0": page_x0,
                    "top": page_top,
                    "x1": page_x1,
                    "bottom": page_bottom,
                    "width": page_x1 - page_x0,
                    "height": page_bottom - page_top,
                    "object_type": "word",
                    "source": "ocr",
                    "confidence": confidence_val,
                    "fontname": "OCR",
                    "size": round(pdf_height) if pdf_height > 0 else 10.0,
                    "page_number": self.page.number,
                    "bold": False,
                    "italic": False,
                    "upright": True,
                    "doctop": page_top + self.page._page.initial_doctop,
                }
                ocr_char_dict = element_data.copy()
                ocr_char_dict["object_type"] = "char"
                ocr_char_dict.setdefault("adv", ocr_char_dict.get("width", 0))
                element_data["_char_dicts"] = [ocr_char_dict]
                from natural_pdf.elements.text import TextElement

                elem = TextElement(element_data, self.page)
                created_elements.append(elem)
                self.page._element_mgr.add_element(elem, element_type="words")
                self.page._element_mgr.add_element(ocr_char_dict, element_type="chars")
            except Exception as e:
                logger.error(
                    f"Failed to convert region OCR result to element: {result}. Error: {e}",
                    exc_info=True,
                )
        logger.info(f"Region {self.bbox}: Added {len(created_elements)} elements from OCR.")
        return self

    def apply_custom_ocr(
        self,
        ocr_function: Callable[["Region"], Optional[str]],
        source_label: str = "custom-ocr",
        replace: bool = True,
        confidence: Optional[float] = None,
        add_to_page: bool = True,
    ) -> "Region":
        """
        Apply a custom OCR function to this region and create text elements from the results.

        This is useful when you want to use a custom OCR method (e.g., an LLM API,
        specialized OCR service, or any custom logic) instead of the built-in OCR engines.

        Args:
            ocr_function: A callable that takes a Region and returns the OCR'd text (or None).
                          The function receives this region as its argument and should return
                          the extracted text as a string, or None if no text was found.
            source_label: Label to identify the source of these text elements (default: "custom-ocr").
                          This will be set as the 'source' attribute on created elements.
            replace: If True (default), removes existing OCR elements in this region before
                     adding new ones. If False, adds new OCR elements alongside existing ones.
            confidence: Optional confidence score for the OCR result (0.0-1.0).
                        If None, defaults to 1.0 if text is returned, 0.0 if None is returned.
            add_to_page: If True (default), adds the created text element to the page.
                         If False, creates the element but doesn't add it to the page.

        Returns:
            Self for method chaining.

        Example:
            # Using with an LLM
            def ocr_with_llm(region):
                image = region.render(resolution=300, crop=True)
                # Call your LLM API here
                return llm_client.ocr(image)

            region.apply_custom_ocr(ocr_with_llm)

            # Using with a custom OCR service
            def ocr_with_service(region):
                img_bytes = region.render(crop=True).tobytes()
                response = ocr_service.process(img_bytes)
                return response.text

            region.apply_custom_ocr(ocr_with_service, source_label="my-ocr-service")
        """
        # If replace is True, remove existing OCR elements in this region
        if replace:
            logger.info(
                f"Region {self.bbox}: Removing existing OCR elements before applying custom OCR."
            )

            removed_count = 0

            # Helper to remove a single element safely
            def _safe_remove(elem):
                nonlocal removed_count
                success = False
                if hasattr(elem, "page") and hasattr(elem.page, "_element_mgr"):
                    etype = getattr(elem, "object_type", "word")
                    if etype == "word":
                        etype_key = "words"
                    elif etype == "char":
                        etype_key = "chars"
                    else:
                        etype_key = etype + "s" if not etype.endswith("s") else etype
                    try:
                        success = elem.page._element_mgr.remove_element(elem, etype_key)
                    except Exception:
                        success = False
                if success:
                    removed_count += 1

            # Remove ALL OCR elements overlapping this region
            # Remove elements with source=="ocr" (built-in OCR) or matching the source_label (previous custom OCR)
            for word in list(self.page._element_mgr.words):
                word_source = getattr(word, "source", "")
                # Match built-in OCR behavior: remove elements with source "ocr" exactly
                # Also remove elements with the same source_label to avoid duplicates
                if (word_source == "ocr" or word_source == source_label) and self.intersects(word):
                    _safe_remove(word)

            # Also remove char dicts if needed (matching built-in OCR)
            for char in list(self.page._element_mgr.chars):
                # char can be dict or TextElement; normalize
                char_src = (
                    char.get("source") if isinstance(char, dict) else getattr(char, "source", None)
                )
                if char_src == "ocr" or char_src == source_label:
                    # Rough bbox for dicts
                    if isinstance(char, dict):
                        cx0, ctop, cx1, cbottom = (
                            char.get("x0", 0),
                            char.get("top", 0),
                            char.get("x1", 0),
                            char.get("bottom", 0),
                        )
                    else:
                        cx0, ctop, cx1, cbottom = char.x0, char.top, char.x1, char.bottom
                    # Quick overlap check
                    if not (
                        cx1 < self.x0 or cx0 > self.x1 or cbottom < self.top or ctop > self.bottom
                    ):
                        _safe_remove(char)

            if removed_count > 0:
                logger.info(f"Region {self.bbox}: Removed {removed_count} existing OCR elements.")

        # Call the custom OCR function
        try:
            logger.debug(f"Region {self.bbox}: Calling custom OCR function...")
            ocr_text = ocr_function(self)

            if ocr_text is not None and not isinstance(ocr_text, str):
                logger.warning(
                    f"Custom OCR function returned non-string type ({type(ocr_text)}). "
                    f"Converting to string."
                )
                ocr_text = str(ocr_text)

        except Exception as e:
            logger.error(
                f"Error calling custom OCR function for region {self.bbox}: {e}", exc_info=True
            )
            return self

        # Create text element if we got text
        if ocr_text is not None:
            # Use the to_text_element method to create the element
            text_element = self.to_text_element(
                text_content=ocr_text,
                source_label=source_label,
                confidence=confidence,
                add_to_page=add_to_page,
            )

            logger.info(
                f"Region {self.bbox}: Created text element with {len(ocr_text)} chars"
                f"{' and added to page' if add_to_page else ''}"
            )
        else:
            logger.debug(f"Region {self.bbox}: Custom OCR function returned None (no text found)")

        return self

    def get_section_between(
        self,
        start_element=None,
        end_element=None,
        include_boundaries="both",
        orientation="vertical",
    ):
        """
        Get a section between two elements within this region.

        Args:
            start_element: Element marking the start of the section
            end_element: Element marking the end of the section
            include_boundaries: How to include boundary elements: 'start', 'end', 'both', or 'none'
            orientation: 'vertical' (default) or 'horizontal' - determines section direction

        Returns:
            Region representing the section
        """
        # Get elements only within this region first
        elements = self.get_elements()

        # If no elements, return self or empty region?
        if not elements:
            logger.warning(
                f"get_section_between called on region {self.bbox} with no contained elements."
            )
            # Return an empty region at the start of the parent region
            return Region(self.page, (self.x0, self.top, self.x0, self.top))

        # Sort elements in reading order
        elements.sort(key=lambda e: (e.top, e.x0))

        # Find start index
        start_idx = 0
        if start_element:
            try:
                start_idx = elements.index(start_element)
            except ValueError:
                # Start element not in region, use first element
                logger.debug("Start element not found in region, using first element.")
                start_element = elements[0]  # Use the actual first element
                start_idx = 0
        else:
            start_element = elements[0]  # Default start is first element

        # Find end index
        end_idx = len(elements) - 1
        if end_element:
            try:
                end_idx = elements.index(end_element)
            except ValueError:
                # End element not in region, use last element
                logger.debug("End element not found in region, using last element.")
                end_element = elements[-1]  # Use the actual last element
                end_idx = len(elements) - 1
        else:
            end_element = elements[-1]  # Default end is last element

        # Validate orientation parameter
        if orientation not in ["vertical", "horizontal"]:
            raise ValueError(f"orientation must be 'vertical' or 'horizontal', got '{orientation}'")

        # Use centralized section utilities
        from natural_pdf.utils.sections import calculate_section_bounds, validate_section_bounds

        # Calculate section boundaries
        bounds = calculate_section_bounds(
            start_element=start_element,
            end_element=end_element,
            include_boundaries=include_boundaries,
            orientation=orientation,
            parent_bounds=self.bbox,
        )

        # Validate boundaries
        if not validate_section_bounds(bounds, orientation):
            # Return an empty region at the start position
            x0, top, _, _ = bounds
            return Region(self.page, (x0, top, x0, top))

        # Create new region
        section = Region(self.page, bounds)

        # Store the original boundary elements and exclusion info
        section.start_element = start_element
        section.end_element = end_element
        section._boundary_exclusions = include_boundaries

        return section

    def get_sections(
        self,
        start_elements=None,
        end_elements=None,
        include_boundaries="both",
        orientation="vertical",
    ) -> "ElementCollection[Region]":
        """
        Get sections within this region based on start/end elements.

        Args:
            start_elements: Elements or selector string that mark the start of sections
            end_elements: Elements or selector string that mark the end of sections
            include_boundaries: How to include boundary elements: 'start', 'end', 'both', or 'none'
            orientation: 'vertical' (default) or 'horizontal' - determines section direction

        Returns:
            List of Region objects representing the extracted sections
        """
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.utils.sections import extract_sections_from_region

        # Use centralized section extraction logic
        sections = extract_sections_from_region(
            region=self,
            start_elements=start_elements,
            end_elements=end_elements,
            include_boundaries=include_boundaries,
            orientation=orientation,
        )

        return ElementCollection(sections)

    def split(self, divider, **kwargs) -> "ElementCollection[Region]":
        """
        Divide this region into sections based on the provided divider elements.

        Args:
            divider: Elements or selector string that mark section boundaries
            **kwargs: Additional parameters passed to get_sections()
                - include_boundaries: How to include boundary elements (default: 'start')
                - orientation: 'vertical' or 'horizontal' (default: 'vertical')

        Returns:
            ElementCollection of Region objects representing the sections

        Example:
            # Split a region by bold text
            sections = region.split("text:bold")

            # Split horizontally by vertical lines
            sections = region.split("line[orientation=vertical]", orientation="horizontal")
        """
        # Default to 'start' boundaries for split (include divider at start of each section)
        if "include_boundaries" not in kwargs:
            kwargs["include_boundaries"] = "start"

        sections = self.get_sections(start_elements=divider, **kwargs)

        # Add section before first divider if there's content
        if sections and hasattr(sections[0], "start_element"):
            first_divider = sections[0].start_element
            if first_divider:
                # Get all elements before the first divider
                all_elements = self.get_elements()
                if all_elements and all_elements[0] != first_divider:
                    # Create section from start to just before first divider
                    initial_section = self.get_section_between(
                        start_element=None,
                        end_element=first_divider,
                        include_boundaries="none",
                        orientation=kwargs.get("orientation", "vertical"),
                    )
                    if initial_section and initial_section.get_elements():
                        sections.insert(0, initial_section)

        return sections

    def create_cells(self):
        """
        Create cell regions for a detected table by intersecting its
        row and column regions, and add them to the page.

        Assumes child row and column regions are already present on the page.

        Returns:
            Self for method chaining.
        """
        # Ensure this is called on a table region
        if self.region_type not in (
            "table",
            "tableofcontents",
        ):  # Allow for ToC which might have structure
            raise ValueError(
                f"create_cells should be called on a 'table' or 'tableofcontents' region, not '{self.region_type}'"
            )

        # Find rows and columns associated with this page
        # Remove the model-specific filter
        rows = self.page.find_all("region[type=table-row]")
        columns = self.page.find_all("region[type=table-column]")

        # Filter to only include those that overlap with this table region
        def is_in_table(element):
            # Use a simple overlap check (more robust than just center point)
            # Check if element's bbox overlaps with self.bbox
            return (
                hasattr(element, "bbox")
                and element.x0 < self.x1  # Ensure element has bbox
                and element.x1 > self.x0
                and element.top < self.bottom
                and element.bottom > self.top
            )

        table_rows = [r for r in rows if is_in_table(r)]
        table_columns = [c for c in columns if is_in_table(c)]

        if not table_rows or not table_columns:
            # Use page's logger if available
            logger_instance = getattr(self._page, "logger", logger)
            logger_instance.warning(
                f"Region {self.bbox}: Cannot create cells. No overlapping row or column regions found."
            )
            return self  # Return self even if no cells created

        # Sort rows and columns
        table_rows.sort(key=lambda r: r.top)
        table_columns.sort(key=lambda c: c.x0)

        # Create cells and add them to the page's element manager
        created_count = 0
        for row in table_rows:
            for column in table_columns:
                # Calculate intersection bbox for the cell
                cell_x0 = max(row.x0, column.x0)
                cell_y0 = max(row.top, column.top)
                cell_x1 = min(row.x1, column.x1)
                cell_y1 = min(row.bottom, column.bottom)

                # Only create a cell if the intersection is valid (positive width/height)
                if cell_x1 > cell_x0 and cell_y1 > cell_y0:
                    # Create cell region at the intersection
                    cell = self.page.create_region(cell_x0, cell_y0, cell_x1, cell_y1)
                    # Set metadata
                    cell.source = "derived"
                    cell.region_type = "table-cell"  # Explicitly set type
                    cell.normalized_type = "table-cell"  # And normalized type
                    # Inherit model from the parent table region
                    cell.model = self.model
                    cell.parent_region = self  # Link cell to parent table region

                    # Add the cell region to the page's element manager
                    self.page._element_mgr.add_region(cell)
                    created_count += 1

        # Optional: Add created cells to the table region's children
        # self.child_regions.extend(cells_created_in_this_call) # Needs list management

        logger_instance = getattr(self._page, "logger", logger)
        logger_instance.info(
            f"Region {self.bbox} (Model: {self.model}): Created and added {created_count} cell regions."
        )

        return self  # Return self for chaining

    def ask(
        self,
        question: Union[str, List[str], Tuple[str, ...]],
        min_confidence: float = 0.1,
        model: str = None,
        debug: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Ask a question about the region content using document QA.

        This method uses a document question answering model to extract answers from the region content.
        It leverages both textual content and layout information for better understanding.

        Args:
            question: The question to ask about the region content
            min_confidence: Minimum confidence threshold for answers (0.0-1.0)
            model: Optional model name to use for QA (if None, uses default model)
            **kwargs: Additional parameters to pass to the QA engine

        Returns:
            Dictionary with answer details: {
                "answer": extracted text,
                "confidence": confidence score,
                "found": whether an answer was found,
                "page_num": page number,
                "region": reference to this region,
                "source_elements": list of elements that contain the answer (if found)
            }
        """
        try:
            from natural_pdf.qa.document_qa import get_qa_engine
        except ImportError:
            logger.error(
                "Question answering requires optional dependencies. Install with `pip install natural-pdf[ai]`"
            )
            return {
                "answer": None,
                "confidence": 0.0,
                "found": False,
                "page_num": self.page.number,
                "source_elements": [],
                "region": self,
            }

        # Get or initialize QA engine with specified model
        try:
            qa_engine = get_qa_engine(model_name=model) if model else get_qa_engine()
        except Exception as e:
            logger.error(f"Failed to initialize QA engine (model: {model}): {e}", exc_info=True)
            return {
                "answer": None,
                "confidence": 0.0,
                "found": False,
                "page_num": self.page.number,
                "source_elements": [],
                "region": self,
            }

        # Ask the question using the QA engine
        try:
            return qa_engine.ask_pdf_region(
                self, question, min_confidence=min_confidence, debug=debug, **kwargs
            )
        except Exception as e:
            logger.error(f"Error during qa_engine.ask_pdf_region: {e}", exc_info=True)
            return {
                "answer": None,
                "confidence": 0.0,
                "found": False,
                "page_num": self.page.number,
                "source_elements": [],
                "region": self,
            }

    def add_child(self, child):
        """
        Add a child region to this region.

        Used for hierarchical document structure when using models like Docling
        that understand document hierarchy.

        Args:
            child: Region object to add as a child

        Returns:
            Self for method chaining
        """
        self.child_regions.append(child)
        child.parent_region = self
        return self

    def get_children(self, selector=None):
        """
        Get immediate child regions, optionally filtered by selector.

        Args:
            selector: Optional selector to filter children

        Returns:
            List of child regions matching the selector
        """
        import logging

        logger = logging.getLogger("natural_pdf.elements.region")

        if selector is None:
            return self.child_regions

        # Use existing selector parser to filter
        try:
            selector_obj = parse_selector(selector)
            filter_func = selector_to_filter_func(selector_obj)  # Removed region=self
            matched = [child for child in self.child_regions if filter_func(child)]
            logger.debug(
                f"get_children: found {len(matched)} of {len(self.child_regions)} children matching '{selector}'"
            )
            return matched
        except Exception as e:
            logger.error(f"Error applying selector in get_children: {e}", exc_info=True)
            return []  # Return empty list on error

    def get_descendants(self, selector=None):
        """
        Get all descendant regions (children, grandchildren, etc.), optionally filtered by selector.

        Args:
            selector: Optional selector to filter descendants

        Returns:
            List of descendant regions matching the selector
        """
        import logging

        logger = logging.getLogger("natural_pdf.elements.region")

        all_descendants = []
        queue = list(self.child_regions)  # Start with direct children

        while queue:
            current = queue.pop(0)
            all_descendants.append(current)
            # Add current's children to the queue for processing
            if hasattr(current, "child_regions"):
                queue.extend(current.child_regions)

        logger.debug(f"get_descendants: found {len(all_descendants)} total descendants")

        # Filter by selector if provided
        if selector is not None:
            try:
                selector_obj = parse_selector(selector)
                filter_func = selector_to_filter_func(selector_obj)  # Removed region=self
                matched = [desc for desc in all_descendants if filter_func(desc)]
                logger.debug(f"get_descendants: filtered to {len(matched)} matching '{selector}'")
                return matched
            except Exception as e:
                logger.error(f"Error applying selector in get_descendants: {e}", exc_info=True)
                return []  # Return empty list on error

        return all_descendants

    def __add__(
        self, other: Union["Element", "Region", "ElementCollection"]
    ) -> "ElementCollection":
        """Add regions/elements together to create an ElementCollection.

        This allows intuitive combination of regions using the + operator:
        ```python
        complainant = section.find("text:contains(Complainant)").right(until='text')
        dob = section.find("text:contains(DOB)").right(until='text')
        combined = complainant + dob  # Creates ElementCollection with both regions
        ```

        Args:
            other: Another Region, Element or ElementCollection to combine

        Returns:
            ElementCollection containing all elements
        """
        from natural_pdf.elements.base import Element
        from natural_pdf.elements.element_collection import ElementCollection

        # Create a list starting with self
        elements = [self]

        # Add the other element(s)
        if isinstance(other, (Element, Region)):
            elements.append(other)
        elif isinstance(other, ElementCollection):
            elements.extend(other)
        elif hasattr(other, "__iter__") and not isinstance(other, (str, bytes)):
            # Handle other iterables but exclude strings
            elements.extend(other)
        else:
            raise TypeError(f"Cannot add Region with {type(other)}")

        return ElementCollection(elements)

    def __radd__(
        self, other: Union["Element", "Region", "ElementCollection"]
    ) -> "ElementCollection":
        """Right-hand addition to support ElementCollection + Region."""
        if other == 0:
            # This handles sum() which starts with 0
            from natural_pdf.elements.element_collection import ElementCollection

            return ElementCollection([self])
        return self.__add__(other)

    def __repr__(self) -> str:
        """String representation of the region."""
        poly_info = " (Polygon)" if self.has_polygon else ""
        name_info = f" name='{self.name}'" if self.name else ""
        type_info = f" type='{self.region_type}'" if self.region_type else ""
        source_info = f" source='{self.source}'" if self.source else ""

        # Add checkbox state if this is a checkbox
        checkbox_info = ""
        if self.region_type == "checkbox" and hasattr(self, "is_checked"):
            state = "checked" if self.is_checked else "unchecked"
            checkbox_info = f" [{state}]"

        return f"<Region{name_info}{type_info}{source_info}{checkbox_info} bbox={self.bbox}{poly_info}>"

    def update_text(
        self,
        transform: Callable[[Any], Optional[str]],
        *,
        selector: str = "text",
        apply_exclusions: bool = False,
    ) -> "Region":
        """Apply *transform* to every text element matched by *selector* inside this region.

        The heavy lifting is delegated to :py:meth:`TextMixin.update_text`; this
        override simply ensures the search is scoped to the region.
        """

        return TextMixin.update_text(
            self, transform, selector=selector, apply_exclusions=apply_exclusions
        )

    # --- Classification Mixin Implementation --- #
    def _get_classification_manager(self) -> "ClassificationManager":
        if (
            not hasattr(self, "page")
            or not hasattr(self.page, "pdf")
            or not hasattr(self.page.pdf, "get_manager")
        ):
            raise AttributeError(
                "ClassificationManager cannot be accessed: Parent Page, PDF, or get_manager method missing."
            )
        try:
            # Use the PDF's manager registry accessor via page
            return self.page.pdf.get_manager("classification")
        except (ValueError, RuntimeError, AttributeError) as e:
            # Wrap potential errors from get_manager for clarity
            raise AttributeError(
                f"Failed to get ClassificationManager from PDF via Page: {e}"
            ) from e

    def _get_classification_content(
        self, model_type: str, **kwargs
    ) -> Union[str, "Image"]:  # Use "Image" for lazy import
        if model_type == "text":
            text_content = self.extract_text(layout=False)  # Simple join for classification
            if not text_content or text_content.isspace():
                raise ValueError("Cannot classify region with 'text' model: No text content found.")
            return text_content
        elif model_type == "vision":
            # Get resolution from manager/kwargs if possible, else default
            # We access manager via the method to ensure it's available
            manager = self._get_classification_manager()
            default_resolution = 150  # Manager doesn't store default res, set here
            # Note: classify() passes resolution via **kwargs if user specifies
            resolution = (
                kwargs.get("resolution", default_resolution)
                if "kwargs" in locals()
                else default_resolution
            )

            img = self.render(
                resolution=resolution,
                crop=True,  # Just the region content
            )
            if img is None:
                raise ValueError(
                    "Cannot classify region with 'vision' model: Failed to render image."
                )
            return img
        else:
            raise ValueError(f"Unsupported model_type for classification: {model_type}")

    def _get_metadata_storage(self) -> Dict[str, Any]:
        # Ensure metadata exists
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}
        return self.metadata

    # --- End Classification Mixin Implementation --- #

    # --- NEW METHOD: analyze_text_table_structure ---
    def analyze_text_table_structure(
        self,
        snap_tolerance: int = 10,
        join_tolerance: int = 3,
        min_words_vertical: int = 3,
        min_words_horizontal: int = 1,
        intersection_tolerance: int = 3,
        expand_bbox: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> Optional[Dict]:
        """
        Analyzes the text elements within the region (or slightly expanded area)
        to find potential table structure (lines, cells) using text alignment logic
        adapted from pdfplumber.

        Args:
            snap_tolerance: Tolerance for snapping parallel lines.
            join_tolerance: Tolerance for joining collinear lines.
            min_words_vertical: Minimum words needed to define a vertical line.
            min_words_horizontal: Minimum words needed to define a horizontal line.
            intersection_tolerance: Tolerance for detecting line intersections.
            expand_bbox: Optional dictionary to expand the search area slightly beyond
                         the region's exact bounds (e.g., {'left': 5, 'right': 5}).
            **kwargs: Additional keyword arguments passed to
                      find_text_based_tables (e.g., specific x/y tolerances).

        Returns:
            A dictionary containing 'horizontal_edges', 'vertical_edges', 'cells' (list of dicts),
            and 'intersections', or None if pdfplumber is unavailable or an error occurs.
        """

        # Determine the search region (expand if requested)
        search_region = self
        if expand_bbox and isinstance(expand_bbox, dict):
            try:
                search_region = self.expand(**expand_bbox)
                logger.debug(
                    f"Expanded search region for text table analysis to: {search_region.bbox}"
                )
            except Exception as e:
                logger.warning(f"Could not expand region bbox: {e}. Using original region.")
                search_region = self

        # Find text elements within the search region
        text_elements = search_region.find_all(
            "text", apply_exclusions=False
        )  # Use unfiltered text
        if not text_elements:
            logger.info(f"Region {self.bbox}: No text elements found for text table analysis.")
            return {"horizontal_edges": [], "vertical_edges": [], "cells": [], "intersections": {}}

        # Extract bounding boxes
        bboxes = [element.bbox for element in text_elements if hasattr(element, "bbox")]
        if not bboxes:
            logger.info(f"Region {self.bbox}: No bboxes extracted from text elements.")
            return {"horizontal_edges": [], "vertical_edges": [], "cells": [], "intersections": {}}

        # Call the utility function
        try:
            analysis_results = find_text_based_tables(
                bboxes=bboxes,
                snap_tolerance=snap_tolerance,
                join_tolerance=join_tolerance,
                min_words_vertical=min_words_vertical,
                min_words_horizontal=min_words_horizontal,
                intersection_tolerance=intersection_tolerance,
                **kwargs,  # Pass through any extra specific tolerance args
            )
            # Store results in the region's analyses cache
            self.analyses["text_table_structure"] = analysis_results
            return analysis_results
        except ImportError:
            logger.error("pdfplumber library is required for 'text' table analysis but not found.")
            return None
        except Exception as e:
            logger.error(f"Error during text-based table analysis: {e}", exc_info=True)
            return None

    # --- END NEW METHOD ---

    # --- NEW METHOD: get_text_table_cells ---
    def get_text_table_cells(
        self,
        snap_tolerance: int = 10,
        join_tolerance: int = 3,
        min_words_vertical: int = 3,
        min_words_horizontal: int = 1,
        intersection_tolerance: int = 3,
        expand_bbox: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> "ElementCollection[Region]":
        """
        Analyzes text alignment to find table cells and returns them as
        temporary Region objects without adding them to the page.

        Args:
            snap_tolerance: Tolerance for snapping parallel lines.
            join_tolerance: Tolerance for joining collinear lines.
            min_words_vertical: Minimum words needed to define a vertical line.
            min_words_horizontal: Minimum words needed to define a horizontal line.
            intersection_tolerance: Tolerance for detecting line intersections.
            expand_bbox: Optional dictionary to expand the search area slightly beyond
                         the region's exact bounds (e.g., {'left': 5, 'right': 5}).
            **kwargs: Additional keyword arguments passed to
                      find_text_based_tables (e.g., specific x/y tolerances).

        Returns:
            An ElementCollection containing temporary Region objects for each detected cell,
            or an empty ElementCollection if no cells are found or an error occurs.
        """
        from natural_pdf.elements.element_collection import ElementCollection

        # 1. Perform the analysis (or use cached results)
        if "text_table_structure" in self.analyses:
            analysis_results = self.analyses["text_table_structure"]
            logger.debug("get_text_table_cells: Using cached analysis results.")
        else:
            analysis_results = self.analyze_text_table_structure(
                snap_tolerance=snap_tolerance,
                join_tolerance=join_tolerance,
                min_words_vertical=min_words_vertical,
                min_words_horizontal=min_words_horizontal,
                intersection_tolerance=intersection_tolerance,
                expand_bbox=expand_bbox,
                **kwargs,
            )

        # 2. Check if analysis was successful and cells were found
        if analysis_results is None or not analysis_results.get("cells"):
            logger.info(f"Region {self.bbox}: No cells found by text table analysis.")
            return ElementCollection([])  # Return empty collection

        # 3. Create temporary Region objects for each cell dictionary
        cell_regions = []
        for cell_data in analysis_results["cells"]:
            try:
                # Use page.region to create the region object
                # It expects left, top, right, bottom keys
                cell_region = self.page.region(**cell_data)

                # Set metadata on the temporary region
                cell_region.region_type = "table-cell"
                cell_region.normalized_type = "table-cell"
                cell_region.model = "pdfplumber-text"
                cell_region.source = "volatile"  # Indicate it's not managed/persistent
                cell_region.parent_region = self  # Link back to the region it came from

                cell_regions.append(cell_region)
            except Exception as e:
                logger.warning(f"Could not create Region object for cell data {cell_data}: {e}")

        # 4. Return the list wrapped in an ElementCollection
        logger.debug(f"get_text_table_cells: Created {len(cell_regions)} temporary cell regions.")
        return ElementCollection(cell_regions)

    # --- END NEW METHOD ---

    def to_text_element(
        self,
        text_content: Optional[Union[str, Callable[["Region"], Optional[str]]]] = None,
        source_label: str = "derived_from_region",
        object_type: str = "word",  # Or "char", controls how it's categorized
        default_font_size: float = 10.0,
        default_font_name: str = "RegionContent",
        confidence: Optional[float] = None,  # Allow overriding confidence
        add_to_page: bool = False,  # NEW: Option to add to page
    ) -> "TextElement":
        """
        Creates a new TextElement object based on this region's geometry.

        The text for the new TextElement can be provided directly,
        generated by a callback function, or left as None.

        Args:
            text_content:
                - If a string, this will be the text of the new TextElement.
                - If a callable, it will be called with this region instance
                  and its return value (a string or None) will be the text.
                - If None (default), the TextElement's text will be None.
            source_label: The 'source' attribute for the new TextElement.
            object_type: The 'object_type' for the TextElement's data dict
                         (e.g., "word", "char").
            default_font_size: Placeholder font size if text is generated.
            default_font_name: Placeholder font name if text is generated.
            confidence: Confidence score for the text. If text_content is None,
                        defaults to 0.0. If text is provided/generated, defaults to 1.0
                        unless specified.
            add_to_page: If True, the created TextElement will be added to the
                         region's parent page. (Default: False)

        Returns:
            A new TextElement instance.

        Raises:
            ValueError: If the region does not have a valid 'page' attribute.
        """
        actual_text: Optional[str] = None
        if isinstance(text_content, str):
            actual_text = text_content
        elif callable(text_content):
            try:
                actual_text = text_content(self)
            except Exception as e:
                logger.error(
                    f"Error executing text_content callback for region {self.bbox}: {e}",
                    exc_info=True,
                )
                actual_text = None  # Ensure actual_text is None on error

        final_confidence = confidence
        if final_confidence is None:
            final_confidence = 1.0 if actual_text is not None and actual_text.strip() else 0.0

        if not hasattr(self, "page") or self.page is None:
            raise ValueError("Region must have a valid 'page' attribute to create a TextElement.")

        # Create character dictionaries for the text
        char_dicts = []
        if actual_text:
            # Create a single character dict that spans the entire region
            # This is a simplified approach - OCR engines typically create one per character
            char_dict = {
                "text": actual_text,
                "x0": self.x0,
                "top": self.top,
                "x1": self.x1,
                "bottom": self.bottom,
                "width": self.width,
                "height": self.height,
                "object_type": "char",
                "page_number": self.page.page_number,
                "fontname": default_font_name,
                "size": default_font_size,
                "upright": True,
                "direction": 1,
                "adv": self.width,
                "source": source_label,
                "confidence": final_confidence,
                "stroking_color": (0, 0, 0),
                "non_stroking_color": (0, 0, 0),
            }
            char_dicts.append(char_dict)

        elem_data = {
            "text": actual_text,
            "x0": self.x0,
            "top": self.top,
            "x1": self.x1,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
            "object_type": object_type,
            "page_number": self.page.page_number,
            "stroking_color": getattr(self, "stroking_color", (0, 0, 0)),
            "non_stroking_color": getattr(self, "non_stroking_color", (0, 0, 0)),
            "fontname": default_font_name,
            "size": default_font_size,
            "upright": True,
            "direction": 1,
            "adv": self.width,
            "source": source_label,
            "confidence": final_confidence,
            "_char_dicts": char_dicts,
        }
        text_element = TextElement(elem_data, self.page)

        if add_to_page:
            if hasattr(self.page, "_element_mgr") and self.page._element_mgr is not None:
                add_as_type = (
                    "words"
                    if object_type == "word"
                    else "chars" if object_type == "char" else object_type
                )
                # REMOVED try-except block around add_element
                self.page._element_mgr.add_element(text_element, element_type=add_as_type)
                logger.debug(
                    f"TextElement created from region {self.bbox} and added to page {self.page.page_number} as {add_as_type}."
                )
                # Also add character dictionaries to the chars collection
                if char_dicts and object_type == "word":
                    for char_dict in char_dicts:
                        self.page._element_mgr.add_element(char_dict, element_type="chars")
            else:
                page_num_str = (
                    str(self.page.page_number) if hasattr(self.page, "page_number") else "N/A"
                )
                logger.warning(
                    f"Cannot add TextElement to page: Page {page_num_str} for region {self.bbox} is missing '_element_mgr'."
                )

        return text_element

    # ------------------------------------------------------------------
    # Unified analysis storage (maps to metadata["analysis"])
    # ------------------------------------------------------------------

    @property
    def analyses(self) -> Dict[str, Any]:
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}
        return self.metadata.setdefault("analysis", {})

    @analyses.setter
    def analyses(self, value: Dict[str, Any]):
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}
        self.metadata["analysis"] = value

    # ------------------------------------------------------------------
    # New helper: build table from pre-computed table_cell regions
    # ------------------------------------------------------------------

    def _extract_table_from_cells(
        self, cell_regions: List["Region"], content_filter=None, apply_exclusions=True
    ) -> List[List[Optional[str]]]:
        """Construct a table (list-of-lists) from table_cell regions.

        This assumes each cell Region has metadata.row_index / col_index as written by
        detect_table_structure_from_lines().  If these keys are missing we will
        fall back to sorting by geometry.

        Args:
            cell_regions: List of table cell Region objects to extract text from
            content_filter: Optional content filter to apply to cell text extraction
        """
        if not cell_regions:
            return []

        # Attempt to use explicit indices first
        all_row_idxs = []
        all_col_idxs = []
        for cell in cell_regions:
            try:
                r_idx = int(cell.metadata.get("row_index"))
                c_idx = int(cell.metadata.get("col_index"))
                all_row_idxs.append(r_idx)
                all_col_idxs.append(c_idx)
            except Exception:
                # Not all cells have indices – clear the lists so we switch to geometric sorting
                all_row_idxs = []
                all_col_idxs = []
                break

        if all_row_idxs and all_col_idxs:
            num_rows = max(all_row_idxs) + 1
            num_cols = max(all_col_idxs) + 1

            # Initialise blank grid
            table_grid: List[List[Optional[str]]] = [[None] * num_cols for _ in range(num_rows)]

            for cell in cell_regions:
                try:
                    r_idx = int(cell.metadata.get("row_index"))
                    c_idx = int(cell.metadata.get("col_index"))
                    text_val = cell.extract_text(
                        layout=False,
                        apply_exclusions=apply_exclusions,
                        content_filter=content_filter,
                    ).strip()
                    table_grid[r_idx][c_idx] = text_val if text_val else None
                except Exception as _err:
                    # Skip problematic cell
                    continue

            return table_grid

        # ------------------------------------------------------------------
        # Fallback: derive order purely from geometry if indices are absent
        # ------------------------------------------------------------------
        # Sort unique centers to define ordering
        try:
            import numpy as np
        except ImportError:
            logger.warning("NumPy required for geometric cell ordering; returning empty result.")
            return []

        # Build arrays of centers
        centers = np.array([[(c.x0 + c.x1) / 2.0, (c.top + c.bottom) / 2.0] for c in cell_regions])
        xs = centers[:, 0]
        ys = centers[:, 1]

        # Cluster unique row Y positions and column X positions with a tolerance
        def _cluster(vals, tol=1.0):
            sorted_vals = np.sort(vals)
            groups = [[sorted_vals[0]]]
            for v in sorted_vals[1:]:
                if abs(v - groups[-1][-1]) <= tol:
                    groups[-1].append(v)
                else:
                    groups.append([v])
            return [np.mean(g) for g in groups]

        row_centers = _cluster(ys)
        col_centers = _cluster(xs)

        num_rows = len(row_centers)
        num_cols = len(col_centers)

        table_grid: List[List[Optional[str]]] = [[None] * num_cols for _ in range(num_rows)]

        # Assign each cell to nearest row & col center
        for cell, (cx, cy) in zip(cell_regions, centers):
            row_idx = int(np.argmin([abs(cy - rc) for rc in row_centers]))
            col_idx = int(np.argmin([abs(cx - cc) for cc in col_centers]))

            text_val = cell.extract_text(
                layout=False, apply_exclusions=apply_exclusions, content_filter=content_filter
            ).strip()
            table_grid[row_idx][col_idx] = text_val if text_val else None

        return table_grid

    def _apply_rtl_processing_to_text(self, text: str) -> str:
        """
        Apply RTL (Right-to-Left) text processing to a string.

        This converts visual order text (as stored in PDFs) to logical order
        for proper display of Arabic, Hebrew, and other RTL scripts.

        Args:
            text: Input text string in visual order

        Returns:
            Text string in logical order
        """
        if not text or not text.strip():
            return text

        # Quick check for RTL characters - if none found, return as-is
        import unicodedata

        def _contains_rtl(s):
            return any(unicodedata.bidirectional(ch) in ("R", "AL", "AN") for ch in s)

        if not _contains_rtl(text):
            return text

        try:
            from bidi.algorithm import get_display  # type: ignore

            from natural_pdf.utils.bidi_mirror import mirror_brackets

            # Apply BiDi algorithm to convert from visual to logical order
            # Process line by line to handle mixed content properly
            processed_lines = []
            for line in text.split("\n"):
                if line.strip():
                    # Determine base direction for this line
                    base_dir = "R" if _contains_rtl(line) else "L"
                    logical_line = get_display(line, base_dir=base_dir)
                    # Apply bracket mirroring for correct logical order
                    processed_lines.append(mirror_brackets(logical_line))
                else:
                    processed_lines.append(line)

            return "\n".join(processed_lines)

        except (ImportError, Exception):
            # If bidi library is not available or fails, return original text
            return text

    def _apply_content_filter_to_text(self, text: str, content_filter) -> str:
        """
        Apply content filter to a text string.

        Args:
            text: Input text string
            content_filter: Content filter (regex, callable, or list of regexes)

        Returns:
            Filtered text string
        """
        if not text or content_filter is None:
            return text

        import re

        if isinstance(content_filter, str):
            # Single regex pattern - remove matching parts
            try:
                return re.sub(content_filter, "", text)
            except re.error:
                return text  # Invalid regex, return original

        elif isinstance(content_filter, list):
            # List of regex patterns - remove parts matching ANY pattern
            try:
                result = text
                for pattern in content_filter:
                    result = re.sub(pattern, "", result)
                return result
            except re.error:
                return text  # Invalid regex, return original

        elif callable(content_filter):
            # Callable filter - apply to individual characters
            try:
                filtered_chars = []
                for char in text:
                    if content_filter(char):
                        filtered_chars.append(char)
                return "".join(filtered_chars)
            except Exception:
                return text  # Function error, return original

        return text

    # ------------------------------------------------------------------
    # Interactive Viewer Support
    # ------------------------------------------------------------------

    def viewer(
        self,
        *,
        resolution: int = 150,
        include_chars: bool = False,
        include_attributes: Optional[List[str]] = None,
    ) -> Optional["InteractiveViewerWidget"]:
        """Create an interactive ipywidget viewer for **this specific region**.

        The method renders the region to an image (cropped to the region bounds) and
        overlays all elements that intersect the region (optionally excluding noisy
        character-level elements).  The resulting widget offers the same zoom / pan
        experience as :py:meth:`Page.viewer` but scoped to the region.

        Parameters
        ----------
        resolution : int, default 150
            Rendering resolution (DPI).  This should match the value used by the
            page-level viewer so element scaling is accurate.
        include_chars : bool, default False
            Whether to include individual *char* elements in the overlay.  These
            are often too dense for a meaningful visualisation so are skipped by
            default.
        include_attributes : list[str], optional
            Additional element attributes to expose in the info panel (on top of
            the default set used by the page viewer).

        Returns
        -------
        InteractiveViewerWidget | None
            The widget instance, or ``None`` if *ipywidgets* is not installed or
            an error occurred during creation.
        """

        # ------------------------------------------------------------------
        # Dependency / environment checks
        # ------------------------------------------------------------------
        if not _IPYWIDGETS_AVAILABLE or InteractiveViewerWidget is None:
            logger.error(
                "Interactive viewer requires 'ipywidgets'. "
                'Please install with: pip install "ipywidgets>=7.0.0,<10.0.0"'
            )
            return None

        try:
            # ------------------------------------------------------------------
            # Render region image (cropped) and encode as data URI
            # ------------------------------------------------------------------
            import base64
            from io import BytesIO

            # Use unified render() with crop=True to obtain just the region
            img = self.render(resolution=resolution, crop=True)
            if img is None:
                logger.error(f"Failed to render image for region {self.bbox} viewer.")
                return None

            buf = BytesIO()
            img.save(buf, format="PNG")
            img_str = base64.b64encode(buf.getvalue()).decode()
            image_uri = f"data:image/png;base64,{img_str}"

            # ------------------------------------------------------------------
            # Prepare element overlay data (coordinates relative to region)
            # ------------------------------------------------------------------
            scale = resolution / 72.0  # Same convention as page viewer

            # Gather elements intersecting the region
            region_elements = self.get_elements(apply_exclusions=False)

            # Optionally filter out chars
            if not include_chars:
                region_elements = [
                    el for el in region_elements if str(getattr(el, "type", "")).lower() != "char"
                ]

            default_attrs = [
                "text",
                "fontname",
                "size",
                "bold",
                "italic",
                "color",
                "linewidth",
                "is_horizontal",
                "is_vertical",
                "source",
                "confidence",
                "label",
                "model",
                "upright",
                "direction",
            ]

            if include_attributes:
                default_attrs.extend([a for a in include_attributes if a not in default_attrs])

            elements_json: List[dict] = []
            for idx, el in enumerate(region_elements):
                try:
                    # Calculate coordinates relative to region bbox and apply scale
                    x0 = (el.x0 - self.x0) * scale
                    y0 = (el.top - self.top) * scale
                    x1 = (el.x1 - self.x0) * scale
                    y1 = (el.bottom - self.top) * scale

                    elem_dict = {
                        "id": idx,
                        "type": getattr(el, "type", "unknown"),
                        "x0": round(x0, 2),
                        "y0": round(y0, 2),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "width": round(x1 - x0, 2),
                        "height": round(y1 - y0, 2),
                    }

                    # Add requested / default attributes
                    for attr_name in default_attrs:
                        if hasattr(el, attr_name):
                            val = getattr(el, attr_name)
                            # Ensure JSON serialisable
                            if not isinstance(val, (str, int, float, bool, list, dict, type(None))):
                                val = str(val)
                            elem_dict[attr_name] = val
                    elements_json.append(elem_dict)
                except Exception as e:
                    logger.warning(f"Error preparing element {idx} for region viewer: {e}")

            viewer_data = {"page_image": image_uri, "elements": elements_json}

            # ------------------------------------------------------------------
            # Instantiate the widget directly using the prepared data
            # ------------------------------------------------------------------
            return InteractiveViewerWidget(pdf_data=viewer_data)

        except Exception as e:
            logger.error(f"Error creating viewer for region {self.bbox}: {e}", exc_info=True)
            return None

    def within(self):
        """Context manager that constrains directional operations to this region.

        When used as a context manager, all directional navigation operations
        (above, below, left, right) will be constrained to the bounds of this region.

        Returns:
            RegionContext: A context manager that yields this region

        Examples:
            ```python
            # Create a column region
            left_col = page.region(right=page.width/2)

            # All directional operations are constrained to left_col
            with left_col.within() as col:
                header = col.find("text[size>14]")
                content = header.below(until="text[size>14]")
                # content will only include elements within left_col

            # Operations outside the context are not constrained
            full_page_below = header.below()  # Searches full page
            ```
        """
        return RegionContext(self)
