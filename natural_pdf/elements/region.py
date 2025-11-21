from __future__ import annotations

import logging
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
    cast,
)

from pdfplumber.utils.geometry import get_bbox_overlap

# New Imports
from tqdm.auto import tqdm

from natural_pdf.analyzers.layout.pdfplumber_table_finder import find_text_based_tables
from natural_pdf.core.context import PDFContext
from natural_pdf.core.crop_utils import resolve_crop_bbox
from natural_pdf.core.exclusion_mixin import ExclusionEntry, ExclusionSpec
from natural_pdf.core.geometry_mixin import RegionGeometryMixin
from natural_pdf.core.interfaces import SupportsGeometry, SupportsSections
from natural_pdf.core.mixins import SinglePageContextMixin
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.elements.base import DirectionalMixin, extract_bbox
from natural_pdf.elements.text import TextElement  # ADDED IMPORT
from natural_pdf.qa.qa_result import QAResult
from natural_pdf.selectors.host_mixin import SelectorHostMixin
from natural_pdf.selectors.parser import (
    build_text_contains_selector,
    parse_selector,
    selector_to_filter_func,
)

# Table utilities
from natural_pdf.services import exclusion_service as _exclusion_service  # noqa: F401
from natural_pdf.services import extraction_service as _extraction_service  # noqa: F401
from natural_pdf.services import guides_service as _guides_service  # noqa: F401
from natural_pdf.services import qa_service as _qa_service  # noqa: F401
from natural_pdf.services import table_service as _table_service  # noqa: F401
from natural_pdf.services.base import ServiceHostMixin, resolve_service
from natural_pdf.tables.result import TableResult

# Import new utils
from natural_pdf.text.operations import (
    apply_bidi_processing,
    filter_chars_spatially,
    generate_text_layout,
)

# Import viewer widget support
from natural_pdf.widgets.viewer import _IPYWIDGETS_AVAILABLE, InteractiveViewerWidget

if TYPE_CHECKING:
    from PIL.Image import Image

    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element  # Added for type hint
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.text import TextElement
    from natural_pdf.flows.region import FlowRegion
    from natural_pdf.widgets.viewer import InteractiveViewerWidget as InteractiveViewerWidgetType
else:  # pragma: no cover - typing fallback
    InteractiveViewerWidgetType: TypeAlias = Any

logger = logging.getLogger(__name__)

CustomOCRCallable = Callable[["Region"], Optional[str]]


class LayoutOptionsProtocol(Protocol):
    directional_offset: float
    directional_within: Optional["Region"]
    auto_multipage: bool


def _layout_options() -> LayoutOptionsProtocol:
    import natural_pdf

    return cast(LayoutOptionsProtocol, natural_pdf.options.layout)


class ImageOptionsProtocol(Protocol):
    resolution: Optional[float]


def _image_options() -> ImageOptionsProtocol:
    import natural_pdf

    return cast(ImageOptionsProtocol, natural_pdf.options.image)


class RegionContext:
    """Context manager for constraining directional operations to a region."""

    def __init__(self, region: "Region"):
        """Initialize the context manager with a region.

        Args:
            region: The Region to use as a constraint for directional operations
        """
        self.region = region
        self.previous_within: Optional["Region"] = None

    def __enter__(self):
        """Enter the context, setting the global directional_within option."""
        layout_options = _layout_options()

        self.previous_within = layout_options.directional_within
        layout_options.directional_within = self.region
        return self.region

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, restoring the previous directional_within option."""
        layout_options = _layout_options()

        layout_options.directional_within = self.previous_within
        return False  # Don't suppress exceptions


class Region(
    SelectorHostMixin,
    DirectionalMixin,
    ServiceHostMixin,
    SinglePageContextMixin,
    RegionGeometryMixin,
    Visualizable,
    SupportsSections,
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
        polygon: Optional[List[Tuple[float, float]]] = None,
        parent: Optional["Region"] = None,
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
        self._page: "Page" = page
        resolved_context = getattr(page, "_context", PDFContext.with_defaults())
        self._init_service_host(resolved_context)
        self._bbox: Tuple[float, float, float, float] = bbox
        self._polygon: Optional[List[Tuple[float, float]]] = polygon

        self.metadata: Dict[str, Any] = {}
        # Analysis results live under self.metadata['analysis'] via property

        # Standard attributes for all elements
        self.object_type: str = "region"  # For selector compatibility
        self.start_element: Optional["Element"] = None
        self.end_element: Optional["Element"] = None
        self._boundary_exclusions: Optional[str] = None

        # Layout detection attributes
        self.region_type: Optional[str] = None
        self.normalized_type: Optional[str] = None
        self.confidence: Optional[float] = None
        self.model: Optional[str] = None
        self.is_checked: Optional[bool] = None
        self.checkbox_state: Optional[str] = None
        self.original_class: Optional[str] = None

        # Region management attributes
        self.name: Optional[str] = None
        self.label = label
        self.source: Optional[str] = None  # Will be set by creation methods

        # Hierarchy support for nested document structure
        self.parent_region: Optional["Region"] = parent
        self.child_regions: List["Region"] = []
        self.text_content: Optional[str] = None  # Direct text content (e.g., from Docling)
        self.associated_text_elements: List[TextElement] = (
            []
        )  # Native text elements that overlap with this region
        self.source_element: Optional[Union["Element", "Region"]] = None
        self.includes_source: bool = False
        self.boundary_element: Optional["Element"] = None

        if self.parent_region is not None:
            self.parent_region.child_regions.append(self)

        self._cached_text: Optional[str] = None
        self._cached_elements: Optional[ElementCollection] = None
        self._cached_bbox: Optional[Tuple[float, float, float, float]] = None
        self._exclusions: List[ExclusionSpec] = []

    def _exclusion_element_manager(self):
        return self.page._element_mgr

    def _element_to_region(self, element: Any, label: Optional[str] = None) -> Optional["Region"]:
        bbox = extract_bbox(element)
        if not bbox:
            return None
        # Clamp to this region's bounds
        x0 = max(self.x0, bbox[0])
        top = max(self.top, bbox[1])
        x1 = min(self.x1, bbox[2])
        bottom = min(self.bottom, bbox[3])
        if x0 >= x1 or top >= bottom:
            return None
        return Region(self.page, (x0, top, x1, bottom), label=label)

    def _invalidate_exclusion_cache(self) -> None:
        self._cached_text = None
        self._cached_elements = None

    def _ocr_element_manager(self):
        return self.page._element_mgr

    def _ocr_scope(self) -> str:
        return "region"

    def _ocr_render_kwargs(self, *, apply_exclusions: bool = True) -> Dict[str, Any]:
        return {"crop": True}

    def _qa_context_page_number(self) -> int:
        return self.page.number

    def _qa_source_elements(self) -> "ElementCollection":
        from natural_pdf.elements.element_collection import ElementCollection

        return ElementCollection([])

    def _qa_target_region(self) -> "Region":
        return self

    def _resolve_element_loader(self):
        """Best-effort lookup for the shared ElementLoader."""
        page = getattr(self, "page", None)
        if page is None:
            return None
        loader_getter = getattr(page, "_get_element_loader", None)
        if callable(loader_getter):
            try:
                return loader_getter()
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Region %s: ElementLoader lookup failed.", self.bbox, exc_info=True)
        return None

    def _resolve_decoration_detector(self):
        """Best-effort lookup for the shared DecorationDetector."""
        page = getattr(self, "page", None)
        if page is None:
            return None
        detector_getter = getattr(page, "_get_decoration_detector", None)
        if callable(detector_getter):
            try:
                return detector_getter()
            except Exception:  # pragma: no cover - defensive guard
                logger.debug(
                    "Region %s: DecorationDetector lookup failed.", self.bbox, exc_info=True
                )
        return None

    def _prepare_char_dicts_with_loader(
        self, char_dicts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        loader = self._resolve_element_loader()
        if loader and char_dicts:
            try:
                return loader.prepare_native_chars(char_dicts)
            except Exception:  # pragma: no cover - defensive guard
                logger.debug(
                    "Region %s: ElementLoader errored while preparing chars; using raw dicts.",
                    self.bbox,
                    exc_info=True,
                )
        return char_dicts

    def _qa_normalize_result(self, result: Any) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        return self._normalize_qa_output(result)

    def _evaluate_exclusion_entries(
        self, entries: Sequence[ExclusionSpec], include_callable: bool, debug: bool
    ) -> List["Region"]:
        if not entries:
            return []
        service = resolve_service(self, "exclusion")
        return service.evaluate_entries(self, entries, include_callable, debug)

    def _evaluate_local_exclusions(
        self, include_callable: bool = True, debug: bool = False
    ) -> List["Region"]:
        if not getattr(self, "_exclusions", None):
            return []
        return self._evaluate_exclusion_entries(self._exclusions, include_callable, debug)

    def _get_exclusion_regions(
        self, include_callable: bool = True, debug: bool = False
    ) -> List["Region"]:
        local = self._evaluate_local_exclusions(include_callable=include_callable, debug=debug)
        page_regions = self.page._get_exclusion_regions(
            include_callable=include_callable, debug=debug
        )
        return local + page_regions

    def to_region(self) -> "Region":
        """Regions already satisfy the section surface; return self."""
        return self

    def _get_render_specs(
        self,
        mode: Literal["show", "render"] = "show",
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        highlights: Optional[Union[List[Dict[str, Any]], bool]] = None,
        crop: Union[bool, int, "Region", Literal["wide"]] = True,  # Default to True for regions
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

        spec = RenderSpec(page=self.page)

        spec.crop_bbox = resolve_crop_bbox(
            width=self.page.width,
            height=self.page.height,
            crop=crop,
            crop_bbox=crop_bbox,
            content_bbox_fn=lambda: self.bbox,
        )

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

    def create_region(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        *,
        relative: bool = True,
        label: Optional[str] = None,
    ) -> "Region":
        """Create a child region anchored to this region.

        Args:
            left: Left coordinate. Interpreted relative to this region when ``relative`` is True.
            top: Top coordinate.
            right: Right coordinate.
            bottom: Bottom coordinate.
            relative: When True (default), coordinates are treated as offsets from this
                region's bounds. Set to False to provide absolute page coordinates.
            label: Optional label to assign to the new region.

        Returns:
            The newly created child region.
        """

        page = getattr(self, "page", None)
        if page is None:
            raise ValueError("Cannot create a sub-region without an associated page")

        if relative:
            abs_left = self.x0 + left
            abs_top = self.top + top
            abs_right = self.x0 + right
            abs_bottom = self.top + bottom
        else:
            abs_left = left
            abs_top = top
            abs_right = right
            abs_bottom = bottom

        child_region = page.region(left=abs_left, top=abs_top, right=abs_right, bottom=abs_bottom)
        child_region.parent_region = self
        self.child_regions.append(child_region)
        if label is not None:
            child_region.label = label
        return child_region

    def _direction(
        self,
        direction: str,
        size: Optional[float] = None,
        cross_size: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ) -> Union["Region", "FlowRegion"]:
        """
        Region-specific wrapper around :py:meth:`DirectionalMixin._direction`.

        It performs any pre-processing required by *Region* (none currently),
        delegates the core geometry work to the mix-in implementation via
        ``super()``, then attaches region-level metadata before returning the
        new :class:`Region` instance.
        """

        effective_offset = offset
        if effective_offset is None:
            effective_offset = _layout_options().directional_offset

        # Delegate to the shared implementation on DirectionalMixin
        result = super()._direction(
            direction=direction,
            size=size,
            cross_size=cross_size,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=effective_offset,
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            within=within,
            anchor=anchor,
            **kwargs,
        )

        if isinstance(result, Region):
            # Post-process: make sure callers can trace lineage and flags
            result.source_element = self
            result.includes_source = include_source

        return result

    def above(
        self,
        height: Optional[float] = None,
        width: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ) -> Union["Region", "FlowRegion"]:
        """
        Select region above this region.

        Args:
            height: Height of the region above, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_source: Whether to include this region in the result (default: False)
            until: Optional selector string to specify an upper boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            offset: Pixel offset when excluding source/endpoint (default: None, uses natural_pdf.options.layout.directional_offset)
            multipage: Override global multipage behaviour; defaults to None meaning use global option.
            **kwargs: Additional parameters

        Returns:
            Region object representing the area above
        """
        return self._direction(
            direction="above",
            size=height,
            cross_size=width,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=offset,
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            within=within,
            anchor=anchor,
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
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ) -> Union["Region", "FlowRegion"]:
        """
        Select region below this region.

        Args:
            height: Height of the region below, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_source: Whether to include this region in the result (default: False)
            until: Optional selector string to specify a lower boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            offset: Pixel offset when excluding source/endpoint (default: None, uses natural_pdf.options.layout.directional_offset)
            multipage: Override global multipage behaviour; defaults to None meaning use global option.
            **kwargs: Additional parameters

        Returns:
            Region object representing the area below
        """
        return self._direction(
            direction="below",
            size=height,
            cross_size=width,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=offset,
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            within=within,
            anchor=anchor,
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
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ) -> Union["Region", "FlowRegion"]:
        """
        Select region to the left of this region.

        Args:
            width: Width of the region to the left, in points
            height: Height mode - "full" for full page height or "element" for element height
            include_source: Whether to include this region in the result (default: False)
            until: Optional selector string to specify a left boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            offset: Pixel offset when excluding source/endpoint (default: None, uses natural_pdf.options.layout.directional_offset)
            multipage: Override global multipage behaviour; defaults to None meaning use global option.
            **kwargs: Additional parameters

        Returns:
            Region object representing the area to the left
        """
        return self._direction(
            direction="left",
            size=width,
            cross_size=height,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=offset,
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            within=within,
            anchor=anchor,
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
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ) -> Union["Region", "FlowRegion"]:
        """
        Select region to the right of this region.

        Args:
            width: Width of the region to the right, in points
            height: Height mode - "full" for full page height or "element" for element height
            include_source: Whether to include this region in the result (default: False)
            until: Optional selector string to specify a right boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            offset: Pixel offset when excluding source/endpoint (default: None, uses natural_pdf.options.layout.directional_offset)
            multipage: Override global multipage behaviour; defaults to None meaning use global option.
            **kwargs: Additional parameters

        Returns:
            Region object representing the area to the right
        """
        return self._direction(
            direction="right",
            size=width,
            cross_size=height,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=offset,
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            within=within,
            anchor=anchor,
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

    def _context_page(self) -> "Page":
        return self.page

    def _context_region_config(self, key: str, sentinel: object) -> Any:
        region_cfg = self.metadata.get("config")
        if isinstance(region_cfg, dict) and key in region_cfg:
            return region_cfg[key]
        return sentinel

    @property
    def origin(self) -> Optional[Union["Element", "Region"]]:
        """The element/region that created this region (if it was created via directional method)."""
        return getattr(self, "source_element", None)

    @property
    def endpoint(self) -> Optional["Element"]:
        """The element where this region stopped (if created with 'until' parameter)."""
        return getattr(self, "boundary_element", None)

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
    ) -> None:
        """
        Highlight this region on the page.

        Args:
            label: Optional label for the highlight
            color: Color tuple/string for the highlight, or None to use automatic color
            use_color_cycling: Force color cycling even with no label (default: False)
            annotate: List of attribute names to display on the highlight (e.g., ['confidence', 'type'])
            existing: How to handle existing highlights ('append' or 'replace').

        Returns:
            None
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

        return None

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
        image_options = _image_options()
        if resolution is None:
            default_resolution = image_options.resolution
            resolution = float(default_resolution) if default_resolution is not None else 144.0
        else:
            resolution = float(resolution)

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
        image_options = _image_options()
        if resolution is None:
            default_resolution = image_options.resolution
            resolution = float(default_resolution) if default_resolution is not None else 144.0
        else:
            resolution = float(resolution)

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
            if image is not None:
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
        image_options = _image_options()
        if resolution is None:
            default_resolution = image_options.resolution
            resolution = float(default_resolution) if default_resolution is not None else 144.0
        else:
            resolution = float(resolution)

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
            raise RuntimeError(f"Region {self.bbox}: render() returned None during trimming.")

        # Convert to grayscale for easier analysis
        import numpy as np

        # Convert PIL image to numpy array
        img_array = np.array(image.convert("L"))  # Convert to grayscale
        height, width = img_array.shape

        if height == 0 or width == 0:
            raise ValueError(f"Region {self.bbox}: rendered image has zero dimensions.")

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
            raise ValueError(f"Region {self.bbox}: no content detected during trimming.")

        top_content_row = max(0, content_row_indices[0] - padding)
        bottom_content_row = min(height - 1, content_row_indices[-1] + padding)

        # Analyze columns (vertical strips) to find left and right boundaries
        col_averages = np.mean(normalized, axis=0)  # Average each column
        content_cols = col_averages < threshold  # True where there's content

        content_col_indices = np.where(content_cols)[0]
        if len(content_col_indices) == 0:
            # No content found in columns either
            raise ValueError(f"Region {self.bbox}: no column content detected during trimming.")

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
            raise ValueError(f"Region {self.bbox}: trimming produced invalid dimensions.")

        # Create the trimmed region
        trimmed_region: Region = Region(self.page, (final_x0, final_top, final_x1, final_bottom))

        # Expand back by the pre_shrink amount to restore original positioning
        if pre_shrink > 0:
            trimmed_region = cast(
                Region,
                trimmed_region.expand(
                    left=pre_shrink, right=pre_shrink, top=pre_shrink, bottom=pre_shrink
                ),
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
            if obj_bbox is None:
                raise TypeError(
                    f"Region {self.bbox}: cannot extract bbox from clipping object {type(obj)}. "
                    "Object must expose bbox or x0/top/x1/bottom attributes."
                )
            obj_x0, obj_top, obj_x1, obj_bottom = obj_bbox
            clip_x0 = max(clip_x0, obj_x0)
            clip_top = max(clip_top, obj_top)
            clip_x1 = min(clip_x1, obj_x1)
            clip_bottom = min(clip_bottom, obj_bottom)

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
            raise ValueError(
                f"Region {self.bbox}: clipping resulted in invalid bounds "
                f"({clip_x0}, {clip_top}, {clip_x1}, {clip_bottom})."
            )

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
        left: Optional[float] = None,
        top: Optional[float] = None,
        right: Optional[float] = None,
        bottom: Optional[float] = None,
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
            left = (self.x0 + left) if left is not None else None
            top = (self.top + top) if top is not None else None
            right = (self.x0 + right) if right is not None else None
            bottom = (self.top + bottom) if bottom is not None else None

            # For numeric width/height with relative coords, we need to handle the calculation
            # in the context of absolute positioning

        # Use the parent page's region method to create the region with all its logic
        region_kwargs: Dict[str, Any] = {}
        if left is not None:
            region_kwargs["left"] = left
        if top is not None:
            region_kwargs["top"] = top
        if right is not None:
            region_kwargs["right"] = right
        if bottom is not None:
            region_kwargs["bottom"] = bottom
        if width is not None:
            region_kwargs["width"] = width
        if height is not None:
            region_kwargs["height"] = height

        new_region = self.page.region(**region_kwargs)

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

    def attr(self, name: str) -> Any:
        """
        Get an attribute value from this region.

        This method provides a consistent interface for attribute access that works
        on both individual regions/elements and collections. When called on a single
        region, it simply returns the attribute value. When called on collections,
        it extracts the attribute from all items.

        Args:
            name: The attribute name to retrieve (e.g., 'text', 'width', 'height')

        Returns:
            The attribute value, or None if the attribute doesn't exist

        Examples:
            # On a single region
            region = page.find('text:contains("Title")').expand(10)
            width = region.attr('width')  # Same as region.width

            # Consistent API across elements and regions
            obj = page.find('*:contains("Title")')  # Could be element or region
            text = obj.attr('text')  # Works for both
        """
        return getattr(self, name, None)

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
            all_page_exclusions = self._get_exclusion_regions(include_callable=True, debug=debug)
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

    def extract_table(self, *args, **kwargs) -> TableResult:
        return self.services.table.extract_table(self, *args, **kwargs)

    def extract_tables(self, *args, **kwargs) -> List[List[List[Optional[str]]]]:
        return self.services.table.extract_tables(self, *args, **kwargs)

    def _filter_elements_by_overlap_mode(
        self,
        elements: Sequence["Element"],
        overlap_mode: str,
    ) -> List["Element"]:
        """
        Apply region overlap filtering while preserving the upstream ordering.
        """
        if not elements:
            return []

        if overlap_mode == "full":
            return [el for el in elements if self.contains(el)]
        if overlap_mode == "partial":
            return [el for el in elements if self.intersects(el)]
        # overlap_mode == "center"
        return [el for el in elements if self.is_element_center_inside(el)]

    def apply_ocr(self, *args, **kwargs) -> "Region":
        """
        Apply OCR to this region and return the created text elements.

        This method supports two modes:
        1. **Built-in/registered OCR engines** – pass parameters like ``engine='easyocr'`` or
           ``languages=['en']`` and the request is routed through the shared
           :class:`~natural_pdf.engine_provider.EngineProvider` registry.
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
        params = dict(kwargs)
        if args and len(args) > 0:
            if len(args) > 1:
                raise TypeError("apply_ocr accepts at most one positional argument (replace).")
            params.setdefault("replace", args[0])

        replace = params.get("replace", True)

        custom_func_candidate = params.pop("function", None) or params.pop("ocr_function", None)
        if callable(custom_func_candidate):
            custom_func = cast(CustomOCRCallable, custom_func_candidate)
            # Delegate to the specialised helper while preserving key kwargs
            return self.apply_custom_ocr(
                ocr_function=custom_func,
                source_label=params.pop("source_label", "custom-ocr"),
                replace=replace,
                confidence=params.pop("confidence", None),
                add_to_page=params.pop("add_to_page", True),
            )

        if replace:
            removed = self.remove_ocr_elements()
            if removed:
                logger.info(
                    f"Region {self.bbox}: Removed {removed} existing OCR elements before re-applying OCR."
                )
            else:
                logger.debug(
                    f"Region {self.bbox}: No overlapping OCR elements found before applying new OCR."
                )

        self.services.ocr.apply_ocr(self, replace=replace, **params)
        return self

    def extract_ocr_elements(
        self,
        *,
        engine: Optional[str] = None,
        options: Optional[Any] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
    ) -> List[Any]:
        """
        Run OCR and return the resulting text elements without mutating this region.

        Args:
            engine: OCR engine name (defaults follow the scope configuration).
            options: Engine-specific options payload or dataclass.
            languages: Optional list of language codes.
            min_confidence: Optional minimum confidence threshold.
            device: Preferred execution device.
            resolution: Explicit render DPI; falls back to config/context when omitted.

        Returns:
            List of text elements created from OCR (not added to the page).
        """

        return self.services.ocr.extract_ocr_elements(
            self,
            engine=engine,
            options=options,
            languages=languages,
            min_confidence=min_confidence,
            device=device,
            resolution=resolution,
        )

    def apply_custom_ocr(
        self,
        ocr_function: CustomOCRCallable,
        source_label: str = "custom-ocr",
        replace: bool = True,
        confidence: Optional[float] = None,
        add_to_page: bool = True,
    ) -> "Region":
        service = resolve_service(self, "ocr")
        service.apply_custom_ocr(
            self,
            ocr_function=ocr_function,
            source_label=source_label,
            replace=replace,
            confidence=confidence,
            add_to_page=add_to_page,
        )
        return self

    def remove_ocr_elements(self, *args, **kwargs) -> int:
        """Remove OCR text from constituent regions."""

        return self.services.ocr.remove_ocr_elements(self, *args, **kwargs)

    def clear_text_layer(self, *args, **kwargs) -> Tuple[int, int]:
        """Clear OCR results from the underlying managers and return totals."""

        return self.services.ocr.clear_text_layer(self, *args, **kwargs)

    def create_text_elements_from_ocr(self, *args, **kwargs):
        """Delegate to the OCR service for text element creation."""

        return self.services.ocr.create_text_elements_from_ocr(self, *args, **kwargs)

    def extract(self, *args, **kwargs):
        self.services.extraction.extract(self, *args, **kwargs)
        return self

    def extract_structured_data(self, *args, **kwargs):
        self.services.extraction.extract(self, *args, **kwargs)
        return self

    def extracted(self, *args, **kwargs):
        return self.services.extraction.extracted(self, *args, **kwargs)

    def update_text(self, *args, **kwargs):
        return self.services.text.update_text(self, *args, **kwargs)

    def correct_ocr(self, *args, **kwargs):
        return self.services.text.correct_ocr(self, *args, **kwargs)

    def classify(self, *args, **kwargs):
        return self.services.classification.classify(self, *args, **kwargs)

    def ask(self, *args, **kwargs):
        return self.services.qa.ask(self, *args, **kwargs)

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

        if not elements:
            raise ValueError(f"Region {self.bbox}: no elements available for section extraction.")

        # Sort elements in reading order
        elements.sort(key=lambda e: (e.top, e.x0))

        if start_element and start_element not in elements:
            logger.debug("Start element not found in region, using first element.")
            start_element = elements[0]
        elif start_element is None:
            start_element = elements[0]

        if end_element and end_element not in elements:
            logger.debug("End element not found in region, using last element.")
            end_element = elements[-1]
        elif end_element is None:
            end_element = elements[-1]

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
        start_elements: Union[str, Sequence["Element"], "ElementCollection", None] = None,
        end_elements: Union[str, Sequence["Element"], "ElementCollection", None] = None,
        include_boundaries: str = "both",
        orientation: str = "vertical",
        **kwargs: Any,
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

        def _normalize_section_boundary(
            arg: Union[str, Sequence["Element"], "ElementCollection", None]
        ) -> Union[str, List["Element"], None]:
            if arg is None or isinstance(arg, str):
                return arg

            if isinstance(arg, ElementCollection):
                return list(arg)

            if isinstance(arg, Sequence):
                return list(arg)

            # Fallback: wrap single element-like object
            return [arg]  # type: ignore[list-item]

        # Legacy tolerances (y_threshold/x_threshold) are now handled internally;
        # accept and ignore them for backward compatibility.
        legacy_thresholds = {"y_threshold", "x_threshold"}
        for legacy_key in list(kwargs.keys()):
            if legacy_key in legacy_thresholds:
                kwargs.pop(legacy_key, None)

        # Use centralized section extraction logic
        sections = extract_sections_from_region(
            region=self,
            start_elements=_normalize_section_boundary(start_elements),
            end_elements=_normalize_section_boundary(end_elements),
            include_boundaries=include_boundaries,
            orientation=orientation,
            **kwargs,
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

                    # Register the cell region with the page (tracks provenance and manager)
                    self.page.add_region(cell, source="derived")
                    created_count += 1

        # Optional: Add created cells to the table region's children
        # self.child_regions.extend(cells_created_in_this_call) # Needs list management

        logger_instance = getattr(self._page, "logger", logger)
        logger_instance.info(
            f"Region {self.bbox} (Model: {self.model}): Created and added {created_count} cell regions."
        )

        return self  # Return self for chaining

    @staticmethod
    def _normalize_qa_output(result: Any) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Convert QA engine results into plain dictionaries for typing compliance."""
        from natural_pdf.qa.qa_result import QAResult

        if isinstance(result, QAResult):
            return dict(result)

        if isinstance(result, list):
            normalized: List[Dict[str, Any]] = []
            for item in result:
                if isinstance(item, QAResult):
                    normalized.append(dict(item))
                elif isinstance(item, MappingABC):
                    normalized.append(dict(item))
                else:
                    normalized.append({"value": item})
            return normalized

        if isinstance(result, MappingABC):
            return dict(result)

        raise TypeError(f"Unexpected QA result type {type(result).__name__}")

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
        elements: List[Union[Element, "Region"]] = [self]

        # Add the other element(s)
        if isinstance(other, (Element, Region)):
            elements.append(other)
        elif isinstance(other, ElementCollection):
            elements.extend(cast(Iterable[Union[Element, "Region"]], list(other)))
        elif hasattr(other, "__iter__") and not isinstance(other, (str, bytes)):
            # Handle other iterables but exclude strings
            elements.extend(cast(Iterable[Union[Element, "Region"]], list(other)))
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

    def _get_classification_content(
        self, model_type: str, **kwargs
    ) -> Union[str, "Image"]:  # Use "Image" for lazy import
        if model_type == "text":
            text_content = self.extract_text(layout=False)  # Simple join for classification
            if not text_content or text_content.isspace():
                raise ValueError("Cannot classify region with 'text' model: No text content found.")
            return text_content
        elif model_type == "vision":
            resolution = kwargs.get("resolution", 150)
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

            def _coerce_expand_value(value: Any) -> Union[float, bool, str]:
                if isinstance(value, (bool, str)):
                    return value
                return float(value)

            sanitized_expand: Dict[str, Any] = {}
            if "amount" in expand_bbox and expand_bbox["amount"] is not None:
                sanitized_expand["amount"] = float(expand_bbox["amount"])
            for key in ("left", "right", "top", "bottom"):
                if key in expand_bbox and expand_bbox[key] is not None:
                    sanitized_expand[key] = _coerce_expand_value(expand_bbox[key])
            if "width_factor" in expand_bbox and expand_bbox["width_factor"] is not None:
                sanitized_expand["width_factor"] = float(expand_bbox["width_factor"])
            if "height_factor" in expand_bbox and expand_bbox["height_factor"] is not None:
                sanitized_expand["height_factor"] = float(expand_bbox["height_factor"])
            if "apply_exclusions" in expand_bbox and expand_bbox["apply_exclusions"] is not None:
                sanitized_expand["apply_exclusions"] = bool(expand_bbox["apply_exclusions"])

            search_region = self.expand(**sanitized_expand)
            logger.debug(f"Expanded search region for text table analysis to: {search_region.bbox}")

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
            cell_region = self.page.region(**cell_data)
            cell_region.region_type = "table-cell"
            cell_region.normalized_type = "table-cell"
            cell_region.model = "pdfplumber-text"
            cell_region.source = "volatile"
            cell_region.parent_region = self
            cell_regions.append(cell_region)

        # 4. Return the list wrapped in an ElementCollection
        logger.debug(f"get_text_table_cells: Created {len(cell_regions)} temporary cell regions.")
        return ElementCollection(cell_regions)

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
        char_dicts: List[Dict[str, Any]] = []
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

        prepared_char_dicts = self._prepare_char_dicts_with_loader(char_dicts)
        detector = self._resolve_decoration_detector()
        if detector and prepared_char_dicts:
            detector.annotate_chars(prepared_char_dicts)

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
            "_char_dicts": prepared_char_dicts,
        }
        text_element = TextElement(elem_data, self.page)
        if detector and prepared_char_dicts:
            detector.propagate_to_words([text_element], prepared_char_dicts)

        if add_to_page:
            add_as_type = (
                "words"
                if object_type == "word"
                else "chars" if object_type == "char" else object_type
            )
            if hasattr(self.page, "add_element"):
                self.page.add_element(text_element, element_type=add_as_type)
                logger.debug(
                    "TextElement created from region %s and added to page %s as %s.",
                    self.bbox,
                    getattr(self.page, "page_number", "N/A"),
                    add_as_type,
                )
                if prepared_char_dicts and object_type == "word":
                    for char_dict in prepared_char_dicts:
                        self.page.add_element(
                            TextElement(char_dict, self.page), element_type="chars"
                        )
            else:
                page_num_str = (
                    str(self.page.page_number) if hasattr(self.page, "page_number") else "N/A"
                )
                raise AttributeError(
                    f"Cannot add TextElement to page {page_num_str}: page does not expose add_element"
                )

        return text_element

    # Unified analysis storage (maps to metadata["analysis"])

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

    # New helper: build table from pre-computed table_cell regions

    def _apply_rtl_processing_to_text(self, text: str) -> str:
        return apply_bidi_processing(text)

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

    # Interactive Viewer Support

    def viewer(
        self,
        *,
        resolution: int = 150,
        include_chars: bool = False,
        include_attributes: Optional[List[str]] = None,
    ) -> Optional[Any]:
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
        InteractiveViewerWidgetType | None
            The widget instance, or ``None`` if *ipywidgets* is not installed or
            an error occurred during creation.
        """

        # Dependency / environment checks
        if not _IPYWIDGETS_AVAILABLE or InteractiveViewerWidget is None:
            logger.error(
                "Interactive viewer requires 'ipywidgets'. "
                'Please install with: pip install "ipywidgets>=7.0.0,<10.0.0"'
            )
            return None

        try:
            # Render region image (cropped) and encode as data URI
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

            # Prepare element overlay data (coordinates relative to region)
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

                for attr_name in default_attrs:
                    if hasattr(el, attr_name):
                        val = getattr(el, attr_name)
                        if not isinstance(val, (str, int, float, bool, list, dict, type(None))):
                            val = str(val)
                        elem_dict[attr_name] = val
                elements_json.append(elem_dict)

            viewer_data = {"page_image": image_uri, "elements": elements_json}

            # Instantiate the widget directly using the prepared data
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

    def detect_lines(self, *args, **kwargs):
        return self.services.shapes.detect_lines(self, *args, **kwargs)

    def detect_checkboxes(self, *args, **kwargs):
        return self.services.checkbox.detect_checkboxes(self, *args, **kwargs)

    def guides(self, *args, **kwargs):
        return self.services.guides.guides(self, *args, **kwargs)

    def describe(self, **kwargs):
        """
        Describe the region content using the describe service.
        """
        return self.services.describe.describe(self, **kwargs)

    def inspect(self, limit: int = 30, **kwargs):
        """
        Inspect the region content using the describe service.
        """
        collection = self.find_all("*")
        return self.services.describe.inspect(collection, limit=limit, **kwargs)


# Flow navigation fallback uses Region directional helpers
from natural_pdf.elements.base import DirectionalMixin as _DirectionalMixin

_REGION_NAV_FALLBACK = {
    name: getattr(_DirectionalMixin, name) for name in ("above", "below", "left", "right")
}
