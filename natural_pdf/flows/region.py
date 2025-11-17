from __future__ import annotations

import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
    overload,
)

from pdfplumber.utils.geometry import merge_bboxes  # Import merge_bboxes directly

from natural_pdf.core.capabilities import MultiRegionAnalysisMixin
from natural_pdf.core.context import PDFContext
from natural_pdf.core.crop_utils import resolve_crop_bbox
from natural_pdf.core.exclusion_mixin import ExclusionSpec
from natural_pdf.core.highlighter_utils import resolve_highlighter
from natural_pdf.core.interfaces import SupportsSections
from natural_pdf.core.mixins import ContextResolverMixin
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.elements.base import extract_bbox
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.selectors.host_mixin import SelectorHostMixin
from natural_pdf.services import exclusion_service as _exclusion_service  # noqa: F401
from natural_pdf.services import guides_service as _guides_service  # noqa: F401
from natural_pdf.services import navigation_service as _navigation_service  # noqa: F401
from natural_pdf.services import qa_service as _qa_service  # noqa: F401
from natural_pdf.services.base import ServiceHostMixin, resolve_service
from natural_pdf.services.delegates import attach_capability
from natural_pdf.services.methods import flow_table_methods as _flow_table_methods
from natural_pdf.tables import TableResult

# For runtime image manipulation


if TYPE_CHECKING:
    from PIL.Image import Image as PIL_Image  # For type hints

    from natural_pdf.core.highlighting_service import HighlightContext, HighlightingService
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element as PhysicalElement
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region as PhysicalRegion

    from .element import FlowElement
    from .flow import Flow

logger = logging.getLogger(__name__)


class FlowRegion(
    ServiceHostMixin,
    SupportsSections,
    SelectorHostMixin,
    Visualizable,
    MultiRegionAnalysisMixin,
    ContextResolverMixin,
):
    """
    Represents a selected area within a Flow, potentially composed of multiple
    physical Region objects (constituent_regions) that might span across
    different original pages or disjoint physical regions defined in the Flow.

    A FlowRegion is the result of a directional operation (e.g., .below(), .above())
    on a FlowElement.
    """

    def __init__(
        self,
        flow: "Flow",
        constituent_regions: List["PhysicalRegion"],
        source_flow_element: Optional["FlowElement"] = None,
        boundary_element_found: Optional[Union["PhysicalElement", "PhysicalRegion"]] = None,
    ):
        """
        Initializes a FlowRegion.

        Args:
            flow: The Flow instance this region belongs to.
            constituent_regions: A list of physical natural_pdf.elements.region.Region
                                 objects that make up this FlowRegion.
            source_flow_element: The FlowElement that created this FlowRegion.
            boundary_element_found: The physical element that stopped an 'until' search,
                                    if applicable.
        """
        self.flow: "Flow" = flow
        self.constituent_regions: List["PhysicalRegion"] = constituent_regions
        context = self._resolve_service_context(flow, constituent_regions)
        self._init_service_host(context)
        self.source_flow_element: Optional["FlowElement"] = source_flow_element
        self.boundary_element_found: Optional[Union["PhysicalElement", "PhysicalRegion"]] = (
            boundary_element_found
        )

        self.start_element: Optional[Union["PhysicalElement", "PhysicalRegion"]] = None
        self.end_element: Optional[Union["PhysicalElement", "PhysicalRegion"]] = None
        self._boundary_exclusions: Optional[str] = None

        # Add attributes for grid building, similar to Region
        self.source: Optional[str] = None
        self.region_type: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

        # Cache for expensive operations
        self._cached_text: Optional[str] = None
        self._cached_elements: Optional["ElementCollection"] = None  # Stringized
        self._cached_bbox: Optional[Tuple[float, float, float, float]] = None
        self._exclusions: List[ExclusionSpec] = []
        self._multi_page_page_warned: bool = False

    @staticmethod
    def _resolve_service_context(
        flow: Optional["Flow"], regions: Sequence["PhysicalRegion"]
    ) -> PDFContext:
        candidates: List[Any] = []
        if flow is not None:
            candidates.append(getattr(flow, "_context", None))
        for region in regions:
            candidates.append(getattr(region, "_context", None))
            page = getattr(region, "page", None)
            if page is not None:
                candidates.append(getattr(page, "_context", None))
                pdf_obj = getattr(page, "pdf", getattr(page, "_parent", None))
                if pdf_obj is not None:
                    candidates.append(getattr(pdf_obj, "_context", None))
        for candidate in candidates:
            if candidate is not None:
                return candidate
        return PDFContext.with_defaults()

    def _qa_context_page_number(self) -> int:
        pages = self.pages
        return pages[0].number if pages else -1

    def _qa_source_elements(self) -> ElementCollection:
        return ElementCollection([])

    def _qa_segments(self) -> Sequence["PhysicalRegion"]:
        return self.constituent_regions

    def _qa_normalize_result(self, result: Any) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        from natural_pdf.elements.region import Region

        return Region._normalize_qa_output(result)

    def _qa_target_region(self):
        if not self.constituent_regions:
            raise RuntimeError("FlowRegion has no constituent regions for QA.")

        return self.constituent_regions[0]

    def to_region(self):
        if not self.constituent_regions:
            raise RuntimeError("FlowRegion has no constituent regions to convert into a Region.")
        return self.constituent_regions[0]

    @staticmethod
    def _qa_confidence(candidate: Any) -> float:
        """Best-effort confidence extraction from normalized QA results."""
        if isinstance(candidate, list) and candidate:
            return FlowRegion._qa_confidence(candidate[0])
        if isinstance(candidate, dict):
            value = candidate.get("confidence")
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("-inf")
        try:
            value = getattr(candidate, "confidence", None)
            if value is None:
                value = candidate.get("confidence")
            return float(value)
        except Exception:
            return float("-inf")

    def _ocr_element_manager(self):
        if not self.constituent_regions:
            raise RuntimeError("FlowRegion has no regions for OCR operations")
        return self.constituent_regions[0].page._element_mgr

    def remove_ocr_elements(self) -> int:
        removed = 0
        for region in self.constituent_regions:
            removed += region.remove_ocr_elements()
        return removed

    def clear_text_layer(self) -> Tuple[int, int]:
        total_chars = 0
        total_words = 0
        seen_pages: Set[int] = set()
        for region in self.constituent_regions:
            page = region.page
            marker = id(page)
            if marker in seen_pages:
                continue
            seen_pages.add(marker)
            cleared_chars, cleared_words = page.clear_text_layer()
            total_chars += cleared_chars
            total_words += cleared_words
        return total_chars, total_words

    def create_text_elements_from_ocr(
        self, ocr_results: Any, scale_x: Optional[float] = None, scale_y: Optional[float] = None
    ) -> List[Any]:
        if not self.constituent_regions:
            return []
        return self.constituent_regions[0].page.create_text_elements_from_ocr(
            ocr_results, scale_x=scale_x, scale_y=scale_y
        )

    def _iter_ocr_regions(self) -> Iterable[Any]:
        return tuple(self.constituent_regions)

    def apply_ocr(self, *args: Any, **kwargs: Any) -> "FlowRegion":
        """
        Apply OCR across all constituent regions using their native implementations.
        """
        for region in self.constituent_regions:
            apply_fn = getattr(region, "apply_ocr", None)
            if callable(apply_fn):
                apply_fn(*args, **kwargs)
        return self

    def extract_ocr_elements(self, *args: Any, **kwargs: Any) -> List[Any]:
        """
        Extract OCR elements from each constituent region and flatten the results.
        """
        extracted: List[Any] = []
        for region in self.constituent_regions:
            extract_fn = getattr(region, "extract_ocr_elements", None)
            if callable(extract_fn):
                result = extract_fn(*args, **kwargs)
                if isinstance(result, list):
                    extracted.extend(result)
                elif result is not None:
                    extracted.append(result)
        return extracted

    def _exclusion_element_manager(self):
        if not self.constituent_regions:
            raise RuntimeError("FlowRegion has no constituent regions for exclusions")
        return self.constituent_regions[0].page._element_mgr

    def _evaluate_exclusion_entries(
        self,
        entries: Sequence[ExclusionSpec],
        include_callable: bool = True,
        debug: bool = False,
    ):
        if not entries:
            return []
        service = resolve_service(self, "exclusion")
        return service.evaluate_entries(self, entries, include_callable, debug)

    def _get_exclusion_regions(
        self,
        include_callable: bool = True,
        debug: bool = False,
    ):
        entries = cast(List[ExclusionSpec], getattr(self, "_exclusions", []))
        local = self._evaluate_exclusion_entries(entries, include_callable, debug)

        aggregated: List["PhysicalRegion"] = []
        for region in self.constituent_regions:
            aggregated.extend(
                region._get_exclusion_regions(include_callable=include_callable, debug=debug)
            )
        return self._dedupe_regions((*local, *aggregated))

    @staticmethod
    def _dedupe_regions(regions: Iterable["PhysicalRegion"]) -> List["PhysicalRegion"]:
        deduped: List["PhysicalRegion"] = []
        seen: set[int] = set()
        for region in regions:
            marker = id(region)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(region)
        return deduped

    def _element_to_region(
        self, element: Any, label: Optional[str] = None
    ) -> Optional["PhysicalRegion"]:
        bbox = extract_bbox(element)
        if not bbox:
            return None

        page = getattr(element, "page", None)
        if page is None and self.constituent_regions:
            page = self.constituent_regions[0].page
        if page is None:
            return None

        from natural_pdf.elements.region import (
            Region as PhysicalRegion,  # Local import to avoid cycles
        )

        clamp_bbox = bbox
        matching_regions = [
            region for region in self.constituent_regions if getattr(region, "page", None) is page
        ]
        if matching_regions:
            min_x0 = min(region.x0 for region in matching_regions)
            min_top = min(region.top for region in matching_regions)
            max_x1 = max(region.x1 for region in matching_regions)
            max_bottom = max(region.bottom for region in matching_regions)

            clamp_bbox = (
                max(min_x0, bbox[0]),
                max(min_top, bbox[1]),
                min(max_x1, bbox[2]),
                min(max_bottom, bbox[3]),
            )

            if clamp_bbox[0] >= clamp_bbox[2] or clamp_bbox[1] >= clamp_bbox[3]:
                return None

        return PhysicalRegion(page, clamp_bbox, label=label)

    def _invalidate_exclusion_cache(self) -> None:
        self._cached_text = None
        self._cached_elements = None

    def _iter_exclusion_regions(self) -> Iterable[Any]:
        return tuple(self.constituent_regions)

    @property
    def pages(self) -> Tuple["Page", ...]:
        """Return the distinct pages covered by this flow region."""
        seen: Set[int] = set()
        ordered_pages: List["Page"] = []
        for region in self.constituent_regions:
            page = getattr(region, "page", None)
            if page is None:
                continue
            marker = id(page)
            if marker not in seen:
                seen.add(marker)
                ordered_pages.append(page)
        return tuple(ordered_pages)

    @property
    def page(self) -> "Page":
        """Return the primary page for this region (first page when multi-page)."""
        pages = self.pages
        if len(pages) == 0:
            raise AttributeError("FlowRegion has no associated pages")
        if len(pages) > 1 and not self._multi_page_page_warned:
            warnings.warn(
                "FlowRegion spans multiple pages; returning the first page for .page access. "
                "Use .pages to inspect all pages.",
                UserWarning,
                stacklevel=2,
            )
            self._multi_page_page_warned = True
        return pages[0]

    def get_highlighter(self) -> "HighlightingService":
        """Resolve a highlighting service from the constituent regions."""
        if not self.constituent_regions:
            raise RuntimeError("FlowRegion has no constituent regions to get highlighter from")
        return cast("HighlightingService", resolve_highlighter(self.constituent_regions, self.flow))

    def _get_highlighter(self):
        """Compatibility hook for Visualizable mixin."""
        return self.get_highlighter()

    def _context_resolution_roots(self) -> Iterable[Any]:
        roots: List[Any] = [self]
        roots.extend(self.constituent_regions)
        if self.flow is not None:
            roots.append(self.flow)
        return roots

    def _get_render_specs(
        self,
        mode: Literal["show", "render"] = "show",
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        highlights: Optional[List[Dict[str, Any]]] = None,
        crop: Union[bool, Literal["content"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> List[RenderSpec]:
        """Get render specifications for this flow region.

        Args:
            mode: Rendering mode - 'show' includes highlights, 'render' is clean
            color: Color for highlighting this region in show mode
            highlights: Additional highlight groups to show
            crop: Whether to crop to constituent regions
            crop_bbox: Explicit crop bounds
            **kwargs: Additional parameters

        Returns:
            List of RenderSpec objects, one per page with constituent regions
        """
        if not self.constituent_regions:
            return []

        label_prefix = kwargs.pop("label_prefix", None)

        regions_by_page = {}
        for region in self.constituent_regions:
            page = getattr(region, "page", None)
            if page is None:
                continue
            regions_by_page.setdefault(page, []).append(region)

        if not regions_by_page:
            return []

        specs = []
        for page, page_regions in regions_by_page.items():
            spec = RenderSpec(page=page)

            def union_bbox() -> Optional[Tuple[float, float, float, float]]:
                x_coords: List[float] = []
                y_coords: List[float] = []
                for region in page_regions:
                    bbox = getattr(region, "bbox", None)
                    if bbox:
                        x0, y0, x1, y1 = bbox
                        x_coords.extend([x0, x1])
                        y_coords.extend([y0, y1])
                if x_coords and y_coords:
                    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                return None

            spec.crop_bbox = resolve_crop_bbox(
                width=page.width,
                height=page.height,
                crop=crop,
                crop_bbox=crop_bbox,
                content_bbox_fn=union_bbox,
            )

            # Add highlights in show mode
            if mode == "show":
                # Highlight constituent regions
                for i, region in enumerate(page_regions):
                    # Label each part if multiple regions
                    label = None
                    if len(self.constituent_regions) > 1:
                        try:
                            global_idx = self.constituent_regions.index(region)
                        except ValueError:
                            global_idx = i
                        if label_prefix:
                            label = f"{label_prefix}_{global_idx + 1}"
                        else:
                            label = f"FlowPart_{global_idx + 1}"
                    else:
                        label = label_prefix or "FlowRegion"

                    spec.add_highlight(
                        bbox=region.bbox,
                        polygon=region.polygon if region.has_polygon else None,
                        color=color or "fuchsia",
                        label=label,
                    )

                # Add additional highlight groups if provided
                if highlights:
                    for group in highlights:
                        group_elements = group.get("elements", [])
                        group_color = group.get("color", color)
                        group_label = group.get("label")

                        for elem in group_elements:
                            # Only add if element is on this page
                            if hasattr(elem, "page") and elem.page == page:
                                spec.add_highlight(
                                    element=elem, color=group_color, label=group_label
                                )

            specs.append(spec)

        return specs

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically proxy attribute access to the source FlowElement for safe attributes only.
        Spatial methods (above, below, left, right) are explicitly implemented to prevent
        silent failures and incorrect behavior.
        """
        if name in self.__dict__:
            return self.__dict__[name]

        # List of methods that should NOT be proxied - they need proper FlowRegion implementation
        spatial_methods = {"above", "below", "left", "right", "to_region"}

        if name in spatial_methods:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                f"This method requires proper FlowRegion implementation to handle spatial relationships correctly."
            )

        # Only proxy safe attributes and methods
        if self.source_flow_element is not None:
            try:
                attr = getattr(self.source_flow_element, name)
                # Only proxy non-callable attributes and explicitly safe methods
                if not callable(attr) or name in {"page", "document"}:  # Add safe methods as needed
                    return attr
                else:
                    raise AttributeError(
                        f"Method '{name}' cannot be safely proxied from FlowElement to FlowRegion. "
                        f"It may need explicit implementation."
                    )
            except AttributeError:
                pass

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @property
    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """
        The bounding box that encloses all constituent regions.
        Calculated dynamically and cached.
        """
        if self._cached_bbox is not None:
            return self._cached_bbox
        if not self.constituent_regions:
            return None

        # Use merge_bboxes from pdfplumber.utils.geometry to merge bboxes
        # Extract bbox tuples from regions first
        region_bboxes = [
            region.bbox for region in self.constituent_regions if hasattr(region, "bbox")
        ]
        if not region_bboxes:
            return None

        self._cached_bbox = merge_bboxes(region_bboxes)
        return self._cached_bbox

    def _require_bbox(self) -> Tuple[float, float, float, float]:
        bbox = self.bbox
        if bbox is None:
            raise ValueError("FlowRegion has no bounding box; ensure it has constituent regions")
        return bbox

    @property
    def x0(self) -> float:
        return self._require_bbox()[0]

    @property
    def top(self) -> float:
        return self._require_bbox()[1]

    @property
    def x1(self) -> float:
        return self._require_bbox()[2]

    @property
    def bottom(self) -> float:
        return self._require_bbox()[3]

    @property
    def width(self) -> Optional[float]:
        bbox = self.bbox
        if not bbox:
            return None
        return bbox[2] - bbox[0]

    @property
    def height(self) -> Optional[float]:
        bbox = self.bbox
        if not bbox:
            return None
        return bbox[3] - bbox[1]

    @property
    def has_polygon(self) -> bool:
        return False

    @property
    def polygon(self) -> List[Tuple[float, float]]:
        bbox = self.bbox
        if not bbox:
            return []
        x0, y0, x1, y1 = bbox
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def extract_text(self, apply_exclusions: bool = True, **kwargs) -> str:
        """Concatenate text from constituent regions while preserving flow order."""
        if self._cached_text is not None and apply_exclusions:
            return self._cached_text

        if not self.constituent_regions:
            return ""

        from natural_pdf.elements.element_collection import ElementCollection

        elements = self.elements(apply_exclusions=apply_exclusions)
        # ElementCollection.extract_text handles ordering and layout-specific kwargs.
        extracted = elements.extract_text(**kwargs)

        if not extracted:
            return ""

        if apply_exclusions:
            self._cached_text = extracted
        return extracted

    def elements(self, apply_exclusions: bool = True) -> "ElementCollection":  # Stringized return
        """
        Collects all unique physical elements from all constituent physical regions.

        Args:
            apply_exclusions: Whether to respect PDF exclusion zones within each
                              constituent physical region when gathering elements.

        Returns:
            An ElementCollection containing all unique elements.
        """
        from natural_pdf.elements.element_collection import ElementCollection

        if self._cached_elements is not None and apply_exclusions:  # Simple cache check
            return self._cached_elements

        if not self.constituent_regions:
            return ElementCollection([])

        all_physical_elements: List["PhysicalElement"] = []  # Stringized item type
        seen_elements = (
            set()
        )  # To ensure uniqueness if elements are shared or duplicated by region definitions

        for region in self.constituent_regions:
            # Region.get_elements() returns a list, not ElementCollection
            elements_in_region: List["PhysicalElement"] = region.get_elements(
                apply_exclusions=apply_exclusions
            )
            for elem in elements_in_region:
                if elem not in seen_elements:  # Check for uniqueness based on object identity
                    all_physical_elements.append(elem)
                    seen_elements.add(elem)

        # Basic reading order sort based on original page and coordinates.
        def get_sort_key(phys_elem: "PhysicalElement"):  # Stringized param type
            page_idx = -1
            if hasattr(phys_elem, "page") and hasattr(phys_elem.page, "index"):
                page_idx = phys_elem.page.index
            return (page_idx, phys_elem.top, phys_elem.x0)

        try:
            sorted_physical_elements = sorted(all_physical_elements, key=get_sort_key)
        except AttributeError as exc:
            raise AttributeError(
                "Could not sort elements in FlowRegion; ensure elements expose page, top, and x0"
            ) from exc

        result_collection = ElementCollection(sorted_physical_elements)
        if apply_exclusions:
            self._cached_elements = result_collection
        return result_collection

    def find(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        overlap: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
        engine: Optional[str] = None,
    ) -> Optional["PhysicalElement"]:
        """Return the first physical element across constituent regions matching selector/text.

        Args:
            selector: CSS-like selector string.
            text: Text shortcut equivalent to ``text:contains(...)``.
            overlap: How constituent regions determine containment (defaults to ``"full"``).
            apply_exclusions: Whether exclusion zones within each region are honoured.
            regex: Whether selector/text filters use regular expressions.
            case: Whether text comparisons are case-sensitive.
            text_tolerance: Optional pdfplumber-style tolerance overrides applied temporarily.
            auto_text_tolerance: Optional overrides for automatic tolerance behaviour.
            reading_order: Whether matches use flow reading order.
            near_threshold: Maximum distance (in points) used by ``:near`` selectors.
            engine: Optional selector engine forwarded to constituent regions.

        Returns:
            The first matching physical element, if any.
        """

        return resolve_service(self, "selector").find(
            self,
            selector=selector,
            text=text,
            overlap=overlap,
            apply_exclusions=apply_exclusions,
            regex=regex,
            case=case,
            text_tolerance=text_tolerance,
            auto_text_tolerance=auto_text_tolerance,
            reading_order=reading_order,
            near_threshold=near_threshold,
            engine=engine,
        )

    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        overlap: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
        engine: Optional[str] = None,
    ) -> "ElementCollection":
        """Return every physical element across constituent regions matching selector/text.

        Args mirror :meth:`find`; ``engine`` is forwarded to constituent regions when provided.
        """

        from natural_pdf.elements.element_collection import ElementCollection

        return resolve_service(self, "selector").find_all(
            self,
            selector=selector,
            text=text,
            overlap=overlap,
            apply_exclusions=apply_exclusions,
            regex=regex,
            case=case,
            text_tolerance=text_tolerance,
            auto_text_tolerance=auto_text_tolerance,
            reading_order=reading_order,
            near_threshold=near_threshold,
            engine=engine,
        )

    def highlight(
        self, label: Optional[str] = None, color: Optional[Union[Tuple, str]] = None, **kwargs
    ) -> Optional["PIL_Image"]:
        """
        Highlights all constituent physical regions on their respective pages.

        Args:
            label: A base label for the highlights. Each constituent region might get an indexed label.
            color: Color for the highlight.
            **kwargs: Additional arguments for the underlying highlight method.

        Returns:
            Image generated by the underlying highlight call, or None if no highlights were added.
        """
        if not self.constituent_regions:
            return None

        base_label = label if label else "FlowRegionPart"
        for i, region in enumerate(self.constituent_regions):
            current_label = (
                f"{base_label}_{i+1}" if len(self.constituent_regions) > 1 else base_label
            )
            region.highlight(label=current_label, color=color, **kwargs)
        return None

    def highlights(self, show: bool = False) -> "HighlightContext":
        """
        Create a highlight context for accumulating highlights.

        This allows for clean syntax to show multiple highlight groups:

        Example:
            with flow_region.highlights() as h:
                h.add(flow_region.find_all('table'), label='tables', color='blue')
                h.add(flow_region.find_all('text:bold'), label='bold text', color='red')
                h.show()

        Or with automatic display:
            with flow_region.highlights(show=True) as h:
                h.add(flow_region.find_all('table'), label='tables')
                h.add(flow_region.find_all('text:bold'), label='bold')
                # Automatically shows when exiting the context

        Args:
            show: If True, automatically show highlights when exiting context

        Returns:
            HighlightContext for accumulating highlights
        """
        from natural_pdf.core.highlighting_service import HighlightContext

        return HighlightContext(self, show_on_exit=show)

    def to_images(
        self,
        resolution: float = 150,
        **kwargs,
    ) -> List["PIL_Image"]:
        """
        Generates and returns a list of cropped PIL Images,
        one for each constituent physical region of this FlowRegion.
        """
        if not self.constituent_regions:
            logger.info("FlowRegion.to_images() called on an empty FlowRegion.")
            return []

        cropped_images: List["PIL_Image"] = []
        for region_part in self.constituent_regions:
            # Use render() for clean image without highlights
            img = region_part.render(resolution=resolution, crop=True, **kwargs)
            if img:
                cropped_images.append(img)

        return cropped_images

    def __repr__(self) -> str:
        return (
            f"<FlowRegion constituents={len(self.constituent_regions)}, flow={self.flow}, "
            f"source_bbox={self.source_flow_element.bbox if self.source_flow_element else 'N/A'}>"
        )

    @overload
    def expand(self, amount: float, *, apply_exclusions: bool = True) -> "FlowRegion": ...

    @overload
    def expand(
        self,
        *,
        left: Union[float, bool, str] = 0,
        right: Union[float, bool, str] = 0,
        top: Union[float, bool, str] = 0,
        bottom: Union[float, bool, str] = 0,
        width_factor: float = 1.0,
        height_factor: float = 1.0,
        apply_exclusions: bool = True,
    ) -> "FlowRegion": ...

    def expand(
        self,
        amount: Optional[float] = None,
        *,
        left: Union[float, bool, str] = 0,
        right: Union[float, bool, str] = 0,
        top: Union[float, bool, str] = 0,
        bottom: Union[float, bool, str] = 0,
        width_factor: float = 1.0,
        height_factor: float = 1.0,
        apply_exclusions: bool = True,
    ) -> "FlowRegion":
        """
        Create a new FlowRegion with all constituent regions expanded.

        Args:
            left: Amount to expand left edge (positive value expands leftwards)
            right: Amount to expand right edge (positive value expands rightwards)
            top: Amount to expand top edge (positive value expands upwards)
            bottom: Amount to expand bottom edge (positive value expands downwards)
            width_factor: Factor to multiply width by (applied after absolute expansion)
            height_factor: Factor to multiply height by (applied after absolute expansion)

        Returns:
            New FlowRegion with expanded constituent regions
        """
        if not self.constituent_regions:
            return self._spawn_from_regions([])

        if amount is not None:
            left = right = top = bottom = amount

        expanded_regions = []
        for idx, region in enumerate(self.constituent_regions):
            # Determine which adjustments to apply based on flow arrangement
            apply_left = left
            apply_right = right
            apply_top = top
            apply_bottom = bottom

            if self.flow.arrangement == "vertical":
                # In a vertical flow, only the *first* region should react to `top`
                # and only the *last* region should react to `bottom`.  This keeps
                # the virtual contiguous area intact while allowing users to nudge
                # the flow boundaries.
                if idx != 0:
                    apply_top = 0
                if idx != len(self.constituent_regions) - 1:
                    apply_bottom = 0
                # left/right apply to every region (same column width change)
            else:  # horizontal flow
                # In a horizontal flow, only the first region reacts to `left`
                # and only the last region reacts to `right`.
                if idx != 0:
                    apply_left = 0
                if idx != len(self.constituent_regions) - 1:
                    apply_right = 0
                # top/bottom apply to every region in horizontal flows

            # Skip no-op expansion to avoid extra Region objects
            needs_expansion = (
                any(v not in (0, 1.0) for v in (apply_left, apply_right, apply_top, apply_bottom))
                or width_factor != 1.0
                or height_factor != 1.0
            )

            expanded_region = (
                region.expand(
                    left=apply_left,
                    right=apply_right,
                    top=apply_top,
                    bottom=apply_bottom,
                    width_factor=width_factor,
                    height_factor=height_factor,
                    apply_exclusions=apply_exclusions,
                )
                if needs_expansion
                else region
            )
            expanded_regions.append(expanded_region)

        # Create new FlowRegion with expanded constituent regions
        return self._spawn_from_regions(expanded_regions)

    def _spawn_from_regions(self, regions: List["PhysicalRegion"]) -> "FlowRegion":
        """Create a FlowRegion clone with the supplied constituent regions."""
        new_flow_region = FlowRegion(
            flow=self.flow,
            constituent_regions=list(regions),
            source_flow_element=self.source_flow_element,
            boundary_element_found=self.boundary_element_found,
        )
        new_flow_region.source = self.source
        new_flow_region.region_type = self.region_type
        if isinstance(self.metadata, dict):
            new_flow_region.metadata = self.metadata.copy()
        else:
            new_flow_region.metadata = self.metadata
        new_flow_region._cached_text = None
        new_flow_region._cached_elements = None
        new_flow_region._cached_bbox = None
        return new_flow_region

    def _direction(
        self,
        direction: str,
        size: Optional[float] = None,
        cross_size: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: float = 0.0,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["PhysicalRegion"] = None,
        anchor: str = "start",
        **kwargs,
    ) -> "FlowRegion":
        """Shared directional helper used by the navigation service."""

        regions = list(self.constituent_regions)
        if not regions:
            return self._spawn_from_regions([])

        arrangement = getattr(self.flow, "arrangement", "vertical") or "vertical"
        normalized = direction.lower()
        base_kwargs: Dict[str, Any] = {
            "include_source": include_source,
            "until": until,
            "include_endpoint": include_endpoint,
            "offset": offset,
            "apply_exclusions": apply_exclusions,
            "multipage": multipage,
            "within": within,
            "anchor": anchor,
        }
        base_kwargs.update(kwargs)

        def _call(region: "PhysicalRegion", **specific_kwargs) -> "PhysicalRegion":
            params = dict(base_kwargs)
            params.update(specific_kwargs)
            nav_fn = getattr(region, normalized)
            return cast("PhysicalRegion", nav_fn(**params))

        new_regions: List["PhysicalRegion"] = []

        if normalized in {"above", "below"}:
            target_index = 0 if normalized == "above" else len(regions) - 1
            width_arg = "element" if arrangement == "vertical" else cross_size

            def _vertical(region: "PhysicalRegion"):
                return _call(region, height=size, width=width_arg)

            if arrangement == "vertical":
                for idx, region in enumerate(regions):
                    if idx == target_index:
                        new_regions.append(_vertical(region))
                    elif include_source:
                        new_regions.append(region)
            else:
                for region in regions:
                    new_regions.append(_vertical(region))
        elif normalized in {"left", "right"}:
            target_index = 0 if normalized == "left" else len(regions) - 1
            height_arg = "element" if arrangement == "horizontal" else cross_size

            def _horizontal(region: "PhysicalRegion"):
                return _call(region, width=size, height=height_arg)

            if arrangement == "horizontal":
                for idx, region in enumerate(regions):
                    if idx == target_index:
                        new_regions.append(_horizontal(region))
                    elif include_source:
                        new_regions.append(region)
            else:
                for region in regions:
                    new_regions.append(_horizontal(region))
        else:
            raise ValueError(f"Unsupported direction '{direction}' for FlowRegion navigation.")

        return self._spawn_from_regions(new_regions)

    # Table extraction helpers (delegates to underlying physical regions)

    def extract_table(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[["PhysicalRegion"], Optional[str]]] = None,
        show_progress: bool = False,
        content_filter: Optional[Union[str, Callable[[str], bool], List[str]]] = None,
        apply_exclusions: bool = True,
        verticals: Optional[List[float]] = None,
        horizontals: Optional[List[float]] = None,
        stitch_rows: Optional[
            Callable[[List[Optional[str]], List[Optional[str]], int, "PhysicalRegion"], bool]
        ] = None,
        merge_headers: Optional[bool] = None,
        structure_engine: Optional[str] = None,
        **kwargs,
    ) -> TableResult:
        return _flow_table_methods.flowregion_extract_table(
            self,
            method=method,
            table_settings=table_settings,
            use_ocr=use_ocr,
            ocr_config=ocr_config,
            text_options=text_options,
            cell_extraction_func=cell_extraction_func,
            show_progress=show_progress,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
            verticals=verticals,
            horizontals=horizontals,
            stitch_rows=stitch_rows,
            merge_headers=merge_headers,
            structure_engine=structure_engine,
            **kwargs,
        )

    def extract_tables(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        **kwargs,
    ) -> List[List[List[Optional[str]]]]:
        return _flow_table_methods.flowregion_extract_tables(
            self,
            method=method,
            table_settings=table_settings,
            **kwargs,
        )

    def get_sections(
        self,
        start_elements=None,
        end_elements=None,
        new_section_on_page_break: bool = False,
        include_boundaries: str = "both",
        orientation: str = "vertical",
    ) -> "ElementCollection":
        """
        Extract logical sections from this FlowRegion based on start/end boundary elements.

        This delegates to the parent Flow's get_sections() method, but only operates
        on the segments that are part of this FlowRegion.

        Args:
            start_elements: Elements or selector string that mark the start of sections
            end_elements: Elements or selector string that mark the end of sections
            new_section_on_page_break: Whether to start a new section at page boundaries
            include_boundaries: How to include boundary elements: 'start', 'end', 'both', or 'none'
            orientation: 'vertical' (default) or 'horizontal' - determines section direction

        Returns:
            ElementCollection of FlowRegion objects representing the extracted sections

        Example:
            # Split a multi-page table region by headers
            table_region = flow.find("text:contains('Table 4')").below(until="text:contains('Table 5')")
            sections = table_region.get_sections(start_elements="text:bold")
        """
        # Create a temporary Flow with just our constituent regions as segments
        from natural_pdf.flows.flow import Flow

        temp_flow = Flow(
            segments=self.constituent_regions,
            arrangement=self.flow.arrangement,
            alignment=self.flow.alignment,
            segment_gap=self.flow.segment_gap,
        )

        # Delegate to Flow's get_sections implementation
        return temp_flow.get_sections(
            start_elements=start_elements,
            end_elements=end_elements,
            new_section_on_page_break=new_section_on_page_break,
            include_boundaries=include_boundaries,
            orientation=orientation,
        )

    def split(
        self, by: Optional[str] = None, page_breaks: bool = True, **kwargs
    ) -> "ElementCollection":
        """
        Split this FlowRegion into sections.

        This is a convenience method that wraps get_sections() with common splitting patterns.

        Args:
            by: Selector string for elements that mark section boundaries (e.g., "text:bold")
            page_breaks: Whether to also split at page boundaries (default: True)
            **kwargs: Additional arguments passed to get_sections()

        Returns:
            ElementCollection of FlowRegion objects representing the sections

        Example:
            # Split by bold headers
            sections = flow_region.split(by="text:bold")

            # Split only by specific text pattern, ignoring page breaks
            sections = flow_region.split(
                by="text:contains('Section')",
                page_breaks=False
            )
        """
        return self.get_sections(start_elements=by, new_section_on_page_break=page_breaks, **kwargs)

    @property
    def normalized_type(self) -> Optional[str]:
        """
        Return the normalized type for selector compatibility.
        This allows FlowRegion to be found by selectors like 'table'.
        """
        if self.region_type:
            # Convert region_type to normalized format (replace spaces with underscores, lowercase)
            return self.region_type.lower().replace(" ", "_")
        return None

    @property
    def type(self) -> Optional[str]:
        """
        Return the type attribute for selector compatibility.
        This is an alias for normalized_type.
        """
        return self.normalized_type

    def get_highlight_specs(self) -> List[Dict[str, Any]]:
        """
        Get highlight specifications for all constituent regions.

        This implements the highlighting protocol for FlowRegions, returning
        specs for each constituent region so they can be highlighted on their
        respective pages.

        Returns:
            List of highlight specification dictionaries, one for each
            constituent region.
        """
        specs = []

        for region in self.constituent_regions:
            if not hasattr(region, "page") or region.page is None:
                continue

            if not hasattr(region, "bbox") or region.bbox is None:
                continue

            spec = {
                "page": region.page,
                "page_index": region.page.index if hasattr(region.page, "index") else 0,
                "bbox": region.bbox,
                "element": region,  # Reference to the constituent region
            }

            # Add polygon if available
            if hasattr(region, "polygon") and hasattr(region, "has_polygon") and region.has_polygon:
                spec["polygon"] = region.polygon

            specs.append(spec)

        return specs


attach_capability(FlowRegion, "navigation")
attach_capability(FlowRegion, "qa")
attach_capability(FlowRegion, "exclusion")
attach_capability(FlowRegion, "guides")
