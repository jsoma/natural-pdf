import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pdfplumber.utils.geometry import merge_bboxes  # Import merge_bboxes directly

# For runtime image manipulation
from PIL import Image as PIL_Image_Runtime

from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.tables import TableResult

if TYPE_CHECKING:
    from PIL.Image import Image as PIL_Image  # For type hints

    from natural_pdf.core.page import Page as PhysicalPage
    from natural_pdf.elements.base import Element as PhysicalElement
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region as PhysicalRegion

    from .element import FlowElement
    from .flow import Flow

logger = logging.getLogger(__name__)


class FlowRegion(Visualizable):
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
        source_flow_element: "FlowElement",
        boundary_element_found: Optional["PhysicalElement"] = None,
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
        self.source_flow_element: "FlowElement" = source_flow_element
        self.boundary_element_found: Optional["PhysicalElement"] = boundary_element_found

        # Add attributes for grid building, similar to Region
        self.source: Optional[str] = None
        self.region_type: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

        # Cache for expensive operations
        self._cached_text: Optional[str] = None
        self._cached_elements: Optional["ElementCollection"] = None  # Stringized
        self._cached_bbox: Optional[Tuple[float, float, float, float]] = None

    def _get_highlighter(self):
        """Get the highlighting service from constituent regions."""
        if not self.constituent_regions:
            raise RuntimeError("FlowRegion has no constituent regions to get highlighter from")

        # Get highlighter from first constituent region
        first_region = self.constituent_regions[0]
        if hasattr(first_region, "_highlighter"):
            return first_region._highlighter
        elif hasattr(first_region, "page") and hasattr(first_region.page, "_highlighter"):
            return first_region.page._highlighter
        else:
            raise RuntimeError(
                f"Cannot find HighlightingService from FlowRegion constituent regions. "
                f"First region type: {type(first_region).__name__}"
            )

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

        # Group constituent regions by page
        regions_by_page = {}
        for region in self.constituent_regions:
            if hasattr(region, "page") and region.page:
                page = region.page
                if page not in regions_by_page:
                    regions_by_page[page] = []
                regions_by_page[page].append(region)

        if not regions_by_page:
            return []

        # Create RenderSpec for each page
        specs = []
        for page, page_regions in regions_by_page.items():
            spec = RenderSpec(page=page)

            # Handle cropping
            if crop_bbox:
                spec.crop_bbox = crop_bbox
            elif crop == "content" or crop is True:
                # Calculate bounds of regions on this page
                x_coords = []
                y_coords = []
                for region in page_regions:
                    if hasattr(region, "bbox") and region.bbox:
                        x0, y0, x1, y1 = region.bbox
                        x_coords.extend([x0, x1])
                        y_coords.extend([y0, y1])

                if x_coords and y_coords:
                    spec.crop_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

            # Add highlights in show mode
            if mode == "show":
                # Highlight constituent regions
                for i, region in enumerate(page_regions):
                    # Label each part if multiple regions
                    label = None
                    if len(self.constituent_regions) > 1:
                        # Find global index
                        try:
                            global_idx = self.constituent_regions.index(region)
                            label = f"FlowPart_{global_idx + 1}"
                        except ValueError:
                            label = f"FlowPart_{i + 1}"
                    else:
                        label = "FlowRegion"

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

    @property
    def x0(self) -> Optional[float]:
        return self.bbox[0] if self.bbox else None

    @property
    def top(self) -> Optional[float]:
        return self.bbox[1] if self.bbox else None

    @property
    def x1(self) -> Optional[float]:
        return self.bbox[2] if self.bbox else None

    @property
    def bottom(self) -> Optional[float]:
        return self.bbox[3] if self.bbox else None

    @property
    def width(self) -> Optional[float]:
        return self.x1 - self.x0 if self.bbox else None

    @property
    def height(self) -> Optional[float]:
        return self.bottom - self.top if self.bbox else None

    def extract_text(self, apply_exclusions: bool = True, **kwargs) -> str:
        """
        Extracts and concatenates text from all constituent physical regions.
        The order of concatenation respects the flow's arrangement.

        Args:
            apply_exclusions: Whether to respect PDF exclusion zones within each
                              constituent physical region during text extraction.
            **kwargs: Additional arguments passed to the underlying extract_text method
                      of each constituent region.

        Returns:
            The combined text content as a string.
        """
        if (
            self._cached_text is not None and apply_exclusions
        ):  # Simple cache check, might need refinement if kwargs change behavior
            return self._cached_text

        if not self.constituent_regions:
            return ""

        texts: List[str] = []
        # For now, simple concatenation. Order depends on how constituent_regions were added.
        # The FlowElement._flow_direction method is responsible for ordering constituent_regions correctly.
        for region in self.constituent_regions:
            texts.append(region.extract_text(apply_exclusions=apply_exclusions, **kwargs))

        # Join based on flow arrangement (e.g., newline for vertical, space for horizontal)
        # This is a simplification; true layout-aware joining would be more complex.
        joiner = (
            "\n" if self.flow.arrangement == "vertical" else " "
        )  # TODO: Consider flow.segment_gap for proportional spacing between segments
        extracted = joiner.join(t for t in texts if t)

        if apply_exclusions:  # Only cache if standard exclusion behavior
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
        from natural_pdf.elements.element_collection import (
            ElementCollection as RuntimeElementCollection,  # Local import
        )

        if self._cached_elements is not None and apply_exclusions:  # Simple cache check
            return self._cached_elements

        if not self.constituent_regions:
            return RuntimeElementCollection([])

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
        except AttributeError:
            logger.warning(
                "Could not sort elements in FlowRegion by reading order; some elements might be missing page, top or x0 attributes."
            )
            sorted_physical_elements = all_physical_elements

        result_collection = RuntimeElementCollection(sorted_physical_elements)
        if apply_exclusions:
            self._cached_elements = result_collection
        return result_collection

    def find(
        self, selector: Optional[str] = None, *, text: Optional[str] = None, **kwargs
    ) -> Optional["PhysicalElement"]:  # Stringized
        """
        Find the first element in flow order that matches the selector or text.

        This implementation iterates through the constituent regions *in the order
        they appear in ``self.constituent_regions`` (i.e. document flow order),
        delegating the search to each region's own ``find`` method.  It therefore
        avoids constructing a huge intermediate ElementCollection and returns as
        soon as a match is found, which is substantially faster and ensures that
        selectors such as 'table' work exactly as they do on an individual
        Region.
        """
        if not self.constituent_regions:
            return None

        for region in self.constituent_regions:
            try:
                result = region.find(selector=selector, text=text, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(
                    f"FlowRegion.find: error searching region {region}: {e}",
                    exc_info=False,
                )
        return None  # No match found

    def find_all(
        self, selector: Optional[str] = None, *, text: Optional[str] = None, **kwargs
    ) -> "ElementCollection":  # Stringized
        """
        Find **all** elements across the constituent regions that match the given
        selector or text.

        Rather than first materialising *every* element in the FlowRegion (which
        can be extremely slow for multi-page flows), this implementation simply
        chains each region's native ``find_all`` call and concatenates their
        results into a single ElementCollection while preserving flow order.
        """
        from natural_pdf.elements.element_collection import (
            ElementCollection as RuntimeElementCollection,
        )

        matched_elements = []  # type: List["PhysicalElement"]

        if not self.constituent_regions:
            return RuntimeElementCollection([])

        for region in self.constituent_regions:
            try:
                region_matches = region.find_all(selector=selector, text=text, **kwargs)
                if region_matches:
                    # ``region_matches`` is an ElementCollection – extend with its
                    # underlying list so we don't create nested collections.
                    matched_elements.extend(
                        region_matches.elements
                        if hasattr(region_matches, "elements")
                        else list(region_matches)
                    )
            except Exception as e:
                logger.warning(
                    f"FlowRegion.find_all: error searching region {region}: {e}",
                    exc_info=False,
                )

        return RuntimeElementCollection(matched_elements)

    def highlight(
        self, label: Optional[str] = None, color: Optional[Union[Tuple, str]] = None, **kwargs
    ) -> "FlowRegion":  # Stringized
        """
        Highlights all constituent physical regions on their respective pages.

        Args:
            label: A base label for the highlights. Each constituent region might get an indexed label.
            color: Color for the highlight.
            **kwargs: Additional arguments for the underlying highlight method.

        Returns:
            Self for method chaining.
        """
        if not self.constituent_regions:
            return self

        base_label = label if label else "FlowRegionPart"
        for i, region in enumerate(self.constituent_regions):
            current_label = (
                f"{base_label}_{i+1}" if len(self.constituent_regions) > 1 else base_label
            )
            region.highlight(label=current_label, color=color, **kwargs)
        return self

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
            try:
                # Use render() for clean image without highlights
                img = region_part.render(resolution=resolution, crop=True, **kwargs)
                if img:
                    cropped_images.append(img)
            except Exception as e:
                logger.error(
                    f"Error generating image for constituent region {region_part.bbox}: {e}",
                    exc_info=True,
                )

        return cropped_images

    def __repr__(self) -> str:
        return (
            f"<FlowRegion constituents={len(self.constituent_regions)}, flow={self.flow}, "
            f"source_bbox={self.source_flow_element.bbox if self.source_flow_element else 'N/A'}>"
        )

    def expand(
        self,
        left: float = 0,
        right: float = 0,
        top: float = 0,
        bottom: float = 0,
        width_factor: float = 1.0,
        height_factor: float = 1.0,
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
            return FlowRegion(
                flow=self.flow,
                constituent_regions=[],
                source_flow_element=self.source_flow_element,
                boundary_element_found=self.boundary_element_found,
            )

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
                any(
                    v not in (0, 1.0)  # compare width/height factor logically later
                    for v in (apply_left, apply_right, apply_top, apply_bottom)
                )
                or width_factor != 1.0
                or height_factor != 1.0
            )

            try:
                expanded_region = (
                    region.expand(
                        left=apply_left,
                        right=apply_right,
                        top=apply_top,
                        bottom=apply_bottom,
                        width_factor=width_factor,
                        height_factor=height_factor,
                    )
                    if needs_expansion
                    else region
                )
                expanded_regions.append(expanded_region)
            except Exception as e:
                logger.warning(
                    f"FlowRegion.expand: Error expanding constituent region {region.bbox}: {e}",
                    exc_info=False,
                )
                expanded_regions.append(region)

        # Create new FlowRegion with expanded constituent regions
        new_flow_region = FlowRegion(
            flow=self.flow,
            constituent_regions=expanded_regions,
            source_flow_element=self.source_flow_element,
            boundary_element_found=self.boundary_element_found,
        )

        # Copy metadata
        new_flow_region.source = self.source
        new_flow_region.region_type = self.region_type
        new_flow_region.metadata = self.metadata.copy()

        # Clear caches since the regions have changed
        new_flow_region._cached_text = None
        new_flow_region._cached_elements = None
        new_flow_region._cached_bbox = None

        return new_flow_region

    def above(
        self,
        height: Optional[float] = None,
        width: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegion":
        """
        Create a FlowRegion with regions above this FlowRegion.

        For vertical flows: Only expands the topmost constituent region upward.
        For horizontal flows: Expands all constituent regions upward.

        Args:
            height: Height of the region above, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_source: Whether to include this FlowRegion in the result
            until: Optional selector string to specify an upper boundary element
            include_endpoint: Whether to include the boundary element in the region
            **kwargs: Additional parameters

        Returns:
            New FlowRegion with regions above
        """
        if not self.constituent_regions:
            return FlowRegion(
                flow=self.flow,
                constituent_regions=[],
                source_flow_element=self.source_flow_element,
                boundary_element_found=self.boundary_element_found,
            )

        new_regions = []

        if self.flow.arrangement == "vertical":
            # For vertical flow, use FLOW ORDER (index 0 is earliest). Only expand the
            # first constituent region in that order.
            first_region = self.constituent_regions[0]
            for idx, region in enumerate(self.constituent_regions):
                if idx == 0:  # Only expand the first region (earliest in flow)
                    above_region = region.above(
                        height=height,
                        width="element",  # Keep original column width
                        include_source=include_source,
                        until=until,
                        include_endpoint=include_endpoint,
                        **kwargs,
                    )
                    new_regions.append(above_region)
                elif include_source:
                    new_regions.append(region)
        else:  # horizontal flow
            # For horizontal flow, expand all regions upward
            for region in self.constituent_regions:
                above_region = region.above(
                    height=height,
                    width=width,
                    include_source=include_source,
                    until=until,
                    include_endpoint=include_endpoint,
                    **kwargs,
                )
                new_regions.append(above_region)

        return FlowRegion(
            flow=self.flow,
            constituent_regions=new_regions,
            source_flow_element=self.source_flow_element,
            boundary_element_found=self.boundary_element_found,
        )

    def below(
        self,
        height: Optional[float] = None,
        width: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegion":
        """
        Create a FlowRegion with regions below this FlowRegion.

        For vertical flows: Only expands the bottommost constituent region downward.
        For horizontal flows: Expands all constituent regions downward.

        Args:
            height: Height of the region below, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_source: Whether to include this FlowRegion in the result
            until: Optional selector string to specify a lower boundary element
            include_endpoint: Whether to include the boundary element in the region
            **kwargs: Additional parameters

        Returns:
            New FlowRegion with regions below
        """
        if not self.constituent_regions:
            return FlowRegion(
                flow=self.flow,
                constituent_regions=[],
                source_flow_element=self.source_flow_element,
                boundary_element_found=self.boundary_element_found,
            )

        new_regions = []

        if self.flow.arrangement == "vertical":
            # For vertical flow, expand only the LAST constituent region in flow order.
            last_idx = len(self.constituent_regions) - 1
            for idx, region in enumerate(self.constituent_regions):
                if idx == last_idx:
                    below_region = region.below(
                        height=height,
                        width="element",
                        include_source=include_source,
                        until=until,
                        include_endpoint=include_endpoint,
                        **kwargs,
                    )
                    new_regions.append(below_region)
                elif include_source:
                    new_regions.append(region)
        else:  # horizontal flow
            # For horizontal flow, expand all regions downward
            for region in self.constituent_regions:
                below_region = region.below(
                    height=height,
                    width=width,
                    include_source=include_source,
                    until=until,
                    include_endpoint=include_endpoint,
                    **kwargs,
                )
                new_regions.append(below_region)

        return FlowRegion(
            flow=self.flow,
            constituent_regions=new_regions,
            source_flow_element=self.source_flow_element,
            boundary_element_found=self.boundary_element_found,
        )

    def left(
        self,
        width: Optional[float] = None,
        height: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegion":
        """
        Create a FlowRegion with regions to the left of this FlowRegion.

        For vertical flows: Expands all constituent regions leftward.
        For horizontal flows: Only expands the leftmost constituent region leftward.

        Args:
            width: Width of the region to the left, in points
            height: Height mode - "full" for full page height or "element" for element height
            include_source: Whether to include this FlowRegion in the result
            until: Optional selector string to specify a left boundary element
            include_endpoint: Whether to include the boundary element in the region
            **kwargs: Additional parameters

        Returns:
            New FlowRegion with regions to the left
        """
        if not self.constituent_regions:
            return FlowRegion(
                flow=self.flow,
                constituent_regions=[],
                source_flow_element=self.source_flow_element,
                boundary_element_found=self.boundary_element_found,
            )

        new_regions = []

        if self.flow.arrangement == "vertical":
            # For vertical flow, expand all regions leftward
            for region in self.constituent_regions:
                left_region = region.left(
                    width=width,
                    height="element",
                    include_source=include_source,
                    until=until,
                    include_endpoint=include_endpoint,
                    **kwargs,
                )
                new_regions.append(left_region)
        else:  # horizontal flow
            # For horizontal flow, only expand the leftmost region leftward
            leftmost_region = min(self.constituent_regions, key=lambda r: r.x0)
            for region in self.constituent_regions:
                if region == leftmost_region:
                    # Expand this region leftward
                    left_region = region.left(
                        width=width,
                        height="element",
                        include_source=include_source,
                        until=until,
                        include_endpoint=include_endpoint,
                        **kwargs,
                    )
                    new_regions.append(left_region)
                elif include_source:
                    # Include other regions unchanged if include_source is True
                    new_regions.append(region)

        return FlowRegion(
            flow=self.flow,
            constituent_regions=new_regions,
            source_flow_element=self.source_flow_element,
            boundary_element_found=self.boundary_element_found,
        )

    def right(
        self,
        width: Optional[float] = None,
        height: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegion":
        """
        Create a FlowRegion with regions to the right of this FlowRegion.

        For vertical flows: Expands all constituent regions rightward.
        For horizontal flows: Only expands the rightmost constituent region rightward.

        Args:
            width: Width of the region to the right, in points
            height: Height mode - "full" for full page height or "element" for element height
            include_source: Whether to include this FlowRegion in the result
            until: Optional selector string to specify a right boundary element
            include_endpoint: Whether to include the boundary element in the region
            **kwargs: Additional parameters

        Returns:
            New FlowRegion with regions to the right
        """
        if not self.constituent_regions:
            return FlowRegion(
                flow=self.flow,
                constituent_regions=[],
                source_flow_element=self.source_flow_element,
                boundary_element_found=self.boundary_element_found,
            )

        new_regions = []

        if self.flow.arrangement == "vertical":
            # For vertical flow, expand all regions rightward
            for region in self.constituent_regions:
                right_region = region.right(
                    width=width,
                    height="element",
                    include_source=include_source,
                    until=until,
                    include_endpoint=include_endpoint,
                    **kwargs,
                )
                new_regions.append(right_region)
        else:  # horizontal flow
            # For horizontal flow, only expand the rightmost region rightward
            rightmost_region = max(self.constituent_regions, key=lambda r: r.x1)
            for region in self.constituent_regions:
                if region == rightmost_region:
                    # Expand this region rightward
                    right_region = region.right(
                        width=width,
                        height="element",
                        include_source=include_source,
                        until=until,
                        include_endpoint=include_endpoint,
                        **kwargs,
                    )
                    new_regions.append(right_region)
                elif include_source:
                    # Include other regions unchanged if include_source is True
                    new_regions.append(region)

        return FlowRegion(
            flow=self.flow,
            constituent_regions=new_regions,
            source_flow_element=self.source_flow_element,
            boundary_element_found=self.boundary_element_found,
        )

    def to_region(self) -> "FlowRegion":
        """
        Convert this FlowRegion to a region (returns a copy).
        This is equivalent to calling expand() with no arguments.

        Returns:
            Copy of this FlowRegion
        """
        return self.expand()

    @property
    def is_empty(self) -> bool:
        """Checks if the FlowRegion contains no constituent regions or if all are empty."""
        if not self.constituent_regions:
            return True
        # A more robust check might see if extract_text() is empty and elements() is empty.
        # For now, if it has regions, it's not considered empty by this simple check.
        # User Point 4: FlowRegion can be empty (no text, no elements). This implies checking content.
        try:
            return not bool(self.extract_text(apply_exclusions=False).strip()) and not bool(
                self.elements(apply_exclusions=False)
            )
        except Exception:
            return True  # If error during check, assume empty to be safe

    # ------------------------------------------------------------------
    # Table extraction helpers (delegates to underlying physical regions)
    # ------------------------------------------------------------------

    def extract_table(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[["PhysicalRegion"], Optional[str]]] = None,
        show_progress: bool = False,
        # Optional row-level merge predicate. If provided, it decides whether
        # the current row (first row of a segment/page) should be merged with
        # the previous one (to handle multi-page spill-overs).
        stitch_rows: Optional[
            Callable[[List[Optional[str]], List[Optional[str]], int, "PhysicalRegion"], bool]
        ] = None,
        merge_headers: Optional[bool] = None,
        **kwargs,
    ) -> TableResult:
        """Extracts a single logical table from the FlowRegion.

        This is a convenience wrapper that iterates through the constituent
        physical regions **in flow order**, calls their ``extract_table``
        method, and concatenates the resulting rows.  It mirrors the public
        interface of :pymeth:`natural_pdf.elements.region.Region.extract_table`.

        Args:
            method, table_settings, use_ocr, ocr_config, text_options, cell_extraction_func, show_progress:
                Same as in :pymeth:`Region.extract_table` and are forwarded as-is
                to each physical region.
            merge_headers: Whether to merge tables by removing repeated headers from subsequent
                pages/segments. If None (default), auto-detects by checking if the first row
                of each segment matches the first row of the first segment. If segments have
                inconsistent header patterns (some repeat, others don't), raises ValueError.
                Useful for multi-page tables where headers repeat on each page.
            **kwargs: Additional keyword arguments forwarded to the underlying
                ``Region.extract_table`` implementation.

        Returns:
            A TableResult object containing the aggregated table data.  Rows returned from
            consecutive constituent regions are appended in document order.  If
            no tables are detected in any region, an empty TableResult is returned.

        stitch_rows parameter:
            Controls whether the first rows of subsequent segments/regions should be merged
            into the previous row (to handle spill-over across page breaks).
            Applied AFTER header removal if merge_headers is enabled.

            • None (default) – no merging (behaviour identical to previous versions).
            • Callable – custom predicate taking
                   (prev_row, cur_row, row_idx_in_segment, segment_object) → bool.
               Return True to merge `cur_row` into `prev_row` (default column-wise merge is used).
        """

        if table_settings is None:
            table_settings = {}
        if text_options is None:
            text_options = {}

        if not self.constituent_regions:
            return TableResult([])

        # Resolve stitch_rows predicate -------------------------------------------------------
        predicate: Optional[
            Callable[[List[Optional[str]], List[Optional[str]], int, "PhysicalRegion"], bool]
        ] = (stitch_rows if callable(stitch_rows) else None)

        def _default_merge(
            prev_row: List[Optional[str]], cur_row: List[Optional[str]]
        ) -> List[Optional[str]]:
            """Column-wise merge – concatenates non-empty strings with a space."""
            from itertools import zip_longest

            merged: List[Optional[str]] = []
            for p, c in zip_longest(prev_row, cur_row, fillvalue=""):
                if (p or "").strip() and (c or "").strip():
                    merged.append(f"{p} {c}".strip())
                else:
                    merged.append((p or "") + (c or ""))
            return merged

        aggregated_rows: List[List[Optional[str]]] = []
        header_row: Optional[List[Optional[str]]] = None
        merge_headers_enabled = False
        headers_warned = False  # Track if we've already warned about dropping headers
        segment_has_repeated_header = []  # Track which segments have repeated headers

        for region_idx, region in enumerate(self.constituent_regions):
            try:
                region_result = region.extract_table(
                    method=method,
                    table_settings=table_settings.copy(),  # Avoid side-effects
                    use_ocr=use_ocr,
                    ocr_config=ocr_config,
                    text_options=text_options.copy(),
                    cell_extraction_func=cell_extraction_func,
                    show_progress=show_progress,
                    **kwargs,
                )

                # Convert result to list of rows
                if not region_result:
                    continue

                if isinstance(region_result, TableResult):
                    segment_rows = list(region_result)
                else:
                    segment_rows = list(region_result)

                # Handle header detection and merging for multi-page tables
                if region_idx == 0:
                    # First segment: capture potential header row
                    if segment_rows:
                        header_row = segment_rows[0]
                        # Determine if we should merge headers
                        if merge_headers is None:
                            # Auto-detect: we'll check all subsequent segments
                            merge_headers_enabled = False  # Will be determined later
                        else:
                            merge_headers_enabled = merge_headers
                        # Track that first segment exists (for consistency checking)
                        segment_has_repeated_header.append(False)  # First segment doesn't "repeat"
                elif region_idx == 1 and merge_headers is None:
                    # Auto-detection: check if first row of second segment matches header
                    has_header = segment_rows and header_row and segment_rows[0] == header_row
                    segment_has_repeated_header.append(has_header)

                    if has_header:
                        merge_headers_enabled = True
                        # Remove the detected repeated header from this segment
                        segment_rows = segment_rows[1:]
                        if not headers_warned:
                            warnings.warn(
                                "Detected repeated headers in multi-page table. Merging by removing "
                                "repeated headers from subsequent pages.",
                                UserWarning,
                                stacklevel=2,
                            )
                            headers_warned = True
                    else:
                        merge_headers_enabled = False
                elif region_idx > 1:
                    # Check consistency: all segments should have same pattern
                    has_header = segment_rows and header_row and segment_rows[0] == header_row
                    segment_has_repeated_header.append(has_header)

                    # Remove header if merging is enabled and header is present
                    if merge_headers_enabled and has_header:
                        segment_rows = segment_rows[1:]
                elif region_idx > 0 and merge_headers_enabled:
                    # Explicit merge_headers=True: remove headers from subsequent segments
                    if segment_rows and header_row and segment_rows[0] == header_row:
                        segment_rows = segment_rows[1:]
                        if not headers_warned:
                            warnings.warn(
                                "Removing repeated headers from multi-page table during merge.",
                                UserWarning,
                                stacklevel=2,
                            )
                            headers_warned = True

                # Process remaining rows with stitch_rows logic
                for row_idx, row in enumerate(segment_rows):
                    if (
                        predicate is not None
                        and aggregated_rows
                        and predicate(aggregated_rows[-1], row, row_idx, region)
                    ):
                        # Merge with previous row
                        aggregated_rows[-1] = _default_merge(aggregated_rows[-1], row)
                    else:
                        aggregated_rows.append(row)
            except Exception as e:
                logger.error(
                    f"FlowRegion.extract_table: Error extracting table from constituent region {region}: {e}",
                    exc_info=True,
                )

        # Check for inconsistent header patterns after processing all segments
        if merge_headers is None and len(segment_has_repeated_header) > 2:
            # During auto-detection, check for consistency across all segments
            expected_pattern = segment_has_repeated_header[1]  # Pattern from second segment
            for seg_idx, has_header in enumerate(segment_has_repeated_header[2:], 2):
                if has_header != expected_pattern:
                    # Inconsistent pattern detected
                    segments_with_headers = [
                        i for i, has_h in enumerate(segment_has_repeated_header[1:], 1) if has_h
                    ]
                    segments_without_headers = [
                        i for i, has_h in enumerate(segment_has_repeated_header[1:], 1) if not has_h
                    ]
                    raise ValueError(
                        f"Inconsistent header pattern in multi-page table: "
                        f"segments {segments_with_headers} have repeated headers, "
                        f"but segments {segments_without_headers} do not. "
                        f"All segments must have the same header pattern for reliable merging."
                    )

        return TableResult(aggregated_rows)

    def extract_tables(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        **kwargs,
    ) -> List[List[List[Optional[str]]]]:
        """Extract **all** tables from the FlowRegion.

        This simply chains :pymeth:`Region.extract_tables` over each physical
        region and concatenates their results, preserving flow order.

        Args:
            method, table_settings: Forwarded to underlying ``Region.extract_tables``.
            **kwargs: Additional keyword arguments forwarded.

        Returns:
            A list where each item is a full table (list of rows).  The order of
            tables follows the order of the constituent regions in the flow.
        """

        if table_settings is None:
            table_settings = {}

        if not self.constituent_regions:
            return []

        all_tables: List[List[List[Optional[str]]]] = []

        for region in self.constituent_regions:
            try:
                region_tables = region.extract_tables(
                    method=method,
                    table_settings=table_settings.copy(),
                    **kwargs,
                )
                # ``region_tables`` is a list (possibly empty).
                if region_tables:
                    all_tables.extend(region_tables)
            except Exception as e:
                logger.error(
                    f"FlowRegion.extract_tables: Error extracting tables from constituent region {region}: {e}",
                    exc_info=True,
                )

        return all_tables

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
