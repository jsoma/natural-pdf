"""
Base Element class for natural-pdf.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, overload

from PIL import Image

from natural_pdf.classification.mixin import ClassificationMixin
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.describe.mixin import DescribeMixin

# Import selector parsing functions
from natural_pdf.selectors.parser import parse_selector, selector_to_filter_func

if TYPE_CHECKING:
    from natural_pdf.classification.manager import ClassificationManager  # noqa: F401
    from natural_pdf.core.page import Page
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region


def extract_bbox(obj: Any) -> Optional[Tuple[float, float, float, float]]:
    """Extract bounding box coordinates from any object that has bbox properties.

    This utility function provides a standardized way to extract bounding box
    coordinates from various object types that may store bbox information in
    different formats (properties, attributes, or dictionary keys).

    Args:
        obj: Object that might have bbox coordinates. Can be an Element, Region,
            dictionary, or any object with bbox-related attributes.

    Returns:
        Tuple of (x0, top, x1, bottom) coordinates as floats, or None if the
        object doesn't have valid bbox properties. Coordinates are in PDF
        coordinate system (points, with origin at bottom-left).

    Example:
        ```python
        # Works with various object types
        element_bbox = extract_bbox(text_element)  # From Element
        region_bbox = extract_bbox(region)         # From Region
        dict_bbox = extract_bbox({                 # From dictionary
            'x0': 100, 'top': 200, 'x1': 300, 'bottom': 250
        })

        if element_bbox:
            x0, top, x1, bottom = element_bbox
            width = x1 - x0
            height = bottom - top
        ```
    """
    # Try bbox property first (most common)
    if hasattr(obj, "bbox") and obj.bbox is not None:
        bbox = obj.bbox
        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            return tuple(float(coord) for coord in bbox)

    # Try individual coordinate properties
    if all(hasattr(obj, attr) for attr in ["x0", "top", "x1", "bottom"]):
        try:
            return (float(obj.x0), float(obj.top), float(obj.x1), float(obj.bottom))
        except (ValueError, TypeError):
            pass

    # If object is a dict with bbox keys
    if isinstance(obj, dict):
        if all(key in obj for key in ["x0", "top", "x1", "bottom"]):
            try:
                return (float(obj["x0"]), float(obj["top"]), float(obj["x1"]), float(obj["bottom"]))
            except (ValueError, TypeError):
                pass

    return None


class DirectionalMixin:
    """Mixin class providing directional methods for both Element and Region classes.

    This mixin provides spatial navigation capabilities that allow elements and regions
    to create new regions in specific directions (left, right, above, below) relative
    to themselves. This forms the foundation of natural-pdf's spatial navigation system.

    The directional methods use the PDF coordinate system where:
    - x increases from left to right
    - y increases from bottom to top (PDF standard)
    - Origin (0, 0) is at the bottom-left of the page

    Methods provided:
    - left(): Create region to the left
    - right(): Create region to the right
    - above(): Create region above
    - below(): Create region below

    Note:
        This mixin requires the implementing class to have 'page', 'x0', 'top',
        'x1', and 'bottom' attributes for coordinate calculations.
    """

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
        Protected helper method to create a region in a specified direction relative to this element/region.

        Args:
            direction: 'left', 'right', 'above', or 'below'
            size: Size in the primary direction (width for horizontal, height for vertical)
            cross_size: Size in the cross direction ('full' or 'element')
            include_source: Whether to include this element/region's area in the result
            until: Optional selector string to specify a boundary element
            include_endpoint: Whether to include the boundary element found by 'until'
            **kwargs: Additional parameters for the 'until' selector search

        Returns:
            Region object
        """
        import math  # Use math.inf for infinity

        is_horizontal = direction in ("left", "right")
        is_positive = direction in ("right", "below")  # right/below are positive directions
        pixel_offset = 1  # Offset for excluding elements/endpoints

        # 1. Determine initial boundaries based on direction and include_source
        if is_horizontal:
            # Initial cross-boundaries (vertical)
            y0 = 0 if cross_size == "full" else self.top
            y1 = self.page.height if cross_size == "full" else self.bottom

            # Initial primary boundaries (horizontal)
            if is_positive:  # right
                x0_initial = self.x0 if include_source else self.x1 + pixel_offset
                x1_initial = self.x1  # This edge moves
            else:  # left
                x0_initial = self.x0  # This edge moves
                x1_initial = self.x1 if include_source else self.x0 - pixel_offset
        else:  # Vertical
            # Initial cross-boundaries (horizontal)
            x0 = 0 if cross_size == "full" else self.x0
            x1 = self.page.width if cross_size == "full" else self.x1

            # Initial primary boundaries (vertical)
            if is_positive:  # below
                y0_initial = self.top if include_source else self.bottom + pixel_offset
                y1_initial = self.bottom  # This edge moves
            else:  # above
                y0_initial = self.top  # This edge moves
                y1_initial = self.bottom if include_source else self.top - pixel_offset

        # 2. Calculate the final primary boundary, considering 'size' or page limits
        if is_horizontal:
            if is_positive:  # right
                x1_final = min(
                    self.page.width,
                    x1_initial + (size if size is not None else (self.page.width - x1_initial)),
                )
                x0_final = x0_initial
            else:  # left
                x0_final = max(0, x0_initial - (size if size is not None else x0_initial))
                x1_final = x1_initial
        else:  # Vertical
            if is_positive:  # below
                y1_final = min(
                    self.page.height,
                    y1_initial + (size if size is not None else (self.page.height - y1_initial)),
                )
                y0_final = y0_initial
            else:  # above
                y0_final = max(0, y0_initial - (size if size is not None else y0_initial))
                y1_final = y1_initial

        # 3. Handle 'until' selector if provided
        target = None
        if until:
            all_matches = self.page.find_all(until, **kwargs)
            matches_in_direction = []

            # Filter and sort matches based on direction
            if direction == "above":
                matches_in_direction = [m for m in all_matches if m.bottom <= self.top]
                matches_in_direction.sort(key=lambda e: e.bottom, reverse=True)
            elif direction == "below":
                matches_in_direction = [m for m in all_matches if m.top >= self.bottom]
                matches_in_direction.sort(key=lambda e: e.top)
            elif direction == "left":
                matches_in_direction = [m for m in all_matches if m.x1 <= self.x0]
                matches_in_direction.sort(key=lambda e: e.x1, reverse=True)
            elif direction == "right":
                matches_in_direction = [m for m in all_matches if m.x0 >= self.x1]
                matches_in_direction.sort(key=lambda e: e.x0)

            if matches_in_direction:
                target = matches_in_direction[0]

                # Adjust the primary boundary based on the target
                if is_horizontal:
                    if is_positive:  # right
                        x1_final = target.x1 if include_endpoint else target.x0 - pixel_offset
                    else:  # left
                        x0_final = target.x0 if include_endpoint else target.x1 + pixel_offset
                else:  # Vertical
                    if is_positive:  # below
                        y1_final = target.bottom if include_endpoint else target.top - pixel_offset
                    else:  # above
                        y0_final = target.top if include_endpoint else target.bottom + pixel_offset

                # Adjust cross boundaries if cross_size is 'element'
                if cross_size == "element":
                    if is_horizontal:  # Adjust y0, y1
                        y0 = min(y0, self.top)
                        y1 = max(y1, self.bottom)
                    else:  # Adjust x0, x1
                        x0 = min(x0, self.x0)
                        x1 = max(x1, self.x1)

        # 4. Finalize bbox coordinates
        if is_horizontal:
            bbox = (x0_final, y0, x1_final, y1)
        else:
            bbox = (x0, y0_final, x1, y1_final)

        # Ensure valid coordinates (x0 <= x1, y0 <= y1)
        final_x0 = min(bbox[0], bbox[2])
        final_y0 = min(bbox[1], bbox[3])
        final_x1 = max(bbox[0], bbox[2])
        final_y1 = max(bbox[1], bbox[3])
        final_bbox = (final_x0, final_y0, final_x1, final_y1)

        # 5. Create and return appropriate object based on self type
        from natural_pdf.elements.region import Region

        result = Region(self.page, final_bbox)
        result.source_element = self
        result.includes_source = include_source
        # Optionally store the boundary element if found
        if target:
            result.boundary_element = target

        return result

    def above(
        self,
        height: Optional[float] = None,
        width: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "Region":
        """
        Select region above this element/region.

        Args:
            height: Height of the region above, in points
            width: Width mode - "full" (default) for full page width or "element" for element width
            include_source: Whether to include this element/region in the result (default: False)
            until: Optional selector string to specify an upper boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters

        Returns:
            Region object representing the area above

        Examples:
            ```python
            # Default: full page width
            signature.above()  # Gets everything above across full page width

            # Match element width
            signature.above(width='element')  # Gets region above matching signature width

            # Stop at specific element
            signature.above(until='text:contains("Date")')  # Region from date to signature
            ```
        """
        return self._direction(
            direction="above",
            size=height,
            cross_size=width,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def below(
        self,
        height: Optional[float] = None,
        width: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "Region":
        """
        Select region below this element/region.

        Args:
            height: Height of the region below, in points
            width: Width mode - "full" (default) for full page width or "element" for element width
            include_source: Whether to include this element/region in the result (default: False)
            until: Optional selector string to specify a lower boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters

        Returns:
            Region object representing the area below

        Examples:
            ```python
            # Default: full page width
            header.below()  # Gets everything below across full page width

            # Match element width
            header.below(width='element')  # Gets region below matching header width

            # Limited height
            header.below(height=200)  # Gets 200pt tall region below header
            ```
        """
        return self._direction(
            direction="below",
            size=height,
            cross_size=width,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def left(
        self,
        width: Optional[float] = None,
        height: str = "element",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "Region":
        """
        Select region to the left of this element/region.

        Args:
            width: Width of the region to the left, in points
            height: Height mode - "element" (default) for element height or "full" for full page height
            include_source: Whether to include this element/region in the result (default: False)
            until: Optional selector string to specify a left boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters

        Returns:
            Region object representing the area to the left

        Examples:
            ```python
            # Default: matches element height
            table.left()  # Gets region to the left at same height as table

            # Full page height
            table.left(height='full')  # Gets entire left side of page

            # Custom height
            table.left(height=100)  # Gets 100pt tall region to the left
            ```
        """
        return self._direction(
            direction="left",
            size=width,
            cross_size=height,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def right(
        self,
        width: Optional[float] = None,
        height: str = "element",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "Region":
        """
        Select region to the right of this element/region.

        Args:
            width: Width of the region to the right, in points
            height: Height mode - "element" (default) for element height or "full" for full page height
            include_source: Whether to include this element/region in the result (default: False)
            until: Optional selector string to specify a right boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters

        Returns:
            Region object representing the area to the right

        Examples:
            ```python
            # Default: matches element height
            label.right()  # Gets region to the right at same height as label

            # Full page height
            label.right(height='full')  # Gets entire right side of page

            # Custom height
            label.right(height=50)  # Gets 50pt tall region to the right
            ```
        """
        return self._direction(
            direction="right",
            size=width,
            cross_size=height,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def to_region(self):
        return self.expand()

    @overload
    def expand(self, amount: float) -> "Region":
        """Expand in all directions by the same amount."""
        ...

    @overload
    def expand(
        self,
        *,
        left: float = 0,
        right: float = 0,
        top: float = 0,
        bottom: float = 0,
        width_factor: float = 1.0,
        height_factor: float = 1.0,
    ) -> "Region":
        """Expand by different amounts in each direction."""
        ...

    def expand(
        self,
        amount: Optional[float] = None,
        left: float = 0,
        right: float = 0,
        top: float = 0,
        bottom: float = 0,
        width_factor: float = 1.0,
        height_factor: float = 1.0,
    ) -> "Region":
        """
        Create a new region expanded from this element/region.

        Args:
            amount: If provided as the first positional argument, expand all edges by this amount
            left: Amount to expand left edge (positive value expands leftwards)
            right: Amount to expand right edge (positive value expands rightwards)
            top: Amount to expand top edge (positive value expands upwards)
            bottom: Amount to expand bottom edge (positive value expands downwards)
            width_factor: Factor to multiply width by (applied after absolute expansion)
            height_factor: Factor to multiply height by (applied after absolute expansion)

        Returns:
            New expanded Region object

        Examples:
            # Expand 5 pixels in all directions
            expanded = element.expand(5)

            # Expand by different amounts in each direction
            expanded = element.expand(left=10, right=5, top=3, bottom=7)

            # Use width/height factors
            expanded = element.expand(width_factor=1.5, height_factor=2.0)
        """
        # If amount is provided as first positional argument, use it for all directions
        if amount is not None:
            left = right = top = bottom = amount
        # Start with current coordinates
        new_x0 = self.x0
        new_x1 = self.x1
        new_top = self.top
        new_bottom = self.bottom

        # Apply absolute expansions first
        new_x0 -= left
        new_x1 += right
        new_top -= top  # Expand upward (decrease top coordinate)
        new_bottom += bottom  # Expand downward (increase bottom coordinate)

        # Apply percentage factors if provided
        if width_factor != 1.0 or height_factor != 1.0:
            # Calculate center point *after* absolute expansion
            center_x = (new_x0 + new_x1) / 2
            center_y = (new_top + new_bottom) / 2

            # Calculate current width and height *after* absolute expansion
            current_width = new_x1 - new_x0
            current_height = new_bottom - new_top

            # Calculate new width and height
            new_width = current_width * width_factor
            new_height = current_height * height_factor

            # Adjust coordinates based on the new dimensions, keeping the center
            new_x0 = center_x - new_width / 2
            new_x1 = center_x + new_width / 2
            new_top = center_y - new_height / 2
            new_bottom = center_y + new_height / 2

        # Clamp coordinates to page boundaries
        new_x0 = max(0, new_x0)
        new_top = max(0, new_top)
        new_x1 = min(self.page.width, new_x1)
        new_bottom = min(self.page.height, new_bottom)

        # Ensure coordinates are valid (x0 <= x1, top <= bottom)
        if new_x0 > new_x1:
            new_x0 = new_x1 = (new_x0 + new_x1) / 2
        if new_top > new_bottom:
            new_top = new_bottom = (new_top + new_bottom) / 2

        # Create new region with expanded bbox
        from natural_pdf.elements.region import Region

        new_region = Region(self.page, (new_x0, new_top, new_x1, new_bottom))

        return new_region

    # ------------------------------------------------------------------
    # Spatial parent lookup
    # ------------------------------------------------------------------

    def parent(
        self,
        selector: Optional[str] = None,
        *,
        mode: str = "contains",  # "contains" | "center" | "overlap"
    ) -> Optional["Element"]:
        """Return the *smallest* element/region that encloses this one.

        The search is purely geometric – no pre-existing hierarchy is assumed.

        Parameters
        ----------
        selector : str, optional
            CSS-style selector used to filter candidate containers first.
        mode : str, default "contains"
            How to decide if a candidate encloses this element.

            • ``"contains"`` – candidate bbox fully contains *self* bbox.
            • ``"center"``   – candidate contains the centroid of *self*.
            • ``"overlap"``  – any bbox intersection > 0 pt².

        Returns
        -------
        Element | Region | None
            The smallest-area container that matches, or *None* if none found.
        """

        from natural_pdf.selectors.parser import parse_selector, selector_to_filter_func

        # --- Gather candidates ------------------------------------------------
        page = getattr(self, "page", None)
        if page is None:
            return None

        # All basic elements
        try:
            candidates: List["Element"] = list(page.get_elements(apply_exclusions=False))
        except Exception:
            candidates = []

        # Add detected regions if present
        if hasattr(page, "_element_mgr") and hasattr(page._element_mgr, "regions"):
            candidates.extend(list(page._element_mgr.regions))

        # Remove self from pool
        candidates = [c for c in candidates if c is not self]

        # Apply selector filtering early if provided
        if selector:
            sel_obj = parse_selector(selector)
            filt = selector_to_filter_func(sel_obj)
            candidates = [c for c in candidates if filt(c)]

        if not candidates:
            return None

        # Helper to extract bbox (x0, top, x1, bottom)
        def _bbox(obj):
            return extract_bbox(obj)

        # Self metrics
        self_bbox = _bbox(self)
        if self_bbox is None:
            return None
        s_x0, s_y0, s_x1, s_y1 = self_bbox
        s_cx = (s_x0 + s_x1) / 2
        s_cy = (s_y0 + s_y1) / 2

        matches: List["Element"] = []

        for cand in candidates:
            c_bbox = _bbox(cand)
            if c_bbox is None:
                continue
            c_x0, c_y0, c_x1, c_y1 = c_bbox

            if mode == "contains":
                if c_x0 <= s_x0 and c_y0 <= s_y0 and c_x1 >= s_x1 and c_y1 >= s_y1:
                    matches.append(cand)
            elif mode == "center":
                if c_x0 <= s_cx <= c_x1 and c_y0 <= s_cy <= c_y1:
                    matches.append(cand)
            elif mode == "overlap":
                # Compute overlap rectangle
                ox0 = max(c_x0, s_x0)
                oy0 = max(c_y0, s_y0)
                ox1 = min(c_x1, s_x1)
                oy1 = min(c_y1, s_y1)
                if ox1 > ox0 and oy1 > oy0:
                    matches.append(cand)

        if not matches:
            return None

        # Pick the smallest-area match
        def _area(obj):
            bb = _bbox(obj)
            if bb is None:
                return float("inf")
            return (bb[2] - bb[0]) * (bb[3] - bb[1])

        matches.sort(key=_area)
        return matches[0]


class HighlightableMixin:
    """
    Mixin that provides the highlighting protocol for elements.

    This protocol enables ElementCollection.show() to work with mixed content
    including FlowRegions and elements from multiple pages by providing a
    standard way to get highlight specifications.
    """

    def get_highlight_specs(self) -> List[Dict[str, Any]]:
        """
        Get highlight specifications for this element.

        Returns a list of dictionaries, each containing:
        - page: The Page object to highlight on
        - page_index: The 0-based index of the page
        - bbox: The bounding box (x0, y0, x1, y1) to highlight
        - polygon: Optional polygon coordinates for non-rectangular highlights
        - element: Reference to the element being highlighted

        For regular elements, this returns a single spec.
        For FlowRegions, this returns specs for all constituent regions.

        Returns:
            List of highlight specification dictionaries
        """
        # Default implementation for regular elements
        if not hasattr(self, "page") or self.page is None:
            return []

        if not hasattr(self, "bbox") or self.bbox is None:
            return []

        spec = {
            "page": self.page,
            "page_index": self.page.index if hasattr(self.page, "index") else 0,
            "bbox": self.bbox,
            "element": self,
        }

        # Add polygon if available
        if hasattr(self, "polygon") and hasattr(self, "has_polygon") and self.has_polygon:
            spec["polygon"] = self.polygon

        return [spec]


class Element(
    DirectionalMixin, ClassificationMixin, DescribeMixin, HighlightableMixin, Visualizable
):
    """Base class for all PDF elements.

    This class provides common properties and methods for all PDF elements,
    including text elements, rectangles, lines, images, and other geometric shapes.
    It serves as the foundation for natural-pdf's element system and provides
    spatial navigation, classification, and description capabilities through mixins.

    The Element class wraps underlying pdfplumber objects and extends them with:
    - Spatial navigation methods (left, right, above, below)
    - Bounding box and coordinate properties
    - Classification and description capabilities
    - Polygon support for complex shapes
    - Metadata storage for analysis results

    All coordinates use the PDF coordinate system where:
    - Origin (0, 0) is at the bottom-left of the page
    - x increases from left to right
    - y increases from bottom to top

    Attributes:
        type: Element type (e.g., 'char', 'line', 'rect', 'image').
        bbox: Bounding box tuple (x0, top, x1, bottom).
        x0: Left x-coordinate.
        top: Top y-coordinate (minimum y).
        x1: Right x-coordinate.
        bottom: Bottom y-coordinate (maximum y).
        width: Element width (x1 - x0).
        height: Element height (bottom - top).
        page: Reference to the parent Page object.
        metadata: Dictionary for storing analysis results and custom data.

    Example:
        ```python
        pdf = npdf.PDF("document.pdf")
        page = pdf.pages[0]

        # Get text elements
        text_elements = page.chars
        for element in text_elements:
            print(f"Text '{element.get_text()}' at {element.bbox}")

        # Spatial navigation
        first_char = page.chars[0]
        region_to_right = first_char.right(size=100)

        # Classification
        element.classify("document_type", model="clip")
        ```

    Note:
        Element objects are typically created automatically when accessing page
        collections (page.chars, page.words, page.rects, etc.). Direct instantiation
        is rarely needed in normal usage.
    """

    def __init__(self, obj: Dict[str, Any], page: "Page"):
        """Initialize base element.

        Creates an Element object that wraps a pdfplumber data object with enhanced
        functionality for spatial navigation, analysis, and classification.

        Args:
            obj: The underlying pdfplumber object dictionary containing element
                properties like coordinates, text, fonts, etc. This typically comes
                from pdfplumber's chars, words, rects, lines, or images collections.
            page: The parent Page object that contains this element and provides
                access to document-level functionality and other elements.

        Note:
            This constructor is typically called automatically when accessing element
            collections through page properties. Direct instantiation is rarely needed.

        Example:
            ```python
            # Elements are usually accessed through page collections
            page = pdf.pages[0]
            chars = page.chars  # Elements created automatically

            # Direct construction (advanced usage)
            pdfplumber_char = page._page.chars[0]  # Raw pdfplumber data
            element = Element(pdfplumber_char, page)
            ```
        """
        self._obj = obj
        self._page = page

        # Containers for per-element metadata and analysis results (e.g., classification)
        self.metadata: Dict[str, Any] = {}
        # Access analysis results via self.analyses property (see below)

    @property
    def type(self) -> str:
        """Element type."""
        return self._obj.get("object_type", "unknown")

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box (x0, top, x1, bottom)."""
        return (self.x0, self.top, self.x1, self.bottom)

    @property
    def x0(self) -> float:
        """Left x-coordinate."""
        if self.has_polygon:
            return min(pt[0] for pt in self.polygon)
        return self._obj.get("x0", 0)

    @property
    def top(self) -> float:
        """Top y-coordinate."""
        if self.has_polygon:
            return min(pt[1] for pt in self.polygon)
        return self._obj.get("top", 0)

    @property
    def x1(self) -> float:
        """Right x-coordinate."""
        if self.has_polygon:
            return max(pt[0] for pt in self.polygon)
        return self._obj.get("x1", 0)

    @property
    def bottom(self) -> float:
        """Bottom y-coordinate."""
        if self.has_polygon:
            return max(pt[1] for pt in self.polygon)
        return self._obj.get("bottom", 0)

    @property
    def width(self) -> float:
        """Element width."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Element height."""
        return self.bottom - self.top

    @property
    def has_polygon(self) -> bool:
        """Check if this element has polygon coordinates."""
        return (
            "polygon" in self._obj and self._obj["polygon"] and len(self._obj["polygon"]) >= 3
        ) or hasattr(self, "_polygon")

    @property
    def polygon(self) -> List[Tuple[float, float]]:
        """Get polygon coordinates if available, otherwise return rectangle corners."""
        if hasattr(self, "_polygon") and self._polygon:
            return self._polygon
        elif "polygon" in self._obj and self._obj["polygon"]:
            return self._obj["polygon"]
        else:
            # Create rectangle corners as fallback
            return [
                (self._obj.get("x0", 0), self._obj.get("top", 0)),  # top-left
                (self._obj.get("x1", 0), self._obj.get("top", 0)),  # top-right
                (self._obj.get("x1", 0), self._obj.get("bottom", 0)),  # bottom-right
                (self._obj.get("x0", 0), self._obj.get("bottom", 0)),  # bottom-left
            ]

    def is_point_inside(self, x: float, y: float) -> bool:
        """
        Check if a point is inside this element using ray casting algorithm for polygons.

        Args:
            x: X-coordinate to check
            y: Y-coordinate to check

        Returns:
            True if the point is inside the element
        """
        if not self.has_polygon:
            # Use simple rectangle check
            return (self.x0 <= x <= self.x1) and (self.top <= y <= self.bottom)

        # Ray casting algorithm for complex polygons
        poly = self.polygon
        n = len(poly)
        inside = False

        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @property
    def page(self) -> "Page":
        """Get the parent page."""
        return self._page

    def next(
        self,
        selector: Optional[str] = None,
        limit: int = 10,
        apply_exclusions: bool = True,
        **kwargs,
    ) -> Optional["Element"]:
        """
        Find next element in reading order.

        Args:
            selector: Optional selector to filter by
            limit: Maximum number of elements to search through (default: 10)
            apply_exclusions: Whether to apply exclusion regions (default: True)
            **kwargs: Additional parameters for selector filtering (e.g., regex, case)

        Returns:
            Next element or None if not found
        """
        # Get all elements in reading order
        all_elements = self.page.find_all("*", apply_exclusions=apply_exclusions)

        # Find our index in the list
        try:
            # Compare by object identity since bbox could match multiple elements
            idx = next(i for i, elem in enumerate(all_elements) if elem is self)
        except StopIteration:
            # If not found, it might have been filtered out by exclusions
            return None

        # Search for next matching element
        if selector:
            # Filter elements after this one
            candidates = all_elements[idx + 1 :]
            # Limit search range for performance
            candidates = candidates[:limit] if limit else candidates

            # Parse the selector and create a filter function
            parsed_selector = parse_selector(selector)
            # Pass relevant kwargs (like regex, case) to the filter function builder
            filter_func = selector_to_filter_func(parsed_selector, **kwargs)

            # Iterate and return the first match
            for candidate in candidates:
                if filter_func(candidate):
                    return candidate
            return None  # No match found

        # No selector, just return the next element if it exists
        elif idx + 1 < len(all_elements):
            return all_elements[idx + 1]

        return None

    def prev(
        self,
        selector: Optional[str] = None,
        limit: int = 10,
        apply_exclusions: bool = True,
        **kwargs,
    ) -> Optional["Element"]:
        """
        Find previous element in reading order.

        Args:
            selector: Optional selector to filter by
            limit: Maximum number of elements to search through (default: 10)
            apply_exclusions: Whether to apply exclusion regions (default: True)
            **kwargs: Additional parameters for selector filtering (e.g., regex, case)

        Returns:
            Previous element or None if not found
        """
        # Get all elements in reading order
        all_elements = self.page.find_all("*", apply_exclusions=apply_exclusions)

        # Find our index in the list
        try:
            # Compare by object identity since bbox could match multiple elements
            idx = next(i for i, elem in enumerate(all_elements) if elem is self)
        except StopIteration:
            # If not found, it might have been filtered out by exclusions
            return None

        # Search for previous matching element
        if selector:
            # Select elements before this one
            candidates = all_elements[:idx]
            # Reverse to search backwards from the current element
            candidates = candidates[::-1]
            # Limit search range for performance
            candidates = candidates[:limit] if limit else candidates

            # Parse the selector and create a filter function
            parsed_selector = parse_selector(selector)
            # Pass relevant kwargs (like regex, case) to the filter function builder
            filter_func = selector_to_filter_func(parsed_selector, **kwargs)

            # Iterate and return the first match (from reversed list)
            for candidate in candidates:
                if filter_func(candidate):
                    return candidate
            return None  # No match found

        # No selector, just return the previous element if it exists
        elif idx > 0:
            return all_elements[idx - 1]

        return None

    def nearest(
        self,
        selector: str,
        max_distance: Optional[float] = None,
        apply_exclusions: bool = True,
        **kwargs,
    ) -> Optional["Element"]:
        """
        Find nearest element matching selector.

        Args:
            selector: CSS-like selector string
            max_distance: Maximum distance to search (default: None = unlimited)
            apply_exclusions: Whether to apply exclusion regions (default: True)
            **kwargs: Additional parameters

        Returns:
            Nearest element or None if not found
        """
        # Find matching elements
        matches = self.page.find_all(selector, apply_exclusions=apply_exclusions, **kwargs)
        if not matches:
            return None

        # Calculate distance to center point of this element
        self_center_x = (self.x0 + self.x1) / 2
        self_center_y = (self.top + self.bottom) / 2

        # Calculate distances to each match
        distances = []
        for match in matches:
            if match is self:  # Skip self
                continue

            match_center_x = (match.x0 + match.x1) / 2
            match_center_y = (match.top + match.bottom) / 2

            # Euclidean distance
            distance = (
                (match_center_x - self_center_x) ** 2 + (match_center_y - self_center_y) ** 2
            ) ** 0.5

            # Filter by max_distance if specified
            if max_distance is None or distance <= max_distance:
                distances.append((match, distance))

        # Sort by distance and return the closest
        if distances:
            distances.sort(key=lambda x: x[1])
            return distances[0][0]

        return None

    def until(
        self, selector: str, include_endpoint: bool = True, width: str = "element", **kwargs
    ) -> "Region":
        """
        Select content from this element until matching selector.

        Args:
            selector: CSS-like selector string
            include_endpoint: Whether to include the endpoint element in the region (default: True)
            width: Width mode - "element" to use element widths or "full" for full page width
            **kwargs: Additional selection parameters

        Returns:
            Region object representing the selected content
        """
        from natural_pdf.elements.region import Region

        # Find the target element
        target = self.page.find(selector, **kwargs)
        if not target:
            # If target not found, return a region with just this element
            return Region(self.page, self.bbox)

        # Use full page width if requested
        if width == "full":
            x0 = 0
            x1 = self.page.width
            # Determine vertical bounds based on element positions
            if target.top >= self.bottom:  # Target is below this element
                top = self.top
                bottom = (
                    target.bottom if include_endpoint else target.top - 1
                )  # Subtract 1 pixel when excluding
            else:  # Target is above this element
                top = (
                    target.top if include_endpoint else target.bottom + 1
                )  # Add 1 pixel when excluding
                bottom = self.bottom
            return Region(self.page, (x0, top, x1, bottom))

        # Otherwise use element-based width
        # Determine the correct order for creating the region
        # If the target is below this element (normal reading order)
        if target.top >= self.bottom:
            x0 = min(self.x0, target.x0 if include_endpoint else target.x1)
            x1 = max(self.x1, target.x1 if include_endpoint else target.x0)
            top = self.top
            bottom = (
                target.bottom if include_endpoint else target.top - 1
            )  # Subtract 1 pixel when excluding
        # If the target is above this element (reverse reading order)
        elif target.bottom <= self.top:
            x0 = min(self.x0, target.x0 if include_endpoint else target.x1)
            x1 = max(self.x1, target.x1 if include_endpoint else target.x0)
            top = (
                target.top if include_endpoint else target.bottom + 1
            )  # Add 1 pixel when excluding
            bottom = self.bottom
        # If they're side by side, use the horizontal version
        elif target.x0 >= self.x1:  # Target is to the right
            x0 = self.x0
            x1 = target.x1 if include_endpoint else target.x0
            top = min(self.top, target.top if include_endpoint else target.bottom)
            bottom = max(self.bottom, target.bottom if include_endpoint else target.top)
        else:  # Target is to the left
            x0 = target.x0 if include_endpoint else target.x1
            x1 = self.x1
            top = min(self.top, target.top if include_endpoint else target.bottom)
            bottom = max(self.bottom, target.bottom if include_endpoint else target.top)

        region = Region(self.page, (x0, top, x1, bottom))
        region.source_element = self
        region.end_element = target
        return region

    # Note: select_until method removed in favor of until()

    def extract_text(self, preserve_whitespace=True, use_exclusions=True, **kwargs) -> str:
        """
        Extract text from this element.

        Args:
            preserve_whitespace: Whether to keep blank characters (default: True)
            use_exclusions: Whether to apply exclusion regions (default: True)
            **kwargs: Additional extraction parameters

        Returns:
            Extracted text as string
        """
        # Default implementation - override in subclasses
        return ""

    # Note: extract_text_compat method removed

    def highlight(
        self,
        label: str = "",
        color: Optional[Tuple[float, float, float]] = None,
        use_color_cycling: bool = True,
        annotate: Optional[List[str]] = None,
        existing: str = "append",
    ) -> "Element":
        """Highlight the element with the specified colour.

        Highlight the element on the page.
        """

        # Access the correct highlighter service
        highlighter = self.page._highlighter

        # Prepare common arguments
        highlight_args = {
            "page_index": self.page.index,
            "color": color,
            "label": label,
            "use_color_cycling": use_color_cycling,
            "element": self,  # Pass the element itself so attributes can be accessed
            "annotate": annotate,
            "existing": existing,
        }

        # Call the appropriate service method based on geometry
        if self.has_polygon:
            highlight_args["polygon"] = self.polygon
            highlighter.add_polygon(**highlight_args)
        else:
            highlight_args["bbox"] = self.bbox
            highlighter.add(**highlight_args)

        return self

    def _get_render_specs(
        self,
        mode: Literal["show", "render"] = "show",
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        highlights: Optional[List[Dict[str, Any]]] = None,
        crop: Union[bool, Literal["content"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        label: Optional[str] = None,
        **kwargs,
    ) -> List[RenderSpec]:
        """Get render specifications for this element.

        Args:
            mode: Rendering mode - 'show' includes highlights, 'render' is clean
            color: Color for highlighting this element in show mode
            highlights: Additional highlight groups to show
            crop: Whether to crop to element bounds
            crop_bbox: Explicit crop bounds
            label: Optional label for this element
            **kwargs: Additional parameters

        Returns:
            List with single RenderSpec for this element's page
        """
        if not hasattr(self, "page") or self.page is None:
            return []

        spec = RenderSpec(page=self.page)

        # Handle cropping
        if crop_bbox:
            spec.crop_bbox = crop_bbox
        elif crop == "content" or crop is True:
            # Crop to element bounds
            if hasattr(self, "bbox") and self.bbox:
                spec.crop_bbox = self.bbox

        # Add highlight in show mode
        if mode == "show":
            # Use provided label or generate one
            element_label = label if label is not None else self.__class__.__name__

            spec.add_highlight(
                element=self,
                color=color or "red",  # Default red for single element
                label=element_label,
            )

            # Add additional highlight groups if provided
            if highlights:
                for group in highlights:
                    group_elements = group.get("elements", [])
                    group_color = group.get("color", color)
                    group_label = group.get("label")

                    for elem in group_elements:
                        # Only add if element is on same page
                        if hasattr(elem, "page") and elem.page == self.page:
                            spec.add_highlight(element=elem, color=group_color, label=group_label)

        return [spec]

    def save(
        self,
        filename: str,
        resolution: Optional[float] = None,
        labels: bool = True,
        legend_position: str = "right",
    ) -> None:
        """
        Save the page with this element highlighted to an image file.

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
        # Save the highlighted image
        self.page.save_image(
            filename, resolution=resolution, labels=labels, legend_position=legend_position
        )
        return self

    # Note: save_image method removed in favor of save()

    def __repr__(self) -> str:
        """String representation of the element."""
        return f"<{self.__class__.__name__} bbox={self.bbox}>"

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
        selector: Optional[str] = None,
        *,
        text: Optional[str] = None,
        overlap: str = "full",
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> Optional["Element"]:
        """
        Find first element within this element's bounds matching the selector OR text.
        Creates a temporary region to perform the search.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for (equivalent to 'text:contains(...)').
            overlap: How to determine if elements overlap with this element: 'full' (fully inside),
                     'partial' (any overlap), or 'center' (center point inside).
                     (default: "full")
            apply_exclusions: Whether to apply exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            **kwargs: Additional parameters for element filtering.

        Returns:
            First matching element or None.
        """
        from natural_pdf.elements.region import Region

        # Create a temporary region from this element's bounds
        temp_region = Region(self.page, self.bbox)
        # Delegate to the region's find method
        return temp_region.find(
            selector=selector,
            text=text,
            overlap=overlap,
            apply_exclusions=apply_exclusions,
            regex=regex,
            case=case,
            **kwargs,
        )

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
        selector: Optional[str] = None,
        *,
        text: Optional[str] = None,
        overlap: str = "full",
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> "ElementCollection":
        """
        Find all elements within this element's bounds matching the selector OR text.
        Creates a temporary region to perform the search.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for (equivalent to 'text:contains(...)').
            overlap: How to determine if elements overlap with this element: 'full' (fully inside),
                     'partial' (any overlap), or 'center' (center point inside).
                     (default: "full")
            apply_exclusions: Whether to apply exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            **kwargs: Additional parameters for element filtering.

        Returns:
            ElementCollection with matching elements.
        """
        from natural_pdf.elements.region import Region

        # Create a temporary region from this element's bounds
        temp_region = Region(self.page, self.bbox)
        # Delegate to the region's find_all method
        return temp_region.find_all(
            selector=selector,
            text=text,
            overlap=overlap,
            apply_exclusions=apply_exclusions,
            regex=regex,
            case=case,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # ClassificationMixin requirements
    # ------------------------------------------------------------------

    def _get_classification_manager(self) -> "ClassificationManager":
        """Access the shared ClassificationManager via the parent PDF."""
        if (
            not hasattr(self, "page")
            or not hasattr(self.page, "pdf")
            or not hasattr(self.page.pdf, "get_manager")
        ):
            raise AttributeError(
                "ClassificationManager cannot be accessed: Parent Page, PDF, or get_manager method missing."
            )

        return self.page.pdf.get_manager("classification")

    def _get_classification_content(self, model_type: str, **kwargs):  # type: ignore[override]
        """Return either text or an image, depending on model_type (text|vision)."""
        if model_type == "text":
            text_content = self.extract_text(layout=False)  # type: ignore[arg-type]
            if not text_content or text_content.isspace():
                raise ValueError(
                    "Cannot classify element with 'text' model: No text content found."
                )
            return text_content

        elif model_type == "vision":
            # Delegate to Region implementation via a temporary expand()
            resolution = kwargs.get("resolution", 150)
            from natural_pdf.elements.region import Region  # Local import to avoid cycles

            # Use render() for clean image without highlights
            return self.expand().render(
                resolution=resolution,
                crop=True,
            )
        else:
            raise ValueError(f"Unsupported model_type for classification: {model_type}")

    # ------------------------------------------------------------------
    # Unified analysis storage (maps to metadata["analysis"])
    # ------------------------------------------------------------------

    @property
    def analyses(self) -> Dict[str, Any]:
        """Dictionary holding model-generated analysis objects (classification, extraction, …)."""
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}
        return self.metadata.setdefault("analysis", {})

    @analyses.setter
    def analyses(self, value: Dict[str, Any]):
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}
        self.metadata["analysis"] = value
