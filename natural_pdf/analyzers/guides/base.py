"""Guide system for table extraction and layout analysis."""

import logging
from collections import UserList
from collections.abc import Iterable as IterableABC
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    SupportsIndex,
    Tuple,
    TypeGuard,
    Union,
    cast,
    overload,
)

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from natural_pdf.core.interfaces import HasPages, HasSinglePage
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.line import LineElement
from natural_pdf.elements.region import Region
from natural_pdf.flows.region import FlowRegion
from natural_pdf.guides.guides_provider import run_guides_detect
from natural_pdf.services.methods import table_methods as _table_methods

from .flow_adapter import FlowGuideAdapter
from .grid_helpers import collect_constituent_pages, register_regions_with_pages
from .helpers import (
    BoolArray,
    Bounds,
    GuidesContext,
    IntArray,
    SupportsGuidesContext,
    _bounds_from_object,
    _collect_line_elements,
    _constituent_regions,
    _ensure_bounds_tuple,
    _has_size,
    _is_flow_region,
    _is_guides_context,
    _label_contiguous_regions,
    _normalize_markers,
    _require_bounds,
    _resolve_single_page,
    _SupportsSize,
)
from .separators import (
    find_min_crossing_separator,
    find_seam_carving_separator,
    stabilize_with_rows,
)
from .text_detect import (
    collect_text_elements,
    find_horizontal_element_gaps,
    find_horizontal_whitespace_gaps,
    find_vertical_element_gaps,
    find_vertical_whitespace_gaps,
)

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.core.page_collection import PageCollection
    from natural_pdf.elements.base import Element
    from natural_pdf.flows.region import FlowRegion

from natural_pdf.tables.result import TableResult

logger = logging.getLogger(__name__)

Bounds = Tuple[float, float, float, float]
OuterBoundaryMode = Union[bool, Literal["first", "last"]]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]


class GuidesList(UserList[float]):
    """A list of guide coordinates that also provides methods for creating guides."""

    def __init__(
        self,
        parent_guides: "Guides",
        axis: Literal["vertical", "horizontal"],
        data: Optional[Iterable[float]] = None,
    ):
        # Always sort the initial data
        values = [float(value) for value in data] if data else []
        super().__init__(sorted(values))
        self._parent = parent_guides
        self._axis: Literal["vertical", "horizontal"] = axis

    if TYPE_CHECKING:
        data: List[float]
    else:

        @property
        def data(self) -> List[float]:
            """Access the underlying coordinate list."""
            return cast(List[float], self.__dict__.setdefault("_data", []))

        @data.setter
        def data(self, value: Iterable[float]) -> None:
            values = [float(v) for v in value] if value else []
            self.__dict__["_data"] = sorted(values)

    def __setitem__(self, i, item):
        """Override to maintain sorted order."""
        self.data[i] = float(item)
        self.data.sort()

    def append(self, item):
        """Override to maintain sorted order."""
        self.data.append(float(item))
        self.data.sort()

    def extend(self, other):
        """Override to maintain sorted order."""
        self.data.extend(float(value) for value in other)
        self.data.sort()

    @overload
    def __getitem__(self, i: SupportsIndex) -> float: ...

    @overload
    def __getitem__(self, i: slice) -> "GuidesList": ...

    def __getitem__(self, i: Union[SupportsIndex, slice]) -> Union[float, "GuidesList"]:
        """Return float for indices and GuidesList for slices."""
        if isinstance(i, slice):
            return self.__class__(self._parent, self._axis, self.data[i])
        return self.data[int(i)]

    def insert(self, i, item):
        """Override to maintain sorted order."""
        self.data.append(float(item))  # Just append and sort
        self.data.sort()

    def __iadd__(self, other):
        """Override to maintain sorted order."""
        self.data.extend(float(value) for value in other)
        self.data.sort()
        return self

    def from_content(
        self,
        markers: Union[str, List[str], "ElementCollection", Callable, None],
        obj: Optional[GuidesContext] = None,
        align: Union[
            Literal["left", "right", "center", "between"], Literal["top", "bottom"]
        ] = "left",
        outer: OuterBoundaryMode = True,
        tolerance: float = 5,
        *,
        append: bool = False,
        apply_exclusions: bool = True,
    ) -> "Guides":
        """
        Create guides from content markers and add to this axis.

        Args:
            markers: Content to search for. Can be:
                - str: single selector (e.g., 'text:contains("Name")') or literal text
                - List[str]: list of selectors or literal text strings
                - ElementCollection: collection of elements to extract text from
                - Callable: function that takes a page and returns markers
                - None: no markers
            obj: Page/Region/FlowRegion to search (uses parent's context if None)
            align: How to align guides relative to found elements:
                - For vertical guides: 'left', 'right', 'center', 'between'
                - For horizontal guides: 'top', 'bottom', 'center', 'between'
                - Note: 'left'/'right' also work for horizontal (mapped to top/bottom)
            outer: Whether to add outer boundary guides
            tolerance: Tolerance for snapping to element edges
            apply_exclusions: Whether to apply exclusion zones when searching for text

        Returns:
            Parent Guides object for chaining
        """
        target_obj = obj or self._parent.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        # Store callable markers for later evaluation
        if callable(markers):
            self._callable = markers
            # For now, evaluate with the current target object to get initial guides
            actual_markers = markers(target_obj)
        else:
            self._callable = None
            actual_markers = markers

        # Normalize alignment for horizontal guides
        if self._axis == "horizontal":
            if align == "top":
                align = "left"
            elif align == "bottom":
                align = "right"

        options = {
            "markers": actual_markers,
            "align": align,
            "outer": outer,
            "tolerance": tolerance,
            "apply_exclusions": apply_exclusions,
        }

        if self._parent.is_flow_region:
            adapter = FlowGuideAdapter(self._parent)
            region_values: Dict[Any, Sequence[float]] = {}

            for region in adapter.regions:
                result = run_guides_detect(
                    axis=self._axis,
                    method="content",
                    context=region,
                    options=options,
                )
                region_values[region] = [float(value) for value in result.coordinates]

            adapter.update_axis_from_regions(self._axis, region_values, append=append)
            return self._parent

        result = run_guides_detect(
            axis=self._axis,
            method="content",
            context=target_obj,
            options=options,
        )

        if append:
            self.data.extend(float(value) for value in result.coordinates)
        else:
            self.data = [float(value) for value in result.coordinates]

        self.data = sorted(set(self.data))

        return self._parent  # Return parent for chaining

    def from_lines(
        self,
        obj: Optional[GuidesContext] = None,
        threshold: Union[float, str] = "auto",
        source_label: Optional[str] = None,
        max_lines: Optional[int] = None,
        outer: bool = False,
        detection_method: str = "pixels",
        resolution: int = 192,
        *,
        n: Optional[int] = None,
        min_gap: Optional[int] = None,
        append: bool = False,
        **detect_kwargs,
    ) -> "Guides":
        """
        Create guides from detected line elements.

        Args:
            obj: Page/Region/FlowRegion to search (uses parent's context if None)
            threshold: Line detection threshold ('auto' or float 0.0-1.0)
            source_label: Filter lines by source label (for vector method)
            max_lines: Maximum lines to use (alias: n)
            n: Convenience alias for max_lines. If provided, overrides max_lines.
            min_gap: Minimum pixel gap enforced between detected lines. Mapped to
                ``min_gap_h`` or ``min_gap_v`` depending on axis (ignored if those
                keys are already supplied via ``detect_kwargs``).
            outer: Whether to add outer boundary guides
            detection_method: 'vector', 'pixels' (default), or 'auto' for hybrid detection.
            resolution: DPI for pixel-based detection (default: 192)
            **detect_kwargs: Additional parameters for pixel-based detection
                (e.g., min_gap_h, min_gap_v, binarization_method, etc.)

        Returns:
            Parent Guides object for chaining
        """
        target_obj = obj or self._parent.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        # Resolve max_lines via alias `n` (n takes priority)
        if n is not None:
            if n <= 0:
                raise ValueError("n must be a positive integer")
            max_lines = n

        # Set appropriate max_lines parameter for underlying API
        max_lines_h = max_lines if self._axis == "horizontal" else None
        max_lines_v = max_lines if self._axis == "vertical" else None

        # Map generic `min_gap` to axis-specific argument expected by detection
        if min_gap is not None:
            if min_gap < 1:
                raise ValueError("min_gap must be ≥ 1 pixel")
            axis_key = "min_gap_h" if self._axis == "horizontal" else "min_gap_v"
            detect_kwargs.setdefault(axis_key, min_gap)

        options = {
            "threshold": threshold,
            "source_label": source_label,
            "max_lines_h": max_lines_h,
            "max_lines_v": max_lines_v,
            "outer": outer,
            "detection_method": detection_method,
            "resolution": resolution,
            **detect_kwargs,
        }

        if self._parent.is_flow_region:
            adapter = FlowGuideAdapter(self._parent)
            region_values: Dict[Any, Sequence[float]] = {}

            for region in adapter.regions:
                result = run_guides_detect(
                    axis=self._axis,
                    method="lines",
                    context=region,
                    options=options,
                )
                region_values[region] = [float(value) for value in result.coordinates]

            adapter.update_axis_from_regions(self._axis, region_values, append=append)
            return self._parent

        result = run_guides_detect(
            axis=self._axis,
            method="lines",
            context=target_obj,
            options=options,
        )

        if append:
            self.data.extend(float(value) for value in result.coordinates)
        else:
            self.data = [float(value) for value in result.coordinates]

        self.data = sorted(set(self.data))

        return self._parent

    def from_whitespace(
        self,
        obj: Optional[GuidesContext] = None,
        min_gap: float = 10,
        *,
        append: bool = False,
    ) -> "Guides":
        target_obj = obj or self._parent.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        options = {"min_gap": min_gap}

        if self._parent.is_flow_region:
            adapter = FlowGuideAdapter(self._parent)
            region_values: Dict[Any, Sequence[float]] = {}
            for region in adapter.regions:
                result = run_guides_detect(
                    axis=self._axis,
                    method="whitespace",
                    context=region,
                    options=options,
                )
                region_values[region] = [float(value) for value in result.coordinates]
            adapter.update_axis_from_regions(self._axis, region_values, append=append)
            return self._parent

        result = run_guides_detect(
            axis=self._axis,
            method="whitespace",
            context=target_obj,
            options=options,
        )

        new_coords = [float(value) for value in result.coordinates]
        if append:
            self.data = sorted(set(self.data).union(new_coords))
        else:
            self.data = sorted(new_coords)

        return self._parent

    def divide(self, n: int = 2, obj: Optional[Union["Page", "Region"]] = None) -> "Guides":
        """
        Divide the space evenly along this axis.

        Args:
            n: Number of divisions (creates n-1 guides)
            obj: Object to divide (uses parent's context if None)

        Returns:
            Parent Guides object for chaining
        """
        target_obj = obj or self._parent.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        axis_literal: Literal["vertical", "horizontal"] = self._axis

        if _is_flow_region(target_obj):
            adapter = FlowGuideAdapter(self._parent)
            region_values: Dict[Any, Sequence[float]] = {}
            for region in adapter.regions:
                region_guides = Guides.divide(obj=region, n=n, axis=axis_literal)
                axis_values = (
                    [float(value) for value in region_guides.vertical]
                    if axis_literal == "vertical"
                    else [float(value) for value in region_guides.horizontal]
                )
                region_values[region] = axis_values

            adapter.update_axis_from_regions(axis_literal, region_values, append=False)
            return self._parent
        else:
            # Create guides using divide
            new_guides = Guides.divide(
                obj=cast(Union["Page", "Region"], target_obj), n=n, axis=axis_literal
            )

            # Replace existing guides instead of extending (no append option here)
            axis_values = (
                new_guides.vertical if axis_literal == "vertical" else new_guides.horizontal
            )
            self.data = sorted(float(value) for value in axis_values)

        # Remove duplicates
        seen: set[float] = set()
        unique: List[float] = []
        for x in self.data:
            if x not in seen:
                seen.add(x)
                unique.append(x)
        self.data = unique

        return self._parent

    def snap_to_whitespace(
        self,
        min_gap: float = 10.0,
        detection_method: str = "pixels",
        threshold: Union[float, str] = "auto",
        on_no_snap: str = "warn",
        obj: Optional[Union["Page", "Region"]] = None,
    ) -> "Guides":
        """
        Snap guides in this axis to whitespace gaps.

        Args:
            min_gap: Minimum gap size to consider
            detection_method: 'pixels' or 'text' for gap detection
            threshold: Threshold for whitespace detection (0.0-1.0) or 'auto'
            on_no_snap: What to do when snapping fails ('warn', 'raise', 'ignore')
            obj: Object to analyze (uses parent's context if None)

        Returns:
            Parent Guides object for chaining
        """
        target_obj = obj or self._parent.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        # Use the parent's snap_to_whitespace but only for this axis
        original_horizontal: List[float] = []
        original_vertical: List[float] = []
        # Temporarily set the parent's guides to only this axis
        if self._axis == "vertical":
            original_horizontal = self._parent.horizontal.data.copy()
            self._parent.horizontal.data = []
        else:
            original_vertical = self._parent.vertical.data.copy()
            self._parent.vertical.data = []

        try:
            # Call the parent's method
            self._parent.snap_to_whitespace(
                axis=self._axis,
                min_gap=min_gap,
                detection_method=detection_method,
                threshold=threshold,
                on_no_snap=on_no_snap,
            )

            # Update our data from the parent
            if self._axis == "vertical":
                self.data = self._parent.vertical.data.copy()
            else:
                self.data = self._parent.horizontal.data.copy()

        finally:
            # Restore the other axis
            if self._axis == "vertical":
                self._parent.horizontal.data = original_horizontal
            else:
                self._parent.vertical.data = original_vertical

        return self._parent

    def snap_to_content(
        self,
        markers: Union[str, List[str], "ElementCollection", None] = "text",
        align: Literal["left", "right", "center"] = "left",
        tolerance: float = 5,
        obj: Optional[Union["Page", "Region"]] = None,
    ) -> "Guides":
        """
        Snap guides in this axis to nearby text content.

        Args:
            markers: Content to snap to. Can be:
                - str: single selector or literal text (default: 'text' for all text)
                - List[str]: list of selectors or literal text strings
                - ElementCollection: collection of elements
                - None: no markers (no snapping)
            align: How to align to the found text
            tolerance: Maximum distance to move when snapping
            obj: Object to search (uses parent's context if None)

        Returns:
            Parent Guides object for chaining
        """
        target_obj = obj or self._parent.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        # Handle special case of 'text' as a selector for all text
        if markers == "text":
            # Get all text elements
            if hasattr(target_obj, "find_all"):
                text_elements = target_obj.find_all("text")
                if hasattr(text_elements, "elements"):
                    text_elements = text_elements.elements

                # Snap each guide to the nearest text element
                for i, guide_pos in enumerate(self.data):
                    best_distance = float("inf")
                    best_pos = guide_pos

                    for elem in text_elements:
                        # Calculate target position based on alignment
                        if self._axis == "vertical":
                            if align == "left":
                                elem_pos = elem.x0
                            elif align == "right":
                                elem_pos = elem.x1
                            else:  # center
                                elem_pos = (elem.x0 + elem.x1) / 2
                        else:  # horizontal
                            if align == "left":  # top for horizontal
                                elem_pos = elem.top
                            elif align == "right":  # bottom for horizontal
                                elem_pos = elem.bottom
                            else:  # center
                                elem_pos = (elem.top + elem.bottom) / 2

                        # Check if this is closer than current best
                        distance = abs(guide_pos - elem_pos)
                        if distance < best_distance and distance <= tolerance:
                            best_distance = distance
                            best_pos = elem_pos

                    # Update guide position if we found a good snap
                    if best_pos != guide_pos:
                        self.data[i] = best_pos
                        logger.debug(
                            f"Snapped {self._axis} guide from {guide_pos:.1f} to {best_pos:.1f}"
                        )
            else:
                logger.warning("Object does not support find_all for text snapping")
        else:
            # Original behavior for specific markers
            marker_texts = _normalize_markers(markers, target_obj)

            # Find each marker and snap guides
            for marker in marker_texts:
                if hasattr(target_obj, "find"):
                    element = target_obj.find(f'text:contains("{marker}")')
                    if not element:
                        logger.warning(f"Could not find text '{marker}' for snapping")
                        continue

                    # Determine target position based on alignment
                    if self._axis == "vertical":
                        if align == "left":
                            target_pos = element.x0
                        elif align == "right":
                            target_pos = element.x1
                        else:  # center
                            target_pos = (element.x0 + element.x1) / 2
                    else:  # horizontal
                        if align == "left":  # top for horizontal
                            target_pos = element.top
                        elif align == "right":  # bottom for horizontal
                            target_pos = element.bottom
                        else:  # center
                            target_pos = (element.top + element.bottom) / 2

                    # Find closest guide and snap if within tolerance
                    if self.data:
                        closest_idx = min(
                            range(len(self.data)), key=lambda i: abs(self.data[i] - target_pos)
                        )
                        if abs(self.data[closest_idx] - target_pos) <= tolerance:
                            self.data[closest_idx] = target_pos

        # Sort after snapping
        self.data.sort()
        return self._parent

    def shift(self, index: int, offset: float) -> "Guides":
        """
        Move a specific guide in this axis by a offset amount.

        Args:
            index: Index of the guide to move
            offset: Amount to move (positive = right/down)

        Returns:
            Parent Guides object for chaining
        """
        if 0 <= index < len(self.data):
            self.data[index] += offset
            self.data.sort()
        else:
            logger.warning(f"Guide index {index} out of range for {self._axis} axis")

        return self._parent

    def add(self, position: Union[float, List[float]]) -> "Guides":
        """
        Add one or more guides at the specified position(s).

        Args:
            position: Coordinate(s) to add guide(s) at. Can be:
                - float: single position
                - List[float]: multiple positions

        Returns:
            Parent Guides object for chaining
        """
        if isinstance(position, (list, tuple)):
            # Add multiple positions
            for pos in position:
                self.append(float(pos))
        else:
            # Add single position
            self.append(float(position))

        self.data.sort()
        return self._parent

    def remove_at(self, index: int) -> "Guides":
        """
        Remove a guide by index.

        Args:
            index: Index of guide to remove

        Returns:
            Parent Guides object for chaining
        """
        if 0 <= index < len(self.data):
            self.data.pop(index)
        return self._parent

    def clear_all(self) -> "Guides":
        """
        Remove all guides from this axis.

        Returns:
            Parent Guides object for chaining
        """
        self.data.clear()
        return self._parent

    def from_headers(
        self,
        headers: Union["ElementCollection", List["Element"], List[str]],
        obj: Optional[Union["Page", "Region"]] = None,
        method: Literal["min_crossings", "seam_carving"] = "min_crossings",
        min_width: Optional[float] = None,
        max_width: Optional[float] = None,
        margin: float = 0.5,
        row_stabilization: bool = True,
        num_samples: int = 400,
        *,
        append: bool = False,
    ) -> "Guides":
        """Create vertical guides for columns based on headers and whitespace valleys.

        This method detects column boundaries by finding optimal vertical separators
        between headers that minimize text crossings, regardless of text alignment.

        Args:
            headers: Column header elements. Can be:
                - ElementCollection: collection of header elements
                - List[Element]: list of header elements
                - List[str]: list of header text to search for
            obj: Page/Region to analyze (uses parent's context if None)
            method: Detection method:
                - 'min_crossings': Fast vector-based minimum intersection count
                - 'seam_carving': Dynamic programming for curved boundaries
            min_width: Minimum column width constraint (pixels)
            max_width: Maximum column width constraint (pixels)
            margin: Buffer space from header edges when searching for separators (default: 0.5)
            row_stabilization: Whether to use row-wise median for stability
            num_samples: Number of x-positions to test per gap (for min_crossings)
            append: Whether to append to existing guides

        Returns:
            Parent Guides object for chaining

        Examples:
            # Create column guides from headers
            headers = page.find_all('text[size=16]')
            guides.vertical.from_headers(headers)

            # From header text strings
            guides.vertical.from_headers(["Statute", "Description", "Level", "Repeat"])

            # With width constraints
            guides.vertical.from_headers(headers, min_width=50, max_width=200)

            # Seam carving for complex layouts
            guides.vertical.from_headers(headers, method='seam_carving')
        """

        if self._axis != "vertical":
            raise ValueError("from_headers() only works for vertical guides (columns)")

        target_obj = obj or self._parent.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        options = {
            "headers": headers,
            "method": method,
            "min_width": min_width,
            "max_width": max_width,
            "margin": margin,
            "row_stabilization": row_stabilization,
            "num_samples": num_samples,
        }

        if self._parent.is_flow_region:
            adapter = FlowGuideAdapter(self._parent)
            region_values: Dict[Any, Sequence[float]] = {}
            for region in adapter.regions:
                result = run_guides_detect(
                    axis="vertical",
                    method="headers",
                    context=region,
                    options=options,
                )
                region_values[region] = [float(value) for value in result.coordinates]
            adapter.update_axis_from_regions("vertical", region_values, append=append)
            return self._parent

        result = run_guides_detect(
            axis="vertical",
            method="headers",
            context=target_obj,
            options=options,
        )

        coords = [float(value) for value in result.coordinates]
        if append:
            self.extend(coords)
        else:
            self.data = coords
        self.data = sorted(set(self.data))

        return self._parent

    @staticmethod
    def _find_min_crossing_separator(
        x0: float,
        x1: float,
        bboxes: List[Tuple[float, float, float, float]],
        num_samples: int,
    ) -> float:
        """Backward-compatible shim for separator helper."""
        return find_min_crossing_separator(x0, x1, bboxes, num_samples)

    @staticmethod
    def _find_seam_carving_separator(
        x0: float,
        x1: float,
        obj,  # Retained for compatibility, unused now
        header_y: float,
        page_bottom: float,
        bboxes: List[Tuple[float, float, float, float]],
    ) -> float:
        return find_seam_carving_separator(x0, x1, header_y, page_bottom, bboxes)

    @staticmethod
    def _stabilize_with_rows(
        separators: List[float],
        obj,
        bboxes: List[Tuple[float, float, float, float]],
        header_y: float,
    ) -> List[float]:
        return stabilize_with_rows(separators, bboxes, header_y)

    def from_stripes(
        self,
        stripes=None,
        color=None,  # Explicitly specify stripe color
    ) -> "Guides":
        """Create guides from striped table rows or columns.

        Creates guides at both edges of stripe elements (e.g., colored table rows).
        Perfect for zebra-striped tables where you need guides at every row boundary.

        Args:
            stripes: Elements representing stripes. If None, auto-detects.
            color: Specific color to look for (e.g., '#00ffff'). If None, finds most common.

        Examples:
            # Auto-detect zebra stripes
            guides.horizontal.from_stripes()

            # Specific color
            guides.horizontal.from_stripes(color='#00ffff')

            # Manual selection
            stripes = page.find_all('rect[fill=#00ffff]')
            guides.horizontal.from_stripes(stripes)

            # Vertical stripes
            guides.vertical.from_stripes(color='#e0e0e0')

        Returns:
            Parent Guides object for chaining
        """
        target_obj = self._parent.context
        if target_obj is None:
            raise ValueError("No context available for stripe detection")

        options = {"stripes": stripes, "color": color}

        if self._parent.is_flow_region:
            adapter = FlowGuideAdapter(self._parent)
            region_values: Dict[Any, Sequence[float]] = {}

            for region in adapter.regions:
                result = run_guides_detect(
                    axis=self._axis,
                    method="stripes",
                    context=region,
                    options=options,
                )
                region_values[region] = [float(value) for value in result.coordinates]

            adapter.update_axis_from_regions(self._axis, region_values, append=True)
            return self._parent

        result = run_guides_detect(
            axis=self._axis,
            method="stripes",
            context=target_obj,
            options=options,
        )

        coords = sorted({float(value) for value in result.coordinates})
        if coords:
            self.extend(coords)

        return self._parent

    def __add__(self, other):
        """Handle addition of GuidesList objects by returning combined data."""
        if isinstance(other, GuidesList):
            return self.data + other.data
        elif isinstance(other, list):
            return self.data + other
        else:
            return NotImplemented


class Guides:
    """
    Manages vertical and horizontal guide lines for table extraction and layout analysis.

    Guides are collections of coordinates that can be used to define table boundaries,
    column positions, or general layout structures. They can be created through various
    detection methods or manually specified.

    Attributes:
        verticals: List of x-coordinates for vertical guide lines
        horizontals: List of y-coordinates for horizontal guide lines
        context: Optional Page/Region that these guides relate to
        bounds: Optional bounding box (x0, y0, x1, y1) for relative coordinate conversion
        snap_behavior: How to handle failed snapping operations ('warn', 'ignore', 'raise')
    """

    def __init__(
        self,
        verticals: Optional[Union[Iterable[float], GuidesContext]] = None,
        horizontals: Optional[Iterable[float]] = None,
        context: Optional[GuidesContext] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        relative: bool = False,
        snap_behavior: Literal["raise", "warn", "ignore"] = "warn",
    ):
        """
        Initialize a Guides object.

        Args:
            verticals: Iterable of x-coordinates for vertical guides, or a context object shorthand
            horizontals: Iterable of y-coordinates for horizontal guides
            context: Object providing spatial context (page, region, flow, etc.)
            bounds: Bounding box (x0, top, x1, bottom) if context not provided
            relative: Whether coordinates are relative (0-1) or absolute
            snap_behavior: How to handle snapping conflicts ('raise', 'warn', or 'ignore')
        """
        context_obj = context
        vertical_seed: Iterable[float] = ()

        # Handle Guides(page) or Guides(flow_region) shorthand
        if (
            verticals is not None
            and horizontals is None
            and context_obj is None
            and _is_guides_context(verticals)
        ):
            context_obj = cast(GuidesContext, verticals)
        elif verticals is not None:
            vertical_seed = cast(Iterable[float], verticals)

        self.context = context_obj
        coerced_bounds = _bounds_from_object(bounds) if bounds is not None else None
        self.bounds: Optional[Bounds] = coerced_bounds
        self.relative = relative
        self.snap_behavior = snap_behavior
        # Backwards compatibility alias for legacy options
        self.on_no_snap = snap_behavior

        # Check if we're dealing with a FlowRegion
        self.is_flow_region = _is_flow_region(context_obj)

        # If FlowRegion, we'll store guides per constituent region
        if self.is_flow_region:
            self._flow_guides: Dict["Region", Tuple[List[float], List[float]]] = {}
            # For unified view across all regions
            self._unified_vertical: List[Tuple[float, "Region"]] = []
            self._unified_horizontal: List[Tuple[float, "Region"]] = []
            # Cache for sorted unique coordinates
            self._vertical_cache: Optional[List[float]] = None
            self._horizontal_cache: Optional[List[float]] = None

        # Initialize with GuidesList instances
        horizontal_seed: Iterable[float] = horizontals if horizontals is not None else ()
        self._vertical = GuidesList(
            self,
            "vertical",
            sorted([float(x) for x in vertical_seed]),
        )
        self._horizontal = GuidesList(
            self,
            "horizontal",
            sorted([float(y) for y in horizontal_seed]),
        )

        # Determine bounds from context if needed
        if self.bounds is None and self.context is not None:
            self.bounds = _bounds_from_object(self.context)

        # Convert relative to absolute if needed
        if self.relative and self.bounds is not None:
            x0, top, x1, bottom = self.bounds
            width = x1 - x0
            height = bottom - top

            self._vertical.data = [x0 + float(v) * width for v in self._vertical]
            self._horizontal.data = [top + float(h) * height for h in self._horizontal]
            self.relative = False

    def _extract_with_table_service(self, host, **kwargs) -> TableResult:
        """Helper to route all table extraction through host helpers when possible."""
        extractor = getattr(host, "extract_table", None)
        if callable(extractor):
            extracted = extractor(**kwargs)
            if isinstance(extracted, TableResult):
                return extracted
            rows_iter = cast(Optional[Iterable[List[Any]]], extracted)
            rows: List[List[Any]] = list(rows_iter or [])
            return TableResult(rows)

        return _table_methods.extract_table(host, **kwargs)

    def _flow_context(self) -> FlowRegion:
        if not _is_flow_region(self.context):
            raise AttributeError("Flow context is not available for these guides")
        return cast(FlowRegion, self.context)

    def _flow_constituent_regions(self) -> Sequence["Region"]:
        return _constituent_regions(self._flow_context())

    @property
    def vertical(self) -> GuidesList:
        """Get vertical guide coordinates."""
        if self.is_flow_region and self._vertical_cache is not None:
            # Return cached unified view
            self._vertical.data = self._vertical_cache
        elif self.is_flow_region and self._unified_vertical:
            # Build unified view from flow guides
            all_verticals = []
            for coord, region in self._unified_vertical:
                all_verticals.append(coord)
            # Remove duplicates and sort
            self._vertical_cache = sorted(list(set(all_verticals)))
            self._vertical.data = self._vertical_cache
        return self._vertical

    @vertical.setter
    def vertical(self, value: Union[List[float], "Guides", None]):
        """Set vertical guides from a list of coordinates or another Guides object."""
        if self.is_flow_region:
            # Invalidate cache when setting new values
            self._vertical_cache = None

        if value is None:
            self._vertical.data = []
        elif isinstance(value, Guides):
            # Extract vertical coordinates from another Guides object
            self._vertical.data = sorted([float(x) for x in value.vertical])
        elif isinstance(value, str):
            # Explicitly reject strings to avoid confusing iteration over characters
            raise TypeError(
                f"vertical cannot be a string, got '{value}'. Use a list of coordinates or Guides object."
            )
        elif hasattr(value, "__iter__"):
            # Handle list/tuple of coordinates
            try:
                self._vertical.data = sorted([float(x) for x in value])
            except (ValueError, TypeError) as e:
                raise TypeError(f"vertical must contain numeric values, got {value}: {e}")
        else:
            raise TypeError(f"vertical must be a list, Guides object, or None, got {type(value)}")

    @property
    def horizontal(self) -> GuidesList:
        """Get horizontal guide coordinates."""
        if self.is_flow_region and self._horizontal_cache is not None:
            # Return cached unified view
            self._horizontal.data = self._horizontal_cache
        elif self.is_flow_region and self._unified_horizontal:
            # Build unified view from flow guides
            all_horizontals = []
            for coord, region in self._unified_horizontal:
                all_horizontals.append(coord)
            # Remove duplicates and sort
            self._horizontal_cache = sorted(list(set(all_horizontals)))
            self._horizontal.data = self._horizontal_cache
        return self._horizontal

    @horizontal.setter
    def horizontal(self, value: Union[List[float], "Guides", None]):
        """Set horizontal guides from a list of coordinates or another Guides object."""
        if self.is_flow_region:
            # Invalidate cache when setting new values
            self._horizontal_cache = None

        if value is None:
            self._horizontal.data = []
        elif isinstance(value, Guides):
            # Extract horizontal coordinates from another Guides object
            self._horizontal.data = sorted([float(y) for y in value.horizontal])
        elif isinstance(value, str):
            # Explicitly reject strings
            raise TypeError(
                f"horizontal cannot be a string, got '{value}'. Use a list of coordinates or Guides object."
            )
        elif hasattr(value, "__iter__"):
            # Handle list/tuple of coordinates
            try:
                self._horizontal.data = sorted([float(y) for y in value])
            except (ValueError, TypeError) as e:
                raise TypeError(f"horizontal must contain numeric values, got {value}: {e}")
        else:
            raise TypeError(f"horizontal must be a list, Guides object, or None, got {type(value)}")

    def _get_context_bounds(self) -> Tuple[float, float, float, float]:
        """Return bounding box for the current context, ensuring it exists."""
        if self.context is None:
            raise ValueError("No context available for bounds computation")
        return _require_bounds(self.context, context="guide context")

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def divide(
        cls,
        obj: Union["Page", "Region", Tuple[float, float, float, float]],
        n: Optional[int] = None,
        cols: Optional[int] = None,
        rows: Optional[int] = None,
        axis: Literal["vertical", "horizontal", "both"] = "both",
    ) -> "Guides":
        """
        Create guides by evenly dividing an object.

        Args:
            obj: Object to divide (Page, Region, or bbox tuple)
            n: Number of divisions (creates n+1 guides). Used if cols/rows not specified.
            cols: Number of columns (creates cols+1 vertical guides)
            rows: Number of rows (creates rows+1 horizontal guides)
            axis: Which axis to divide along

        Returns:
            New Guides object with evenly spaced lines

        Examples:
            # Divide into 3 columns
            guides = Guides.divide(page, cols=3)

            # Divide into 5 rows
            guides = Guides.divide(region, rows=5)

            # Divide both axes
            guides = Guides.divide(page, cols=3, rows=5)
        """
        # Extract bounds from object
        if isinstance(obj, tuple) and len(obj) == 4:
            bounds = _ensure_bounds_tuple(obj)
            context = None
        else:
            context = obj
            bounds = _require_bounds(obj, context="object to divide")

        x0, y0, x1, y1 = bounds
        verticals = []
        horizontals = []

        # Handle vertical guides
        if axis in ("vertical", "both"):
            n_vertical = cols + 1 if cols is not None else (n + 1 if n is not None else 0)
            if n_vertical > 0:
                for i in range(n_vertical):
                    x = x0 + (x1 - x0) * i / (n_vertical - 1)
                    verticals.append(float(x))

        # Handle horizontal guides
        if axis in ("horizontal", "both"):
            n_horizontal = rows + 1 if rows is not None else (n + 1 if n is not None else 0)
            if n_horizontal > 0:
                for i in range(n_horizontal):
                    y = y0 + (y1 - y0) * i / (n_horizontal - 1)
                    horizontals.append(float(y))

        return cls(verticals=verticals, horizontals=horizontals, context=context, bounds=bounds)

    @classmethod
    def from_lines(
        cls,
        obj: GuidesContext,
        axis: Literal["vertical", "horizontal", "both"] = "both",
        threshold: Union[float, str] = "auto",
        source_label: Optional[str] = None,
        max_lines_h: Optional[int] = None,
        max_lines_v: Optional[int] = None,
        outer: bool = False,
        detection_method: str = "auto",
        resolution: int = 192,
        **detect_kwargs,
    ) -> "Guides":
        """
        Create guides from detected line elements.

        Args:
            obj: Page, Region, or FlowRegion to detect lines from
            axis: Which orientations to detect
            threshold: Detection threshold ('auto' or float 0.0-1.0) - used for pixel detection
            source_label: Filter for line source (vector method) or label for detected lines (pixel method)
            max_lines_h: Maximum number of horizontal lines to keep
            max_lines_v: Maximum number of vertical lines to keep
            outer: Whether to add outer boundary guides
            detection_method: 'vector', 'pixels' (default), or 'auto' for hybrid detection.
            resolution: DPI for pixel-based detection (default: 192)
            **detect_kwargs: Additional parameters for pixel-based detection:
                - min_gap_h: Minimum gap between horizontal lines (pixels)
                - min_gap_v: Minimum gap between vertical lines (pixels)
                - binarization_method: 'adaptive' or 'otsu'
                - morph_op_h/v: Morphological operations ('open', 'close', 'none')
                - smoothing_sigma_h/v: Gaussian smoothing sigma
                - method: 'projection' (default) or 'lsd' (requires opencv)

        Returns:
            New Guides object with detected line positions
        """
        if axis == "both":
            vertical_guides = cls.from_lines(
                obj,
                axis="vertical",
                threshold=threshold,
                source_label=source_label,
                max_lines_h=max_lines_h,
                max_lines_v=max_lines_v,
                outer=outer,
                detection_method=detection_method,
                resolution=resolution,
                **detect_kwargs,
            )
            horizontal_guides = cls.from_lines(
                obj,
                axis="horizontal",
                threshold=threshold,
                source_label=source_label,
                max_lines_h=max_lines_h,
                max_lines_v=max_lines_v,
                outer=outer,
                detection_method=detection_method,
                resolution=resolution,
                **detect_kwargs,
            )
            bounds = _bounds_from_object(obj)
            return cls(
                verticals=list(vertical_guides.vertical),
                horizontals=list(horizontal_guides.horizontal),
                context=obj,
                bounds=bounds,
            )

        options = {
            "threshold": threshold,
            "source_label": source_label,
            "max_lines_h": max_lines_h if axis == "horizontal" else None,
            "max_lines_v": max_lines_v if axis == "vertical" else None,
            "outer": outer,
            "detection_method": detection_method,
            "resolution": resolution,
            **detect_kwargs,
        }

        if _is_flow_region(obj):
            guides = cls(context=obj)
            adapter = FlowGuideAdapter(guides)
            axis_values: Dict[Any, Sequence[float]] = {}
            for region in adapter.regions:
                result = run_guides_detect(
                    axis=axis,
                    method="lines",
                    context=region,
                    options=options,
                )
                axis_values[region] = [float(value) for value in result.coordinates]

            adapter.update_axis_from_regions(axis, axis_values, append=False)
            return guides

        result = run_guides_detect(
            axis=axis,
            method="lines",
            context=obj,
            options=options,
        )

        coords = [float(value) for value in result.coordinates]
        bounds = _bounds_from_object(obj)
        if axis == "vertical":
            return cls(verticals=coords, horizontals=[], context=obj, bounds=bounds)
        return cls(verticals=[], horizontals=coords, context=obj, bounds=bounds)

    @classmethod
    def from_content(
        cls,
        obj: GuidesContext,
        axis: Literal["vertical", "horizontal"] = "vertical",
        markers: Union[str, List[str], "ElementCollection", None] = None,
        align: Union[
            Literal["left", "right", "center", "between"], Literal["top", "bottom"]
        ] = "left",
        outer: OuterBoundaryMode = True,
        tolerance: float = 5,
        apply_exclusions: bool = True,
    ) -> "Guides":
        """
        Create guides based on text content positions.

        Args:
            obj: Page, Region, or FlowRegion to search for content
            axis: Whether to create vertical or horizontal guides
            markers: Content to search for. Can be:
                - str: single selector (e.g., 'text:contains("Name")') or literal text
                - List[str]: list of selectors or literal text strings
                - ElementCollection: collection of elements to extract text from
                - None: no markers
            align: Where to place guides relative to found text:
                - For vertical guides: 'left', 'right', 'center', 'between'
                - For horizontal guides: 'top', 'bottom', 'center', 'between'
            outer: Whether to add guides at the boundaries
            tolerance: Maximum distance to search for text
            apply_exclusions: Whether to apply exclusion zones when searching for text

        Returns:
            New Guides object aligned to text content
        """
        if axis == "horizontal":
            if align == "top":
                align = "left"
            elif align == "bottom":
                align = "right"

        options = {
            "markers": markers,
            "align": align,
            "outer": outer,
            "tolerance": tolerance,
            "apply_exclusions": apply_exclusions,
        }

        if _is_flow_region(obj):
            guides = cls(context=obj)
            adapter = FlowGuideAdapter(guides)
            axis_values: Dict[Any, Sequence[float]] = {}
            for region in adapter.regions:
                result = run_guides_detect(
                    axis=axis,
                    method="content",
                    context=region,
                    options=options,
                )
                axis_values[region] = [float(value) for value in result.coordinates]

            adapter.update_axis_from_regions(axis, axis_values, append=False)
            return guides

        result = run_guides_detect(
            axis=axis,
            method="content",
            context=obj,
            options=options,
        )

        coords = [float(value) for value in result.coordinates]
        bounds = _bounds_from_object(obj)
        if axis == "vertical":
            return cls(verticals=coords, context=obj, bounds=bounds)
        return cls(horizontals=coords, context=obj, bounds=bounds)

    @classmethod
    def from_whitespace(
        cls,
        obj: GuidesContext,
        axis: Literal["vertical", "horizontal", "both"] = "both",
        min_gap: float = 10,
    ) -> "Guides":
        """Create guides by detecting whitespace gaps (divide + snap placeholder)."""
        if axis == "both":
            vertical_guides = cls.from_whitespace(obj, axis="vertical", min_gap=min_gap)
            horizontal_guides = cls.from_whitespace(obj, axis="horizontal", min_gap=min_gap)
            bounds = _bounds_from_object(obj)
            return cls(
                verticals=list(vertical_guides.vertical),
                horizontals=list(horizontal_guides.horizontal),
                context=obj,
                bounds=bounds,
            )

        options = {"min_gap": min_gap}

        if _is_flow_region(obj):
            guides = cls(context=obj)
            adapter = FlowGuideAdapter(guides)
            axis_values: Dict[Any, Sequence[float]] = {}
            for region in adapter.regions:
                result = run_guides_detect(
                    axis=axis,
                    method="whitespace",
                    context=region,
                    options=options,
                )
                axis_values[region] = [float(value) for value in result.coordinates]

            adapter.update_axis_from_regions(axis, axis_values, append=False)
            return guides

        result = run_guides_detect(
            axis=axis,
            method="whitespace",
            context=obj,
            options=options,
        )

        coords = [float(value) for value in result.coordinates]
        bounds = _bounds_from_object(obj)
        if axis == "vertical":
            return cls(verticals=coords, horizontals=[], context=obj, bounds=bounds)
        return cls(verticals=[], horizontals=coords, context=obj, bounds=bounds)

    @classmethod
    def from_headers(
        cls,
        obj: GuidesContext,
        axis: Literal["vertical", "horizontal"] = "vertical",
        headers: Union["ElementCollection", Sequence[Any], None] = None,
        method: Literal["min_crossings", "seam_carving"] = "min_crossings",
        min_width: Optional[float] = None,
        max_width: Optional[float] = None,
        margin: float = 0.5,
        row_stabilization: bool = True,
        num_samples: int = 400,
    ) -> "Guides":
        """Create vertical guides by analyzing header elements."""

        if axis != "vertical":
            raise ValueError("from_headers() only works for vertical guides (columns)")

        options = {
            "headers": headers,
            "method": method,
            "min_width": min_width,
            "max_width": max_width,
            "margin": margin,
            "row_stabilization": row_stabilization,
            "num_samples": num_samples,
        }

        if _is_flow_region(obj):
            guides = cls(context=obj)
            adapter = FlowGuideAdapter(guides)
            region_values: Dict[Any, Sequence[float]] = {}

            for region in adapter.regions:
                result = run_guides_detect(
                    axis="vertical",
                    method="headers",
                    context=region,
                    options=options,
                )
                region_values[region] = [float(value) for value in result.coordinates]

            adapter.update_axis_from_regions("vertical", region_values, append=False)
            return guides

        result = run_guides_detect(
            axis="vertical",
            method="headers",
            context=obj,
            options=options,
        )

        coords = [float(value) for value in result.coordinates]
        bounds = _bounds_from_object(obj)
        return cls(verticals=coords, horizontals=[], context=obj, bounds=bounds)

    @classmethod
    def from_stripes(
        cls,
        obj: GuidesContext,
        axis: Literal["vertical", "horizontal"] = "horizontal",
        stripes: Optional[Union["ElementCollection", Sequence[Any]]] = None,
        color: Optional[str] = None,
    ) -> "Guides":
        """Create guides from zebra stripes or colored bands."""

        axis_lower = axis.lower()
        if axis_lower not in {"vertical", "horizontal"}:
            raise ValueError("axis must be 'vertical' or 'horizontal'")
        axis = cast(Literal["vertical", "horizontal"], axis_lower)

        options = {"stripes": stripes, "color": color}

        if _is_flow_region(obj):
            guides = cls(context=obj)
            adapter = FlowGuideAdapter(guides)
            region_values: Dict[Any, Sequence[float]] = {}

            for region in adapter.regions:
                result = run_guides_detect(
                    axis=axis, method="stripes", context=region, options=options
                )
                region_values[region] = [float(value) for value in result.coordinates]

            adapter.update_axis_from_regions(axis, region_values, append=True)
            return guides

        result = run_guides_detect(
            axis=axis,
            method="stripes",
            context=obj,
            options=options,
        )

        coords = [float(value) for value in result.coordinates]
        bounds = _bounds_from_object(obj)
        if axis == "vertical":
            return cls(verticals=coords, horizontals=[], context=obj, bounds=bounds)
        return cls(verticals=[], horizontals=coords, context=obj, bounds=bounds)

    @classmethod
    def new(cls, context: Optional[Union["Page", "Region"]] = None) -> "Guides":
        """
        Create a new empty Guides object, optionally with a context.

        This provides a clean way to start building guides through chaining:
        guides = Guides.new(page).add_content(axis='vertical', markers=[...])

        Args:
            context: Optional Page or Region to use as default context for operations

        Returns:
            New empty Guides object
        """
        return cls(verticals=[], horizontals=[], context=context)

    # -------------------------------------------------------------------------
    # Manipulation Methods
    # -------------------------------------------------------------------------

    def snap_to_whitespace(
        self,
        axis: str = "vertical",
        min_gap: float = 10.0,
        detection_method: str = "pixels",  # 'pixels' or 'text'
        threshold: Union[
            float, str
        ] = "auto",  # threshold for what counts as a trough (0.0-1.0) or 'auto'
        on_no_snap: str = "warn",
    ) -> "Guides":
        """
        Snap guides to nearby whitespace gaps (troughs) using optimal assignment.
        Modifies this Guides object in place.

        Args:
            axis: Direction to snap ('vertical' or 'horizontal')
            min_gap: Minimum gap size to consider as a valid trough
            detection_method: Method for detecting troughs:
                            'pixels' - use pixel-based density analysis (default)
                            'text' - use text element spacing analysis
            threshold: Threshold for what counts as a trough:
                      - float (0.0-1.0): areas with this fraction or less of max density count as troughs
                      - 'auto': automatically find threshold that creates enough troughs for guides
                      (only applies when detection_method='pixels')
            on_no_snap: Action when snapping fails ('warn', 'ignore', 'raise')

        Returns:
            Self for method chaining.
        """
        if not self.context:
            logger.warning("No context available for whitespace detection")
            return self

        detection_mode = detection_method.lower()
        if detection_mode not in {"pixels", "text"}:
            raise ValueError("detection_method must be 'pixels' or 'text'")

        text_elements = collect_text_elements(self.context)

        def _compute_gaps(axis: str, guide_positions: Sequence[float]) -> List[Tuple[float, float]]:
            if detection_mode == "pixels":
                if axis == "vertical":
                    return find_vertical_whitespace_gaps(
                        self.bounds,
                        text_elements,
                        min_gap,
                        threshold,
                        guide_positions=guide_positions,
                    )
                return find_horizontal_whitespace_gaps(
                    self.bounds,
                    text_elements,
                    min_gap,
                    threshold,
                    guide_positions=guide_positions,
                )
            if axis == "vertical":
                return find_vertical_element_gaps(self.bounds, text_elements, min_gap)
            return find_horizontal_element_gaps(self.bounds, text_elements, min_gap)

        # Handle FlowRegion case - collect all text elements across regions
        if self.is_flow_region:
            if not text_elements:
                logger.warning(
                    "No text elements found across flow regions for whitespace detection"
                )
                return self

            if axis == "vertical":
                gaps = _compute_gaps("vertical", self.vertical.data)
                all_guides = []
                guide_to_region_map = {}
                for coord, region in self._unified_vertical:
                    all_guides.append(coord)
                    guide_to_region_map.setdefault(coord, []).append(region)

                if gaps and all_guides:
                    original_guides = all_guides.copy()
                    self._snap_guides_to_gaps(all_guides, gaps, axis)

                    self._unified_vertical = []
                    for i, new_coord in enumerate(all_guides):
                        original_coord = original_guides[i]
                        regions = guide_to_region_map.get(original_coord, [])
                        for region in regions:
                            self._unified_vertical.append((new_coord, region))

                    for region in self._flow_guides:
                        region_verticals = [
                            coord for coord, r in self._unified_vertical if r == region
                        ]
                        self._flow_guides[region] = (
                            sorted(list(set(region_verticals))),
                            self._flow_guides[region][1],
                        )

                    self._vertical_cache = None

            elif axis == "horizontal":
                gaps = _compute_gaps("horizontal", self.horizontal.data)
                all_guides = []
                guide_to_region_map = {}
                for coord, region in self._unified_horizontal:
                    all_guides.append(coord)
                    guide_to_region_map.setdefault(coord, []).append(region)

                if gaps and all_guides:
                    original_guides = all_guides.copy()
                    self._snap_guides_to_gaps(all_guides, gaps, axis)

                    self._unified_horizontal = []
                    for i, new_coord in enumerate(all_guides):
                        original_coord = original_guides[i]
                        regions = guide_to_region_map.get(original_coord, [])
                        for region in regions:
                            self._unified_horizontal.append((new_coord, region))

                    for region in self._flow_guides:
                        region_horizontals = [
                            coord for coord, r in self._unified_horizontal if r == region
                        ]
                        self._flow_guides[region] = (
                            self._flow_guides[region][0],
                            sorted(list(set(region_horizontals))),
                        )

                    self._horizontal_cache = None

            else:
                raise ValueError("axis must be 'vertical' or 'horizontal'")

            return self

        if not text_elements:
            logger.warning("No text elements found for whitespace detection")
            return self

        if axis == "vertical":
            gaps = _compute_gaps("vertical", self.vertical.data)
            if gaps:
                self._snap_guides_to_gaps(self.vertical.data, gaps, axis)
        elif axis == "horizontal":
            gaps = _compute_gaps("horizontal", self.horizontal.data)
            if gaps:
                self._snap_guides_to_gaps(self.horizontal.data, gaps, axis)
        else:
            raise ValueError("axis must be 'vertical' or 'horizontal'")

        # Ensure all coordinates are Python floats (not numpy types)
        self.vertical.data[:] = [float(x) for x in self.vertical.data]
        self.horizontal.data[:] = [float(y) for y in self.horizontal.data]

        return self

    def shift(
        self, index: int, offset: float, axis: Literal["vertical", "horizontal"] = "vertical"
    ) -> "Guides":
        """
        Move a specific guide by a offset amount.

        Args:
            index: Index of the guide to move
            offset: Amount to move (positive = right/down)
            axis: Which guide list to modify

        Returns:
            Self for method chaining
        """
        if axis == "vertical":
            if 0 <= index < len(self.vertical):
                self.vertical[index] += offset
                self.vertical = sorted(self.vertical)
            else:
                logger.warning(f"Vertical guide index {index} out of range")
        else:
            if 0 <= index < len(self.horizontal):
                self.horizontal[index] += offset
                self.horizontal = sorted(self.horizontal)
            else:
                logger.warning(f"Horizontal guide index {index} out of range")

        return self

    def add_vertical(self, x: float) -> "Guides":
        """Add a vertical guide at the specified x-coordinate."""
        self.vertical.append(x)
        self.vertical = sorted(self.vertical)
        return self

    def add_horizontal(self, y: float) -> "Guides":
        """Add a horizontal guide at the specified y-coordinate."""
        self.horizontal.append(y)
        self.horizontal = sorted(self.horizontal)
        return self

    def remove_vertical(self, index: int) -> "Guides":
        """Remove a vertical guide by index."""
        if 0 <= index < len(self.vertical):
            self.vertical.pop(index)
        return self

    def remove_horizontal(self, index: int) -> "Guides":
        """Remove a horizontal guide by index."""
        if 0 <= index < len(self.horizontal):
            self.horizontal.pop(index)
        return self

    # -------------------------------------------------------------------------
    # Region extraction properties
    # -------------------------------------------------------------------------

    @property
    def columns(self):
        """Access columns by index like guides.columns[0]."""
        return _ColumnAccessor(self)

    @property
    def rows(self):
        """Access rows by index like guides.rows[0]."""
        return _RowAccessor(self)

    @property
    def cells(self):
        """Access cells by index like guides.cells[row][col] or guides.cells[row, col]."""
        return _CellAccessor(self)

    # -------------------------------------------------------------------------
    # Region extraction methods (alternative API)
    # -------------------------------------------------------------------------

    def column(self, index: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a column region from the guides.

        Args:
            index: Column index (0-based)
            obj: Page or Region to create the column on (uses self.context if None)

        Returns:
            Region representing the specified column

        Raises:
            IndexError: If column index is out of range
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        vertical_guides = list(self.vertical.data)
        if not vertical_guides or index < 0 or index >= len(vertical_guides) - 1:
            raise IndexError(
                f"Column index {index} out of range (have {len(vertical_guides)-1} columns)"
            )

        # Get bounds from context
        _, y0, _, y1 = self._get_context_bounds()

        # Get column boundaries
        x0 = vertical_guides[index]
        x1 = vertical_guides[index + 1]

        # Create region using absolute coordinates
        if hasattr(target, "create_region"):
            if isinstance(target, Region):
                return target.create_region(x0, y0, x1, y1, relative=False)
            return target.create_region(x0, y0, x1, y1)

        try:
            page = _resolve_single_page(target)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Cannot create region on {type(target)}") from exc

        return page.create_region(x0, y0, x1, y1)

    def row(self, index: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a row region from the guides.

        Args:
            index: Row index (0-based)
            obj: Page or Region to create the row on (uses self.context if None)

        Returns:
            Region representing the specified row

        Raises:
            IndexError: If row index is out of range
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        horizontal_guides = list(self.horizontal.data)
        if not horizontal_guides or index < 0 or index >= len(horizontal_guides) - 1:
            raise IndexError(
                f"Row index {index} out of range (have {len(horizontal_guides)-1} rows)"
            )

        # Get bounds from context
        x0, _, x1, _ = self._get_context_bounds()

        # Get row boundaries
        y0 = horizontal_guides[index]
        y1 = horizontal_guides[index + 1]

        # Create region using absolute coordinates
        if hasattr(target, "create_region"):
            if isinstance(target, Region):
                return target.create_region(x0, y0, x1, y1, relative=False)
            return target.create_region(x0, y0, x1, y1)

        try:
            page = _resolve_single_page(target)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Cannot create region on {type(target)}") from exc

        return page.create_region(x0, y0, x1, y1)

    def cell(self, row: int, col: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a cell region from the guides.

        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            obj: Page or Region to create the cell on (uses self.context if None)

        Returns:
            Region representing the specified cell

        Raises:
            IndexError: If row or column index is out of range
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        vertical_guides = list(self.vertical.data)
        horizontal_guides = list(self.horizontal.data)
        if not vertical_guides or col < 0 or col >= len(vertical_guides) - 1:
            raise IndexError(
                f"Column index {col} out of range (have {len(vertical_guides)-1} columns)"
            )
        if not horizontal_guides or row < 0 or row >= len(horizontal_guides) - 1:
            raise IndexError(f"Row index {row} out of range (have {len(horizontal_guides)-1} rows)")

        # Get cell boundaries
        x0 = vertical_guides[col]
        x1 = vertical_guides[col + 1]
        y0 = horizontal_guides[row]
        y1 = horizontal_guides[row + 1]

        # Create region using absolute coordinates
        if hasattr(target, "create_region"):
            if isinstance(target, Region):
                return target.create_region(x0, y0, x1, y1, relative=False)
            return target.create_region(x0, y0, x1, y1)

        try:
            page = _resolve_single_page(target)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Cannot create region on {type(target)}") from exc

        return page.create_region(x0, y0, x1, y1)

    def left_of(self, guide_index: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a region to the left of a vertical guide.

        Args:
            guide_index: Vertical guide index
            obj: Page or Region to create the region on (uses self.context if None)

        Returns:
            Region to the left of the specified guide
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        if not self.vertical or guide_index < 0 or guide_index >= len(self.vertical):
            raise IndexError(f"Guide index {guide_index} out of range")

        # Get bounds from context
        bounds = self._get_context_bounds()
        if not bounds:
            raise ValueError("Could not determine bounds")
        x0, y0, _, y1 = bounds

        # Create region from left edge to guide
        x1 = self.vertical[guide_index]

        if hasattr(target, "region"):
            return target.region(x0, y0, x1, y1)
        else:
            raise TypeError(f"Cannot create region on {type(target)}")

    def right_of(self, guide_index: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a region to the right of a vertical guide.

        Args:
            guide_index: Vertical guide index
            obj: Page or Region to create the region on (uses self.context if None)

        Returns:
            Region to the right of the specified guide
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        if not self.vertical or guide_index < 0 or guide_index >= len(self.vertical):
            raise IndexError(f"Guide index {guide_index} out of range")

        # Get bounds from context
        bounds = self._get_context_bounds()
        if not bounds:
            raise ValueError("Could not determine bounds")
        _, y0, x1, y1 = bounds

        # Create region from guide to right edge
        x0 = self.vertical[guide_index]

        if hasattr(target, "region"):
            return target.region(x0, y0, x1, y1)
        else:
            raise TypeError(f"Cannot create region on {type(target)}")

    def above(self, guide_index: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a region above a horizontal guide.

        Args:
            guide_index: Horizontal guide index
            obj: Page or Region to create the region on (uses self.context if None)

        Returns:
            Region above the specified guide
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        if not self.horizontal or guide_index < 0 or guide_index >= len(self.horizontal):
            raise IndexError(f"Guide index {guide_index} out of range")

        # Get bounds from context
        bounds = self._get_context_bounds()
        if not bounds:
            raise ValueError("Could not determine bounds")
        x0, y0, x1, _ = bounds

        # Create region from top edge to guide
        y1 = self.horizontal[guide_index]

        if hasattr(target, "region"):
            return target.region(x0, y0, x1, y1)
        else:
            raise TypeError(f"Cannot create region on {type(target)}")

    def below(self, guide_index: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a region below a horizontal guide.

        Args:
            guide_index: Horizontal guide index
            obj: Page or Region to create the region on (uses self.context if None)

        Returns:
            Region below the specified guide
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        if not self.horizontal or guide_index < 0 or guide_index >= len(self.horizontal):
            raise IndexError(f"Guide index {guide_index} out of range")

        # Get bounds from context
        bounds = self._get_context_bounds()
        if not bounds:
            raise ValueError("Could not determine bounds")
        x0, _, x1, y1 = bounds

        # Create region from guide to bottom edge
        y0 = self.horizontal[guide_index]

        if hasattr(target, "region"):
            return target.region(x0, y0, x1, y1)
        else:
            raise TypeError(f"Cannot create region on {type(target)}")

    def between_vertical(
        self, start_index: int, end_index: int, obj: Optional[Union["Page", "Region"]] = None
    ) -> "Region":
        """
        Get a region between two vertical guides.

        Args:
            start_index: Starting vertical guide index
            end_index: Ending vertical guide index
            obj: Page or Region to create the region on (uses self.context if None)

        Returns:
            Region between the specified guides
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        if not self.vertical:
            raise ValueError("No vertical guides available")
        if start_index < 0 or start_index >= len(self.vertical):
            raise IndexError(f"Start index {start_index} out of range")
        if end_index < 0 or end_index >= len(self.vertical):
            raise IndexError(f"End index {end_index} out of range")
        if start_index >= end_index:
            raise ValueError("Start index must be less than end index")

        # Get bounds from context
        bounds = self._get_context_bounds()
        if not bounds:
            raise ValueError("Could not determine bounds")
        _, y0, _, y1 = bounds

        # Get horizontal boundaries
        x0 = self.vertical[start_index]
        x1 = self.vertical[end_index]

        if hasattr(target, "region"):
            return target.region(x0, y0, x1, y1)
        else:
            raise TypeError(f"Cannot create region on {type(target)}")

    def between_horizontal(
        self, start_index: int, end_index: int, obj: Optional[Union["Page", "Region"]] = None
    ) -> "Region":
        """
        Get a region between two horizontal guides.

        Args:
            start_index: Starting horizontal guide index
            end_index: Ending horizontal guide index
            obj: Page or Region to create the region on (uses self.context if None)

        Returns:
            Region between the specified guides
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        if not self.horizontal:
            raise ValueError("No horizontal guides available")
        if start_index < 0 or start_index >= len(self.horizontal):
            raise IndexError(f"Start index {start_index} out of range")
        if end_index < 0 or end_index >= len(self.horizontal):
            raise IndexError(f"End index {end_index} out of range")
        if start_index >= end_index:
            raise ValueError("Start index must be less than end index")

        # Get bounds from context
        bounds = self._get_context_bounds()
        if not bounds:
            raise ValueError("Could not determine bounds")
        x0, _, x1, _ = bounds

        # Get vertical boundaries
        y0 = self.horizontal[start_index]
        y1 = self.horizontal[end_index]

        if hasattr(target, "region"):
            return target.region(x0, y0, x1, y1)
        else:
            raise TypeError(f"Cannot create region on {type(target)}")

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def __add__(self, other: "Guides") -> "Guides":
        """
        Combine two guide sets.

        Returns:
            New Guides object with combined coordinates
        """
        # Combine and deduplicate coordinates, ensuring Python floats
        combined_verticals = sorted([float(x) for x in set(self.vertical + other.vertical)])
        combined_horizontals = sorted([float(y) for y in set(self.horizontal + other.horizontal)])

        # Handle FlowRegion context merging
        new_context = self.context or other.context

        # If both are flow regions, we might need a more complex merge,
        # but for now, just picking one context is sufficient.

        # Create the new Guides object
        new_guides = Guides(
            verticals=combined_verticals,
            horizontals=combined_horizontals,
            context=new_context,
            bounds=self.bounds or other.bounds,
        )

        # If the new context is a FlowRegion, we need to rebuild the flow-related state
        if new_guides.is_flow_region:
            # Re-initialize flow guides from both sources
            # This is a simplification; a true merge would be more complex.
            # For now, we combine the flow_guides dictionaries.
            if hasattr(self, "_flow_guides"):
                new_guides._flow_guides.update(self._flow_guides)
            if hasattr(other, "_flow_guides"):
                new_guides._flow_guides.update(other._flow_guides)

            # Re-initialize unified views
            if hasattr(self, "_unified_vertical"):
                new_guides._unified_vertical.extend(self._unified_vertical)
            if hasattr(other, "_unified_vertical"):
                new_guides._unified_vertical.extend(other._unified_vertical)

            if hasattr(self, "_unified_horizontal"):
                new_guides._unified_horizontal.extend(self._unified_horizontal)
            if hasattr(other, "_unified_horizontal"):
                new_guides._unified_horizontal.extend(other._unified_horizontal)

            # Invalidate caches to force rebuild
            new_guides._vertical_cache = None
            new_guides._horizontal_cache = None

        return new_guides

    def show(self, on=None, **kwargs):
        """
        Display the guides overlaid on a page or region.

        Args:
            on: Page, Region, PIL Image, or string to display guides on.
                If None, uses self.context (the object guides were created from).
                If string 'page', uses the page from self.context.
            **kwargs: Additional arguments passed to to_image() if applicable.

        Returns:
            PIL Image with guides drawn on it.
        """
        # Handle FlowRegion case
        if self.is_flow_region and (on is None or on == self.context):
            if not self._flow_guides:
                raise ValueError("No guides to show for FlowRegion")

            # Get stacking parameters from kwargs or use defaults
            stack_direction = kwargs.get("stack_direction", "vertical")
            stack_gap = kwargs.get("stack_gap", 5)
            stack_background_color = kwargs.get("stack_background_color", (255, 255, 255))

            # First, render all constituent regions without guides to get base images
            base_images = []
            region_infos = []  # Store region info for guide coordinate mapping

            for region in list(self._flow_constituent_regions()):
                render_fn = getattr(region, "render", None)
                if render_fn is None:
                    raise AttributeError(f"Region {region} does not support rendering")
                img = render_fn(
                    resolution=kwargs.get("resolution", 150),
                    width=kwargs.get("width", None),
                    crop=True,
                )
                if img:
                    base_images.append(img)

                    scale_x = img.width / region.width
                    scale_y = img.height / region.height

                    region_infos.append(
                        {
                            "region": region,
                            "img_width": img.width,
                            "img_height": img.height,
                            "scale_x": scale_x,
                            "scale_y": scale_y,
                            "pdf_x0": region.x0,
                            "pdf_top": region.top,
                            "pdf_x1": region.x1,
                            "pdf_bottom": region.bottom,
                        }
                    )

            if not base_images:
                raise ValueError("Failed to render any images for FlowRegion")

            # Calculate final canvas size based on stacking direction
            if stack_direction == "vertical":
                final_width = max(img.width for img in base_images)
                final_height = (
                    sum(img.height for img in base_images) + (len(base_images) - 1) * stack_gap
                )
            else:  # horizontal
                final_width = (
                    sum(img.width for img in base_images) + (len(base_images) - 1) * stack_gap
                )
                final_height = max(img.height for img in base_images)

            # Create unified canvas
            canvas = Image.new("RGB", (final_width, final_height), stack_background_color)
            draw = ImageDraw.Draw(canvas)

            # Paste base images and track positions
            region_positions = []  # (region_info, paste_x, paste_y)

            if stack_direction == "vertical":
                current_y = 0
                for i, (img, info) in enumerate(zip(base_images, region_infos)):
                    paste_x = (final_width - img.width) // 2  # Center horizontally
                    canvas.paste(img, (paste_x, current_y))
                    region_positions.append((info, paste_x, current_y))
                    current_y += img.height + stack_gap
            else:  # horizontal
                current_x = 0
                for i, (img, info) in enumerate(zip(base_images, region_infos)):
                    paste_y = (final_height - img.height) // 2  # Center vertically
                    canvas.paste(img, (current_x, paste_y))
                    region_positions.append((info, current_x, paste_y))
                    current_x += img.width + stack_gap

            # Now draw guides on the unified canvas
            # Draw vertical guides (blue) - these extend through the full canvas height
            for v_coord in self.vertical:
                # Find which region(s) this guide intersects
                for info, paste_x, paste_y in region_positions:
                    if info["pdf_x0"] <= v_coord <= info["pdf_x1"]:
                        # This guide is within this region's x-bounds
                        # Convert PDF coordinate to pixel coordinate relative to the region
                        adjusted_x = v_coord - info["pdf_x0"]
                        pixel_x = adjusted_x * info["scale_x"] + paste_x

                        # Draw full-height line on canvas (not clipped to region)
                        if 0 <= pixel_x <= final_width:
                            x_pixel = int(pixel_x)
                            draw.line(
                                [(x_pixel, 0), (x_pixel, final_height - 1)],
                                fill=(0, 0, 255, 200),
                                width=2,
                            )
                        break  # Only draw once per guide

            # Draw horizontal guides (red) - these extend through the full canvas width
            for h_coord in self.horizontal:
                # Find which region(s) this guide intersects
                for info, paste_x, paste_y in region_positions:
                    if info["pdf_top"] <= h_coord <= info["pdf_bottom"]:
                        # This guide is within this region's y-bounds
                        # Convert PDF coordinate to pixel coordinate relative to the region
                        adjusted_y = h_coord - info["pdf_top"]
                        pixel_y = adjusted_y * info["scale_y"] + paste_y

                        # Draw full-width line on canvas (not clipped to region)
                        if 0 <= pixel_y <= final_height:
                            y_pixel = int(pixel_y)
                            draw.line(
                                [(0, y_pixel), (final_width - 1, y_pixel)],
                                fill=(255, 0, 0, 200),
                                width=2,
                            )
                        break  # Only draw once per guide

            return canvas

        # Original single-region logic follows...
        # Determine what to display guides on
        target = on if on is not None else self.context

        # Handle string shortcuts
        if isinstance(target, str):
            if target == "page":
                if self.context is None:
                    raise ValueError("Cannot resolve 'page' without a guides context")
                try:
                    target = _resolve_single_page(self.context)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Cannot resolve 'page' from the current guides context"
                    ) from exc
            else:
                raise ValueError(f"Unknown string target: {target}. Only 'page' is supported.")

        if target is None:
            raise ValueError("No target specified and no context available for guides display")

        # Prepare kwargs for image generation
        image_kwargs = {}

        # Extract only the parameters that the new render() method accepts
        if "resolution" in kwargs:
            image_kwargs["resolution"] = kwargs["resolution"]
        if "width" in kwargs:
            image_kwargs["width"] = kwargs["width"]
        if "crop" in kwargs:
            image_kwargs["crop"] = kwargs["crop"]

        # If target is a region-like object, crop to just that region
        try:
            _resolve_single_page(target)
            target_is_single_page = True
        except (TypeError, ValueError):
            target_is_single_page = False

        if hasattr(target, "bbox") and target_is_single_page:
            image_kwargs["crop"] = True

        # Get base image
        if hasattr(target, "render"):
            rendered = cast(Any, target).render(**image_kwargs)
            if rendered is None:
                raise ValueError("Failed to generate base image")
            img = cast(Image.Image, rendered)
        elif hasattr(target, "mode") and hasattr(target, "size"):
            # It's already a PIL Image
            img = cast(Image.Image, target)
        else:
            raise ValueError(f"Object {target} does not support render() and is not a PIL Image")

        # Create a copy to draw on
        img = cast(Image.Image, img.copy())
        draw = ImageDraw.Draw(img)

        # Determine scale factor for coordinate conversion
        if _has_size(target) and not (hasattr(target, "mode") and hasattr(target, "size")):
            # target is a PDF object (Page/Region) with PDF coordinates
            size_target = cast(_SupportsSize, target)
            scale_x = img.width / size_target.width
            scale_y = img.height / size_target.height

            # If we're showing guides on a region, we need to adjust coordinates
            # to be relative to the region's origin
            if hasattr(target, "bbox") and target_is_single_page:
                # This is a Region - adjust guide coordinates to be relative to region
                region_like = cast(Any, target)
                region_x0 = float(getattr(region_like, "x0", 0.0))
                region_top = float(getattr(region_like, "top", 0.0))
            else:
                # This is a Page - no adjustment needed
                region_x0, region_top = 0, 0
        else:
            # target is already an image, no scaling needed
            scale_x = 1.0
            scale_y = 1.0
            region_x0, region_top = 0, 0

        # Draw vertical guides (blue)
        for x_coord in self.vertical:
            # Adjust coordinate if we're showing on a region
            adjusted_x = x_coord - region_x0
            pixel_x = adjusted_x * scale_x
            # Ensure guides at the edge are still visible by clamping to valid range
            if 0 <= pixel_x <= img.width - 1:
                x_pixel = int(min(pixel_x, img.width - 1))
                draw.line([(x_pixel, 0), (x_pixel, img.height - 1)], fill=(0, 0, 255, 200), width=2)

        # Draw horizontal guides (red)
        for y_coord in self.horizontal:
            # Adjust coordinate if we're showing on a region
            adjusted_y = y_coord - region_top
            pixel_y = adjusted_y * scale_y
            # Ensure guides at the edge are still visible by clamping to valid range
            if 0 <= pixel_y <= img.height - 1:
                y_pixel = int(min(pixel_y, img.height - 1))
                draw.line([(0, y_pixel), (img.width - 1, y_pixel)], fill=(255, 0, 0, 200), width=2)

        return img

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_cells(self) -> List[Tuple[float, float, float, float]]:
        """
        Get all cell bounding boxes from guide intersections.

        Returns:
            List of (x0, y0, x1, y1) tuples for each cell
        """
        cells = []

        # Create cells from guide intersections
        for i in range(len(self.vertical) - 1):
            for j in range(len(self.horizontal) - 1):
                x0 = self.vertical[i]
                x1 = self.vertical[i + 1]
                y0 = self.horizontal[j]
                y1 = self.horizontal[j + 1]
                cells.append((x0, y0, x1, y1))

        return cells

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format suitable for pdfplumber table_settings.

        Returns:
            Dictionary with explicit_vertical_lines and explicit_horizontal_lines
        """
        return {
            "explicit_vertical_lines": self.vertical,
            "explicit_horizontal_lines": self.horizontal,
        }

    def to_relative(self) -> "Guides":
        """
        Convert absolute coordinates to relative (0-1) coordinates.

        Returns:
            New Guides object with relative coordinates
        """
        if self.relative:
            return self  # Already relative

        if not self.bounds:
            raise ValueError("Cannot convert to relative without bounds")

        x0, y0, x1, y1 = self.bounds
        width = x1 - x0
        height = y1 - y0

        rel_verticals = [(x - x0) / width for x in self.vertical]
        rel_horizontals = [(y - y0) / height for y in self.horizontal]

        return Guides(
            verticals=rel_verticals,
            horizontals=rel_horizontals,
            context=self.context,
            bounds=(0, 0, 1, 1),
            relative=True,
        )

    def to_absolute(self, bounds: Tuple[float, float, float, float]) -> "Guides":
        """
        Convert relative coordinates to absolute coordinates.

        Args:
            bounds: Target bounding box (x0, y0, x1, y1)

        Returns:
            New Guides object with absolute coordinates
        """
        if not self.relative:
            return self  # Already absolute

        x0, y0, x1, y1 = bounds
        width = x1 - x0
        height = y1 - y0

        abs_verticals = [x0 + x * width for x in self.vertical]
        abs_horizontals = [y0 + y * height for y in self.horizontal]

        return Guides(
            verticals=abs_verticals,
            horizontals=abs_horizontals,
            context=self.context,
            bounds=bounds,
            relative=False,
        )

    @property
    def n_rows(self) -> int:
        """Number of rows defined by horizontal guides."""
        return max(0, len(self.horizontal) - 1)

    @property
    def n_cols(self) -> int:
        """Number of columns defined by vertical guides."""
        return max(0, len(self.vertical) - 1)

    def _handle_snap_failure(self, message: str):
        """Handle cases where snapping cannot be performed."""
        behavior = getattr(self, "snap_behavior", getattr(self, "on_no_snap", "warn"))
        if behavior == "warn":
            logger.warning(message)
        elif behavior == "raise":
            raise ValueError(message)
        # 'ignore' case: do nothing

    def _optimal_guide_assignment(
        self, guides: List[float], trough_ranges: List[Tuple[float, float]]
    ) -> Dict[int, int]:
        """
        Assign guides to trough ranges using the user's desired logic:
        - Guides already in a trough stay put
        - Only guides NOT in any trough get moved to available troughs
        - Prefer closest assignment for guides that need to move
        """
        if not guides or not trough_ranges:
            return {}

        assignments = {}

        # Step 1: Identify which guides are already in troughs
        guides_in_troughs = set()
        for i, guide_pos in enumerate(guides):
            for trough_start, trough_end in trough_ranges:
                if trough_start <= guide_pos <= trough_end:
                    guides_in_troughs.add(i)
                    logger.debug(
                        f"Guide {i} (pos {guide_pos:.1f}) is already in trough ({trough_start:.1f}-{trough_end:.1f}), keeping in place"
                    )
                    break

        # Step 2: Identify which troughs are already occupied
        occupied_troughs = set()
        for i in guides_in_troughs:
            guide_pos = guides[i]
            for j, (trough_start, trough_end) in enumerate(trough_ranges):
                if trough_start <= guide_pos <= trough_end:
                    occupied_troughs.add(j)
                    break

        # Step 3: Find guides that need reassignment (not in any trough)
        guides_to_move = []
        for i, guide_pos in enumerate(guides):
            if i not in guides_in_troughs:
                guides_to_move.append(i)
                logger.debug(
                    f"Guide {i} (pos {guide_pos:.1f}) is NOT in any trough, needs reassignment"
                )

        # Step 4: Find available troughs (not occupied by existing guides)
        available_troughs = []
        for j, (trough_start, trough_end) in enumerate(trough_ranges):
            if j not in occupied_troughs:
                available_troughs.append(j)
                logger.debug(f"Trough {j} ({trough_start:.1f}-{trough_end:.1f}) is available")

        # Step 5: Assign guides to move to closest available troughs
        if guides_to_move and available_troughs:
            # Calculate distances for all combinations
            distances = []
            for guide_idx in guides_to_move:
                guide_pos = guides[guide_idx]
                for trough_idx in available_troughs:
                    trough_start, trough_end = trough_ranges[trough_idx]
                    trough_center = (trough_start + trough_end) / 2
                    distance = abs(guide_pos - trough_center)
                    distances.append((distance, guide_idx, trough_idx))

            # Sort by distance and assign greedily
            distances.sort()
            used_troughs = set()

            for distance, guide_idx, trough_idx in distances:
                if guide_idx not in assignments and trough_idx not in used_troughs:
                    assignments[guide_idx] = trough_idx
                    used_troughs.add(trough_idx)
                    logger.debug(
                        f"Assigned guide {guide_idx} (pos {guides[guide_idx]:.1f}) to trough {trough_idx} (distance: {distance:.1f})"
                    )

        logger.debug(f"Final assignments: {assignments}")
        return assignments

    def _snap_guides_to_gaps(self, guides: List[float], gaps: List[Tuple[float, float]], axis: str):
        """
        Snap guides to nearby gaps using optimal assignment.
        Only moves guides that are NOT already in a trough.
        """
        if not guides or not gaps:
            return

        logger.debug(f"Snapping {len(guides)} {axis} guides to {len(gaps)} trough ranges")
        for i, (start, end) in enumerate(gaps):
            center = (start + end) / 2
            logger.debug(f"  Trough {i}: {start:.1f} to {end:.1f} (center: {center:.1f})")

        # Get optimal assignments
        assignments = self._optimal_guide_assignment(guides, gaps)

        # Apply assignments (modify guides list in-place)
        for guide_idx, trough_idx in assignments.items():
            trough_start, trough_end = gaps[trough_idx]
            new_pos = (trough_start + trough_end) / 2  # Move to trough center
            old_pos = guides[guide_idx]
            guides[guide_idx] = new_pos
            logger.info(f"Snapped {axis} guide from {old_pos:.1f} to {new_pos:.1f}")

    def build_grid(
        self,
        target: Optional[GuidesContext] = None,
        source: str = "guides",
        cell_padding: float = 0.5,
        include_outer_boundaries: bool = False,
        *,
        multi_page: Literal["auto", True, False] = "auto",
    ) -> Dict[str, Any]:
        """
        Create table structure (table, rows, columns, cells) from guide coordinates.

        Args:
            target: Page or Region to create regions on (uses self.context if None)
            source: Source label for created regions (for identification)
            cell_padding: Internal padding for cell regions in points
            include_outer_boundaries: Whether to add boundaries at edges if missing
            multi_page: Controls multi-region table creation for FlowRegions.
                - "auto": (default) Creates a unified grid if there are multiple regions or guides span pages.
                - True: Forces creation of a unified multi-region grid.
                - False: Creates separate grids for each region.

        Returns:
            Dictionary with 'counts' and 'regions' created.
        """
        # Dispatch to appropriate implementation based on context and flags
        if self.is_flow_region:
            # Check if we should create a unified multi-region grid
            has_multiple_regions = len(self._flow_constituent_regions()) > 1
            spans_pages = self._spans_pages()

            # Create unified grid if:
            # - multi_page is explicitly True, OR
            # - multi_page is "auto" AND (spans pages OR has multiple regions)
            if multi_page is True or (
                multi_page == "auto" and (spans_pages or has_multiple_regions)
            ):
                return self._build_grid_multi_page(
                    source=source,
                    cell_padding=cell_padding,
                    include_outer_boundaries=include_outer_boundaries,
                )
            else:
                # Single region FlowRegion or multi_page=False: create separate tables per region
                total_counts = {"table": 0, "rows": 0, "columns": 0, "cells": 0}
                all_regions = {"table": [], "rows": [], "columns": [], "cells": []}

                for region in self._flow_constituent_regions():
                    if region in self._flow_guides:
                        verticals, horizontals = self._flow_guides[region]

                        region_guides = Guides(
                            verticals=verticals, horizontals=horizontals, context=region
                        )

                        result = region_guides._build_grid_single_page(
                            target=region,
                            source=source,
                            cell_padding=cell_padding,
                            include_outer_boundaries=include_outer_boundaries,
                        )

                        for key in total_counts:
                            total_counts[key] += result["counts"][key]

                        if result["regions"]["table"]:
                            all_regions["table"].append(result["regions"]["table"])
                        all_regions["rows"].extend(result["regions"]["rows"])
                        all_regions["columns"].extend(result["regions"]["columns"])
                        all_regions["cells"].extend(result["regions"]["cells"])

                logger.info(
                    f"Created {total_counts['table']} tables, {total_counts['rows']} rows, "
                    f"{total_counts['columns']} columns, and {total_counts['cells']} cells "
                    f"from guides across {len(self._flow_guides)} regions"
                )

                return {"counts": total_counts, "regions": all_regions}

        # Fallback for single page/region
        return self._build_grid_single_page(
            target=target,
            source=source,
            cell_padding=cell_padding,
            include_outer_boundaries=include_outer_boundaries,
        )

    def _build_grid_multi_page(
        self,
        source: str,
        cell_padding: float,
        include_outer_boundaries: bool,
    ) -> Dict[str, Any]:
        """
        Builds a single, coherent grid across multiple regions of a FlowRegion.

        Creates physical Region objects for each constituent region with _fragment
        region types (e.g., table_column_fragment), then stitches them into logical
        FlowRegion objects. Both are registered with pages, but the fragment types
        allow easy differentiation:
        - find_all('table_column') returns only logical columns
        - find_all('table_column_fragment') returns only physical fragments
        """
        from natural_pdf.flows.region import FlowRegion

        if not self.is_flow_region:
            raise ValueError("Multi-page grid building requires a FlowRegion with a valid Flow.")
        flow_context = self._flow_context()
        if not getattr(flow_context, "flow", None):
            raise ValueError("Multi-page grid building requires a FlowRegion with a valid Flow.")

        orientation = self._get_flow_orientation()
        adapter = FlowGuideAdapter(self)
        region_grids = adapter.build_region_grids(source=source, cell_padding=cell_padding)

        if not region_grids:
            return {
                "counts": {"table": 0, "rows": 0, "columns": 0, "cells": 0},
                "regions": {"table": None, "rows": [], "columns": [], "cells": []},
            }

        flow_region = self._flow_context()
        flow = flow_region.flow

        physical_tables: List[Any] = [grid.table for grid in region_grids if grid.table is not None]
        flattened_tables = adapter._flatten_region_likes(physical_tables)
        multi_page_table = FlowRegion(
            flow=flow, constituent_regions=flattened_tables, source_flow_element=None
        )
        multi_page_table.source = source
        multi_page_table.region_type = "table"
        multi_page_table.metadata.update(
            {"is_multi_page": True, "num_rows": self.n_rows, "num_cols": self.n_cols}
        )

        final_rows, final_cols, final_cells = adapter.stitch_region_results(
            region_grids, orientation, source
        )

        constituent_pages = collect_constituent_pages(self._flow_constituent_regions())
        register_regions_with_pages(
            constituent_pages,
            multi_page_table,
            final_rows,
            final_cols,
            final_cells,
            log=logger,
        )

        final_counts = {
            "table": 1,
            "rows": len(final_rows),
            "columns": len(final_cols),
            "cells": len(final_cells),
        }
        final_regions = {
            "table": multi_page_table,
            "rows": final_rows,
            "columns": final_cols,
            "cells": final_cells,
        }

        logger.info(
            f"Created 1 multi-page table, {final_counts['rows']} logical rows, "
            f"{final_counts['columns']} logical columns from guides and registered with all constituent pages"
        )

        return {"counts": final_counts, "regions": final_regions}

    def _build_grid_single_page(
        self,
        target: Optional[GuidesContext] = None,
        source: str = "guides",
        cell_padding: float = 0.5,
        include_outer_boundaries: bool = False,
    ) -> Dict[str, Any]:
        """
        Private method to create table structure on a single page or region.
        (Refactored from the original public build_grid method).
        """
        # This method now only handles a single page/region context.
        # Looping for FlowRegions is handled by the public `build_grid` method.

        # Original single-region logic follows...
        target_obj = target or self.context
        if not target_obj:
            raise ValueError("No target object available. Provide target parameter or context.")
        if _is_flow_region(target_obj):
            raise ValueError(
                "FlowRegion targets require multi-page handling – call build_grid with multi_page=True."
            )

        bounds = _require_bounds(target_obj, context="grid target")
        origin_x, origin_y, max_x, max_y = bounds
        context_width = max_x - origin_x
        context_height = max_y - origin_y

        from natural_pdf.core.page import Page

        page: Page
        if isinstance(target_obj, Page):
            page = target_obj
        elif isinstance(target_obj, Region):
            if target_obj.page is None:
                raise ValueError("Region is not associated with a page")
            page = target_obj.page
        else:
            try:
                page = _resolve_single_page(target_obj)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Target object {target_obj} does not expose a single page reference"
                ) from exc
            if not isinstance(page, Page):
                raise ValueError("Resolved page does not implement the Page interface")

        if not hasattr(page, "add_element") or not hasattr(page, "remove_element"):
            raise ValueError("Target page does not expose element registration helpers")

        # Setup boundaries
        row_boundaries = [float(v) for v in self.horizontal]
        col_boundaries = [float(v) for v in self.vertical]

        # Add outer boundaries if requested and missing
        if include_outer_boundaries:
            if not row_boundaries or row_boundaries[0] > origin_y:
                row_boundaries.insert(0, origin_y)
            if not row_boundaries or row_boundaries[-1] < origin_y + context_height:
                row_boundaries.append(origin_y + context_height)

            if not col_boundaries or col_boundaries[0] > origin_x:
                col_boundaries.insert(0, origin_x)
            if not col_boundaries or col_boundaries[-1] < origin_x + context_width:
                col_boundaries.append(origin_x + context_width)

        # Remove duplicates and sort
        row_boundaries = sorted(list(set(row_boundaries)))
        col_boundaries = sorted(list(set(col_boundaries)))

        def _collect_regions_from_page_obj(page_obj: Any) -> List[Any]:
            iterator = getattr(page_obj, "iter_regions", None)
            regions: Optional[Iterable[Any]]
            if callable(iterator):
                regions = cast(Optional[Iterable[Any]], iterator())
            else:
                regions = cast(Optional[Iterable[Any]], iterator)

            if regions is None:
                return []
            if not isinstance(regions, IterableABC):
                return []
            return list(regions)

        # ------------------------------------------------------------------
        # Clean-up: remove any previously created grid regions (table, rows,
        # columns, cells) that were generated by the same `source` label and
        # overlap the area we are about to populate.  This prevents the page's
        # `ElementManager` from accumulating stale/duplicate regions when the
        # user rebuilds the grid multiple times.
        # ------------------------------------------------------------------
        if row_boundaries and col_boundaries:
            grid_bbox = (
                col_boundaries[0],  # x0
                row_boundaries[0],  # top
                col_boundaries[-1],  # x1
                row_boundaries[-1],  # bottom
            )

            def _bbox_overlap(b1, b2):
                """Return True if two (x0, top, x1, bottom) bboxes overlap."""
                return not (
                    b1[2] <= b2[0]  # b1 right ≤ b2 left
                    or b1[0] >= b2[2]  # b1 left ≥ b2 right
                    or b1[3] <= b2[1]  # b1 bottom ≤ b2 top
                    or b1[1] >= b2[3]  # b1 top ≥ b2 bottom
                )

            existing_regions = _collect_regions_from_page_obj(page)
            regions_to_remove = []
            for r in existing_regions:
                if getattr(r, "source", None) != source:
                    continue
                if getattr(r, "region_type", None) not in {
                    "table",
                    "table_row",
                    "table_column",
                    "table_cell",
                }:
                    continue
                if not hasattr(r, "bbox"):
                    continue
                if _bbox_overlap(r.bbox, grid_bbox):
                    regions_to_remove.append(r)

            for r in regions_to_remove:
                page.remove_element(r, element_type="regions")

            if regions_to_remove:
                logger.debug(
                    f"Removed {len(regions_to_remove)} existing grid region(s) prior to rebuild"
                )

        logger.debug(
            f"Building grid with {len(row_boundaries)} row and {len(col_boundaries)} col boundaries"
        )

        # Track creation counts and regions
        counts = {"table": 0, "rows": 0, "columns": 0, "cells": 0}
        created_regions = {"table": None, "rows": [], "columns": [], "cells": []}

        # Create overall table region
        if len(row_boundaries) >= 2 and len(col_boundaries) >= 2:
            table_region = page.create_region(
                col_boundaries[0], row_boundaries[0], col_boundaries[-1], row_boundaries[-1]
            )
            table_region.source = source
            table_region.region_type = "table"
            table_region.normalized_type = "table"
            table_region.metadata.update(
                {
                    "source_guides": True,
                    "num_rows": len(row_boundaries) - 1,
                    "num_cols": len(col_boundaries) - 1,
                    "boundaries": {"rows": row_boundaries, "cols": col_boundaries},
                }
            )
            page.add_region(table_region, source=source)
            counts["table"] = 1
            created_regions["table"] = table_region

        # Create row regions
        if len(row_boundaries) >= 2 and len(col_boundaries) >= 2:
            for i in range(len(row_boundaries) - 1):
                row_region = page.create_region(
                    col_boundaries[0], row_boundaries[i], col_boundaries[-1], row_boundaries[i + 1]
                )
                row_region.source = source
                row_region.region_type = "table_row"
                row_region.normalized_type = "table_row"
                row_region.metadata.update({"row_index": i, "source_guides": True})
                page.add_region(row_region, source=source)
                counts["rows"] += 1
                created_regions["rows"].append(row_region)

        # Create column regions
        if len(col_boundaries) >= 2 and len(row_boundaries) >= 2:
            for j in range(len(col_boundaries) - 1):
                col_region = page.create_region(
                    col_boundaries[j], row_boundaries[0], col_boundaries[j + 1], row_boundaries[-1]
                )
                col_region.source = source
                col_region.region_type = "table_column"
                col_region.normalized_type = "table_column"
                col_region.metadata.update({"col_index": j, "source_guides": True})
                page.add_region(col_region, source=source)
                counts["columns"] += 1
                created_regions["columns"].append(col_region)

        # Create cell regions
        if len(row_boundaries) >= 2 and len(col_boundaries) >= 2:
            for i in range(len(row_boundaries) - 1):
                for j in range(len(col_boundaries) - 1):
                    # Apply padding
                    cell_x0 = col_boundaries[j] + cell_padding
                    cell_top = row_boundaries[i] + cell_padding
                    cell_x1 = col_boundaries[j + 1] - cell_padding
                    cell_bottom = row_boundaries[i + 1] - cell_padding

                    # Skip invalid cells
                    if cell_x1 <= cell_x0 or cell_bottom <= cell_top:
                        continue

                    cell_region = page.create_region(cell_x0, cell_top, cell_x1, cell_bottom)
                    cell_region.source = source
                    cell_region.region_type = "table_cell"
                    cell_region.normalized_type = "table_cell"
                    cell_region.metadata.update(
                        {
                            "row_index": i,
                            "col_index": j,
                            "source_guides": True,
                            "original_boundaries": {
                                "left": col_boundaries[j],
                                "top": row_boundaries[i],
                                "right": col_boundaries[j + 1],
                                "bottom": row_boundaries[i + 1],
                            },
                        }
                    )
                    page.add_region(cell_region, source=source)
                    counts["cells"] += 1
                    created_regions["cells"].append(cell_region)

        logger.info(
            f"Created {counts['table']} table, {counts['rows']} rows, "
            f"{counts['columns']} columns, and {counts['cells']} cells from guides"
        )

        return {"counts": counts, "regions": created_regions}

    def __repr__(self) -> str:
        """String representation of the guides."""
        return (
            f"Guides(verticals={len(self.vertical)}, "
            f"horizontals={len(self.horizontal)}, "
            f"cells={len(self.get_cells())})"
        )

    def _spans_pages(self) -> bool:
        """Check if any guides are defined across multiple pages in a FlowRegion."""
        if not self.is_flow_region:
            return False

        # Check vertical guides
        v_guide_pages = {}
        for coord, region in self._unified_vertical:
            v_guide_pages.setdefault(coord, set()).add(region.page.page_number)

        for pages in v_guide_pages.values():
            if len(pages) > 1:
                return True

        # Check horizontal guides
        h_guide_pages = {}
        for coord, region in self._unified_horizontal:
            h_guide_pages.setdefault(coord, set()).add(region.page.page_number)

        for pages in h_guide_pages.values():
            if len(pages) > 1:
                return True

        return False

    # -------------------------------------------------------------------------
    # Instance methods for fluent chaining (avoid name conflicts with class methods)
    # -------------------------------------------------------------------------

    def add_content(
        self,
        axis: Literal["vertical", "horizontal"] = "vertical",
        markers: Union[str, List[str], "ElementCollection", None] = None,
        obj: Optional[Union["Page", "Region"]] = None,
        align: Literal["left", "right", "center", "between"] = "left",
        outer: OuterBoundaryMode = True,
        tolerance: float = 5,
        apply_exclusions: bool = True,
    ) -> "Guides":
        """
        Instance method: Add guides from content, allowing chaining.
        This allows: Guides.new(page).add_content(axis='vertical', markers=[...])

        Args:
            axis: Which axis to create guides for
            markers: Content to search for. Can be:
                - str: single selector or literal text
                - List[str]: list of selectors or literal text strings
                - ElementCollection: collection of elements to extract text from
                - None: no markers
            obj: Page or Region to search (uses self.context if None)
            align: How to align guides relative to found elements
            outer: Whether to add outer boundary guides. Can be:
                - bool: True/False to add/not add both
                - "first": To add boundary before the first element
                - "last": To add boundary before the last element
            tolerance: Tolerance for snapping to element edges
            apply_exclusions: Whether to apply exclusion zones when searching for text

        Returns:
            Self for method chaining
        """
        # Use provided object or fall back to stored context
        target_obj = obj or self.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        # Create new guides using the class method
        new_guides = Guides.from_content(
            obj=target_obj,
            axis=axis,
            markers=markers,
            align=align,
            outer=outer,
            tolerance=tolerance,
            apply_exclusions=apply_exclusions,
        )

        # Add the appropriate coordinates to this object
        if axis == "vertical":
            self.vertical = list(set(self.vertical + new_guides.vertical))
        else:
            self.horizontal = list(set(self.horizontal + new_guides.horizontal))

        return self

    def add_lines(
        self,
        axis: Literal["vertical", "horizontal", "both"] = "both",
        obj: Optional[Union["Page", "Region"]] = None,
        threshold: Union[float, str] = "auto",
        source_label: Optional[str] = None,
        max_lines_h: Optional[int] = None,
        max_lines_v: Optional[int] = None,
        outer: bool = False,
        detection_method: str = "vector",
        resolution: int = 192,
        **detect_kwargs,
    ) -> "Guides":
        """
        Instance method: Add guides from lines, allowing chaining.
        This allows: Guides.new(page).add_lines(axis='horizontal')

        Args:
            axis: Which axis to detect lines for
            obj: Page or Region to search (uses self.context if None)
            threshold: Line detection threshold ('auto' or float 0.0-1.0)
            source_label: Filter lines by source label (vector) or label for detected lines (pixels)
            max_lines_h: Maximum horizontal lines to use
            max_lines_v: Maximum vertical lines to use
            outer: Whether to add outer boundary guides
            detection_method: 'vector', 'pixels', or 'auto' (default). 'auto' uses vector line
                information when available and falls back to pixel detection otherwise.
            resolution: DPI for pixel-based detection (default: 192)
            **detect_kwargs: Additional parameters for pixel detection (see from_lines)

        Returns:
            Self for method chaining
        """
        # Use provided object or fall back to stored context
        target_obj = obj or self.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        # Create new guides using the class method
        new_guides = Guides.from_lines(
            obj=target_obj,
            axis=axis,
            threshold=threshold,
            source_label=source_label,
            max_lines_h=max_lines_h,
            max_lines_v=max_lines_v,
            outer=outer,
            detection_method=detection_method,
            resolution=resolution,
            **detect_kwargs,
        )

        # Add the appropriate coordinates to this object
        if axis in ("vertical", "both"):
            self.vertical = list(set(self.vertical + new_guides.vertical))
        if axis in ("horizontal", "both"):
            self.horizontal = list(set(self.horizontal + new_guides.horizontal))

        return self

    def add_whitespace(
        self,
        axis: Literal["vertical", "horizontal", "both"] = "both",
        obj: Optional[Union["Page", "Region"]] = None,
        min_gap: float = 10,
    ) -> "Guides":
        """
        Instance method: Add guides from whitespace, allowing chaining.
        This allows: Guides.new(page).add_whitespace(axis='both')

        Args:
            axis: Which axis to create guides for
            obj: Page or Region to search (uses self.context if None)
            min_gap: Minimum gap size to consider

        Returns:
            Self for method chaining
        """
        # Use provided object or fall back to stored context
        target_obj = obj or self.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        # Create new guides using the class method
        new_guides = Guides.from_whitespace(obj=target_obj, axis=axis, min_gap=min_gap)

        # Add the appropriate coordinates to this object
        if axis in ("vertical", "both"):
            self.vertical = list(set(self.vertical + new_guides.vertical))
        if axis in ("horizontal", "both"):
            self.horizontal = list(set(self.horizontal + new_guides.horizontal))

        return self

    def extract_table(
        self,
        target: Optional[
            Union[
                "Page",
                "Region",
                "PageCollection",
                "ElementCollection",
                List[Union["Page", "Region"]],
            ]
        ] = None,
        source: str = "guides_temp",
        cell_padding: float = 0.5,
        include_outer_boundaries: bool = False,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[["Region"], Optional[str]]] = None,
        show_progress: bool = False,
        content_filter: Optional[Union[str, Callable[[str], bool], List[str]]] = None,
        apply_exclusions: bool = True,
        *,
        multi_page: Literal["auto", True, False] = "auto",
        header: Union[str, List[str], None] = "first",
        skip_repeating_headers: Optional[bool] = None,
        structure_engine: Optional[str] = None,
    ) -> "TableResult":
        """
        Extract table data directly from guides without leaving temporary regions.

        This method:
        1. Creates table structure using build_grid()
        2. Extracts table data from the created table region
        3. Cleans up all temporary regions
        4. Returns the TableResult

        When passed a collection (PageCollection, ElementCollection, or list), this method
        will extract tables from each element and combine them into a single result.

        Args:
            target: Page, Region, or collection of Pages/Regions to extract from (uses self.context if None)
            source: Source label for temporary regions (will be cleaned up)
            cell_padding: Internal padding for cell regions in points
            include_outer_boundaries: Whether to add boundaries at edges if missing
            method: Table extraction method ('tatr', 'pdfplumber', 'text', etc.)
            table_settings: Settings for pdfplumber table extraction
            use_ocr: Whether to use OCR for text extraction
            ocr_config: OCR configuration parameters
            text_options: Dictionary of options for the 'text' method
            cell_extraction_func: Optional callable for custom cell text extraction
            show_progress: Controls progress bar for text method
            content_filter: Content filtering function or patterns
            apply_exclusions: Whether to apply exclusion regions during text extraction (default: True)
            multi_page: Controls multi-region table creation for FlowRegions
            header: How to handle headers when extracting from collections:
                - "first": Use first row of first element as headers (default)
                - "all": Expect headers on each element, use from first element
                - None: No headers, use numeric indices
                - List[str]: Custom column names
            skip_repeating_headers: Whether to remove duplicate header rows when extracting from collections.
                Defaults to True when header is "first" or "all", False otherwise.
            structure_engine: Optional structure detection engine name passed to the underlying
                region extraction to leverage provider-backed table structure results.

        Returns:
            TableResult: Extracted table data

        Raises:
            ValueError: If no table region is created from the guides

        Example:
            ```python
            from natural_pdf.analyzers import Guides

            # Single page extraction
            guides = Guides.from_lines(page, source_label="detected")
            table_data = guides.extract_table()
            df = table_data.to_df()

            # Multiple page extraction
            guides = Guides(pages[0])
            guides.vertical.from_content(['Column 1', 'Column 2'])
            table_result = guides.extract_table(pages, header=['Col1', 'Col2'])
            df = table_result.to_df()

            # Region collection extraction
            regions = pdf.find_all('region[type=table]')
            guides = Guides(regions[0])
            guides.vertical.from_lines(n=3)
            table_result = guides.extract_table(regions)
            ```
        """
        from natural_pdf.core.page_collection import PageCollection

        target_obj = target if target is not None else self.context
        if target_obj is None:
            raise ValueError("No target object available. Provide target parameter or context.")

        # Check if target is a collection - if so, delegate to _extract_table_from_collection
        if isinstance(target_obj, (PageCollection, ElementCollection, list)):
            # For collections, pass through most parameters as-is
            return self._extract_table_from_collection(
                elements=target_obj,
                header=header,
                skip_repeating_headers=skip_repeating_headers,
                method=method,
                table_settings=table_settings,
                use_ocr=use_ocr,
                ocr_config=ocr_config,
                text_options=text_options,
                cell_extraction_func=cell_extraction_func,
                show_progress=show_progress,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
                structure_engine=structure_engine,
            )

        from natural_pdf.core.page import Page

        # Get the page for cleanup later
        if hasattr(target_obj, "x0") and hasattr(target_obj, "top"):  # Region
            page_obj = getattr(target_obj, "_page", None)
            if page_obj is None:
                raise ValueError("Region is not associated with a page")
            page = cast(Page, page_obj)
        elif hasattr(target_obj, "add_element"):  # Page-like
            page = cast(Page, target_obj)
        else:
            raise ValueError(f"Target object {target_obj} is not a Page or Region")

        # Check if we have guides in only one dimension
        has_verticals = len(self.vertical) > 0
        has_horizontals = len(self.horizontal) > 0

        # If we have guides in only one dimension, use direct extraction with explicit lines
        if (has_verticals and not has_horizontals) or (has_horizontals and not has_verticals):
            logger.debug(
                f"Partial guides detected - using direct extraction (v={has_verticals}, h={has_horizontals})"
            )

            return self._extract_with_table_service(
                target_obj,
                method=method,  # Let auto-detection work when None
                table_settings=table_settings,
                use_ocr=use_ocr,
                ocr_config=ocr_config,
                text_options=text_options,
                cell_extraction_func=cell_extraction_func,
                show_progress=show_progress,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
                verticals=list(self.vertical) if has_verticals else None,
                horizontals=list(self.horizontal) if has_horizontals else None,
                structure_engine=structure_engine,
            )

        # Both dimensions have guides - use normal grid-based extraction
        try:
            # Step 1: Build grid structure (creates temporary regions)
            grid_result = self.build_grid(
                target=target_obj,
                source=source,
                cell_padding=cell_padding,
                include_outer_boundaries=include_outer_boundaries,
                multi_page=multi_page,
            )

            # Step 2: Get the table region and extract table data
            table_region = grid_result["regions"]["table"]
            if table_region is None:
                raise ValueError(
                    "No table region was created from the guides. Check that you have both vertical and horizontal guides."
                )

            # Handle multi-page case where table_region might be a list
            if isinstance(table_region, list):
                if not table_region:
                    raise ValueError("No table regions were created from the guides.")
                # Use the first table region for extraction
                table_region = table_region[0]

            table_result = self._extract_with_table_service(
                table_region,
                method=method,
                table_settings=table_settings,
                use_ocr=use_ocr,
                ocr_config=ocr_config,
                text_options=text_options,
                cell_extraction_func=cell_extraction_func,
                show_progress=show_progress,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
                structure_engine=structure_engine,
            )
            self._assign_headers_from_rows(table_result, header)
            return table_result

        finally:
            # Step 4: Clean up all temporary regions created by build_grid
            # This ensures no regions are left behind regardless of success/failure
            iter_regions = getattr(page, "iter_regions", None)
            raw_regions: Optional[Iterable[Any]]
            if callable(iter_regions):
                raw_regions = cast(Optional[Iterable[Any]], iter_regions())
            else:
                raw_regions = cast(Optional[Iterable[Any]], iter_regions)

            if raw_regions is None or not isinstance(raw_regions, IterableABC):
                existing_regions = []
            else:
                existing_regions = list(raw_regions)

            regions_to_remove = [
                r
                for r in existing_regions
                if getattr(r, "source", None) == source
                and getattr(r, "region_type", None)
                in {"table", "table_row", "table_column", "table_cell"}
            ]

            for region in regions_to_remove:
                page.remove_element(region, element_type="regions")

            if regions_to_remove:
                logger.debug("Cleaned up %d temporary regions", len(regions_to_remove))

    def _extract_table_from_collection(
        self,
        elements: Union["PageCollection", "ElementCollection", List[Union["Page", "Region"]]],
        header: Union[str, List[str], None] = "first",
        skip_repeating_headers: Optional[bool] = None,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[["Region"], Optional[str]]] = None,
        show_progress: bool = True,
        content_filter: Optional[Union[str, Callable[[str], bool], List[str]]] = None,
        apply_exclusions: bool = True,
        structure_engine: Optional[str] = None,
    ) -> "TableResult":
        """
        Extract tables from multiple pages or regions using this guide pattern.

        This method applies the guide to each element, extracts tables, and combines
        them into a single TableResult. Dynamic guides (using lambdas) are evaluated
        for each element.

        Args:
            elements: PageCollection, ElementCollection, or list of Pages/Regions to extract from
            header: How to handle headers:
                - "first": Use first row of first element as headers (default)
                - "all": Expect headers on each element, use from first element
                - None: No headers, use numeric indices
                - List[str]: Custom column names
            skip_repeating_headers: Whether to remove duplicate header rows.
                Defaults to True when header is "first" or "all", False otherwise.
            method: Table extraction method (passed to extract_table)
            table_settings: Settings for pdfplumber table extraction
            use_ocr: Whether to use OCR for text extraction
            ocr_config: OCR configuration parameters
            text_options: Dictionary of options for the 'text' method
            cell_extraction_func: Optional callable for custom cell text extraction
            show_progress: Show progress bar for multi-element extraction (default: True)
            content_filter: Content filtering function or patterns
            apply_exclusions: Whether to apply exclusion regions during extraction
            structure_engine: Optional structure engine forwarded to each element extraction.

        Returns:
            TableResult: Combined table data from all elements

        Example:
            ```python
            # Create guide with static vertical, dynamic horizontal
            guide = Guides(regions[0])
            guide.vertical.from_content(columns, outer="last")
            guide.horizontal.from_content(lambda r: r.find_all('text:starts-with(NF-)'))

            # Extract from all regions
            table_result = guide._extract_table_from_collection(regions, header=columns)
            df = table_result.to_df()
            ```
        """
        from natural_pdf.core.page_collection import PageCollection
        from natural_pdf.tables.result import TableResult

        # Convert to list if it's a collection
        if isinstance(elements, (PageCollection, ElementCollection)):
            element_list = list(elements)
        else:
            element_list = elements

        if not element_list:
            return TableResult([])

        # Determine header handling
        if skip_repeating_headers is None:
            skip_repeating_headers = header in ["first", "all"] or isinstance(header, list)

        all_rows = []
        header_row = None

        # Configure progress bar
        iterator = element_list
        if show_progress and len(element_list) > 1:
            from tqdm.auto import tqdm

            iterator = tqdm(element_list, desc="Extracting tables from elements", unit="element")

        for i, element in enumerate(iterator):
            # Create a new Guides object for this element
            element_guide = Guides(element)

            # Copy vertical guides (usually static)
            if hasattr(self.vertical, "_callable") and self.vertical._callable is not None:
                # If vertical is dynamic (lambda), evaluate it
                element_guide.vertical.from_content(self.vertical._callable(element))
            else:
                # Copy static vertical positions
                element_guide.vertical.data = self.vertical.data.copy()

            # Handle horizontal guides
            if hasattr(self.horizontal, "_callable") and self.horizontal._callable is not None:
                # If horizontal is dynamic (lambda), evaluate it
                element_guide.horizontal.from_content(self.horizontal._callable(element))
            else:
                # Copy static horizontal positions
                element_guide.horizontal.data = self.horizontal.data.copy()

            # Extract table from this element
            table_result = element_guide.extract_table(
                method=method,
                table_settings=table_settings,
                use_ocr=use_ocr,
                ocr_config=ocr_config,
                text_options=text_options,
                cell_extraction_func=cell_extraction_func,
                show_progress=False,  # Don't show nested progress
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
                structure_engine=structure_engine,
            )

            # Convert to list of rows
            rows = list(table_result)

            # Handle headers based on strategy
            if i == 0:  # First element
                if header == "first" or header == "all":
                    # Use first row as header
                    if rows:
                        header_row = rows[0]
                        rows = rows[1:]  # Remove header from data
                elif isinstance(header, list):
                    # Custom headers provided
                    header_row = header
            else:  # Subsequent elements
                if header == "all" and skip_repeating_headers and rows:
                    # Expect and remove header row
                    if rows and header_row and rows[0] == header_row:
                        rows = rows[1:]
                    elif rows:
                        # Still remove first row if it looks like a header
                        rows = rows[1:]

            # Add rows to combined result
            all_rows.extend(rows)

        # Create final TableResult
        if isinstance(header, list):
            final_result = TableResult(all_rows)
            final_result.headers = header
        elif header_row is not None:
            final_result = TableResult([header_row] + all_rows)
            final_result.headers = header_row
        else:
            final_result = TableResult(all_rows)
        self._assign_headers_from_rows(final_result, header)
        return final_result

    @staticmethod
    def _assign_headers_from_rows(
        table_result: TableResult, header: Union[str, int, List[str], None]
    ) -> None:
        """Normalize headers on a TableResult and drop empty leading rows."""

        def _row_is_blank(row: Sequence[Any]) -> bool:
            return all(cell is None or (isinstance(cell, str) and not cell.strip()) for cell in row)

        rows = getattr(table_result, "_rows", None)
        if not isinstance(rows, list) or not rows or header in (None, False):
            return

        if isinstance(header, list):
            table_result.headers = header
            return

        # Remove leading empty rows that can appear due to outer boundaries
        while rows and _row_is_blank(rows[0]):
            rows.pop(0)

        if not rows:
            return

        header_row: Optional[List[Any]] = None
        if header == "first":
            header_row = list(rows[0])
        elif isinstance(header, int):
            idx = max(0, min(len(rows) - 1, header))
            header_row = list(rows[idx])

        if header_row:
            table_result.headers = header_row

    def _get_flow_orientation(self) -> Literal["vertical", "horizontal", "unknown"]:
        """Determines if a FlowRegion's constituent parts are arranged vertically or horizontally."""
        if not self.is_flow_region or len(self._flow_constituent_regions()) < 2:
            return "unknown"

        r1 = self._flow_constituent_regions()[0]
        r2 = self._flow_constituent_regions()[1]  # Compare first two regions

        if not r1.bbox or not r2.bbox:
            return "unknown"

        # Calculate non-overlapping distances.
        # This determines the primary direction of separation.
        x_dist = max(0, max(r1.x0, r2.x0) - min(r1.x1, r2.x1))
        y_dist = max(0, max(r1.top, r2.top) - min(r1.bottom, r2.bottom))

        if y_dist > x_dist:
            return "vertical"
        else:
            return "horizontal"


# -------------------------------------------------------------------------
# Accessor classes for property-based access
# -------------------------------------------------------------------------


class _ColumnAccessor:
    """Provides indexed access to columns via guides.columns[index]."""

    def __init__(self, guides: "Guides"):
        self._guides = guides

    def __len__(self):
        """Return number of columns (vertical guides - 1)."""
        return max(0, len(self._guides.vertical) - 1)

    def __getitem__(self, index: Union[int, slice]) -> Union["Region", "ElementCollection"]:
        """Get column at the specified index or slice."""

        if isinstance(index, slice):
            # Handle slice notation - return multiple columns
            columns = []
            num_cols = len(self)

            # Convert slice to range of indices
            start, stop, step = index.indices(num_cols)
            for i in range(start, stop, step):
                columns.append(self._guides.column(i))

            return ElementCollection(columns)
        else:
            # Handle negative indexing
            if index < 0:
                index = len(self) + index
            return self._guides.column(index)


class _RowAccessor:
    """Provides indexed access to rows via guides.rows[index]."""

    def __init__(self, guides: "Guides"):
        self._guides = guides

    def __len__(self):
        """Return number of rows (horizontal guides - 1)."""
        return max(0, len(self._guides.horizontal) - 1)

    def __getitem__(self, index: Union[int, slice]) -> Union["Region", "ElementCollection"]:
        """Get row at the specified index or slice."""

        if isinstance(index, slice):
            # Handle slice notation - return multiple rows
            rows = []
            num_rows = len(self)

            # Convert slice to range of indices
            start, stop, step = index.indices(num_rows)
            for i in range(start, stop, step):
                rows.append(self._guides.row(i))

            return ElementCollection(rows)
        else:
            # Handle negative indexing
            if index < 0:
                index = len(self) + index
            return self._guides.row(index)


class _CellAccessor:
    """Provides indexed access to cells via guides.cells[row][col] or guides.cells[row, col]."""

    def __init__(self, guides: "Guides"):
        self._guides = guides

    def __getitem__(self, key) -> Union["Region", "_CellRowAccessor", "ElementCollection"]:
        """
        Get cell(s) at the specified position.

        Supports:
        - guides.cells[row, col] - single cell
        - guides.cells[row][col] - single cell (nested)
        - guides.cells[row, :] - all cells in a row
        - guides.cells[:, col] - all cells in a column
        - guides.cells[:, :] - all cells
        - guides.cells[row][:] - all cells in a row (nested)
        """

        if isinstance(key, tuple) and len(key) == 2:
            row, col = key

            # Handle slices for row and/or column
            if isinstance(row, slice) or isinstance(col, slice):
                cells = []
                num_rows = len(self._guides.rows)
                num_cols = len(self._guides.columns)

                # Convert slices to ranges
                if isinstance(row, slice):
                    row_indices = range(*row.indices(num_rows))
                else:
                    # Single row index
                    if row < 0:
                        row = num_rows + row
                    row_indices = [row]

                if isinstance(col, slice):
                    col_indices = range(*col.indices(num_cols))
                else:
                    # Single column index
                    if col < 0:
                        col = num_cols + col
                    col_indices = [col]

                # Collect all cells in the specified ranges
                for r in row_indices:
                    for c in col_indices:
                        cells.append(self._guides.cell(r, c))

                return ElementCollection(cells)
            else:
                # Both are integers - single cell access
                # Handle negative indexing for both row and col
                if row < 0:
                    row = len(self._guides.rows) + row
                if col < 0:
                    col = len(self._guides.columns) + col
                return self._guides.cell(row, col)
        elif isinstance(key, slice):
            # First level slice: guides.cells[:] - return all rows as accessors
            # For now, let's return all cells flattened
            cells = []
            num_rows = len(self._guides.rows)
            row_indices = range(*key.indices(num_rows))

            for r in row_indices:
                for c in range(len(self._guides.columns)):
                    cells.append(self._guides.cell(r, c))

            return ElementCollection(cells)
        elif isinstance(key, int):
            # First level of nested access: guides.cells[row]
            # Handle negative indexing for row
            if key < 0:
                key = len(self._guides.rows) + key
            # Return a row accessor that allows [col] or [:] indexing
            return _CellRowAccessor(self._guides, key)
        else:
            raise TypeError(
                f"Cell indices must be integers, slices, or tuple of two integers/slices, got {type(key)}"
            )


class _CellRowAccessor:
    """Provides column access for a specific row in nested cell indexing."""

    def __init__(self, guides: "Guides", row: int):
        self._guides = guides
        self._row = row

    def __getitem__(self, col: Union[int, slice]) -> Union["Region", "ElementCollection"]:
        """Get cell at [row][col] or all cells in row with [row][:]."""

        if isinstance(col, slice):
            # Handle slice notation - return all cells in this row
            cells = []
            num_cols = len(self._guides.columns)

            # Convert slice to range of indices
            start, stop, step = col.indices(num_cols)
            for c in range(start, stop, step):
                cells.append(self._guides.cell(self._row, c))

            return ElementCollection(cells)
        else:
            # Handle single column index
            # Handle negative indexing for column
            if col < 0:
                col = len(self._guides.columns) + col
            return self._guides.cell(self._row, col)
