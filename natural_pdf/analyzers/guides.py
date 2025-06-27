"""Guide system for table extraction and layout analysis."""

import json
import logging
from collections import UserList
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.collections import ElementCollection
    from natural_pdf.elements.region import Region
    from natural_pdf.flows.region import FlowRegion

logger = logging.getLogger(__name__)


def _normalize_markers(
    markers: Union[str, List[str], "ElementCollection", None], obj: Union["Page", "Region", "FlowRegion"]
) -> List[str]:
    """
    Normalize markers parameter to a list of text strings for guide creation.

    Args:
        markers: Can be:
            - str: single selector or text string
            - List[str]: list of selectors or text strings
            - ElementCollection: collection of elements to extract text from
            - None: empty list
        obj: Object to search for elements if markers contains selectors

    Returns:
        List of text strings to search for
    """
    if markers is None:
        return []

    # Handle FlowRegion by collecting markers from all constituent regions
    if hasattr(obj, "constituent_regions"):
        all_markers = []
        for region in obj.constituent_regions:
            region_markers = _normalize_markers(markers, region)
            all_markers.extend(region_markers)
        # Remove duplicates while preserving order
        seen = set()
        unique_markers = []
        for m in all_markers:
            if m not in seen:
                seen.add(m)
                unique_markers.append(m)
        return unique_markers

    if isinstance(markers, str):
        # Single selector or text string
        if markers.startswith(("text", "region", "line", "rect", "blob", "image")):
            # It's a CSS selector, find elements and extract text
            if hasattr(obj, "find_all"):
                elements = obj.find_all(markers)
                return [elem.text if hasattr(elem, "text") else str(elem) for elem in elements]
            else:
                logger.warning(f"Object {obj} doesn't support find_all for selector '{markers}'")
                return [markers]  # Treat as literal text
        else:
            # Treat as literal text
            return [markers]

    elif hasattr(markers, "__iter__") and not isinstance(markers, str):
        # It might be an ElementCollection or list
        if hasattr(markers, "extract_each_text"):
            # It's an ElementCollection
            try:
                return markers.extract_each_text()
            except Exception as e:
                logger.warning(f"Failed to extract text from ElementCollection: {e}")
                # Fallback: try to get text from individual elements
                texts = []
                for elem in markers:
                    if hasattr(elem, "text"):
                        texts.append(elem.text)
                    elif hasattr(elem, "extract_text"):
                        texts.append(elem.extract_text())
                    else:
                        texts.append(str(elem))
                return texts
        else:
            # It's a regular list - process each item
            result = []
            for marker in markers:
                if isinstance(marker, str):
                    if marker.startswith(("text", "region", "line", "rect", "blob", "image")):
                        # It's a selector
                        if hasattr(obj, "find_all"):
                            elements = obj.find_all(marker)
                            result.extend(
                                [
                                    elem.text if hasattr(elem, "text") else str(elem)
                                    for elem in elements
                                ]
                            )
                        else:
                            result.append(marker)  # Treat as literal
                    else:
                        # Literal text
                        result.append(marker)
                elif hasattr(marker, "text"):
                    # It's an element object
                    result.append(marker.text)
                elif hasattr(marker, "extract_text"):
                    # It's an element that can extract text
                    result.append(marker.extract_text())
                else:
                    result.append(str(marker))
            return result

    else:
        # Unknown type, try to convert to string
        return [str(markers)]


class GuidesList(UserList):
    """A list of guide coordinates that also provides methods for creating guides."""

    def __init__(self, parent_guides: "Guides", axis: Literal["vertical", "horizontal"], data=None):
        super().__init__(data or [])
        self._parent = parent_guides
        self._axis = axis

    def from_content(
        self,
        markers: Union[str, List[str], "ElementCollection", None],
        obj: Optional[Union["Page", "Region", "FlowRegion"]] = None,
        align: Literal["left", "right", "center", "between"] = "left",
        outer: bool = True,
        tolerance: float = 5,
        *,
        append: bool = False,
    ) -> "Guides":
        """
        Create guides from content markers and add to this axis.

        Args:
            markers: Content to search for. Can be:
                - str: single selector (e.g., 'text:contains("Name")') or literal text
                - List[str]: list of selectors or literal text strings
                - ElementCollection: collection of elements to extract text from
                - None: no markers
            obj: Page/Region/FlowRegion to search (uses parent's context if None)
            align: How to align guides relative to found elements
            outer: Whether to add outer boundary guides
            tolerance: Tolerance for snapping to element edges

        Returns:
            Parent Guides object for chaining
        """
        target_obj = obj or self._parent.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        # Check if parent is in flow mode
        if self._parent.is_flow_region:
            # Create guides across all constituent regions
            all_guides = []
            for region in self._parent.context.constituent_regions:
                # Normalize markers for this region
                marker_texts = _normalize_markers(markers, region)
                
                # Create guides for this region
                region_guides = Guides.from_content(
                    obj=region,
                    axis=self._axis,
                    markers=marker_texts,
                    align=align,
                    outer=outer,
                    tolerance=tolerance,
                )
                
                # Collect guides from this region
                if self._axis == "vertical":
                    all_guides.extend(region_guides.vertical)
                else:
                    all_guides.extend(region_guides.horizontal)
            
            # Update parent's flow guides structure
            if append:
                # Append to existing
                existing = [coord for coord, _ in 
                           (self._parent._unified_vertical if self._axis == "vertical" 
                            else self._parent._unified_horizontal)]
                all_guides = existing + all_guides
            
            # Remove duplicates and sort
            unique_guides = sorted(list(set(all_guides)))
            
            # Clear and rebuild unified view
            if self._axis == "vertical":
                self._parent._unified_vertical = []
                for coord in unique_guides:
                    # Find which region(s) this guide belongs to
                    for region in self._parent.context.constituent_regions:
                        if hasattr(region, "bbox"):
                            x0, _, x1, _ = region.bbox
                            if x0 <= coord <= x1:
                                self._parent._unified_vertical.append((coord, region))
                                break
                self._parent._vertical_cache = None
                self.data = unique_guides
            else:
                self._parent._unified_horizontal = []
                for coord in unique_guides:
                    # Find which region(s) this guide belongs to
                    for region in self._parent.context.constituent_regions:
                        if hasattr(region, "bbox"):
                            _, y0, _, y1 = region.bbox
                            if y0 <= coord <= y1:
                                self._parent._unified_horizontal.append((coord, region))
                                break
                self._parent._horizontal_cache = None
                self.data = unique_guides
            
            # Update per-region guides
            for region in self._parent.context.constituent_regions:
                region_verticals = []
                region_horizontals = []
                
                for coord, r in self._parent._unified_vertical:
                    if r == region:
                        region_verticals.append(coord)
                        
                for coord, r in self._parent._unified_horizontal:
                    if r == region:
                        region_horizontals.append(coord)
                
                self._parent._flow_guides[region] = (
                    sorted(region_verticals),
                    sorted(region_horizontals)
                )
            
            return self._parent
        
        # Original single-region logic
        # Normalize markers to list of text strings
        marker_texts = _normalize_markers(markers, target_obj)

        # Create guides for this axis
        new_guides = Guides.from_content(
            obj=target_obj,
            axis=self._axis,
            markers=marker_texts,
            align=align,
            outer=outer,
            tolerance=tolerance,
        )

        # Replace or append based on parameter
        if append:
            if self._axis == "vertical":
                self.extend(new_guides.vertical)
            else:
                self.extend(new_guides.horizontal)
        else:
            if self._axis == "vertical":
                self.data = list(new_guides.vertical)
            else:
                self.data = list(new_guides.horizontal)

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for x in self.data:
            if x not in seen:
                seen.add(x)
                unique.append(x)
        self.data = unique

        return self._parent  # Return parent for chaining

    def from_lines(
        self,
        obj: Optional[Union["Page", "Region", "FlowRegion"]] = None,
        threshold: Union[float, str] = "auto",
        source_label: Optional[str] = None,
        max_lines: Optional[int] = None,
        outer: bool = False,
        detection_method: str = "vector",
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
            detection_method: 'vector' (use existing LineElements) or 'pixels' (detect from image)
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

        # Check if parent is in flow mode
        if self._parent.is_flow_region:
            # Create guides across all constituent regions
            all_guides = []
            
            for region in self._parent.context.constituent_regions:
                # Create guides for this specific region
                region_guides = Guides.from_lines(
                    obj=region,
                    axis=self._axis,
                    threshold=threshold,
                    source_label=source_label,
                    max_lines_h=max_lines_h,
                    max_lines_v=max_lines_v,
                    outer=outer,
                    detection_method=detection_method,
                    resolution=resolution,
                    **detect_kwargs
                )
                
                # Collect guides from this region
                if self._axis == "vertical":
                    all_guides.extend(region_guides.vertical)
                else:
                    all_guides.extend(region_guides.horizontal)
            
            # Update parent's flow guides structure
            if append:
                # Append to existing
                existing = [coord for coord, _ in 
                           (self._parent._unified_vertical if self._axis == "vertical" 
                            else self._parent._unified_horizontal)]
                all_guides = existing + all_guides
            
            # Remove duplicates and sort
            unique_guides = sorted(list(set(all_guides)))
            
            # Clear and rebuild unified view
            if self._axis == "vertical":
                self._parent._unified_vertical = []
                for coord in unique_guides:
                    # Find which region(s) this guide belongs to
                    for region in self._parent.context.constituent_regions:
                        if hasattr(region, "bbox"):
                            x0, _, x1, _ = region.bbox
                            if x0 <= coord <= x1:
                                self._parent._unified_vertical.append((coord, region))
                                break
                self._parent._vertical_cache = None
                self.data = unique_guides
            else:
                self._parent._unified_horizontal = []
                for coord in unique_guides:
                    # Find which region(s) this guide belongs to
                    for region in self._parent.context.constituent_regions:
                        if hasattr(region, "bbox"):
                            _, y0, _, y1 = region.bbox
                            if y0 <= coord <= y1:
                                self._parent._unified_horizontal.append((coord, region))
                                break
                self._parent._horizontal_cache = None
                self.data = unique_guides
            
            # Update per-region guides
            for region in self._parent.context.constituent_regions:
                region_verticals = []
                region_horizontals = []
                
                for coord, r in self._parent._unified_vertical:
                    if r == region:
                        region_verticals.append(coord)
                        
                for coord, r in self._parent._unified_horizontal:
                    if r == region:
                        region_horizontals.append(coord)
                
                self._parent._flow_guides[region] = (
                    sorted(region_verticals),
                    sorted(region_horizontals)
                )
            
            return self._parent

        # Original single-region logic
        # Create guides for this axis
        new_guides = Guides.from_lines(
            obj=target_obj,
            axis=self._axis,
            threshold=threshold,
            source_label=source_label,
            max_lines_h=max_lines_h,
            max_lines_v=max_lines_v,
            outer=outer,
            detection_method=detection_method,
            resolution=resolution,
            **detect_kwargs,
        )

        # Replace or append based on parameter
        if append:
            if self._axis == "vertical":
                self.extend(new_guides.vertical)
            else:
                self.extend(new_guides.horizontal)
        else:
            if self._axis == "vertical":
                self.data = list(new_guides.vertical)
            else:
                self.data = list(new_guides.horizontal)

        # Remove duplicates
        seen = set()
        unique = []
        for x in self.data:
            if x not in seen:
                seen.add(x)
                unique.append(x)
        self.data = unique

        return self._parent

    def from_whitespace(
        self, obj: Optional[Union["Page", "Region", "FlowRegion"]] = None, min_gap: float = 10,
        *, append: bool = False
    ) -> "Guides":
        """
        Create guides from whitespace gaps.

        Args:
            obj: Page/Region/FlowRegion to analyze (uses parent's context if None)
            min_gap: Minimum gap size to consider

        Returns:
            Parent Guides object for chaining
        """
        target_obj = obj or self._parent.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        # Check if parent is in flow mode
        if self._parent.is_flow_region:
            # Create guides across all constituent regions
            all_guides = []
            
            for region in self._parent.context.constituent_regions:
                # Create guides for this specific region
                region_guides = Guides.from_whitespace(
                    obj=region,
                    axis=self._axis,
                    min_gap=min_gap
                )
                
                # Collect guides from this region
                if self._axis == "vertical":
                    all_guides.extend(region_guides.vertical)
                else:
                    all_guides.extend(region_guides.horizontal)
            
            # Update parent's flow guides structure
            if append:
                # Append to existing
                existing = [coord for coord, _ in 
                           (self._parent._unified_vertical if self._axis == "vertical" 
                            else self._parent._unified_horizontal)]
                all_guides = existing + all_guides
            
            # Remove duplicates and sort
            unique_guides = sorted(list(set(all_guides)))
            
            # Clear and rebuild unified view
            if self._axis == "vertical":
                self._parent._unified_vertical = []
                for coord in unique_guides:
                    # Find which region(s) this guide belongs to
                    for region in self._parent.context.constituent_regions:
                        if hasattr(region, "bbox"):
                            x0, _, x1, _ = region.bbox
                            if x0 <= coord <= x1:
                                self._parent._unified_vertical.append((coord, region))
                                break
                self._parent._vertical_cache = None
                self.data = unique_guides
            else:
                self._parent._unified_horizontal = []
                for coord in unique_guides:
                    # Find which region(s) this guide belongs to
                    for region in self._parent.context.constituent_regions:
                        if hasattr(region, "bbox"):
                            _, y0, _, y1 = region.bbox
                            if y0 <= coord <= y1:
                                self._parent._unified_horizontal.append((coord, region))
                                break
                self._parent._horizontal_cache = None
                self.data = unique_guides
            
            # Update per-region guides
            for region in self._parent.context.constituent_regions:
                region_verticals = []
                region_horizontals = []
                
                for coord, r in self._parent._unified_vertical:
                    if r == region:
                        region_verticals.append(coord)
                        
                for coord, r in self._parent._unified_horizontal:
                    if r == region:
                        region_horizontals.append(coord)
                
                self._parent._flow_guides[region] = (
                    sorted(region_verticals),
                    sorted(region_horizontals)
                )
            
            return self._parent

        # Original single-region logic
        # Create guides for this axis
        new_guides = Guides.from_whitespace(obj=target_obj, axis=self._axis, min_gap=min_gap)

        # Replace or append
        if append:
            if self._axis == "vertical":
                self.extend(new_guides.vertical)
            else:
                self.extend(new_guides.horizontal)
        else:
            if self._axis == "vertical":
                self.data = list(new_guides.vertical)
            else:
                self.data = list(new_guides.horizontal)

        # Remove duplicates
        seen = set()
        unique = []
        for x in self.data:
            if x not in seen:
                seen.add(x)
                unique.append(x)
        self.data = unique

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

        # Create guides using divide
        new_guides = Guides.divide(obj=target_obj, n=n, axis=self._axis)

        # Replace existing guides instead of extending (no append option here)
        if self._axis == "vertical":
            self.data = list(new_guides.vertical)
        else:
            self.data = list(new_guides.horizontal)

        # Remove duplicates
        seen = set()
        unique = []
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
        original_guides = self.data.copy()

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
        verticals: Optional[Union[List[float], "Page", "Region", "FlowRegion"]] = None,
        horizontals: Optional[List[float]] = None,
        context: Optional[Union["Page", "Region", "FlowRegion"]] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        relative: bool = False,
        snap_behavior: Literal["raise", "warn", "ignore"] = "warn",
    ):
        """
        Initialize a Guides object.

        Args:
            verticals: List of x-coordinates for vertical guides, or a Page/Region/FlowRegion as context
            horizontals: List of y-coordinates for horizontal guides
            context: Page, Region, or FlowRegion object these guides were created from
            bounds: Bounding box (x0, top, x1, bottom) if context not provided
            relative: Whether coordinates are relative (0-1) or absolute
            snap_behavior: How to handle snapping conflicts ('raise', 'warn', or 'ignore')
        """
        # Handle Guides(page) or Guides(flow_region) shorthand
        if (
            verticals is not None
            and not isinstance(verticals, (list, tuple))
            and horizontals is None
            and context is None
        ):
            # First argument is a page/region/flow_region, not coordinates
            context = verticals
            verticals = None

        self.context = context
        self.bounds = bounds
        self.relative = relative
        self.snap_behavior = snap_behavior
        
        # Check if we're dealing with a FlowRegion
        self.is_flow_region = hasattr(context, "constituent_regions")
        
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
        self._vertical = GuidesList(self, "vertical", sorted([float(x) for x in (verticals or [])]))
        self._horizontal = GuidesList(
            self, "horizontal", sorted([float(y) for y in (horizontals or [])])
        )

        # Determine bounds from context if needed
        if self.bounds is None and self.context is not None:
            if hasattr(self.context, "bbox"):
                self.bounds = self.context.bbox
            elif hasattr(self.context, "x0"):
                self.bounds = (
                    self.context.x0,
                    self.context.top,
                    self.context.x1,
                    self.context.bottom,
                )

        # Convert relative to absolute if needed
        if self.relative and self.bounds:
            x0, top, x1, bottom = self.bounds
            width = x1 - x0
            height = bottom - top

            self._vertical.data = [x0 + v * width for v in self._vertical]
            self._horizontal.data = [top + h * height for h in self._horizontal]
            self.relative = False

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

    def _get_context_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get bounds from context if available."""
        if self.context is None:
            return None

        if hasattr(self.context, "bbox"):
            return self.context.bbox
        elif hasattr(self.context, "x0") and hasattr(self.context, "top"):
            return (self.context.x0, self.context.top, self.context.x1, self.context.bottom)
        elif hasattr(self.context, "width") and hasattr(self.context, "height"):
            return (0, 0, self.context.width, self.context.height)
        return None

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
            bounds = obj
            context = None
        else:
            context = obj
            if hasattr(obj, "bbox"):
                bounds = obj.bbox
            elif hasattr(obj, "x0"):
                bounds = (obj.x0, obj.top, obj.x1, obj.bottom)
            else:
                bounds = (0, 0, obj.width, obj.height)

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
        obj: Union["Page", "Region", "FlowRegion"],
        axis: Literal["vertical", "horizontal", "both"] = "both",
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
        Create guides from detected line elements.

        Args:
            obj: Page, Region, or FlowRegion to detect lines from
            axis: Which orientations to detect
            threshold: Detection threshold ('auto' or float 0.0-1.0) - used for pixel detection
            source_label: Filter for line source (vector method) or label for detected lines (pixel method)
            max_lines_h: Maximum number of horizontal lines to keep
            max_lines_v: Maximum number of vertical lines to keep
            outer: Whether to add outer boundary guides
            detection_method: 'vector' (use existing LineElements) or 'pixels' (detect from image)
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
        # Handle FlowRegion
        if hasattr(obj, "constituent_regions"):
            guides = cls(context=obj)
            
            # Process each constituent region
            for region in obj.constituent_regions:
                # Create guides for this specific region
                region_guides = cls.from_lines(
                    region,
                    axis=axis,
                    threshold=threshold,
                    source_label=source_label,
                    max_lines_h=max_lines_h,
                    max_lines_v=max_lines_v,
                    outer=outer,
                    detection_method=detection_method,
                    resolution=resolution,
                    **detect_kwargs
                )
                
                # Store in flow guides
                guides._flow_guides[region] = (
                    list(region_guides.vertical),
                    list(region_guides.horizontal)
                )
                
                # Add to unified view
                for v in region_guides.vertical:
                    guides._unified_vertical.append((v, region))
                for h in region_guides.horizontal:
                    guides._unified_horizontal.append((h, region))
            
            # Invalidate caches to force rebuild on next access
            guides._vertical_cache = None
            guides._horizontal_cache = None
            
            return guides
        
        # Original single-region logic follows...
        # Get bounds for potential outer guides
        if hasattr(obj, "bbox"):
            bounds = obj.bbox
        elif hasattr(obj, "x0"):
            bounds = (obj.x0, obj.top, obj.x1, obj.bottom)
        elif hasattr(obj, "width"):
            bounds = (0, 0, obj.width, obj.height)
        else:
            bounds = None

        verticals = []
        horizontals = []

        if detection_method == "pixels":
            # Use pixel-based line detection
            if not hasattr(obj, "detect_lines"):
                raise ValueError(f"Object {obj} does not support pixel-based line detection")

            # Set up detection parameters
            detect_params = {
                "resolution": resolution,
                "source_label": source_label or "guides_detection",
                "horizontal": axis in ("horizontal", "both"),
                "vertical": axis in ("vertical", "both"),
                "replace": True,  # Replace any existing lines with this source
                "method": detect_kwargs.get("method", "projection"),
            }

            # Handle threshold parameter
            if threshold == "auto":
                # Auto mode: use very low thresholds with max_lines constraints
                detect_params["peak_threshold_h"] = 0.0
                detect_params["peak_threshold_v"] = 0.0
                detect_params["max_lines_h"] = max_lines_h
                detect_params["max_lines_v"] = max_lines_v
            else:
                # Fixed threshold mode
                detect_params["peak_threshold_h"] = (
                    float(threshold) if axis in ("horizontal", "both") else 1.0
                )
                detect_params["peak_threshold_v"] = (
                    float(threshold) if axis in ("vertical", "both") else 1.0
                )
                detect_params["max_lines_h"] = max_lines_h
                detect_params["max_lines_v"] = max_lines_v

            # Add any additional detection parameters
            for key in [
                "min_gap_h",
                "min_gap_v",
                "binarization_method",
                "adaptive_thresh_block_size",
                "adaptive_thresh_C_val",
                "morph_op_h",
                "morph_kernel_h",
                "morph_op_v",
                "morph_kernel_v",
                "smoothing_sigma_h",
                "smoothing_sigma_v",
                "peak_width_rel_height",
            ]:
                if key in detect_kwargs:
                    detect_params[key] = detect_kwargs[key]

            # Perform the detection
            obj.detect_lines(**detect_params)

            # Now get the detected lines and use them
            if hasattr(obj, "lines"):
                lines = obj.lines
            elif hasattr(obj, "find_all"):
                lines = obj.find_all("line")
            else:
                lines = []

            # Filter by the source we just used
            lines = [
                l for l in lines if getattr(l, "source", None) == detect_params["source_label"]
            ]

        else:  # detection_method == 'vector' (default)
            # Get existing lines from the object
            if hasattr(obj, "lines"):
                lines = obj.lines
            elif hasattr(obj, "find_all"):
                lines = obj.find_all("line")
            else:
                logger.warning(f"Object {obj} has no lines or find_all method")
                lines = []

            # Filter by source if specified
            if source_label:
                lines = [l for l in lines if getattr(l, "source", None) == source_label]

        # Process lines (same logic for both methods)
        # Separate lines by orientation and collect with metadata for ranking
        h_line_data = []  # (y_coord, length, line_obj)
        v_line_data = []  # (x_coord, length, line_obj)

        for line in lines:
            if hasattr(line, "is_horizontal") and hasattr(line, "is_vertical"):
                if line.is_horizontal and axis in ("horizontal", "both"):
                    # Use the midpoint y-coordinate for horizontal lines
                    y = (line.top + line.bottom) / 2
                    # Calculate line length for ranking
                    length = getattr(
                        line, "width", abs(getattr(line, "x1", 0) - getattr(line, "x0", 0))
                    )
                    h_line_data.append((y, length, line))
                elif line.is_vertical and axis in ("vertical", "both"):
                    # Use the midpoint x-coordinate for vertical lines
                    x = (line.x0 + line.x1) / 2
                    # Calculate line length for ranking
                    length = getattr(
                        line, "height", abs(getattr(line, "bottom", 0) - getattr(line, "top", 0))
                    )
                    v_line_data.append((x, length, line))

        # Process horizontal lines
        if max_lines_h is not None and h_line_data:
            # Sort by length (longer lines are typically more significant)
            h_line_data.sort(key=lambda x: x[1], reverse=True)
            # Take the top N by length
            selected_h = h_line_data[:max_lines_h]
            # Extract just the coordinates and sort by position
            horizontals = sorted([coord for coord, _, _ in selected_h])
            logger.debug(
                f"Selected {len(horizontals)} horizontal lines from {len(h_line_data)} candidates"
            )
        else:
            # Use all horizontal lines (original behavior)
            horizontals = [coord for coord, _, _ in h_line_data]
            horizontals = sorted(list(set(horizontals)))

        # Process vertical lines
        if max_lines_v is not None and v_line_data:
            # Sort by length (longer lines are typically more significant)
            v_line_data.sort(key=lambda x: x[1], reverse=True)
            # Take the top N by length
            selected_v = v_line_data[:max_lines_v]
            # Extract just the coordinates and sort by position
            verticals = sorted([coord for coord, _, _ in selected_v])
            logger.debug(
                f"Selected {len(verticals)} vertical lines from {len(v_line_data)} candidates"
            )
        else:
            # Use all vertical lines (original behavior)
            verticals = [coord for coord, _, _ in v_line_data]
            verticals = sorted(list(set(verticals)))

        # Add outer guides if requested
        if outer and bounds:
            if axis in ("vertical", "both"):
                if not verticals or verticals[0] > bounds[0]:
                    verticals.insert(0, bounds[0])  # x0
                if not verticals or verticals[-1] < bounds[2]:
                    verticals.append(bounds[2])  # x1
            if axis in ("horizontal", "both"):
                if not horizontals or horizontals[0] > bounds[1]:
                    horizontals.insert(0, bounds[1])  # y0
                if not horizontals or horizontals[-1] < bounds[3]:
                    horizontals.append(bounds[3])  # y1

        # Remove duplicates and sort again
        verticals = sorted(list(set(verticals)))
        horizontals = sorted(list(set(horizontals)))

        return cls(verticals=verticals, horizontals=horizontals, context=obj, bounds=bounds)

    @classmethod
    def from_content(
        cls,
        obj: Union["Page", "Region", "FlowRegion"],
        axis: Literal["vertical", "horizontal"] = "vertical",
        markers: Union[str, List[str], "ElementCollection", None] = None,
        align: Literal["left", "right", "center", "between"] = "left",
        outer: bool = True,
        tolerance: float = 5,
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
            align: Where to place guides relative to found text
            outer: Whether to add guides at the boundaries
            tolerance: Maximum distance to search for text

        Returns:
            New Guides object aligned to text content
        """
        # Handle FlowRegion
        if hasattr(obj, "constituent_regions"):
            guides = cls(context=obj)
            
            # Process each constituent region
            for region in obj.constituent_regions:
                # Create guides for this specific region
                region_guides = cls.from_content(
                    region,
                    axis=axis,
                    markers=markers,
                    align=align,
                    outer=outer,
                    tolerance=tolerance
                )
                
                # Store in flow guides
                guides._flow_guides[region] = (
                    list(region_guides.vertical),
                    list(region_guides.horizontal)
                )
                
                # Add to unified view
                for v in region_guides.vertical:
                    guides._unified_vertical.append((v, region))
                for h in region_guides.horizontal:
                    guides._unified_horizontal.append((h, region))
            
            # Invalidate caches
            guides._vertical_cache = None
            guides._horizontal_cache = None
            
            return guides
        
        # Original single-region logic follows...
        guides_coords = []
        bounds = None

        # Get bounds from object
        if hasattr(obj, "bbox"):
            bounds = obj.bbox
        elif hasattr(obj, "x0"):
            bounds = (obj.x0, obj.top, obj.x1, obj.bottom)
        elif hasattr(obj, "width"):
            bounds = (0, 0, obj.width, obj.height)

        # Normalize markers to list of text strings
        marker_texts = _normalize_markers(markers, obj)

        # Find each marker and determine guide position
        for marker in marker_texts:
            if hasattr(obj, "find"):
                element = obj.find(f'text:contains("{marker}")')
                if element:
                    if axis == "vertical":
                        if align == "left":
                            guides_coords.append(element.x0)
                        elif align == "right":
                            guides_coords.append(element.x1)
                        elif align == "center":
                            guides_coords.append((element.x0 + element.x1) / 2)
                        elif align == "between":
                            # For between, collect left edges for processing later
                            guides_coords.append(element.x0)
                    else:  # horizontal
                        if align == "left":  # top for horizontal
                            guides_coords.append(element.top)
                        elif align == "right":  # bottom for horizontal
                            guides_coords.append(element.bottom)
                        elif align == "center":
                            guides_coords.append((element.top + element.bottom) / 2)
                        elif align == "between":
                            # For between, collect top edges for processing later
                            guides_coords.append(element.top)

        # Handle 'between' alignment - find midpoints between adjacent markers
        if align == "between" and len(guides_coords) >= 2:
            # We need to get the right and left edges of each marker
            marker_bounds = []
            for marker in marker_texts:
                if hasattr(obj, "find"):
                    element = obj.find(f'text:contains("{marker}")')
                    if element:
                        if axis == "vertical":
                            marker_bounds.append((element.x0, element.x1))
                        else:  # horizontal
                            marker_bounds.append((element.top, element.bottom))

            # Sort markers by their left edge (or top edge for horizontal)
            marker_bounds.sort(key=lambda x: x[0])

            # Create guides at midpoints between adjacent markers
            between_coords = []
            for i in range(len(marker_bounds) - 1):
                # Midpoint between right edge of current marker and left edge of next marker
                right_edge_current = marker_bounds[i][1]
                left_edge_next = marker_bounds[i + 1][0]
                midpoint = (right_edge_current + left_edge_next) / 2
                between_coords.append(midpoint)

            guides_coords = between_coords

        # Add outer guides if requested
        if outer and bounds:
            if axis == "vertical":
                guides_coords.insert(0, bounds[0])  # x0
                guides_coords.append(bounds[2])  # x1
            else:
                guides_coords.insert(0, bounds[1])  # y0
                guides_coords.append(bounds[3])  # y1

        # Remove duplicates and sort
        guides_coords = sorted(list(set(guides_coords)))

        # Create guides object
        if axis == "vertical":
            return cls(verticals=guides_coords, context=obj, bounds=bounds)
        else:
            return cls(horizontals=guides_coords, context=obj, bounds=bounds)

    @classmethod
    def from_whitespace(
        cls,
        obj: Union["Page", "Region", "FlowRegion"],
        axis: Literal["vertical", "horizontal", "both"] = "both",
        min_gap: float = 10,
    ) -> "Guides":
        """
        Create guides by detecting whitespace gaps.

        Args:
            obj: Page or Region to analyze
            min_gap: Minimum gap size to consider as whitespace
            axis: Which axes to analyze for gaps

        Returns:
            New Guides object positioned at whitespace gaps
        """
        # This is a placeholder - would need sophisticated gap detection
        logger.info("Whitespace detection not yet implemented, using divide instead")
        return cls.divide(obj, n=3, axis=axis)

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
            on_no_snap: Action when snapping fails ('warn', 'ignore', 'raise')

        Returns:
            Self for method chaining.
        """
        if not self.context:
            logger.warning("No context available for whitespace detection")
            return self

        # Handle FlowRegion case - collect all text elements across regions
        if self.is_flow_region:
            all_text_elements = []
            region_bounds = {}
            
            for region in self.context.constituent_regions:
                # Get text elements from this region
                if hasattr(region, "find_all"):
                    try:
                        text_elements = region.find_all("text", apply_exclusions=False)
                        elements = text_elements.elements if hasattr(text_elements, "elements") else text_elements
                        all_text_elements.extend(elements)
                        
                        # Store bounds for each region
                        if hasattr(region, "bbox"):
                            region_bounds[region] = region.bbox
                        elif hasattr(region, "x0"):
                            region_bounds[region] = (region.x0, region.top, region.x1, region.bottom)
                    except Exception as e:
                        logger.warning(f"Error getting text elements from region: {e}")
            
            if not all_text_elements:
                logger.warning("No text elements found across flow regions for whitespace detection")
                return self
            
            # Find whitespace gaps across all regions
            if axis == "vertical":
                gaps = self._find_vertical_whitespace_gaps(all_text_elements, min_gap, threshold)
                # Get all vertical guides across regions
                all_guides = []
                guide_to_region_map = {}  # Map guide coordinate to region
                for coord, region in self._unified_vertical:
                    all_guides.append(coord)
                    guide_to_region_map[coord] = region
                    
                if gaps and all_guides:
                    # Keep a copy of original guides to maintain mapping
                    original_guides = all_guides.copy()
                    
                    # Snap guides to gaps
                    self._snap_guides_to_gaps(all_guides, gaps, axis)
                    
                    # Update the unified view with snapped positions
                    self._unified_vertical = []
                    for i, new_coord in enumerate(all_guides):
                        # Find the original region for this guide using the original position
                        original_coord = original_guides[i]
                        region = guide_to_region_map[original_coord]
                        self._unified_vertical.append((new_coord, region))
                    
                    # Update individual region guides
                    for region in self._flow_guides:
                        region_verticals = []
                        for coord, r in self._unified_vertical:
                            if r == region:
                                region_verticals.append(coord)
                        self._flow_guides[region] = (
                            sorted(region_verticals),
                            self._flow_guides[region][1]
                        )
                    
                    # Invalidate cache
                    self._vertical_cache = None
                    
            elif axis == "horizontal":
                gaps = self._find_horizontal_whitespace_gaps(all_text_elements, min_gap, threshold)
                # Get all horizontal guides across regions
                all_guides = []
                guide_to_region_map = {}  # Map guide coordinate to region
                for coord, region in self._unified_horizontal:
                    all_guides.append(coord)
                    guide_to_region_map[coord] = region
                    
                if gaps and all_guides:
                    # Keep a copy of original guides to maintain mapping
                    original_guides = all_guides.copy()
                    
                    # Snap guides to gaps
                    self._snap_guides_to_gaps(all_guides, gaps, axis)
                    
                    # Update the unified view with snapped positions
                    self._unified_horizontal = []
                    for i, new_coord in enumerate(all_guides):
                        # Find the original region for this guide using the original position
                        original_coord = original_guides[i]
                        region = guide_to_region_map[original_coord]
                        self._unified_horizontal.append((new_coord, region))
                    
                    # Update individual region guides
                    for region in self._flow_guides:
                        region_horizontals = []
                        for coord, r in self._unified_horizontal:
                            if r == region:
                                region_horizontals.append(coord)
                        self._flow_guides[region] = (
                            self._flow_guides[region][0],
                            sorted(region_horizontals)
                        )
                    
                    # Invalidate cache
                    self._horizontal_cache = None
            
            else:
                raise ValueError("axis must be 'vertical' or 'horizontal'")
            
            return self
        
        # Original single-region logic
        # Get elements for trough detection
        text_elements = self._get_text_elements()
        if not text_elements:
            logger.warning("No text elements found for whitespace detection")
            return self

        if axis == "vertical":
            gaps = self._find_vertical_whitespace_gaps(text_elements, min_gap, threshold)
            if gaps:
                self._snap_guides_to_gaps(self.vertical.data, gaps, axis)
        elif axis == "horizontal":
            gaps = self._find_horizontal_whitespace_gaps(text_elements, min_gap, threshold)
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

        # Use context from self if available
        return Guides(
            verticals=combined_verticals,
            horizontals=combined_horizontals,
            context=self.context or other.context,
            bounds=self.bounds or other.bounds,
        )

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
            stack_direction = kwargs.get('stack_direction', 'vertical')
            stack_gap = kwargs.get('stack_gap', 5)
            stack_background_color = kwargs.get('stack_background_color', (255, 255, 255))
            
            # First, render all constituent regions without guides to get base images
            base_images = []
            region_infos = []  # Store region info for guide coordinate mapping
            
            for region in self.context.constituent_regions:
                try:
                    # Render region without guides
                    img = region.to_image(**kwargs)
                    if img:
                        base_images.append(img)
                        
                        # Calculate scaling factors for this region
                        scale_x = img.width / region.width
                        scale_y = img.height / region.height
                        
                        region_infos.append({
                            'region': region,
                            'img_width': img.width,
                            'img_height': img.height,
                            'scale_x': scale_x,
                            'scale_y': scale_y,
                            'pdf_x0': region.x0,
                            'pdf_top': region.top,
                            'pdf_x1': region.x1,
                            'pdf_bottom': region.bottom
                        })
                except Exception as e:
                    logger.warning(f"Failed to render region: {e}")
            
            if not base_images:
                raise ValueError("Failed to render any images for FlowRegion")
            
            # Calculate final canvas size based on stacking direction
            if stack_direction == "vertical":
                final_width = max(img.width for img in base_images)
                final_height = (
                    sum(img.height for img in base_images)
                    + (len(base_images) - 1) * stack_gap
                )
            else:  # horizontal
                final_width = (
                    sum(img.width for img in base_images)
                    + (len(base_images) - 1) * stack_gap
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
                    if info['pdf_x0'] <= v_coord <= info['pdf_x1']:
                        # This guide is within this region's x-bounds
                        # Convert PDF coordinate to pixel coordinate relative to the region
                        adjusted_x = v_coord - info['pdf_x0']
                        pixel_x = adjusted_x * info['scale_x'] + paste_x
                        
                        # Draw full-height line on canvas (not clipped to region)
                        if 0 <= pixel_x <= final_width:
                            x_pixel = int(pixel_x)
                            draw.line([(x_pixel, 0), (x_pixel, final_height - 1)], 
                                    fill=(0, 0, 255, 200), width=2)
                        break  # Only draw once per guide
            
            # Draw horizontal guides (red) - these extend through the full canvas width
            for h_coord in self.horizontal:
                # Find which region(s) this guide intersects
                for info, paste_x, paste_y in region_positions:
                    if info['pdf_top'] <= h_coord <= info['pdf_bottom']:
                        # This guide is within this region's y-bounds
                        # Convert PDF coordinate to pixel coordinate relative to the region
                        adjusted_y = h_coord - info['pdf_top']
                        pixel_y = adjusted_y * info['scale_y'] + paste_y
                        
                        # Draw full-width line on canvas (not clipped to region)
                        if 0 <= pixel_y <= final_height:
                            y_pixel = int(pixel_y)
                            draw.line([(0, y_pixel), (final_width - 1, y_pixel)], 
                                    fill=(255, 0, 0, 200), width=2)
                        break  # Only draw once per guide
            
            return canvas
        
        # Original single-region logic follows...
        # Determine what to display guides on
        target = on if on is not None else self.context

        # Handle string shortcuts
        if isinstance(target, str):
            if target == "page":
                if hasattr(self.context, "page"):
                    target = self.context.page
                elif hasattr(self.context, "_page"):
                    target = self.context._page
                else:
                    raise ValueError("Cannot resolve 'page' - context has no page attribute")
            else:
                raise ValueError(f"Unknown string target: {target}. Only 'page' is supported.")

        if target is None:
            raise ValueError("No target specified and no context available for guides display")

        # Prepare kwargs for image generation
        image_kwargs = kwargs.copy()

        # Always turn off highlights to avoid visual clutter
        image_kwargs["include_highlights"] = False

        # If target is a region-like object, crop to just that region
        if hasattr(target, "bbox") and hasattr(target, "page"):
            # This is likely a Region
            image_kwargs["crop"] = True

        # Get base image
        if hasattr(target, "to_image"):
            img = target.to_image(**image_kwargs)
        elif hasattr(target, "mode") and hasattr(target, "size"):
            # It's already a PIL Image
            img = target
        else:
            raise ValueError(f"Object {target} does not support to_image() and is not a PIL Image")

        if img is None:
            raise ValueError("Failed to generate base image")

        # Create a copy to draw on
        img = img.copy()
        draw = ImageDraw.Draw(img)

        # Determine scale factor for coordinate conversion
        if (
            hasattr(target, "width")
            and hasattr(target, "height")
            and not (hasattr(target, "mode") and hasattr(target, "size"))
        ):
            # target is a PDF object (Page/Region) with PDF coordinates
            scale_x = img.width / target.width
            scale_y = img.height / target.height

            # If we're showing guides on a region, we need to adjust coordinates
            # to be relative to the region's origin
            if hasattr(target, "bbox") and hasattr(target, "page"):
                # This is a Region - adjust guide coordinates to be relative to region
                region_x0, region_top = target.x0, target.top
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
        if hasattr(self, "on_no_snap"):
            if self.on_no_snap == "warn":
                logger.warning(message)
            elif self.on_no_snap == "raise":
                raise ValueError(message)
            # 'ignore' case: do nothing
        else:
            logger.warning(message)  # Default behavior

    def _find_vertical_whitespace_gaps(
        self, text_elements, min_gap: float, threshold: Union[float, str] = "auto"
    ) -> List[Tuple[float, float]]:
        """
        Find vertical whitespace gaps using bbox-based density analysis.
        Returns list of (start, end) tuples representing trough ranges.
        """
        if not self.bounds:
            return []

        x0, _, x1, _ = self.bounds
        width_pixels = int(x1 - x0)

        if width_pixels <= 0:
            return []

        # Create density histogram: count bbox overlaps per x-coordinate
        density = np.zeros(width_pixels)

        for element in text_elements:
            if not hasattr(element, "x0") or not hasattr(element, "x1"):
                continue

            # Clip coordinates to bounds
            elem_x0 = max(x0, element.x0) - x0
            elem_x1 = min(x1, element.x1) - x0

            if elem_x1 > elem_x0:
                start_px = int(elem_x0)
                end_px = int(elem_x1)
                density[start_px:end_px] += 1

        if density.max() == 0:
            return []

        # Determine the threshold value
        if threshold == "auto":
            # Auto mode: try different thresholds with step 0.05 until we have enough troughs
            guides_needing_troughs = len(
                [g for i, g in enumerate(self.vertical) if 0 < i < len(self.vertical) - 1]
            )
            if guides_needing_troughs == 0:
                threshold_val = 0.5  # Default when no guides need placement
            else:
                threshold_val = None
                for test_threshold in np.arange(0.1, 1.0, 0.05):
                    test_gaps = self._find_gaps_with_threshold(density, test_threshold, min_gap, x0)
                    if len(test_gaps) >= guides_needing_troughs:
                        threshold_val = test_threshold
                        logger.debug(
                            f"Auto threshold found: {test_threshold:.2f} (found {len(test_gaps)} troughs for {guides_needing_troughs} guides)"
                        )
                        break

                if threshold_val is None:
                    threshold_val = 0.8  # Fallback to permissive threshold
                    logger.debug(f"Auto threshold fallback to {threshold_val}")
        else:
            # Fixed threshold mode
            if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
                raise ValueError("threshold must be a number between 0.0 and 1.0, or 'auto'")
            threshold_val = float(threshold)

        return self._find_gaps_with_threshold(density, threshold_val, min_gap, x0)

    def _find_gaps_with_threshold(self, density, threshold_val, min_gap, x0):
        """Helper method to find gaps given a specific threshold value."""
        max_density = density.max()
        threshold_density = threshold_val * max_density

        # Smooth the density for better trough detection
        from scipy.ndimage import gaussian_filter1d

        smoothed_density = gaussian_filter1d(density.astype(float), sigma=1.0)

        # Find regions below threshold
        below_threshold = smoothed_density <= threshold_density

        # Find contiguous regions
        from scipy.ndimage import label as nd_label

        labeled_regions, num_regions = nd_label(below_threshold)

        gaps = []
        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            region_indices = np.where(region_mask)[0]

            if len(region_indices) == 0:
                continue

            start_px = region_indices[0]
            end_px = region_indices[-1] + 1

            # Convert back to PDF coordinates
            start_pdf = x0 + start_px
            end_pdf = x0 + end_px

            # Check minimum gap size
            if end_pdf - start_pdf >= min_gap:
                gaps.append((start_pdf, end_pdf))

        return gaps

    def _find_horizontal_whitespace_gaps(
        self, text_elements, min_gap: float, threshold: Union[float, str] = "auto"
    ) -> List[Tuple[float, float]]:
        """
        Find horizontal whitespace gaps using bbox-based density analysis.
        Returns list of (start, end) tuples representing trough ranges.
        """
        if not self.bounds:
            return []

        _, y0, _, y1 = self.bounds
        height_pixels = int(y1 - y0)

        if height_pixels <= 0:
            return []

        # Create density histogram: count bbox overlaps per y-coordinate
        density = np.zeros(height_pixels)

        for element in text_elements:
            if not hasattr(element, "top") or not hasattr(element, "bottom"):
                continue

            # Clip coordinates to bounds
            elem_top = max(y0, element.top) - y0
            elem_bottom = min(y1, element.bottom) - y0

            if elem_bottom > elem_top:
                start_px = int(elem_top)
                end_px = int(elem_bottom)
                density[start_px:end_px] += 1

        if density.max() == 0:
            return []

        # Determine the threshold value (same logic as vertical)
        if threshold == "auto":
            guides_needing_troughs = len(
                [g for i, g in enumerate(self.horizontal) if 0 < i < len(self.horizontal) - 1]
            )
            if guides_needing_troughs == 0:
                threshold_val = 0.5  # Default when no guides need placement
            else:
                threshold_val = None
                for test_threshold in np.arange(0.1, 1.0, 0.05):
                    test_gaps = self._find_gaps_with_threshold_horizontal(
                        density, test_threshold, min_gap, y0
                    )
                    if len(test_gaps) >= guides_needing_troughs:
                        threshold_val = test_threshold
                        logger.debug(
                            f"Auto threshold found: {test_threshold:.2f} (found {len(test_gaps)} troughs for {guides_needing_troughs} guides)"
                        )
                        break

                if threshold_val is None:
                    threshold_val = 0.8  # Fallback to permissive threshold
                    logger.debug(f"Auto threshold fallback to {threshold_val}")
        else:
            # Fixed threshold mode
            if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
                raise ValueError("threshold must be a number between 0.0 and 1.0, or 'auto'")
            threshold_val = float(threshold)

        return self._find_gaps_with_threshold_horizontal(density, threshold_val, min_gap, y0)

    def _find_gaps_with_threshold_horizontal(self, density, threshold_val, min_gap, y0):
        """Helper method to find horizontal gaps given a specific threshold value."""
        max_density = density.max()
        threshold_density = threshold_val * max_density

        # Smooth the density for better trough detection
        from scipy.ndimage import gaussian_filter1d

        smoothed_density = gaussian_filter1d(density.astype(float), sigma=1.0)

        # Find regions below threshold
        below_threshold = smoothed_density <= threshold_density

        # Find contiguous regions
        from scipy.ndimage import label as nd_label

        labeled_regions, num_regions = nd_label(below_threshold)

        gaps = []
        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            region_indices = np.where(region_mask)[0]

            if len(region_indices) == 0:
                continue

            start_px = region_indices[0]
            end_px = region_indices[-1] + 1

            # Convert back to PDF coordinates
            start_pdf = y0 + start_px
            end_pdf = y0 + end_px

            # Check minimum gap size
            if end_pdf - start_pdf >= min_gap:
                gaps.append((start_pdf, end_pdf))

        return gaps

    def _find_vertical_element_gaps(
        self, text_elements, min_gap: float
    ) -> List[Tuple[float, float]]:
        """
        Find vertical whitespace gaps using text element spacing analysis.
        Returns list of (start, end) tuples representing trough ranges.
        """
        if not self.bounds or not text_elements:
            return []

        x0, _, x1, _ = self.bounds

        # Get all element right and left edges
        element_edges = []
        for element in text_elements:
            if not hasattr(element, "x0") or not hasattr(element, "x1"):
                continue
            # Only include elements that overlap vertically with our bounds
            if hasattr(element, "top") and hasattr(element, "bottom"):
                if element.bottom < self.bounds[1] or element.top > self.bounds[3]:
                    continue
            element_edges.extend([element.x0, element.x1])

        if not element_edges:
            return []

        # Sort edges and find gaps
        element_edges = sorted(set(element_edges))

        trough_ranges = []
        for i in range(len(element_edges) - 1):
            gap_start = element_edges[i]
            gap_end = element_edges[i + 1]
            gap_width = gap_end - gap_start

            if gap_width >= min_gap:
                # Check if this gap actually contains no text (is empty space)
                gap_has_text = False
                for element in text_elements:
                    if (
                        hasattr(element, "x0")
                        and hasattr(element, "x1")
                        and element.x0 < gap_end
                        and element.x1 > gap_start
                    ):
                        gap_has_text = True
                        break

                if not gap_has_text:
                    trough_ranges.append((gap_start, gap_end))

        return trough_ranges

    def _find_horizontal_element_gaps(
        self, text_elements, min_gap: float
    ) -> List[Tuple[float, float]]:
        """
        Find horizontal whitespace gaps using text element spacing analysis.
        Returns list of (start, end) tuples representing trough ranges.
        """
        if not self.bounds or not text_elements:
            return []

        _, y0, _, y1 = self.bounds

        # Get all element top and bottom edges
        element_edges = []
        for element in text_elements:
            if not hasattr(element, "top") or not hasattr(element, "bottom"):
                continue
            # Only include elements that overlap horizontally with our bounds
            if hasattr(element, "x0") and hasattr(element, "x1"):
                if element.x1 < self.bounds[0] or element.x0 > self.bounds[2]:
                    continue
            element_edges.extend([element.top, element.bottom])

        if not element_edges:
            return []

        # Sort edges and find gaps
        element_edges = sorted(set(element_edges))

        trough_ranges = []
        for i in range(len(element_edges) - 1):
            gap_start = element_edges[i]
            gap_end = element_edges[i + 1]
            gap_width = gap_end - gap_start

            if gap_width >= min_gap:
                # Check if this gap actually contains no text (is empty space)
                gap_has_text = False
                for element in text_elements:
                    if (
                        hasattr(element, "top")
                        and hasattr(element, "bottom")
                        and element.top < gap_end
                        and element.bottom > gap_start
                    ):
                        gap_has_text = True
                        break

                if not gap_has_text:
                    trough_ranges.append((gap_start, gap_end))

        return trough_ranges

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
        target: Optional[Union["Page", "Region"]] = None,
        source: str = "guides",
        cell_padding: float = 0.5,
        include_outer_boundaries: bool = False,
    ) -> Dict[str, int]:
        """
        Create table structure (table, rows, columns, cells) from guide coordinates.

        Args:
            target: Page or Region to create regions on (uses self.context if None)
            source: Source label for created regions (for identification)
            cell_padding: Internal padding for cell regions in points
            include_outer_boundaries: Whether to add boundaries at edges if missing

        Returns:
            Dictionary with counts: {'table': 1, 'rows': N, 'columns': M, 'cells': N*M}
        """
        # Handle FlowRegion case
        if self.is_flow_region:
            if not self._flow_guides:
                raise ValueError("No guides available for FlowRegion")
            
            total_counts = {"table": 0, "rows": 0, "columns": 0, "cells": 0}
            
            # Build grid on each constituent region
            for region in self.context.constituent_regions:
                if region in self._flow_guides:
                    verticals, horizontals = self._flow_guides[region]
                    
                    # Create temporary Guides object for this region
                    region_guides = Guides(
                        verticals=verticals,
                        horizontals=horizontals,
                        context=region
                    )
                    
                    # Build grid on this region
                    try:
                        counts = region_guides.build_grid(
                            target=region,  # Use the region itself as target
                            source=source,
                            cell_padding=cell_padding,
                            include_outer_boundaries=include_outer_boundaries
                        )
                        
                        # Aggregate counts
                        for key in total_counts:
                            total_counts[key] += counts[key]
                    except Exception as e:
                        logger.warning(f"Failed to build grid on region: {e}")
            
            logger.info(
                f"Created {total_counts['table']} tables, {total_counts['rows']} rows, "
                f"{total_counts['columns']} columns, and {total_counts['cells']} cells "
                f"from guides across {len(self._flow_guides)} regions"
            )
            
            return total_counts
        
        # Original single-region logic follows...
        # Determine target object
        target_obj = target or self.context
        if not target_obj:
            raise ValueError("No target object available. Provide target parameter or context.")

        # Get the page for creating regions
        if hasattr(target_obj, "x0") and hasattr(
            target_obj, "top"
        ):  # Region (has bbox coordinates)
            page = target_obj._page
            origin_x, origin_y = target_obj.x0, target_obj.top
            context_width, context_height = target_obj.width, target_obj.height
        elif hasattr(target_obj, "_element_mgr") or hasattr(target_obj, "width"):  # Page
            page = target_obj
            origin_x, origin_y = 0.0, 0.0
            context_width, context_height = page.width, page.height
        else:
            raise ValueError(f"Target object {target_obj} is not a Page or Region")

        element_manager = page._element_mgr

        # Setup boundaries
        row_boundaries = list(self.horizontal)
        col_boundaries = list(self.vertical)

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

        # ------------------------------------------------------------------
        # Clean-up: remove any previously created grid regions (table, rows,
        # columns, cells) that were generated by the same `source` label and
        # overlap the area we are about to populate.  This prevents the page's
        # `ElementManager` from accumulating stale/duplicate regions when the
        # user rebuilds the grid multiple times.
        # ------------------------------------------------------------------
        try:
            # Bounding box of the grid we are about to create
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

                # Collect existing regions that match the source & region types
                regions_to_remove = [
                    r
                    for r in element_manager.regions
                    if getattr(r, "source", None) == source
                    and getattr(r, "region_type", None)
                    in {"table", "table_row", "table_column", "table_cell"}
                    and hasattr(r, "bbox")
                    and _bbox_overlap(r.bbox, grid_bbox)
                ]

                for r in regions_to_remove:
                    element_manager.remove_element(r, element_type="regions")

                if regions_to_remove:
                    logger.debug(
                        f"Removed {len(regions_to_remove)} existing grid region(s) prior to rebuild"
                    )
        except Exception as cleanup_err:  # pragma: no cover – cleanup must never crash
            logger.warning(f"Grid cleanup failed: {cleanup_err}")

        logger.debug(
            f"Building grid with {len(row_boundaries)} row and {len(col_boundaries)} col boundaries"
        )

        # Track creation counts
        counts = {"table": 0, "rows": 0, "columns": 0, "cells": 0}

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
            element_manager.add_element(table_region, element_type="regions")
            counts["table"] = 1

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
                element_manager.add_element(row_region, element_type="regions")
                counts["rows"] += 1

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
                element_manager.add_element(col_region, element_type="regions")
                counts["columns"] += 1

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
                    element_manager.add_element(cell_region, element_type="regions")
                    counts["cells"] += 1

        logger.info(
            f"Created {counts['table']} table, {counts['rows']} rows, "
            f"{counts['columns']} columns, and {counts['cells']} cells from guides"
        )

        return counts

    def __repr__(self) -> str:
        """String representation of the guides."""
        return (
            f"Guides(verticals={len(self.vertical)}, "
            f"horizontals={len(self.horizontal)}, "
            f"cells={len(self.get_cells())})"
        )

    def _get_text_elements(self):
        """Get text elements from the context."""
        if not self.context:
            return []

        # Handle FlowRegion context
        if self.is_flow_region:
            all_text_elements = []
            for region in self.context.constituent_regions:
                if hasattr(region, "find_all"):
                    try:
                        text_elements = region.find_all("text", apply_exclusions=False)
                        elements = (
                            text_elements.elements if hasattr(text_elements, "elements") else text_elements
                        )
                        all_text_elements.extend(elements)
                    except Exception as e:
                        logger.warning(f"Error getting text elements from region: {e}")
            return all_text_elements

        # Original single-region logic
        # Get text elements from the context
        if hasattr(self.context, "find_all"):
            try:
                text_elements = self.context.find_all("text", apply_exclusions=False)
                return (
                    text_elements.elements if hasattr(text_elements, "elements") else text_elements
                )
            except Exception as e:
                logger.warning(f"Error getting text elements: {e}")
                return []
        else:
            logger.warning("Context does not support text element search")
            return []

    # -------------------------------------------------------------------------
    # Instance methods for fluent chaining (avoid name conflicts with class methods)
    # -------------------------------------------------------------------------

    def add_content(
        self,
        axis: Literal["vertical", "horizontal"] = "vertical",
        markers: Union[str, List[str], "ElementCollection", None] = None,
        obj: Optional[Union["Page", "Region"]] = None,
        align: Literal["left", "right", "center", "between"] = "left",
        outer: bool = True,
        tolerance: float = 5,
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
            outer: Whether to add outer boundary guides
            tolerance: Tolerance for snapping to element edges

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
            detection_method: 'vector' (use existing LineElements) or 'pixels' (detect from image)
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
