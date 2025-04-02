"""
Base Element class for natural-pdf.
"""
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union, Tuple
from PIL import Image

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.region import Region
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.collections import ElementCollection


class DirectionalMixin:
    """
    Mixin class providing directional methods for both Element and Region classes.
    """
    
    def _direction(self, direction: str, size: Optional[float] = None,
                   cross_size: str = "full", include_element: bool = False,
                   until: Optional[str] = None, include_endpoint: bool = True, **kwargs) -> 'Region':
        """
        Protected helper method to create a region in a specified direction relative to this element/region.

        Args:
            direction: 'left', 'right', 'above', or 'below'
            size: Size in the primary direction (width for horizontal, height for vertical)
            cross_size: Size in the cross direction ('full' or 'element')
            include_element: Whether to include this element/region's area in the result
            until: Optional selector string to specify a boundary element
            include_endpoint: Whether to include the boundary element found by 'until'
            **kwargs: Additional parameters for the 'until' selector search

        Returns:
            Region object
        """
        import math # Use math.inf for infinity

        is_horizontal = direction in ('left', 'right')
        is_positive = direction in ('right', 'below') # right/below are positive directions
        pixel_offset = 1 # Offset for excluding elements/endpoints

        # 1. Determine initial boundaries based on direction and include_element
        if is_horizontal:
            # Initial cross-boundaries (vertical)
            y0 = 0 if cross_size == "full" else self.top
            y1 = self.page.height if cross_size == "full" else self.bottom

            # Initial primary boundaries (horizontal)
            if is_positive: # right
                x0_initial = self.x0 if include_element else self.x1 + pixel_offset
                x1_initial = self.x1 # This edge moves
            else: # left
                x0_initial = self.x0 # This edge moves
                x1_initial = self.x1 if include_element else self.x0 - pixel_offset
        else: # Vertical
            # Initial cross-boundaries (horizontal)
            x0 = 0 if cross_size == "full" else self.x0
            x1 = self.page.width if cross_size == "full" else self.x1

            # Initial primary boundaries (vertical)
            if is_positive: # below
                y0_initial = self.top if include_element else self.bottom + pixel_offset
                y1_initial = self.bottom # This edge moves
            else: # above
                y0_initial = self.top # This edge moves
                y1_initial = self.bottom if include_element else self.top - pixel_offset

        # 2. Calculate the final primary boundary, considering 'size' or page limits
        if is_horizontal:
            if is_positive: # right
                x1_final = min(self.page.width, x1_initial + (size if size is not None else (self.page.width - x1_initial)))
                x0_final = x0_initial
            else: # left
                x0_final = max(0, x0_initial - (size if size is not None else x0_initial))
                x1_final = x1_initial
        else: # Vertical
            if is_positive: # below
                y1_final = min(self.page.height, y1_initial + (size if size is not None else (self.page.height - y1_initial)))
                y0_final = y0_initial
            else: # above
                y0_final = max(0, y0_initial - (size if size is not None else y0_initial))
                y1_final = y1_initial

        # 3. Handle 'until' selector if provided
        target = None
        if until:
            all_matches = self.page.find_all(until, **kwargs)
            matches_in_direction = []

            # Filter and sort matches based on direction
            if direction == 'above':
                matches_in_direction = [m for m in all_matches if m.bottom <= self.top]
                matches_in_direction.sort(key=lambda e: e.bottom, reverse=True)
            elif direction == 'below':
                matches_in_direction = [m for m in all_matches if m.top >= self.bottom]
                matches_in_direction.sort(key=lambda e: e.top)
            elif direction == 'left':
                matches_in_direction = [m for m in all_matches if m.x1 <= self.x0]
                matches_in_direction.sort(key=lambda e: e.x1, reverse=True)
            elif direction == 'right':
                matches_in_direction = [m for m in all_matches if m.x0 >= self.x1]
                matches_in_direction.sort(key=lambda e: e.x0)

            if matches_in_direction:
                target = matches_in_direction[0]

                # Adjust the primary boundary based on the target
                if is_horizontal:
                    if is_positive: # right
                        x1_final = target.x1 if include_endpoint else target.x0 - pixel_offset
                    else: # left
                        x0_final = target.x0 if include_endpoint else target.x1 + pixel_offset
                else: # Vertical
                    if is_positive: # below
                        y1_final = target.bottom if include_endpoint else target.top - pixel_offset
                    else: # above
                        y0_final = target.top if include_endpoint else target.bottom + pixel_offset

                # Adjust cross boundaries if cross_size is 'element'
                if cross_size == "element":
                    if is_horizontal: # Adjust y0, y1
                        target_y0 = target.top if include_endpoint else target.bottom # Use opposite boundary if excluding
                        target_y1 = target.bottom if include_endpoint else target.top
                        y0 = min(y0, target_y0)
                        y1 = max(y1, target_y1)
                    else: # Adjust x0, x1
                        target_x0 = target.x0 if include_endpoint else target.x1 # Use opposite boundary if excluding
                        target_x1 = target.x1 if include_endpoint else target.x0
                        x0 = min(x0, target_x0)
                        x1 = max(x1, target_x1)

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
        result.includes_source = include_element
        # Optionally store the boundary element if found
        if target:
            result.boundary_element = target

        return result

    def above(self, height: Optional[float] = None, width: str = "full", include_element: bool = False,
             until: Optional[str] = None, include_endpoint: bool = True, **kwargs) -> 'Region':
        """
        Select region above this element/region.
        
        Args:
            height: Height of the region above, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_element: Whether to include this element/region in the result (default: False)
            until: Optional selector string to specify an upper boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters
            
        Returns:
            Region object representing the area above
        """
        return self._direction(
            direction='above',
            size=height,
            cross_size=width,
            include_element=include_element,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs
        )

    def below(self, height: Optional[float] = None, width: str = "full", include_element: bool = False,
              until: Optional[str] = None, include_endpoint: bool = True, **kwargs) -> 'Region':
        """
        Select region below this element/region.
        
        Args:
            height: Height of the region below, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_element: Whether to include this element/region in the result (default: False)
            until: Optional selector string to specify a lower boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters
            
        Returns:
            Region object representing the area below
        """
        return self._direction(
            direction='below',
            size=height,
            cross_size=width,
            include_element=include_element,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs
        )

    def left(self, width: Optional[float] = None, height: str = "full", include_element: bool = False,
             until: Optional[str] = None, include_endpoint: bool = True, **kwargs) -> 'Region':
        """
        Select region to the left of this element/region.
        
        Args:
            width: Width of the region to the left, in points
            height: Height mode - "full" for full page height or "element" for element height
            include_element: Whether to include this element/region in the result (default: False)
            until: Optional selector string to specify a left boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters
            
        Returns:
            Region object representing the area to the left
        """
        return self._direction(
            direction='left',
            size=width,
            cross_size=height,
            include_element=include_element,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs
        )

    def right(self, width: Optional[float] = None, height: str = "full", include_element: bool = False,
              until: Optional[str] = None, include_endpoint: bool = True, **kwargs) -> 'Region':
        """
        Select region to the right of this element/region.
        
        Args:
            width: Width of the region to the right, in points
            height: Height mode - "full" for full page height or "element" for element height
            include_element: Whether to include this element/region in the result (default: False)
            until: Optional selector string to specify a right boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters
            
        Returns:
            Region object representing the area to the right
        """
        return self._direction(
            direction='right',
            size=width,
            cross_size=height,
            include_element=include_element,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs
        )

    def expand(self, 
              left: float = 0, 
              right: float = 0, 
              top_expand: float = 0,  # Renamed to avoid conflict
              bottom_expand: float = 0,  # Renamed to avoid conflict
              width_factor: float = 1.0,
              height_factor: float = 1.0,
              # Keep original parameter names for backward compatibility
              top: float = None,
              bottom: float = None) -> 'Region':
        """
        Create a new region expanded from this element/region.
        
        Args:
            left: Amount to expand left edge (positive value expands leftwards)
            right: Amount to expand right edge (positive value expands rightwards)
            top_expand: Amount to expand top edge (positive value expands upwards)
            bottom_expand: Amount to expand bottom edge (positive value expands downwards)
            width_factor: Factor to multiply width by (applied after absolute expansion)
            height_factor: Factor to multiply height by (applied after absolute expansion)
            top: (DEPRECATED, use top_expand) Amount to expand top edge (upward)
            bottom: (DEPRECATED, use bottom_expand) Amount to expand bottom edge (downward)
            
        Returns:
            New expanded Region object
        """
        # Start with current coordinates
        new_x0 = self.x0
        new_x1 = self.x1
        new_top = self.top
        new_bottom = self.bottom
        
        # Handle the deprecated parameter names for backward compatibility
        if top is not None:
            top_expand = top
        if bottom is not None:
            bottom_expand = bottom
            
        # Apply absolute expansions first
        new_x0 -= left
        new_x1 += right
        new_top -= top_expand  # Expand upward (decrease top coordinate)
        new_bottom += bottom_expand  # Expand downward (increase bottom coordinate)
        
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
        if new_x0 > new_x1: new_x0 = new_x1 = (new_x0 + new_x1) / 2
        if new_top > new_bottom: new_top = new_bottom = (new_top + new_bottom) / 2

        # Create new region with expanded bbox
        from natural_pdf.elements.region import Region
        new_region = Region(self.page, (new_x0, new_top, new_x1, new_bottom))
                
        return new_region


class Element(DirectionalMixin):
    """
    Base class for all PDF elements.
    
    This class provides common properties and methods for all PDF elements,
    such as text, rectangles, lines, etc.
    """
    
    def __init__(self, obj: Dict[str, Any], page: 'Page'):
        """
        Initialize base element.
        
        Args:
            obj: The underlying pdfplumber object
            page: The parent Page object
        """
        self._obj = obj
        self._page = page
        
    @property
    def type(self) -> str:
        """Element type."""
        return self._obj.get('object_type', 'unknown')
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box (x0, top, x1, bottom)."""
        return (self.x0, self.top, self.x1, self.bottom)
    
    @property
    def x0(self) -> float:
        """Left x-coordinate."""
        if self.has_polygon:
            return min(pt[0] for pt in self.polygon)
        return self._obj.get('x0', 0)
    
    @property
    def top(self) -> float:
        """Top y-coordinate."""
        if self.has_polygon:
            return min(pt[1] for pt in self.polygon)
        return self._obj.get('top', 0)
    
    @property
    def x1(self) -> float:
        """Right x-coordinate."""
        if self.has_polygon:
            return max(pt[0] for pt in self.polygon)
        return self._obj.get('x1', 0)
    
    @property
    def bottom(self) -> float:
        """Bottom y-coordinate."""
        if self.has_polygon:
            return max(pt[1] for pt in self.polygon)
        return self._obj.get('bottom', 0)
    
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
        return ('polygon' in self._obj and self._obj['polygon'] and len(self._obj['polygon']) >= 3) or hasattr(self, '_polygon')
    
    @property
    def polygon(self) -> List[Tuple[float, float]]:
        """Get polygon coordinates if available, otherwise return rectangle corners."""
        if hasattr(self, '_polygon') and self._polygon:
            return self._polygon
        elif 'polygon' in self._obj and self._obj['polygon']:
            return self._obj['polygon']
        else:
            # Create rectangle corners as fallback
            return [
                (self._obj.get('x0', 0), self._obj.get('top', 0)),        # top-left
                (self._obj.get('x1', 0), self._obj.get('top', 0)),        # top-right
                (self._obj.get('x1', 0), self._obj.get('bottom', 0)),     # bottom-right
                (self._obj.get('x0', 0), self._obj.get('bottom', 0))      # bottom-left
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
    def page(self) -> 'Page':
        """Get the parent page."""
        return self._page
        
    def next(self, selector: Optional[str] = None, limit: int = 10, apply_exclusions: bool = True, **kwargs) -> Optional['Element']:
        """
        Find next element in reading order.
        
        Args:
            selector: Optional selector to filter by
            limit: Maximum number of elements to search through (default: 10)
            apply_exclusions: Whether to apply exclusion regions (default: True)
            **kwargs: Additional parameters
            
        Returns:
            Next element or None if not found
        """
        # Get all elements in reading order
        all_elements = self.page.find_all('*', apply_exclusions=apply_exclusions)
        
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
            candidates = all_elements[idx+1:]
            # Limit search range for performance
            candidates = candidates[:limit] if limit else candidates
            
            # Find matching elements
            from natural_pdf.elements.collections import ElementCollection
            matches = ElementCollection(candidates).find_all(selector, **kwargs)
            return matches[0] if matches else None
        elif idx + 1 < len(all_elements):
            # No selector, just return the next element
            return all_elements[idx + 1]
        
        return None
    
    def prev(self, selector: Optional[str] = None, limit: int = 10, apply_exclusions: bool = True, **kwargs) -> Optional['Element']:
        """
        Find previous element in reading order.
        
        Args:
            selector: Optional selector to filter by
            limit: Maximum number of elements to search through (default: 10)
            apply_exclusions: Whether to apply exclusion regions (default: True)
            **kwargs: Additional parameters
            
        Returns:
            Previous element or None if not found
        """
        # Get all elements in reading order
        all_elements = self.page.find_all('*', apply_exclusions=apply_exclusions)
        
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
            
            # Find matching elements using ElementCollection
            from natural_pdf.elements.collections import ElementCollection
            matches = ElementCollection(candidates).find_all(selector, **kwargs)
            return matches[0] if matches else None # find_all returns a collection
        elif idx > 0:
            # No selector, just return the previous element
            return all_elements[idx - 1]
        
        return None
    
    def nearest(self, selector: str, max_distance: Optional[float] = None, apply_exclusions: bool = True, **kwargs) -> Optional['Element']:
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
            distance = ((match_center_x - self_center_x) ** 2 + 
                        (match_center_y - self_center_y) ** 2) ** 0.5
            
            # Filter by max_distance if specified
            if max_distance is None or distance <= max_distance:
                distances.append((match, distance))
        
        # Sort by distance and return the closest
        if distances:
            distances.sort(key=lambda x: x[1])
            return distances[0][0]
            
        return None
    
    def until(self, selector: str, include_endpoint: bool = True, width: str = "element", **kwargs) -> 'Region':
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
                bottom = target.bottom if include_endpoint else target.top - 1  # Subtract 1 pixel when excluding
            else:  # Target is above this element
                top = target.top if include_endpoint else target.bottom + 1  # Add 1 pixel when excluding
                bottom = self.bottom
            return Region(self.page, (x0, top, x1, bottom))
            
        # Otherwise use element-based width
        # Determine the correct order for creating the region
        # If the target is below this element (normal reading order)
        if target.top >= self.bottom:
            x0 = min(self.x0, target.x0 if include_endpoint else target.x1)
            x1 = max(self.x1, target.x1 if include_endpoint else target.x0)
            top = self.top
            bottom = target.bottom if include_endpoint else target.top - 1  # Subtract 1 pixel when excluding
        # If the target is above this element (reverse reading order)
        elif target.bottom <= self.top:
            x0 = min(self.x0, target.x0 if include_endpoint else target.x1)
            x1 = max(self.x1, target.x1 if include_endpoint else target.x0)
            top = target.top if include_endpoint else target.bottom + 1  # Add 1 pixel when excluding
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
    
    def highlight(self, 
                 label: Optional[str] = None,
                 color: Optional[Union[Tuple, str]] = None, # Allow string color
                 use_color_cycling: bool = False,
                 include_attrs: Optional[List[str]] = None,
                 existing: str = 'append') -> 'Element':
        """
        Highlight this element on the page.
        
        Args:
            label: Optional label for the highlight
            color: Color tuple/string for the highlight, or None to use automatic color
            use_color_cycling: Force color cycling even with no label (default: False)
            include_attrs: List of attribute names to display on the highlight (e.g., ['confidence', 'type'])
            existing: How to handle existing highlights - 'append' (default) or 'replace'
            
        Returns:
            Self for method chaining
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
            "include_attrs": include_attrs,
            "existing": existing
        }

        # Call the appropriate service method based on geometry
        if self.has_polygon:
            highlight_args["polygon"] = self.polygon
            highlighter.add_polygon(**highlight_args)
        else:
            highlight_args["bbox"] = self.bbox
            highlighter.add(**highlight_args)

        return self
    
    def show(self, 
            scale: float = 2.0, 
            labels: bool = True,
            legend_position: str = 'right',
            color: Optional[Union[Tuple, str]] = "red", # Default color for single element
            label: Optional[str] = None) -> Optional['Image.Image']:
        """
        Show the page with only this element highlighted temporarily.

        Args:
            scale: Scale factor for rendering
            labels: Whether to include a legend for the highlight
            legend_position: Position of the legend
            color: Color to highlight this element (default: red)
            label: Optional label for this element in the legend

        Returns:
            PIL Image of the page with only this element highlighted, or None if error.
        """
        if not hasattr(self, 'page') or not self.page:
            logger.warning(f"Cannot show element, missing 'page' attribute: {self}")
            return None
        if not hasattr(self.page, '_highlighter') or not self.page._highlighter:
             logger.warning(f"Cannot show element, page lacks highlighter service: {self}")
             return None

        service = self.page._highlighter

        # Determine the label if not provided
        display_label = label if label is not None else f"{self.__class__.__name__}"

        # Prepare temporary highlight data for just this element
        temp_highlight_data = {
            "page_index": self.page.index,
            "bbox": self.bbox if not self.has_polygon else None,
            "polygon": self.polygon if self.has_polygon else None,
            "color": color, # Use provided or default color
            "label": display_label,
            "use_color_cycling": False # Explicitly false for single preview
        }

        # Check if we actually got geometry data
        if temp_highlight_data['bbox'] is None and temp_highlight_data['polygon'] is None:
             logger.warning(f"Cannot show element, failed to get bbox or polygon: {self}")
             return None

        # Use render_preview to show only this highlight
        try:
            return service.render_preview(
                page_index=self.page.index,
                temporary_highlights=[temp_highlight_data],
                scale=scale,
                labels=labels,
                legend_position=legend_position
            )
        except Exception as e:
            logger.error(f"Error calling render_preview for element {self}: {e}", exc_info=True)
            return None
    
    def save(self, 
            filename: str, 
            scale: float = 2.0, 
            labels: bool = True,
            legend_position: str = 'right') -> None:
        """
        Save the page with this element highlighted to an image file.
        
        Args:
            filename: Path to save the image to
            scale: Scale factor for rendering
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            
        Returns:
            Self for method chaining
        """
        # Save the highlighted image
        self.page.save_image(filename, scale=scale, labels=labels, legend_position=legend_position)
        return self
    
    # Note: save_image method removed in favor of save()
        
    def __repr__(self) -> str:
        """String representation of the element."""
        return f"<{self.__class__.__name__} bbox={self.bbox}>"

    def find(self, selector: str, apply_exclusions=True, **kwargs) -> Optional['Element']:
        """
        Find first element within this element's bounds matching the selector.
        Creates a temporary region to perform the search.
        
        Args:
            selector: CSS-like selector string
            apply_exclusions: Whether to apply exclusion regions
            **kwargs: Additional parameters for element filtering
            
        Returns:
            First matching element or None
        """
        from natural_pdf.elements.region import Region

        # Create a temporary region from this element's bounds
        temp_region = Region(self.page, self.bbox)
        return temp_region.find(selector, apply_exclusions=apply_exclusions, **kwargs)

    def find_all(self, selector: str, apply_exclusions=True, **kwargs) -> 'ElementCollection':
        """
        Find all elements within this element's bounds matching the selector.
        Creates a temporary region to perform the search.
        
        Args:
            selector: CSS-like selector string
            apply_exclusions: Whether to apply exclusion regions
            **kwargs: Additional parameters for element filtering
            
        Returns:
            ElementCollection with matching elements
        """
        from natural_pdf.elements.region import Region

        # Create a temporary region from this element's bounds
        temp_region = Region(self.page, self.bbox)
        return temp_region.find_all(selector, apply_exclusions=apply_exclusions, **kwargs)