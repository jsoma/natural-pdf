"""
Base Element class for natural-pdf.
"""
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union, Tuple
from PIL import Image

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.region import Region


class Element:
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
        
    def above(self, height: Optional[float] = None, width: str = "full", include_element: bool = False, 
             until: Optional[str] = None, include_endpoint: bool = True, **kwargs) -> 'Region':
        """
        Select region above this element.
        
        Args:
            height: Height of the region above, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_element: Whether to include this element in the region (default: False)
            until: Optional selector string to specify an upper boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters
            
        Returns:
            Region object representing the area above
        """
        from natural_pdf.elements.region import Region
        
        # Determine bottom boundary based on include_element
        bottom = self.bottom if include_element else self.top - 1  # Subtract 1 pixel offset to create a gap
            
        # Calculate initial bounding box for region
        if width == "full":
            x0 = 0
            x1 = self.page.width
        elif width == "element":
            x0 = self.x0
            x1 = self.x1
        else:
            raise ValueError("Width must be 'full' or 'element'")
            
        # If an "until" selector is specified, find the target element
        if until:
            # Need to find all matches and find the first one above this element
            # instead of just page.find() which might return any match
            all_matches = self.page.find_all(until, **kwargs)
            
            # Sort by vertical position (bottom to top)
            matches_above = [m for m in all_matches if m.bottom <= self.top]
            matches_above.sort(key=lambda e: e.bottom, reverse=True)
            
            if matches_above:
                # Use the first match above this element (closest one)
                target = matches_above[0]
                
                # Target is above this element - use it for the top boundary
                top = target.top if include_endpoint else target.bottom + 1  # Add 1 pixel offset when excluding
                
                # Use the selector match for width if not using full width
                if width == "element":
                    x0 = min(x0, target.x0 if include_endpoint else target.x1)
                    x1 = max(x1, target.x1 if include_endpoint else target.x0)
            else:
                # No targets found above this element - use requested height
                top = max(0, bottom - (height or bottom))
        else:
            # No "until" selector - use requested height
            top = max(0, bottom - (height or bottom))
        
        bbox = (x0, top, x1, bottom)
        region = Region(self.page, bbox)
        region.source_element = self  # Reference to element that created this region
        region.includes_source = include_element  # Whether region includes the source element
        return region
    
    def below(self, height: Optional[float] = None, width: str = "full", include_element: bool = False,
              until: Optional[str] = None, include_endpoint: bool = True, **kwargs) -> 'Region':
        """
        Select region below this element.
        
        Args:
            height: Height of the region below, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_element: Whether to include this element in the region (default: False)
            until: Optional selector string to specify a lower boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters
            
        Returns:
            Region object representing the area below
        """
        from natural_pdf.elements.region import Region
        
        # Determine top boundary based on include_element
        top = self.top if include_element else self.bottom + 1  # Add 1 pixel offset to create a gap
            
        # Calculate initial bounding box for region
        if width == "full":
            x0 = 0
            x1 = self.page.width
        elif width == "element":
            x0 = self.x0
            x1 = self.x1
        else:
            raise ValueError("Width must be 'full' or 'element'")
            
        # If an "until" selector is specified, find the target element
        if until:
            # Need to find all matches and find the first one below this element
            # instead of just page.find() which might return any match
            all_matches = self.page.find_all(until, **kwargs)
            
            # Sort by vertical position (top to bottom)
            matches_below = [m for m in all_matches if m.top >= self.bottom]
            matches_below.sort(key=lambda e: e.top)
            
            if matches_below:
                # Use the first match below this element
                target = matches_below[0]
                
                # Target is below this element - use it for the bottom boundary
                bottom = target.bottom if include_endpoint else target.top - 1  # Subtract 1 pixel offset when excluding
                
                # Use the selector match for width if not using full width
                if width == "element":
                    x0 = min(x0, target.x0 if include_endpoint else target.x1)
                    x1 = max(x1, target.x1 if include_endpoint else target.x0)
            else:
                # No targets found below this element - use requested height
                bottom = min(self.page.height, top + (height or (self.page.height - top)))
        else:
            # No "until" selector - use requested height
            bottom = min(self.page.height, top + (height or (self.page.height - top)))
        
        bbox = (x0, top, x1, bottom)
        region = Region(self.page, bbox)
        region.source_element = self  # Reference to element that created this region
        region.includes_source = include_element  # Whether region includes the source element
        return region
    
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
            matches = self.page.filter_elements(candidates, selector, **kwargs)
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
            # Filter elements before this one
            candidates = all_elements[:idx]
            # Reverse to start from closest to this element
            candidates = candidates[::-1]
            # Limit search range for performance
            candidates = candidates[:limit] if limit else candidates
            
            # Find matching elements
            matches = self.page.filter_elements(candidates, selector, **kwargs)
            return matches[0] if matches else None
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
                 color: Optional[Tuple[int, int, int, int]] = None, 
                 use_color_cycling: bool = False,
                 include_attrs: Optional[List[str]] = None,
                 existing: str = 'append') -> 'Element':
        """
        Highlight this element on the page.
        
        Args:
            label: Optional label for the highlight
            color: RGBA color tuple for the highlight, or None to use automatic color
            use_color_cycling: Force color cycling even with no label (default: False)
            include_attrs: List of attribute names to display on the highlight (e.g., ['confidence', 'type'])
            existing: How to handle existing highlights - 'append' (default) or 'replace'
            
        Returns:
            Self for method chaining
        """
        # Add highlight to the page's highlight manager
        self.page._highlight_mgr.add_highlight(
            self.bbox, 
            color, 
            label, 
            use_color_cycling, 
            element=self,  # Pass the element itself so attributes can be accessed
            include_attrs=include_attrs,
            existing=existing
        )
        return self
    
    def show(self, 
            scale: float = 2.0, 
            labels: bool = True,
            legend_position: str = 'right') -> Image.Image:
        """
        Show the page with this element highlighted.
        
        Args:
            scale: Scale factor for rendering
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            
        Returns:
            PIL Image of the page with this element highlighted
        """
        # Get the highlighted image from the page
        return self.page.show(scale=scale, labels=labels, legend_position=legend_position)
    
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