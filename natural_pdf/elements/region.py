from typing import Optional, Union, List, Dict, Tuple, Any, Callable, TYPE_CHECKING
from natural_pdf.elements.base import DirectionalMixin

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.text import TextElement

# Import OCRManager conditionally to avoid circular imports
try:
    from natural_pdf.ocr import OCRManager
except ImportError:
    # OCRManager will be imported directly in methods that use it
    pass


class Region(DirectionalMixin):
    """
    Represents a rectangular region on a page.
    """
    
    def __init__(self, page: 'Page', bbox: Tuple[float, float, float, float], polygon: List[Tuple[float, float]] = None, parent=None, label: Optional[str] = None):
        """
        Initialize a region.
        
        Args:
            page: Parent page
            bbox: Bounding box as (x0, top, x1, bottom)
            polygon: Optional list of coordinate points [(x1,y1), (x2,y2), ...] for non-rectangular regions
            parent: Optional parent region (for hierarchical document structure)
            label: Optional label for the region (e.g., for exclusions)
        """
        self._page = page
        self._bbox = bbox
        self._polygon = polygon
        self._multi_page_elements = None
        self._spans_pages = False
        self._page_range = None
        self.start_element = None
        self.end_element = None
        
        # Standard attributes for all elements
        self.object_type = 'region'  # For selector compatibility
        
        # Layout detection attributes
        self.region_type = None
        self.normalized_type = None
        self.confidence = None
        self.model = None
        
        # Region management attributes
        self.name = None
        self.source = None  # Will be set by creation methods
        self.label = label
        
        # Hierarchy support for nested document structure
        self.parent_region = parent
        self.child_regions = []
        self.text_content = None  # Direct text content (e.g., from Docling)
        self.associated_text_elements = []  # Native text elements that overlap with this region
    
    def _direction(self, direction: str, size: Optional[float] = None,
                   cross_size: str = "full", include_element: bool = False,
                   until: Optional[str] = None, include_endpoint: bool = True, **kwargs) -> 'Region':
        """
        Protected helper method to create a region in a specified direction relative to this region.

        Args:
            direction: 'left', 'right', 'above', or 'below'
            size: Size in the primary direction (width for horizontal, height for vertical)
            cross_size: Size in the cross direction ('full' or 'element')
            include_element: Whether to include this region's area in the result
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

        # 5. Create and return Region
        region = Region(self.page, final_bbox)
        region.source_element = self
        region.includes_source = include_element
        # Optionally store the boundary element if found
        if target:
            region.boundary_element = target

        return region

    def above(self, height: Optional[float] = None, width: str = "full", include_element: bool = False,
             until: Optional[str] = None, include_endpoint: bool = True, **kwargs) -> 'Region':
        """
        Select region above this region.
        
        Args:
            height: Height of the region above, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_element: Whether to include this region in the result (default: False)
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
        Select region below this region.
        
        Args:
            height: Height of the region below, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_element: Whether to include this region in the result (default: False)
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
        Select region to the left of this region.
        
        Args:
            width: Width of the region to the left, in points
            height: Height mode - "full" for full page height or "element" for element height
            include_element: Whether to include this region in the result (default: False)
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
        Select region to the right of this region.
        
        Args:
            width: Width of the region to the right, in points
            height: Height mode - "full" for full page height or "element" for element height
            include_element: Whether to include this region in the result (default: False)
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
    
    @property
    def type(self) -> str:
        """Element type."""
        # Return the specific type if detected (e.g., from layout analysis)
        # or 'region' as a default.
        return self.region_type or 'region' # Prioritize specific region_type if set
    
    @property
    def page(self) -> 'Page':
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
                (self.x0, self.top),       # top-left
                (self.x1, self.top),       # top-right
                (self.x1, self.bottom),    # bottom-right
                (self.x0, self.bottom)     # bottom-left
            ]
    
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
            if ((self.polygon[i][1] > y) != (self.polygon[j][1] > y)) and \
               (x < (self.polygon[j][0] - self.polygon[i][0]) * (y - self.polygon[i][1]) / \
                (self.polygon[j][1] - self.polygon[i][1]) + self.polygon[i][0]):
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
            if ((self.polygon[i][1] > y) != (self.polygon[j][1] > y)) and \
               (x < (self.polygon[j][0] - self.polygon[i][0]) * (y - self.polygon[i][1]) / \
                (self.polygon[j][1] - self.polygon[i][1]) + self.polygon[i][0]):
                inside = not inside
            j = i
            
        return inside

    def _is_element_in_region(self, element: 'Element', use_boundary_tolerance=True) -> bool:
        """
        Check if an element is within this region.
        
        Args:
            element: Element to check
            use_boundary_tolerance: Whether to apply a small tolerance for boundary elements
            
        Returns:
            True if the element is in the region, False otherwise
        """
        # If we have multi-page elements cached, check if the element is in the list
        if self._spans_pages and self._multi_page_elements is not None:
            return element in self._multi_page_elements
            
        # Check if element is on the same page
        if element.page != self._page:
            return False
            
        # Calculate element center
        element_center_x = (element.x0 + element.x1) / 2
        element_center_y = (element.top + element.bottom) / 2
        
        # If this is a boundary region with exclusions, apply strict boundary checking
        # This helps enforce boundary_inclusion behavior in get_sections
        if hasattr(self, 'start_element') or hasattr(self, 'end_element'):
            # Apply a small tolerance to avoid border cases
            # When an element is right at the border, we want to be more strict
            tolerance = 2.0 if use_boundary_tolerance else 0.0
            
            # Check if element center is strictly within the region (not just on border)
            if (self.x0 + tolerance <= element_center_x <= self.x1 - tolerance and 
                self.top + tolerance <= element_center_y <= self.bottom - tolerance):
                return True
            
            # For elements right at the boundary, be more conservative
            return False
            
        # If the element itself has a polygon, check if ANY corner is in this region
        if hasattr(element, 'has_polygon') and element.has_polygon:
            for point in element.polygon:
                if self.is_point_inside(point[0], point[1]):
                    return True
            # If no point is inside, check if the center is inside
            return self.is_point_inside(element_center_x, element_center_y)
            
        # For regular elements, check if center is in the region
        # Add a small tolerance (1 pixel) to avoid including elements that are exactly on the boundary
        # This ensures consistent behavior with the below() and above() method fixes
        tolerance = 1.0 if use_boundary_tolerance else 0.0
        
        # Check if within region with the tolerance applied
        if self.has_polygon:
            return self.is_point_inside(element_center_x, element_center_y)
        else:
            # For rectangular regions, apply tolerance to all sides
            return (self.x0 + tolerance <= element_center_x <= self.x1 - tolerance and 
                    self.top + tolerance <= element_center_y <= self.bottom - tolerance)

    def highlight(self, 
                 label: Optional[str] = None,
                 color: Optional[Union[Tuple, str]] = None,
                 use_color_cycling: bool = False,
                 include_attrs: Optional[List[str]] = None,
                 existing: str = 'append') -> 'Region':
        """
        Highlight this region on the page.
        
        Args:
            label: Optional label for the highlight
            color: Color tuple/string for the highlight, or None to use automatic color
            use_color_cycling: Force color cycling even with no label (default: False)
            include_attrs: List of attribute names to display on the highlight (e.g., ['confidence', 'type'])
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
            "include_attrs": include_attrs,
            "existing": existing
        }

        # Call the appropriate service method
        if self.has_polygon:
            highlight_args["polygon"] = self.polygon
            highlighter.add_polygon(**highlight_args)
        else:
            highlight_args["bbox"] = self.bbox
            highlighter.add(**highlight_args)

        return self
    
    def to_image(self, 
               scale: float = 2.0,
               resolution: float = 150,
               crop_only: bool = False,
               include_highlights: bool = True,
               **kwargs) -> 'Image.Image':
        """
        Generate an image of just this region.
        
        Args:
            resolution: Resolution in DPI for rendering (default: 150)
            crop_only: If True, only crop the region without highlighting its boundaries
            include_highlights: Whether to include existing highlights (default: True)
            **kwargs: Additional parameters for page.to_image()
            
        Returns:
            PIL Image of just this region
        """
        # First get the full page image with highlights if requested
        page_image = self._page.to_image(scale=scale, resolution=resolution, include_highlights=include_highlights, **kwargs)
        
        # Calculate the crop coordinates - apply resolution scaling factor
        # PDF coordinates are in points (1/72 inch), but image is scaled by resolution
        scale_factor = scale
        
        # Apply scaling to the coordinates
        x0 = int(self.x0 * scale_factor)
        top = int(self.top * scale_factor)
        x1 = int(self.x1 * scale_factor)
        bottom = int(self.bottom * scale_factor)
        
        # Crop the image to just this region
        region_image = page_image.crop((x0, top, x1, bottom))
        
        # If not crop_only, add a border to highlight the region boundaries
        if not crop_only:
            from PIL import ImageDraw
            
            # Create a 1px border around the region
            draw = ImageDraw.Draw(region_image)
            draw.rectangle((0, 0, region_image.width-1, region_image.height-1), 
                          outline=(255, 0, 0), width=1)
        
        return region_image
    
    def show(self,
            scale: float = 2.0,
            labels: bool = True,
            legend_position: str = 'right',
            # Add a default color for standalone show
            color: Optional[Union[Tuple, str]] = "blue",
            label: Optional[str] = None) -> 'Image.Image':
        """
        Show the page with just this region highlighted temporarily.

        Args:
            scale: Scale factor for rendering
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            color: Color to highlight this region (default: blue)
            label: Optional label for this region in the legend

        Returns:
            PIL Image of the page with only this region highlighted
        """
        if not self._page:
            raise ValueError("Region must be associated with a page to show.")

        # Use the highlighting service via the page's property
        service = self._page._highlighter

        # Determine the label if not provided
        display_label = label if label is not None else f"Region ({self.type})" if self.type else "Region"

        # Prepare temporary highlight data for just this region
        temp_highlight_data = {
            "page_index": self._page.index,
            "bbox": self.bbox,
            "polygon": self.polygon if self.has_polygon else None,
            "color": color, # Use provided or default color
            "label": display_label,
            "use_color_cycling": False # Explicitly false for single preview
        }

        # Use render_preview to show only this highlight
        return service.render_preview(
            page_index=self._page.index,
            temporary_highlights=[temp_highlight_data],
            scale=scale,
            labels=labels,
            legend_position=legend_position
        )

    def save(self, 
            filename: str, 
            scale: float = 2.0, 
            labels: bool = True,
            legend_position: str = 'right') -> 'Region':
        """
        Save the page with this region highlighted to an image file.
        
        Args:
            filename: Path to save the image to
            scale: Scale factor for rendering
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            
        Returns:
            Self for method chaining
        """
        # Highlight this region if not already highlighted
        self.highlight()
        
        # Save the highlighted image
        self._page.save_image(filename, scale=scale, labels=labels, legend_position=legend_position)
        return self
        
    def save_image(self, 
                  filename: str,
                  resolution: float = 150,
                  crop_only: bool = False,
                  include_highlights: bool = True,
                  **kwargs) -> 'Region':
        """
        Save an image of just this region to a file.
        
        Args:
            filename: Path to save the image to
            resolution: Resolution in DPI for rendering (default: 150)
            crop_only: If True, only crop the region without highlighting its boundaries
            include_highlights: Whether to include existing highlights (default: True)
            **kwargs: Additional parameters for page.to_image()
            
        Returns:
            Self for method chaining
        """
        # Get the region image
        image = self.to_image(
            resolution=resolution, 
            crop_only=crop_only, 
            include_highlights=include_highlights,
            **kwargs
        )
        
        # Save the image
        image.save(filename)
        return self
        
    def get_elements(self, selector: Optional[str] = None, apply_exclusions=True, **kwargs) -> List['Element']:
        """
        Get all elements within this region.
        
        Args:
            selector: Optional selector to filter elements
            apply_exclusions: Whether to apply exclusion regions
            **kwargs: Additional parameters for element filtering
            
        Returns:
            List of elements in the region
        """
        # If we have multi-page elements, return those
        if self._spans_pages and self._multi_page_elements is not None:
            return self._multi_page_elements
            
        # Otherwise, get elements from the page
        if selector:
            elements = self.page.find_all(selector, apply_exclusions=apply_exclusions, **kwargs)
        else:
            elements = self.page.get_elements(apply_exclusions=apply_exclusions)
            
        # Filter to elements in this region
        return [e for e in elements if self._is_element_in_region(e)]
    
    def extract_text(self, keep_blank_chars=True, apply_exclusions=True, ocr=None, preserve_whitespace=None, debug=False, **kwargs) -> str:
        """
        Extract text from this region using pdfplumber's native functionality.
        
        For regions created by Docling, this will first try to use:
        1. Associated text elements from the PDF (if available)
        2. Direct text content from Docling (if available)
        3. Fall back to standard pdfplumber extraction
        
        Args:
            keep_blank_chars: Whether to keep blank characters (legacy parameter)
            apply_exclusions: Whether to apply exclusion regions
            ocr: OCR configuration. If None, uses PDF settings
            preserve_whitespace: Synonym for keep_blank_chars (for compatibility with page.extract_text)
            debug: Enable verbose debugging for exclusion handling
            **kwargs: Additional parameters for text extraction
            
        Returns:
            Extracted text as string
        """
        import logging
        logger = logging.getLogger("natural_pdf.elements.region")
        
        # Check for Docling model or if we have direct text content
        if self.model == 'docling' or hasattr(self, 'text_content'):
            # First priority: check if we have associated native text elements
            if hasattr(self, 'associated_text_elements') and self.associated_text_elements:
                source_count = len(self.associated_text_elements)
                logger.info(f"Region {self.region_type}: Using {source_count} native PDF text elements")
                # Sort elements in reading order
                sorted_elements = sorted(self.associated_text_elements, key=lambda e: (e.top, e.x0))
                # Extract and join their text
                text_result = " ".join(elem.text for elem in sorted_elements)
                return text_result
                
            # Second priority: use direct text content from Docling
            elif self.text_content:
                logger.info(f"Region {self.region_type}: Using Docling OCR text content")
                return self.text_content
                
            logger.debug(f"Region {self.region_type}: No Docling text found, falling back to standard extraction")
                
        # Handle preserve_whitespace parameter for consistency with Page.extract_text
        if preserve_whitespace is not None:
            keep_blank_chars = preserve_whitespace
            
        # If we span multiple pages, use the original implementation
        if self._spans_pages and self._multi_page_elements is not None:
            # Sort elements in reading order - only include text-like elements
            text_elements = [e for e in self._multi_page_elements if hasattr(e, 'text')]
            
            # Sort in reading order (by page, then top-to-bottom, left-to-right)
            sorted_elements = sorted(text_elements, key=lambda e: (e.page.index, e.top, e.x0))
            
            # Extract text directly from elements to avoid recursion
            texts = []
            for element in sorted_elements:
                if hasattr(element, 'text'):
                    texts.append(element.text)
            
            text_result = " ".join(texts)
            return text_result
        
        # Check if we have exclusions to apply
        exclusion_regions = []
        if apply_exclusions and self._page._exclusions:
            exclusion_regions = self._page._get_exclusion_regions(include_callable=True)
            
            if debug:
                logger.debug(f"Region {self.bbox} with {len(exclusion_regions)} exclusion regions")
        
        # IMPROVEMENT 1: Check if the region intersects with any exclusion zone
        # If not, ignore exclusions entirely
        if exclusion_regions:
            has_intersection = False
            for i, exclusion in enumerate(exclusion_regions):
                # Use a simple bbox overlap check
                overlap = (self.x0 < exclusion.x1 and self.x1 > exclusion.x0 and 
                           self.top < exclusion.bottom and self.bottom > exclusion.top)
                
                if overlap:
                    has_intersection = True
                    if debug:
                        logger.debug(f"  Region intersects with exclusion {i}: {exclusion.bbox}")
                    break
            
            # If no intersection, process without exclusions
            if not has_intersection:
                if debug:
                    logger.debug(f"  No intersection with any exclusion, ignoring exclusions")
                apply_exclusions = False
                exclusion_regions = []
        
        # IMPROVEMENT 2: If rectangular region + full-width exclusions (headers/footers),
        # we can use the simpler cropping approach
        # Only use crop for simple cases
        can_use_crop = not self.has_polygon
        result = ""  # Default empty result
        if can_use_crop and apply_exclusions and exclusion_regions:
            # We'll keep track of exclusions that are full-width horizontal bands (headers/footers)
            # and those that are not
            footer_header_exclusions = []
            other_exclusions = []
            
            for i, exclusion in enumerate(exclusion_regions):
                # Check if exclusion spans the full width of the page
                # and is either at the top or bottom
                full_width = (abs(exclusion.x0) < 5 and 
                             abs(exclusion.x1 - self.page.width) < 5)
                
                if debug:
                    logger.debug(f"  Exclusion {i}: {exclusion.bbox}, full width: {full_width}")
                
                if full_width:
                    footer_header_exclusions.append(exclusion)
                else:
                    other_exclusions.append(exclusion)
            
            # If we have only header/footer exclusions, we can use the cropping approach
            all_are_bands = len(other_exclusions) == 0 and len(footer_header_exclusions) > 0
            
            if all_are_bands:
                # Find the actual content area after excluding header/footer
                top_bound = self.top
                bottom_bound = self.bottom
                
                if debug:
                    logger.debug(f"  Using cropping approach, initial bounds: ({self.x0}, {top_bound}, {self.x1}, {bottom_bound})")
                
                # Process only header/footer exclusions for cropping
                for exclusion in footer_header_exclusions:
                    # If exclusion is at the top of our region
                    if exclusion.bottom > self.top and exclusion.top <= self.top:
                        # Move top bound to exclude the header
                        top_bound = max(top_bound, exclusion.bottom)
                        if debug:
                            logger.debug(f"  Adjusted top bound to {top_bound} due to header exclusion")
                    
                    # If exclusion is at the bottom of our region
                    if exclusion.top < self.bottom and exclusion.bottom >= self.bottom:
                        # Move bottom bound to exclude the footer
                        bottom_bound = min(bottom_bound, exclusion.top)
                        if debug:
                            logger.debug(f"  Adjusted bottom bound to {bottom_bound} due to footer exclusion")
                        
                
                if debug:
                    logger.debug(f"  Final bounds after exclusion adjustment: ({self.x0}, {top_bound}, {self.x1}, {bottom_bound})")
                
                # If we still have a valid region after exclusions
                if top_bound < bottom_bound:
                    # Use direct crop with adjusted bounds
                    crop_bbox = (self.x0, top_bound, self.x1, bottom_bound)
                    cropped = self.page._page.crop(crop_bbox)
                    result = cropped.extract_text(keep_blank_chars=keep_blank_chars, **kwargs)
                    
                    if debug:
                        logger.debug(f"  Successfully extracted text using crop, got {len(result)} characters")
                    
                    # Skip the complex filtering approach
                    return result
                else:
                    # This would only happen if the region is entirely inside an exclusion zone
                    # or if both top and bottom of the region are excluded leaving no valid area
                    logger.debug(f"Region {self.bbox} completely covered by exclusions, returning empty string")
                    return ""
            # We have exclusions, but not all are headers/footers,
            # or we have a non-rectangular region
            else:
                if debug:
                    logger.debug(f"  Mixed exclusion types or non-rectangular region, switching to filtering")
                
                # Don't use crop for mixed exclusion types
                can_use_crop = False
        
        # If we got a result from header/footer cropping, return it
        if result:
            return result
            
        # For single-page regions without exclusions, or when exclusions don't apply, use direct cropping 
        if can_use_crop and not apply_exclusions:
            # Simple case: use direct crop
            crop_bbox = self.bbox
            cropped = self.page._page.crop(crop_bbox)
            result = cropped.extract_text(keep_blank_chars=keep_blank_chars, **kwargs)
            return result
            
        # For all other cases (complex exclusions, polygons), we use element filtering
        if debug:
            logger.debug(f"Using element filtering approach for region {self.bbox}")
        
        # Get only word elements in this region first (instead of ALL elements)
        # This prevents duplication from joining both char and word text
        all_elements = [e for e in self.page.words if self._is_element_in_region(e)]

        if apply_exclusions and exclusion_regions:
            if debug:
                logger.debug(f"Filtering with {len(exclusion_regions)} exclusion zones")
                
            # Filter out elements in exclusion zones
            filtered_elements = []
            for elem in all_elements:
                in_exclusion = False
                # For each element, check if it's in any exclusion zone
                element_center_x = (elem.x0 + elem.x1) / 2
                element_center_y = (elem.top + elem.bottom) / 2
                
                for exclusion in exclusion_regions:
                    if (exclusion.x0 <= element_center_x <= exclusion.x1 and
                        exclusion.top <= element_center_y <= exclusion.bottom):
                        in_exclusion = True
                        break
                
                if not in_exclusion:
                    filtered_elements.append(elem)
        else:
            # No exclusions, use all elements
            filtered_elements = all_elements
        
        # Now extract text from the filtered elements
        if filtered_elements:
            from natural_pdf.elements.collections import ElementCollection
            collection = ElementCollection(filtered_elements)
            # Sort in reading order
            collection = collection.sort(key=lambda e: (e.top, e.x0))
            # Extract text
            result = " ".join(e.text for e in collection if hasattr(e, 'text'))
            
            if debug:
                logger.debug(f"Got {len(result)} characters from element-based extraction")
                
            # Return the result
            return result
        else:
            if debug:
                logger.debug(f"No elements found after filtering")
            return ""
            
        # Handle OCR if needed
        use_ocr = ocr is True or (isinstance(ocr, dict) and ocr.get('enabled', False))
        auto_ocr = ocr is None and self.page._parent._ocr_config.get('enabled') == 'auto'
        
        # Run OCR if explicitly requested or if in auto mode and no text found
        if use_ocr or (auto_ocr and not result.strip()):
            ocr_config = self.page._get_ocr_config(ocr or {}) if use_ocr else self.page._get_ocr_config({'enabled': 'auto'})
            ocr_elements = self.apply_ocr(**ocr_config)
            
            if ocr_elements:
                # Filter OCR elements by exclusions if needed
                if apply_exclusions and exclusion_regions:
                    filtered_ocr = []
                    for element in ocr_elements:
                        exclude = False
                        for region in exclusion_regions:
                            if region._is_element_in_region(element):
                                exclude = True
                                break
                        if not exclude:
                            filtered_ocr.append(element)
                else:
                    filtered_ocr = ocr_elements
                
                # Extract text from OCR elements
                from natural_pdf.elements.collections import ElementCollection
                ocr_collection = ElementCollection(filtered_ocr)
                ocr_text = ocr_collection.extract_text(preserve_whitespace=keep_blank_chars, **kwargs)
                
                # Use OCR text if it's not empty
                if ocr_text.strip():
                    return ocr_text
        
        return result
        
    def extract_table(self, method: str = None, table_settings: dict = None, 
                 use_ocr: bool = False, ocr_config: dict = None) -> List[List[str]]:
        """
        Extract a table from this region.
        
        Args:
            method: Method to use for extraction ('tatr', 'plumber', or None for auto-detection)
            table_settings: Settings for pdfplumber table extraction (used only with 'plumber' method)
            use_ocr: Whether to use OCR for text extraction (only applicable with 'tatr' method)
            ocr_config: OCR configuration parameters
            
        Returns:
            Table data as a list of rows, where each row is a list of cell values
        """
        # Default settings if none provided
        if table_settings is None:
            table_settings = {}
            
        # Auto-detect method if not specified
        if method is None:
            # If this is a TATR-detected region, use TATR method
            if hasattr(self, 'model') and self.model == 'tatr' and self.region_type == 'table':
                method = 'tatr'
            else:
                method = 'plumber'
                
        # Use the selected method
        if method == 'tatr':
            return self._extract_table_tatr(use_ocr=use_ocr, ocr_config=ocr_config)
        else:  # Default to pdfplumber
            return self._extract_table_plumber(table_settings)
    
    def _extract_table_plumber(self, table_settings: dict) -> List[List[str]]:
        """
        Extract table using pdfplumber's table extraction.
        
        Args:
            table_settings: Settings for pdfplumber table extraction
            
        Returns:
            Table data as a list of rows, where each row is a list of cell values
        """
        # Create a crop of the page for this region
        cropped = self.page._page.crop(self.bbox)
        
        # Extract table from the cropped area
        tables = cropped.extract_tables(table_settings)
        
        # Return the first table or an empty list if none found
        if tables:
            return tables[0]
        return []
    
    def _extract_table_tatr(self, use_ocr=False, ocr_config=None) -> List[List[str]]:
        """
        Extract table using TATR structure detection.
        
        Args:
            use_ocr: Whether to apply OCR to each cell for better text extraction
            ocr_config: Optional OCR configuration parameters
            
        Returns:
            Table data as a list of rows, where each row is a list of cell values
        """
        # Find all rows and headers in this table
        rows = self.page.find_all(f'region[type=table-row][model=tatr]')
        headers = self.page.find_all(f'region[type=table-column-header][model=tatr]')
        columns = self.page.find_all(f'region[type=table-column][model=tatr]')
        
        # Filter to only include rows/headers/columns that overlap with this table region
        def is_in_table(region):
            # Check for overlap - simplifying to center point for now
            region_center_x = (region.x0 + region.x1) / 2
            region_center_y = (region.top + region.bottom) / 2
            return (self.x0 <= region_center_x <= self.x1 and
                    self.top <= region_center_y <= self.bottom)
        
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
                    "link_threshold": 0.1  # Lower threshold for connecting text components
                }
            }
            
            # Merge with provided config if any
            if ocr_config:
                if isinstance(ocr_config, dict):
                    # Update default config with provided values
                    for key, value in ocr_config.items():
                        if isinstance(value, dict) and key in default_ocr_config and isinstance(default_ocr_config[key], dict):
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
                header_texts.append(header.extract_text().strip())
            table_data.append(header_texts)
        
        # Process rows
        for row in rows:
            row_cells = []
            
            # If we have columns, use them to extract cells
            if columns:
                for column in columns:
                    # Create a cell region at the intersection of row and column
                    cell_bbox = (
                        column.x0,
                        row.top,
                        column.x1,
                        row.bottom
                    )
                    
                    # Create a region for this cell
                    from natural_pdf.elements.region import Region  # Import here to avoid circular imports
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
                    cell_text = cell_region.extract_text().strip()
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
                row_cells.append(row.extract_text().strip())
            
            table_data.append(row_cells)
        
        return table_data
    
    def find(self, selector: str, apply_exclusions=True, **kwargs) -> Optional['Element']:
        """
        Find the first element in this region matching the selector.
        
        Args:
            selector: CSS-like selector string
            apply_exclusions: Whether to apply exclusion regions
            **kwargs: Additional parameters for element filtering
            
        Returns:
            First matching element or None
        """
        elements = self.find_all(selector, apply_exclusions=apply_exclusions, **kwargs)
        return elements[0] if elements else None
    
    def _find_all(self, selector: str, apply_exclusions=True, **kwargs) -> 'ElementCollection':
        """
        Find all elements in this region matching the selector.
        
        Args:
            selector: CSS-like selector string
            apply_exclusions: Whether to apply exclusion regions
            **kwargs: Additional parameters for element filtering
            
        Returns:
            ElementCollection with matching elements
        """
        from natural_pdf.elements.collections import ElementCollection

        # If we span multiple pages, filter our elements
        if self._spans_pages and self._multi_page_elements is not None:
            # Parse the selector
            from natural_pdf.selectors.parser import parse_selector
            selector_obj = parse_selector(selector)
            
            # Rather than using matches_selector, let each page's find_all handle the matching
            # since that method is already properly implemented
            all_matching_elements = []
            page_ranges = {}
            
            # Group elements by page
            for element in self._multi_page_elements:
                if element.page not in page_ranges:
                    page_ranges[element.page] = []
                page_ranges[element.page].append(element)
            
            # For each page, use its find_all to match elements, then filter to our collection
            for page, page_elements in page_ranges.items():
                # Get all matching elements from the page
                page_matches = page.find_all(selector, apply_exclusions=apply_exclusions, **kwargs)
                
                # Filter to just the elements that are in our collection
                for element in page_matches:
                    if element in page_elements:
                        all_matching_elements.append(element)
            
            return ElementCollection(all_matching_elements)

        # Otherwise, get elements from the page and filter by selector and region
        page_elements = self.page.find_all(selector, apply_exclusions=apply_exclusions, **kwargs)
        filtered_elements = [e for e in page_elements if self._is_element_in_region(e)]
        return ElementCollection(filtered_elements)
    
    def apply_ocr(self, **ocr_params) -> List['TextElement']:
        """
        Apply OCR to this region and return the created text elements.
        
        Args:
            **ocr_params: OCR parameters to override defaults
            
        Returns:
            List of created text elements
        """
        from natural_pdf.ocr import OCRManager
        
        # Get OCR configuration but suppress verbose output
        if isinstance(ocr_params, dict):
            ocr_params["verbose"] = False
        else:
            ocr_params = {"enabled": True, "verbose": False}
            
        ocr_config = self.page._get_ocr_config(ocr_params)
        
        # Skip if OCR is disabled
        if not ocr_config.get('enabled'):
            return []
        
        # Render the page
        page_image = self.page.to_image()
        
        # Crop to this region
        region_image = page_image.crop((self.x0, self.top, self.x1, self.bottom))
        
        # Run OCR on this region
        ocr_mgr = OCRManager.get_instance()
        results = ocr_mgr.recognize_region(region_image, ocr_config)
        
        # Adjust coordinates to be relative to the page
        for result in results:
            # Calculate bbox in page coordinates
            result['bbox'] = (
                result['bbox'][0] + self.x0,
                result['bbox'][1] + self.top,
                result['bbox'][2] + self.x0,
                result['bbox'][3] + self.top
            )
        
        # Create text elements with adjusted coordinates
        elements = []
        for result in results:
            # Only include results that are fully within the region
            if (result['bbox'][0] >= self.x0 and 
                result['bbox'][1] >= self.top and 
                result['bbox'][2] <= self.x1 and 
                result['bbox'][3] <= self.bottom):
                # Create a TextElement object with the appropriate fields
                from natural_pdf.elements.text import TextElement
                element_data = {
                    'text': result['text'],
                    'x0': result['bbox'][0],
                    'top': result['bbox'][1],
                    'x1': result['bbox'][2],
                    'bottom': result['bbox'][3],
                    'width': result['bbox'][2] - result['bbox'][0],
                    'height': result['bbox'][3] - result['bbox'][1],
                    'object_type': 'text',
                    'source': 'ocr',
                    'confidence': result['confidence'],
                    # Add default font information to work with existing expectations
                    'fontname': 'OCR-detected',
                    'size': 10.0,
                    'page_number': self.page.number
                }
                
                elem = TextElement(element_data, self.page)
                elements.append(elem)
                
                # Add to page's elements
                if hasattr(self.page, '_elements') and self.page._elements is not None:
                    # Add to words list to make it accessible via standard API
                    if 'words' in self.page._elements:
                        self.page._elements['words'].append(elem)
                    else:
                        self.page._elements['words'] = [elem]
        
        return elements
    
    def get_section_between(self, start_element=None, end_element=None, boundary_inclusion='both'):
        """
        Get a section between two elements within this region.
        
        Args:
            start_element: Element marking the start of the section
            end_element: Element marking the end of the section
            boundary_inclusion: How to include boundary elements: 'start', 'end', 'both', or 'none'
            
        Returns:
            Region representing the section
        """
        elements = self.get_elements()
        
        # If no elements, return self
        if not elements:
            return self
            
        # Sort elements in reading order
        elements.sort(key=lambda e: (e.top, e.x0))
        
        # Find start index
        start_idx = 0
        if start_element:
            try:
                start_idx = elements.index(start_element)
            except ValueError:
                # Start element not in region, use first element
                pass
                
        # Find end index
        end_idx = len(elements) - 1
        if end_element:
            try:
                end_idx = elements.index(end_element)
            except ValueError:
                # End element not in region, use last element
                pass
                
        # Adjust indexes based on boundary inclusion
        if boundary_inclusion == 'none':
            start_idx += 1
            end_idx -= 1
        elif boundary_inclusion == 'start':
            end_idx -= 1
        elif boundary_inclusion == 'end':
            start_idx += 1
            
        # Ensure valid indexes
        start_idx = max(0, start_idx)
        end_idx = min(len(elements) - 1, end_idx)
        
        # If no valid elements in range, return empty region
        if start_idx > end_idx:
            return Region(self.page, (0, 0, 0, 0))
            
        # Get elements in range
        section_elements = elements[start_idx:end_idx+1]
        
        # Create bounding box around elements
        x0 = min(e.x0 for e in section_elements)
        top = min(e.top for e in section_elements)
        x1 = max(e.x1 for e in section_elements)
        bottom = max(e.bottom for e in section_elements)
        
        # Adjust boundaries for better boundary inclusion/exclusion
        pixel_adjustment = 2.0  # Amount to adjust for avoiding boundary elements
        
        # Only proceed with adjustments if we have elements in the section
        if section_elements:
            # Adjust top boundary if start element should be excluded
            if start_element and boundary_inclusion not in ('start', 'both') and start_idx > 0:
                # If start element is just above the section, move the top down
                # Use a larger threshold (10 points) to catch more cases
                if abs(top - start_element.bottom) < 10:
                    top += pixel_adjustment
                    
            # Adjust bottom boundary if end element should be excluded
            if end_element and boundary_inclusion not in ('end', 'both') and end_idx < len(elements) - 1:
                # If end element is just below the section, move the bottom up
                # Use a larger threshold (10 points) to catch more cases
                if abs(bottom - end_element.top) < 10:
                    bottom -= pixel_adjustment
            
            # Ensure top is always less than bottom (valid region)
            if top >= bottom:
                # Reset to original if adjustment would create an invalid region
                top = min(e.top for e in section_elements)
                bottom = max(e.bottom for e in section_elements)
        
        # Create new region
        section = Region(self.page, (x0, top, x1, bottom))
        section.start_element = start_element if boundary_inclusion in ('start', 'both') else None
        section.end_element = end_element if boundary_inclusion in ('end', 'both') else None
        
        return section
    
    def get_sections(self, start_elements=None, end_elements=None, boundary_inclusion='both') -> List['Region']:
        """
        Get sections within this region based on start/end elements.
        
        Args:
            start_elements: Elements or selector string that mark the start of sections
            end_elements: Elements or selector string that mark the end of sections
            boundary_inclusion: How to include boundary elements: 'start', 'end', 'both', or 'none'
            
        Returns:
            List of Region objects representing the extracted sections
        """
        from natural_pdf.elements.collections import ElementCollection
            
        # Process string selectors to find elements
        if isinstance(start_elements, str):
            start_elements = self.find_all(start_elements)
            if hasattr(start_elements, 'elements'):
                start_elements = start_elements.elements
                
        if isinstance(end_elements, str):
            end_elements = self.find_all(end_elements)
            if hasattr(end_elements, 'elements'):
                end_elements = end_elements.elements
                
        # If no start elements, return empty list
        if not start_elements:
            return []
            
        # Sort elements in reading order
        all_elements = self.get_elements()
        all_elements.sort(key=lambda e: (e.top, e.x0))
        
        # Get all indexes in the sorted list
        section_boundaries = []
        
        # Add start element indexes
        for element in start_elements:
            try:
                idx = all_elements.index(element)
                section_boundaries.append({
                    'index': idx,
                    'element': element,
                    'type': 'start'
                })
            except ValueError:
                # Element not in this region, skip
                continue
                
        # Add end element indexes if provided
        if end_elements:
            for element in end_elements:
                try:
                    idx = all_elements.index(element)
                    section_boundaries.append({
                        'index': idx,
                        'element': element,
                        'type': 'end'
                    })
                except ValueError:
                    # Element not in this region, skip
                    continue
        
        # Sort boundaries by index (document order)
        section_boundaries.sort(key=lambda x: x['index'])
        
        # Generate sections
        sections = []
        current_start = None
        
        for i, boundary in enumerate(section_boundaries):
            # If it's a start boundary and we don't have a current start
            if boundary['type'] == 'start' and current_start is None:
                current_start = boundary
            
            # If it's an end boundary and we have a current start
            elif boundary['type'] == 'end' and current_start is not None:
                # Create a section from current_start to this boundary
                start_element = current_start['element']
                end_element = boundary['element']
                section = self.get_section_between(
                    start_element,
                    end_element,
                    boundary_inclusion
                )
                sections.append(section)
                current_start = None
                
            # If it's another start boundary and we have a current start (for splitting by starts only)
            elif boundary['type'] == 'start' and current_start is not None and not end_elements:
                # Create a section from current_start to just before this boundary
                start_element = current_start['element']
                end_element = all_elements[boundary['index'] - 1] if boundary['index'] > 0 else None
                section = self.get_section_between(
                    start_element,
                    end_element,
                    boundary_inclusion
                )
                sections.append(section)
                current_start = boundary
        
        # Handle the last section if we have a current start
        if current_start is not None:
            start_element = current_start['element']
            # Use the last element in the region as the end
            end_element = all_elements[-1] if all_elements else None
            section = self.get_section_between(
                start_element,
                end_element,
                boundary_inclusion
            )
            sections.append(section)
        
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
        if self.region_type not in ('table', 'tableofcontents'): # Allow for ToC which might have structure
            raise ValueError(f"create_cells should be called on a 'table' or 'tableofcontents' region, not '{self.region_type}'")
        
        # Find rows and columns associated with this page
        # Remove the model-specific filter
        rows = self.page.find_all('region[type=table-row]')
        columns = self.page.find_all('region[type=table-column]')
        
        # Filter to only include those that overlap with this table region
        def is_in_table(element):
            # Use a simple overlap check (more robust than just center point)
            # Check if element's bbox overlaps with self.bbox
            return (element.x0 < self.x1 and element.x1 > self.x0 and
                    element.top < self.bottom and element.bottom > self.top)
        
        table_rows = [r for r in rows if is_in_table(r)]
        table_columns = [c for c in columns if is_in_table(c)]
        
        if not table_rows or not table_columns:
            self._page.logger.warning(f"Region {self.bbox}: Cannot create cells. No overlapping row or column regions found.")
            return self # Return self even if no cells created
            
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
                    cell = self.page.create_region(
                        cell_x0, cell_y0, cell_x1, cell_y1
                    )
                    # Set metadata
                    cell.source = 'derived'
                    cell.region_type = 'table-cell' # Explicitly set type
                    cell.normalized_type = 'table-cell' # And normalized type
                    # Inherit model from the parent table region
                    cell.model = self.model 
                    cell.parent_region = self # Link cell to parent table region
                    
                    # Add the cell region to the page's element manager
                    self.page._element_mgr.add_region(cell)
                    created_count += 1
        
        # Optional: Add created cells to the table region's children
        # self.child_regions.extend(cells_created_in_this_call) # Needs list management

        self._page.logger.info(f"Region {self.bbox} (Model: {self.model}): Created and added {created_count} cell regions.")

        return self # Return self for chaining
        
    def ask(self, question: str, min_confidence: float = 0.1, model: str = None, debug: bool = False, **kwargs) -> Dict[str, Any]:
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
        from natural_pdf.qa.document_qa import get_qa_engine
        
        # Get or initialize QA engine with specified model
        qa_engine = get_qa_engine(model_name=model) if model else get_qa_engine()
        
        # Ask the question using the QA engine
        return qa_engine.ask_pdf_region(self, question, min_confidence=min_confidence, debug=debug, **kwargs)

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
        from natural_pdf.selectors.parser import match_elements_with_selector
        matched = match_elements_with_selector(self.child_regions, selector)
        logger.debug(f"get_children: found {len(matched)} of {len(self.child_regions)} children matching '{selector}'")
        return matched
        
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
        
        # First add direct children
        all_descendants.extend(self.child_regions)
        
        # Then recursively add their descendants
        for child in self.child_regions:
            all_descendants.extend(child.get_descendants())
            
        logger.debug(f"get_descendants: found {len(all_descendants)} total descendants")
            
        # Filter by selector if provided
        if selector is not None:
            from natural_pdf.selectors.parser import match_elements_with_selector
            matched = match_elements_with_selector(all_descendants, selector)
            logger.debug(f"get_descendants: filtered to {len(matched)} matching '{selector}'")
            return matched
        
        return all_descendants
        
    def find_all(self, selector, recursive=True, **kwargs):
        """
        Find all matching elements within this region, with optional recursion through child regions.
        
        Args:
            selector: The selector to find elements with
            recursive: Whether to search recursively through child regions
            **kwargs: Additional parameters to pass to the selector parser
            
        Returns:
            Collection of matching elements
        """
        # Get direct matches 
        direct_matches = self._find_all(selector, region=self, **kwargs)
        
        if not recursive or not self.child_regions:
            return direct_matches
            
        # Get recursive matches from children
        from natural_pdf.elements.collections import ElementCollection
        all_matches = list(direct_matches)
        
        for child in self.child_regions:
            child_matches = child.find_all(selector, recursive=True, **kwargs)
            for match in child_matches:
                if match not in all_matches:
                    all_matches.append(match)
                    
        return ElementCollection(all_matches)
