from typing import Optional, Union, List, Dict, Tuple, Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.text import TextElement

# Import OCRManager conditionally to avoid circular imports
try:
    from natural_pdf.utils.ocr import OCRManager
except ImportError:
    # OCRManager will be imported directly in methods that use it
    pass


class Region:
    """
    Represents a rectangular region on a page.
    """
    
    def __init__(self, page: 'Page', bbox: Tuple[float, float, float, float], polygon: List[Tuple[float, float]] = None, parent=None):
        """
        Initialize a region.
        
        Args:
            page: Parent page
            bbox: Bounding box as (x0, top, x1, bottom)
            polygon: Optional list of coordinate points [(x1,y1), (x2,y2), ...] for non-rectangular regions
            parent: Optional parent region (for hierarchical document structure)
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
        
        # Hierarchy support for nested document structure
        self.parent_region = parent
        self.child_regions = []
        self.text_content = None  # Direct text content (e.g., from Docling)
        self.associated_text_elements = []  # Native text elements that overlap with this region
    
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
            x: X-coordinate to check
            y: Y-coordinate to check
            
        Returns:
            True if the point is inside the polygon
        """
        # If no polygon, use simple rectangle check
        if not self.has_polygon:
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
                if self._is_point_in_polygon(point[0], point[1]):
                    return True
            # If no point is inside, check if the center is inside
            return self._is_point_in_polygon(element_center_x, element_center_y)
            
        # For regular elements, check if center is in the region
        # Add a small tolerance (1 pixel) to avoid including elements that are exactly on the boundary
        # This ensures consistent behavior with the below() and above() method fixes
        tolerance = 1.0 if use_boundary_tolerance else 0.0
        
        # Check if within region with the tolerance applied
        if self.has_polygon:
            return self._is_point_in_polygon(element_center_x, element_center_y)
        else:
            # For rectangular regions, apply tolerance to all sides
            return (self.x0 + tolerance <= element_center_x <= self.x1 - tolerance and 
                    self.top + tolerance <= element_center_y <= self.bottom - tolerance)
                
    def highlight(self, 
                 label: Optional[str] = None,
                 color: Optional[Tuple[int, int, int, int]] = None, 
                 use_color_cycling: bool = False,
                 include_attrs: Optional[List[str]] = None) -> 'Region':
        """
        Highlight this region on the page.
        
        Args:
            label: Optional label for the highlight
            color: RGBA color tuple for the highlight, or None to use automatic color
            use_color_cycling: Force color cycling even with no label (default: False)
            include_attrs: List of attribute names to display on the highlight (e.g., ['confidence', 'type'])
            
        Returns:
            Self for method chaining
        """
        # Add highlight to the page's highlight manager
        if self.has_polygon:
            self._page._highlight_mgr.add_polygon_highlight(
                self.polygon, 
                color, 
                label, 
                use_color_cycling,
                element=self,  # Pass the region itself so attributes can be accessed
                include_attrs=include_attrs
            )
        else:
            self._page._highlight_mgr.add_highlight(
                self.bbox, 
                color, 
                label, 
                use_color_cycling,
                element=self,  # Pass the region itself so attributes can be accessed
                include_attrs=include_attrs
            )
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
            legend_position: str = 'right') -> 'Image.Image':
        """
        Show the page with this region highlighted.
        
        Args:
            scale: Scale factor for rendering
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            
        Returns:
            PIL Image of the page with this region highlighted
        """
        # Highlight this region if not already highlighted
        self.highlight()
        
        # Get and display the highlighted image
        return self._page.show(scale, labels=labels, legend_position=legend_position)
        
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
                import logging
                logger = logging.getLogger("natural_pdf.elements.region")
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
                        import logging
                        logger = logging.getLogger("natural_pdf.elements.region")
                        logger.debug(f"  Region intersects with exclusion {i}: {exclusion.bbox}")
                    break
            
            # If no intersection, process without exclusions
            if not has_intersection:
                if debug:
                    import logging
                    logger = logging.getLogger("natural_pdf.elements.region")
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
                    import logging
                    logger = logging.getLogger("natural_pdf.elements.region")
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
                    import logging
                    logger = logging.getLogger("natural_pdf.elements.region")
                    logger.debug(f"  Using cropping approach, initial bounds: ({self.x0}, {top_bound}, {self.x1}, {bottom_bound})")
                
                # Process only header/footer exclusions for cropping
                for exclusion in footer_header_exclusions:
                    # If exclusion is at the top of our region
                    if exclusion.bottom > self.top and exclusion.top <= self.top:
                        # Move top bound to exclude the header
                        top_bound = max(top_bound, exclusion.bottom)
                        if debug:
                            import logging
                            logger = logging.getLogger("natural_pdf.elements.region")
                            logger.debug(f"  Adjusted top bound to {top_bound} due to header exclusion")
                    
                    # If exclusion is at the bottom of our region
                    if exclusion.top < self.bottom and exclusion.bottom >= self.bottom:
                        # Move bottom bound to exclude the footer
                        bottom_bound = min(bottom_bound, exclusion.top)
                        if debug:
                            import logging
                            logger = logging.getLogger("natural_pdf.elements.region")
                            logger.debug(f"  Adjusted bottom bound to {bottom_bound} due to footer exclusion")
                        
                
                if debug:
                    import logging
                    logger = logging.getLogger("natural_pdf.elements.region")
                    logger.debug(f"  Final bounds after exclusion adjustment: ({self.x0}, {top_bound}, {self.x1}, {bottom_bound})")
                
                # If we still have a valid region after exclusions
                if top_bound < bottom_bound:
                    # Use direct crop with adjusted bounds
                    crop_bbox = (self.x0, top_bound, self.x1, bottom_bound)
                    cropped = self.page._page.crop(crop_bbox)
                    result = cropped.extract_text(keep_blank_chars=keep_blank_chars, **kwargs)
                    
                    if debug:
                        import logging
                        logger = logging.getLogger("natural_pdf.elements.region")
                        logger.debug(f"  Successfully extracted text using crop, got {len(result)} characters")
                    
                    # Skip the complex filtering approach
                    return result
                else:
                    # This would only happen if the region is entirely inside an exclusion zone
                    # or if both top and bottom of the region are excluded leaving no valid area
                    import logging
                    logger = logging.getLogger("natural_pdf.elements.region")
                    logger.debug(f"Region {self.bbox} completely covered by exclusions, returning empty string")
                    return ""
            # We have exclusions, but not all are headers/footers,
            # or we have a non-rectangular region
            else:
                if debug:
                    import logging
                    logger = logging.getLogger("natural_pdf.elements.region")
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
        import warnings
        import logging
        logger = logging.getLogger("natural_pdf.elements.region")
        
        if debug:
            logger.debug(f"Using element filtering approach for region {self.bbox}")
        
        # Get all elements in this region first
        all_elements = self.get_elements(apply_exclusions=False)
        
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
    
    def find_all(self, selector: str, apply_exclusions=True, **kwargs) -> 'ElementCollection':
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
        from natural_pdf.utils.ocr import OCRManager
        
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
        Create a new region expanded from this one.
        
        Args:
            left: Amount to expand left edge
            right: Amount to expand right edge
            top_expand: Amount to expand top edge (upward)
            bottom_expand: Amount to expand bottom edge (downward)
            width_factor: Factor to multiply width by
            height_factor: Factor to multiply height by
            top: (DEPRECATED, use top_expand) Amount to expand top edge (upward)
            bottom: (DEPRECATED, use bottom_expand) Amount to expand bottom edge (downward)
            
        Returns:
            New expanded Region
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
            # Current width and height
            current_width = new_x1 - new_x0
            current_height = new_bottom - new_top
            
            # Calculate new width and height
            new_width = current_width * width_factor
            new_height = current_height * height_factor
            
            # Calculate width and height differences
            width_diff = new_width - current_width
            height_diff = new_height - current_height
            
            # Adjust coordinates to maintain center point
            new_x0 -= width_diff / 2
            new_x1 += width_diff / 2
            new_top -= height_diff / 2
            new_bottom += height_diff / 2
        
        # Create new region with expanded bbox
        new_region = Region(self.page, (new_x0, new_top, new_x1, new_bottom))
        
        # Copy multi-page properties if present
        if self._spans_pages:
            new_region._spans_pages = True
            new_region._multi_page_elements = self._multi_page_elements
            new_region._page_range = self._page_range
            new_region.start_element = self.start_element
            new_region.end_element = self.end_element
            
        return new_region
    
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
        Create cell regions for a TATR-detected table.
        
        Returns:
            List of cell regions
        """
        if not (self.region_type == 'table' and self.model == 'tatr'):
            raise ValueError("Only works for TATR-detected table regions")
        
        # Find rows and columns that belong to this table
        rows = self.page.find_all(f'region[type=table-row][model=tatr]')
        columns = self.page.find_all(f'region[type=table-column][model=tatr]')
        
        # Filter to only include those that overlap with this table
        def is_in_table(element):
            element_center_x = (element.x0 + element.x1) / 2
            element_center_y = (element.top + element.bottom) / 2
            return (self.x0 <= element_center_x <= self.x1 and
                    self.top <= element_center_y <= self.bottom)
        
        table_rows = [r for r in rows if is_in_table(r)]
        table_columns = [c for c in columns if is_in_table(c)]
        
        # Sort rows and columns
        table_rows.sort(key=lambda r: r.top)
        table_columns.sort(key=lambda c: c.x0)
        
        # Create cells
        cells = []
        for row in table_rows:
            for column in table_columns:
                # Create cell region at the intersection
                cell = self.page.create_region(
                    column.x0, row.top, column.x1, row.bottom
                )
                # Set minimal metadata
                cell.source = 'derived'
                cell.region_type = 'table-cell'
                cell.model = 'tatr'
                
                cells.append(cell)
        
        return cells
        
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
        direct_matches = self.page.find_all(selector, region=self, **kwargs)
        
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
