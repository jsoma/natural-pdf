from typing import List, Optional, Dict, Any, Union, Callable, TypeVar, Generic, Iterator, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.region import Region

T = TypeVar('T')
P = TypeVar('P', bound='Page')

class ElementCollection(Generic[T]):
    """
    Collection of PDF elements with batch operations.
    """
    
    def __init__(self, elements: List[T]):
        """
        Initialize a collection of elements.
        
        Args:
            elements: List of Element objects
        """
        self._elements = elements or []
    
    def __len__(self) -> int:
        """Get the number of elements in the collection."""
        return len(self._elements)
    
    def __getitem__(self, index: int) -> 'Element':
        """Get an element by index."""
        return self._elements[index]
    
    def __iter__(self):
        """Iterate over elements."""
        return iter(self._elements)
    
    @property
    def elements(self) -> List['Element']:
        """Get the elements in this collection."""
        return self._elements
    
    @property
    def first(self) -> Optional['Element']:
        """Get the first element in the collection."""
        return self._elements[0] if self._elements else None
    
    @property
    def last(self) -> Optional['Element']:
        """Get the last element in the collection."""
        return self._elements[-1] if self._elements else None
        
    def highest(self) -> Optional['Element']:
        """
        Get element with the smallest top y-coordinate (highest on page).
        
        Raises:
            ValueError: If elements are on multiple pages
            
        Returns:
            Element with smallest top value or None if empty
        """
        if not self._elements:
            return None
            
        # Check if elements are on multiple pages
        if self._are_on_multiple_pages():
            raise ValueError("Cannot determine highest element across multiple pages")
        
        return min(self._elements, key=lambda e: e.top)
    
    def lowest(self) -> Optional['Element']:
        """
        Get element with the largest bottom y-coordinate (lowest on page).
        
        Raises:
            ValueError: If elements are on multiple pages
            
        Returns:
            Element with largest bottom value or None if empty
        """
        if not self._elements:
            return None
            
        # Check if elements are on multiple pages
        if self._are_on_multiple_pages():
            raise ValueError("Cannot determine lowest element across multiple pages")
        
        return max(self._elements, key=lambda e: e.bottom)
    
    def leftmost(self) -> Optional['Element']:
        """
        Get element with the smallest x0 coordinate (leftmost on page).
        
        Raises:
            ValueError: If elements are on multiple pages
            
        Returns:
            Element with smallest x0 value or None if empty
        """
        if not self._elements:
            return None
            
        # Check if elements are on multiple pages
        if self._are_on_multiple_pages():
            raise ValueError("Cannot determine leftmost element across multiple pages")
        
        return min(self._elements, key=lambda e: e.x0)
    
    def rightmost(self) -> Optional['Element']:
        """
        Get element with the largest x1 coordinate (rightmost on page).
        
        Raises:
            ValueError: If elements are on multiple pages
            
        Returns:
            Element with largest x1 value or None if empty
        """
        if not self._elements:
            return None
            
        # Check if elements are on multiple pages
        if self._are_on_multiple_pages():
            raise ValueError("Cannot determine rightmost element across multiple pages")
        
        return max(self._elements, key=lambda e: e.x1)
    
    def _are_on_multiple_pages(self) -> bool:
        """
        Check if elements in this collection span multiple pages.
        
        Returns:
            True if elements are on different pages, False otherwise
        """
        if not self._elements:
            return False
        
        # Get the page index of the first element
        if not hasattr(self._elements[0], 'page'):
            return False
            
        first_page_idx = self._elements[0].page.index
        
        # Check if any element is on a different page
        return any(hasattr(e, 'page') and e.page.index != first_page_idx for e in self._elements)
    
    def exclude_regions(self, regions: List['Region']) -> 'ElementCollection':
        """
        Remove elements that are within any of the specified regions.
        
        Args:
            regions: List of Region objects to exclude
            
        Returns:
            New ElementCollection with filtered elements
        """
        if not regions:
            return ElementCollection(self._elements)
        
        filtered = []
        for element in self._elements:
            exclude = False
            for region in regions:
                if region._is_element_in_region(element):
                    exclude = True
                    break
            if not exclude:
                filtered.append(element)
        
        return ElementCollection(filtered)
    
    def extract_text(self, preserve_whitespace=True, use_exclusions=True, **kwargs) -> str:
        """
        Extract text from all elements in the collection.
        
        Args:
            preserve_whitespace: Whether to keep blank characters (default: True)
            use_exclusions: Whether to apply exclusion regions (default: True)
            **kwargs: Additional extraction parameters
            
        Returns:
            Combined text from all elements
        """
        # Filter to just text-like elements
        text_elements = [e for e in self._elements if hasattr(e, 'extract_text')]
        
        # Sort elements in reading order (top-to-bottom, left-to-right)
        sorted_elements = sorted(text_elements, key=lambda e: (e.top, e.x0))
        
        # Extract text from each element
        texts = []
        for element in sorted_elements:
            # Extract text with new parameter names
            text = element.extract_text(preserve_whitespace=preserve_whitespace, use_exclusions=use_exclusions, **kwargs)
                
            if text:
                texts.append(text)
        
        return " ".join(texts)
    
    def filter(self, func: Callable[['Element'], bool]) -> 'ElementCollection':
        """
        Filter elements using a function.
        
        Args:
            func: Function that takes an element and returns True to keep it
            
        Returns:
            New ElementCollection with filtered elements
        """
        return ElementCollection([e for e in self._elements if func(e)])
    
    def sort(self, key=None, reverse=False) -> 'ElementCollection':
        """
        Sort elements by the given key function.
        
        Args:
            key: Function to generate a key for sorting
            reverse: Whether to sort in descending order
            
        Returns:
            Self for method chaining
        """
        self._elements.sort(key=key, reverse=reverse)
        return self
    
    def highlight(self, 
                 label: Optional[str] = None,
                 color: Optional[tuple] = None, 
                 use_color_cycling: bool = False,
                 cycle_colors: bool = False,
                 include_attrs: Optional[List[str]] = None,
                 existing: str = 'append') -> 'ElementCollection':  # Added for backward compatibility
        """
        Highlight all elements in the collection.
        
        Args:
            label: Optional label for the highlight
            color: Optional color for the highlight (RGBA tuple)
            use_color_cycling: Force color cycling even with no label (default: False)
            cycle_colors: Alias for use_color_cycling (deprecated, for backward compatibility)
            include_attrs: List of attribute names to display on the highlight (e.g., ['confidence', 'type'])
            existing: How to handle existing highlights - 'append' (default) or 'replace'
            
        Returns:
            Self for method chaining
        """
        # Use cycle_colors if provided (backward compatibility)
        color_cycle = use_color_cycling or cycle_colors
        
        # Get the highlight manager from the first element's page (if available)
        if self._elements and hasattr(self._elements[0], 'page'):
            page = self._elements[0].page
            if hasattr(page, '_highlight_mgr'):
                highlight_mgr = page._highlight_mgr
                
                # Add highlights for each element
                for element in self._elements:
                    # Check if element has polygon coordinates
                    if hasattr(element, 'has_polygon') and element.has_polygon:
                        # Use polygon highlight
                        highlight_mgr.add_polygon_highlight(
                            element.polygon, 
                            color, 
                            label, 
                            color_cycle,
                            element=element, 
                            include_attrs=include_attrs,
                            existing=existing if element is self._elements[0] else 'append'
                        )
                    else:
                        # Get the element's bounding box
                        bbox = (element.x0, element.top, element.x1, element.bottom)
                        # Add the highlight
                        highlight_mgr.add_highlight(
                            bbox, 
                            color, 
                            label, 
                            color_cycle,
                            element=element, 
                            include_attrs=include_attrs,
                            existing=existing if element is self._elements[0] else 'append'
                        )
        
        return self
        
    def show(self, 
            scale: float = 2.0,
            width: Optional[int] = None,
            labels: bool = True,
            legend_position: str = 'right',
            render_ocr: bool = False) -> 'Image.Image':
        """
        Show the page with this collection's elements highlighted.
        
        Args:
            scale: Scale factor for rendering
            width: Optional width for the output image in pixels
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            render_ocr: Whether to render OCR text with white background boxes
            
        Returns:
            PIL Image of the page with elements highlighted
        """
        # Use to_image to get the image
        return self.to_image(
            scale=scale,
            width=width,
            labels=labels,
            legend_position=legend_position,
            render_ocr=render_ocr
        )
        
    def save(self, 
            filename: str, 
            scale: float = 2.0,
            width: Optional[int] = None,
            labels: bool = True,
            legend_position: str = 'right',
            render_ocr: bool = False) -> 'ElementCollection':
        """
        Save the page with this collection's elements highlighted to an image file.
        
        Args:
            filename: Path to save the image to
            scale: Scale factor for rendering
            width: Optional width for the output image in pixels
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            render_ocr: Whether to render OCR text with white background boxes
            
        Returns:
            Self for method chaining
        """
        # Use to_image to generate and save the image
        self.to_image(
            path=filename, 
            scale=scale,
            width=width,
            labels=labels, 
            legend_position=legend_position,
            render_ocr=render_ocr
        )
        return self
        
    def to_image(self, 
            path: Optional[str] = None,
            scale: float = 2.0,
            width: Optional[int] = None,
            labels: bool = True,
            legend_position: str = 'right',
            render_ocr: bool = False) -> Optional['Image.Image']:
        """
        Generate an image of the page with this collection's elements highlighted,
        optionally saving it to a file.
        
        Args:
            path: Optional path to save the image to
            scale: Scale factor for rendering
            width: Optional width for the output image in pixels (height calculated to maintain aspect ratio)
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            render_ocr: Whether to render OCR text with white background boxes
            
        Returns:
            PIL Image of the page with elements highlighted, or None if no valid page
        """
        # Get the page from the first element (if available)
        if self._elements and hasattr(self._elements[0], 'page'):
            page = self._elements[0].page
            # Generate the image using to_image
            return page.to_image(
                path=path, 
                scale=scale,
                width=width,
                labels=labels, 
                legend_position=legend_position,
                render_ocr=render_ocr
            )
        return None

class PageCollection(Generic[P]):
    """
    A collection of PDF pages with cross-page operations.
    
    This class provides methods for working with multiple pages, such as finding
    elements across pages, extracting text from page ranges, and more.
    """
    
    def __init__(self, pages: List[P]):
        """
        Initialize a page collection.
        
        Args:
            pages: List of Page objects
        """
        self.pages = pages
    
    def __len__(self) -> int:
        """Return the number of pages in the collection."""
        return len(self.pages)
    
    def __getitem__(self, idx) -> Union[P, 'PageCollection[P]']:
        """Support indexing and slicing."""
        if isinstance(idx, slice):
            return PageCollection(self.pages[idx])
        return self.pages[idx]
    
    def __iter__(self) -> Iterator[P]:
        """Support iteration."""
        return iter(self.pages)
    
    def extract_text(self, keep_blank_chars=True, apply_exclusions=True, **kwargs) -> str:
        """
        Extract text from all pages in the collection.
        
        Args:
            keep_blank_chars: Whether to keep blank characters (default: True)
            apply_exclusions: Whether to apply exclusion regions (default: True)
            **kwargs: Additional extraction parameters
            
        Returns:
            Combined text from all pages
        """
        texts = []
        for page in self.pages:
            text = page.extract_text(
                keep_blank_chars=keep_blank_chars, 
                apply_exclusions=apply_exclusions,
                **kwargs
            )
            texts.append(text)
        
        return "\n".join(texts)
    
    def find(self, selector: str, apply_exclusions=True, **kwargs) -> Optional[T]:
        """
        Find the first element matching the selector across all pages.
        
        Args:
            selector: CSS-like selector string
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True)
            **kwargs: Additional filter parameters
            
        Returns:
            First matching element or None
        """
        for page in self.pages:
            element = page.find(selector, apply_exclusions=apply_exclusions, **kwargs)
            if element:
                return element
        return None
    
    def find_all(self, selector: str, apply_exclusions=True, **kwargs) -> ElementCollection:
        """
        Find all elements matching the selector across all pages.
        
        Args:
            selector: CSS-like selector string
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True)
            **kwargs: Additional filter parameters
            
        Returns:
            ElementCollection with matching elements from all pages
        """
        all_elements = []
        for page in self.pages:
            elements = page.find_all(selector, apply_exclusions=apply_exclusions, **kwargs)
            if elements:
                all_elements.extend(elements.elements)
        
        return ElementCollection(all_elements)
        
    def debug_ocr(self, output_path):
        """
        Generate an interactive HTML debug report for OCR results.
        
        This creates a single-file HTML report with:
        - Side-by-side view of image regions and OCR text
        - Confidence scores with color coding
        - Editable correction fields
        - Filtering and sorting options
        - Export functionality for corrected text
        
        Args:
            output_path: Path to save the HTML report
            
        Returns:
            Path to the generated HTML file
        """
        from natural_pdf.utils.ocr import debug_ocr_to_html
        return debug_ocr_to_html(self.pages, output_path)
    
    def get_sections(self, 
                   start_elements=None, 
                   end_elements=None,
                   new_section_on_page_break=False,
                   boundary_inclusion='both') -> List['Region']:
        """
        Extract sections from a page collection based on start/end elements.
        
        Args:
            start_elements: Elements or selector string that mark the start of sections
            end_elements: Elements or selector string that mark the end of sections
            new_section_on_page_break: Whether to start a new section at page boundaries (default: False)
            boundary_inclusion: How to include boundary elements: 'start', 'end', 'both', or 'none' (default: 'both')
            
        Returns:
            List of Region objects representing the extracted sections
        """
        # Find start and end elements across all pages
        if isinstance(start_elements, str):
            start_elements = self.find_all(start_elements).elements
            
        if isinstance(end_elements, str):
            end_elements = self.find_all(end_elements).elements
            
        # If no start elements, return empty list
        if not start_elements:
            return []
        
        # If there are page break boundaries, we'll need to add them
        if new_section_on_page_break:
            # For each page boundary, create virtual "end" and "start" elements
            for i in range(len(self.pages) - 1):
                # Add a virtual "end" element at the bottom of the current page
                page = self.pages[i]
                # If end_elements is None, initialize it as an empty list
                if end_elements is None:
                    end_elements = []
                
                # Create a region at the bottom of the page as an artificial end marker
                from natural_pdf.elements.region import Region
                bottom_region = Region(page, (0, page.height - 1, page.width, page.height))
                bottom_region.is_page_boundary = True  # Mark it as a special boundary
                end_elements.append(bottom_region)
                
                # Add a virtual "start" element at the top of the next page
                next_page = self.pages[i + 1]
                top_region = Region(next_page, (0, 0, next_page.width, 1))
                top_region.is_page_boundary = True  # Mark it as a special boundary
                start_elements.append(top_region)
        
        # Get all elements from all pages and sort them in document order
        all_elements = []
        for page in self.pages:
            elements = page.get_elements()
            all_elements.extend(elements)
            
        # Sort by page index, then vertical position, then horizontal position
        all_elements.sort(key=lambda e: (e.page.index, e.top, e.x0))
        
        # Mark section boundaries
        section_boundaries = []
        
        # Add start element boundaries
        for element in start_elements:
            if element in all_elements:
                idx = all_elements.index(element)
                section_boundaries.append({
                    'index': idx,
                    'element': element,
                    'type': 'start',
                    'page_idx': element.page.index
                })
            elif hasattr(element, 'is_page_boundary') and element.is_page_boundary:
                # This is a virtual page boundary element
                section_boundaries.append({
                    'index': -1,  # Special index for page boundaries
                    'element': element,
                    'type': 'start',
                    'page_idx': element.page.index
                })
        
        # Add end element boundaries if provided
        if end_elements:
            for element in end_elements:
                if element in all_elements:
                    idx = all_elements.index(element)
                    section_boundaries.append({
                        'index': idx,
                        'element': element,
                        'type': 'end',
                        'page_idx': element.page.index
                    })
                elif hasattr(element, 'is_page_boundary') and element.is_page_boundary:
                    # This is a virtual page boundary element
                    section_boundaries.append({
                        'index': -1,  # Special index for page boundaries
                        'element': element,
                        'type': 'end',
                        'page_idx': element.page.index
                    })
        
        # Sort boundaries by page index, then by actual document position
        section_boundaries.sort(key=lambda x: (x['page_idx'], 
                                             x['index'] if x['index'] != -1 else 
                                             (0 if x['type'] == 'start' else float('inf'))))
        
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
                
                # If both elements are on the same page, use the page's get_section_between
                if start_element.page == end_element.page:
                    section = start_element.page.get_section_between(
                        start_element,
                        end_element,
                        boundary_inclusion
                    )
                    sections.append(section)
                else:
                    # Create a multi-page section
                    from natural_pdf.elements.region import Region
                    
                    # Get the start and end pages
                    start_page = start_element.page
                    end_page = end_element.page
                    
                    # Create a combined region
                    combined_region = Region(
                        start_page,
                        (0, start_element.top, start_page.width, start_page.height)
                    )
                    combined_region._spans_pages = True
                    combined_region._page_range = (start_page.index, end_page.index)
                    combined_region.start_element = start_element
                    combined_region.end_element = end_element
                    
                    # Get all elements that fall within this multi-page region
                    combined_elements = []
                    
                    # Get elements from the first page
                    first_page_elements = [e for e in all_elements 
                                         if e.page == start_page and e.top >= start_element.top]
                    combined_elements.extend(first_page_elements)
                    
                    # Get elements from middle pages (if any)
                    for page_idx in range(start_page.index + 1, end_page.index):
                        middle_page_elements = [e for e in all_elements if e.page.index == page_idx]
                        combined_elements.extend(middle_page_elements)
                    
                    # Get elements from the last page
                    last_page_elements = [e for e in all_elements 
                                        if e.page == end_page and e.bottom <= end_element.bottom]
                    combined_elements.extend(last_page_elements)
                    
                    # Store the elements in the combined region
                    combined_region._multi_page_elements = combined_elements
                    
                    sections.append(combined_region)
                
                current_start = None
                
            # If it's another start boundary and we have a current start (for splitting by starts only)
            elif boundary['type'] == 'start' and current_start is not None and not end_elements:
                # Create a section from current_start to just before this boundary
                start_element = current_start['element']
                
                # Find the last element before this boundary on the same page
                if start_element.page == boundary['element'].page:
                    # Find elements on this page
                    page_elements = [e for e in all_elements if e.page == start_element.page]
                    # Sort by position
                    page_elements.sort(key=lambda e: (e.top, e.x0))
                    
                    # Find the last element before the boundary
                    end_idx = page_elements.index(boundary['element']) - 1 if boundary['element'] in page_elements else -1
                    end_element = page_elements[end_idx] if end_idx >= 0 else None
                    
                    # Create the section
                    section = start_element.page.get_section_between(
                        start_element,
                        end_element,
                        boundary_inclusion
                    )
                    sections.append(section)
                else:
                    # Cross-page section - create from current_start to the end of its page
                    from natural_pdf.elements.region import Region
                    start_page = start_element.page
                    
                    region = Region(
                        start_page, 
                        (0, start_element.top, start_page.width, start_page.height)
                    )
                    region.start_element = start_element
                    sections.append(region)
                
                current_start = boundary
        
        # Handle the last section if we have a current start
        if current_start is not None:
            start_element = current_start['element']
            start_page = start_element.page
            
            if end_elements:
                # With end_elements, we need an explicit end - use the last element
                # on the last page of the collection
                last_page = self.pages[-1]
                last_page_elements = [e for e in all_elements if e.page == last_page]
                last_page_elements.sort(key=lambda e: (e.top, e.x0))
                end_element = last_page_elements[-1] if last_page_elements else None
                
                # Create a multi-page section
                from natural_pdf.elements.region import Region
                
                if start_page == last_page:
                    # Simple case - both on same page
                    section = start_page.get_section_between(
                        start_element,
                        end_element,
                        boundary_inclusion
                    )
                    sections.append(section)
                else:
                    # Create a multi-page section
                    combined_region = Region(
                        start_page,
                        (0, start_element.top, start_page.width, start_page.height)
                    )
                    combined_region._spans_pages = True
                    combined_region._page_range = (start_page.index, last_page.index)
                    combined_region.start_element = start_element
                    combined_region.end_element = end_element
                    
                    # Get all elements that fall within this multi-page region
                    combined_elements = []
                    
                    # Get elements from the first page
                    first_page_elements = [e for e in all_elements 
                                         if e.page == start_page and e.top >= start_element.top]
                    combined_elements.extend(first_page_elements)
                    
                    # Get elements from middle pages (if any)
                    for page_idx in range(start_page.index + 1, last_page.index):
                        middle_page_elements = [e for e in all_elements if e.page.index == page_idx]
                        combined_elements.extend(middle_page_elements)
                    
                    # Get elements from the last page
                    last_page_elements = [e for e in all_elements 
                                        if e.page == last_page and (end_element is None or e.bottom <= end_element.bottom)]
                    combined_elements.extend(last_page_elements)
                    
                    # Store the elements in the combined region
                    combined_region._multi_page_elements = combined_elements
                    
                    sections.append(combined_region)
            else:
                # With start_elements only, create a section to the end of the current page
                from natural_pdf.elements.region import Region
                region = Region(
                    start_page, 
                    (0, start_element.top, start_page.width, start_page.height)
                )
                region.start_element = start_element
                sections.append(region)
                
        return sections