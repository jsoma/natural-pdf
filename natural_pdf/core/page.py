import pdfplumber
import os
import logging
import tempfile
from typing import List, Optional, Union, Any, Dict, Callable, TYPE_CHECKING, Tuple
from PIL import Image
import base64
import io
import json

from natural_pdf.elements.collections import ElementCollection

if TYPE_CHECKING:
    import pdfplumber
    from natural_pdf.core.pdf import PDF
    from natural_pdf.elements.collections import ElementCollection
    from natural_pdf.core.highlighting_service import HighlightingService
    from natural_pdf.elements.base import Element

from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement
from natural_pdf.analyzers.layout.layout_manager import LayoutManager
from natural_pdf.analyzers.layout.layout_options import LayoutOptions
from natural_pdf.ocr import OCROptions
from natural_pdf.ocr import OCRManager
from natural_pdf.core.element_manager import ElementManager
from natural_pdf.analyzers.layout.layout_analyzer import LayoutAnalyzer
from natural_pdf.analyzers.text_structure import TextStyleAnalyzer
from natural_pdf.analyzers.text_options import TextStyleOptions
from natural_pdf.widgets import InteractiveViewerWidget
from natural_pdf.widgets.viewer import SimpleInteractiveViewerWidget

logger = logging.getLogger(__name__)

class Page:
    """
    Enhanced Page wrapper built on top of pdfplumber.Page.
    
    This class provides a fluent interface for working with PDF pages,
    with improved selection, navigation, extraction, and question-answering capabilities.
    """
    
    def __init__(self, page: 'pdfplumber.page.Page', parent: 'PDF', index: int, font_attrs=None):
        """
        Initialize a page wrapper.
        
        Args:
            page: pdfplumber page object
            parent: Parent PDF object
            index: Index of this page in the PDF (0-based)
            font_attrs: Font attributes to consider when grouping characters into words.
        """
        self._page = page
        self._parent = parent
        self._index = index
        self._text_styles = None  # Lazy-loaded text style analyzer results
        self._exclusions = []  # List to store exclusion functions/regions
        
        # Region management
        self._regions = {
            'detected': [],  # Layout detection results
            'named': {},     # Named regions (name -> region)
        }
        
        # Initialize ElementManager
        self._element_mgr = ElementManager(self, font_attrs)

        # --- Get OCR Manager Instance ---
        if OCRManager and hasattr(parent, '_ocr_manager') and isinstance(parent._ocr_manager, OCRManager):
            self._ocr_manager = parent._ocr_manager
            logger.debug(f"Page {self.number}: Using OCRManager instance from parent PDF.")
        else:
            self._ocr_manager = None
            if OCRManager:
                 logger.warning(f"Page {self.number}: OCRManager instance not found on parent PDF object.")

        # --- Get Layout Manager Instance ---
        if LayoutManager and hasattr(parent, '_layout_manager') and isinstance(parent._layout_manager, LayoutManager):
            self._layout_manager = parent._layout_manager
            logger.debug(f"Page {self.number}: Using LayoutManager instance from parent PDF.")
        else:
            self._layout_manager = None
            if LayoutManager:
                 logger.warning(f"Page {self.number}: LayoutManager instance not found on parent PDF object. Layout analysis will fail.")

        # Initialize the internal variable with a single underscore
        self._layout_analyzer = None 

    @property
    def number(self) -> int:
        """Get page number (1-based)."""
        return self._page.page_number
    
    @property
    def index(self) -> int:
        """Get page index (0-based)."""
        return self._index
    
    @property
    def width(self) -> float:
        """Get page width."""
        return self._page.width
    
    @property
    def height(self) -> float:
        """Get page height."""
        return self._page.height

    # --- Highlighting Service Accessor ---
    @property
    def _highlighter(self) -> 'HighlightingService':
         """Provides access to the parent PDF's HighlightingService."""
         if not hasattr(self._parent, 'highlighter'):
              # This should ideally not happen if PDF.__init__ works correctly
              raise AttributeError("Parent PDF object does not have a 'highlighter' attribute.")
         return self._parent.highlighter

    def add_exclusion(self, exclusion_func_or_region: Union[Callable[['Page'], Region], Region]) -> 'Page':
        """
        Add an exclusion to the page. Text from these regions will be excluded from extraction.
        
        Args:
            exclusion_func_or_region: Either a Region object or a function that takes a Page
                                      and returns a Region to exclude
            
        Returns:
            Self for method chaining
        """
        self._exclusions.append(exclusion_func_or_region)
        return self
        
    def add_region(self, region: Region, name: Optional[str] = None) -> 'Page':
        """
        Add a region to the page.
        
        Args:
            region: Region object to add
            name: Optional name for the region
            
        Returns:
            Self for method chaining
        """
        # Check if it's actually a Region object
        if not isinstance(region, Region):
            raise TypeError("region must be a Region object")
            
        # Set the source and name
        region.source = 'named'
        
        if name:
            region.name = name
            # Add to named regions dictionary (overwriting if name already exists)
            self._regions['named'][name] = region
        else:
            # Add to detected regions list (unnamed but registered)
            self._regions['detected'].append(region)
            
        # Add to element manager for selector queries
        self._element_mgr.add_region(region)
                
        return self
                
    def add_regions(self, regions: List[Region], prefix: Optional[str] = None) -> 'Page':
        """
        Add multiple regions to the page.
        
        Args:
            regions: List of Region objects to add
            prefix: Optional prefix for automatic naming (regions will be named prefix_1, prefix_2, etc.)
            
        Returns:
            Self for method chaining
        """
        if prefix:
            # Add with automatic sequential naming
            for i, region in enumerate(regions):
                self.add_region(region, name=f"{prefix}_{i+1}")
        else:
            # Add without names
            for region in regions:
                self.add_region(region)
                
        return self
    
    def _get_exclusion_regions(self, include_callable=True, debug=False) -> List[Region]:
        """
        Get all exclusion regions for this page.
        
        Args:
            include_callable: Whether to evaluate callable exclusion functions
            debug: Enable verbose debug logging for exclusion evaluation
            
        Returns:
            List of Region objects to exclude
        """
        regions = []
        
        # Track exclusion results for debugging
        if debug:
            print(f"\nPage {self.index}: Evaluating {len(self._exclusions)} exclusions")
            
        for i, exclusion in enumerate(self._exclusions):
            # Get exclusion label if it's a tuple from PDF level
            exclusion_label = f"exclusion {i}"
            original_exclusion = exclusion
            
            # Check if it's a tuple from PDF.add_exclusion
            if isinstance(exclusion, tuple) and len(exclusion) == 2 and callable(exclusion[0]):
                # This is likely from PDF.add_exclusion with (func, label)
                exclusion_func, label = exclusion
                if label:
                    exclusion_label = label
                exclusion = exclusion_func
            
            # Process callable exclusion functions
            if callable(exclusion) and include_callable:
                # It's a function, call it with this page
                try:
                    if debug:
                        print(f"  - Evaluating callable {exclusion_label}...")
                    
                    # Create a temporary copy of exclusions to avoid recursion
                    original_exclusions = self._exclusions
                    self._exclusions = []  # Temporarily clear exclusions
                    
                    # Call the function
                    region = exclusion(self)
                    
                    # Restore exclusions
                    self._exclusions = original_exclusions
                    
                    if region:
                        regions.append(region)
                        if debug:
                            print(f"    ✓ Added region: {region}")
                    else:
                        if debug:
                            print(f"    ✗ Function returned None, no region added")
                            
                except Exception as e:
                    error_msg = f"Error in {exclusion_label} for page {self.index}: {e}"
                    print(error_msg)
                    # Print more detailed traceback for debugging
                    import traceback
                    print(f"    Traceback: {traceback.format_exc().splitlines()[-3:]}")
            
            # Process direct Region objects
            elif not callable(exclusion):
                # It's already a Region object
                regions.append(exclusion)
                if debug:
                    print(f"  - Added direct region: {exclusion}")
        
        if debug:
            print(f"Page {self.index}: Found {len(regions)} valid exclusion regions")
            
        return regions

    def _filter_elements_by_exclusions(self, elements: List['Element'], debug_exclusions: bool = False) -> List['Element']:
        """
        Filters a list of elements, removing those within the page's exclusion regions.

        Args:
            elements: The list of elements to filter.
            debug_exclusions: Whether to output detailed exclusion debugging info (default: False).

        Returns:
            A new list containing only the elements not falling within any exclusion region.
        """
        if not self._exclusions:
            if debug_exclusions:
                print(f"Page {self.index}: No exclusions defined, returning all {len(elements)} elements.")
            return elements

        # Get all exclusion regions, including evaluating callable functions
        exclusion_regions = self._get_exclusion_regions(include_callable=True, debug=debug_exclusions)

        if not exclusion_regions:
            if debug_exclusions:
                print(f"Page {self.index}: No valid exclusion regions found, returning all {len(elements)} elements.")
            return elements

        if debug_exclusions:
            print(f"Page {self.index}: Applying {len(exclusion_regions)} exclusion regions to {len(elements)} elements.")

        filtered_elements = []
        excluded_count = 0
        for element in elements:
            exclude = False
            for region in exclusion_regions:
                # Use the region's method to check if the element is inside
                if region._is_element_in_region(element):
                    exclude = True
                    excluded_count += 1
                    break  # No need to check other regions for this element
            if not exclude:
                filtered_elements.append(element)

        if debug_exclusions:
            print(f"Page {self.index}: Excluded {excluded_count} elements, keeping {len(filtered_elements)}.")

        return filtered_elements

    def find(self, selector: str, apply_exclusions=True, regex=False, case=True, **kwargs) -> Any:
        """
        Find first element on this page matching selector.

        Args:
            selector: CSS-like selector string
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True)
            regex: Whether to use regex for text search in :contains (default: False)
            case: Whether to do case-sensitive text search (default: True)
            **kwargs: Additional filter parameters

        Returns:
            Element object or None if not found
        """
        from natural_pdf.selectors.parser import parse_selector
        selector_obj = parse_selector(selector)
        
        # Pass regex and case flags to selector function
        kwargs['regex'] = regex
        kwargs['case'] = case
        
        # First get all matching elements without applying exclusions initially within _apply_selector
        results_collection = self._apply_selector(selector_obj, **kwargs) # _apply_selector doesn't filter
        
        # Filter the results based on exclusions if requested
        if apply_exclusions and self._exclusions and results_collection:
            filtered_elements = self._filter_elements_by_exclusions(results_collection.elements)
            # Return the first element from the filtered list
            return filtered_elements[0] if filtered_elements else None
        elif results_collection:
            # Return the first element from the unfiltered results
            return results_collection.first
        else:
            return None

    def find_all(self, selector: str, apply_exclusions=True, regex=False, case=True, **kwargs) -> 'ElementCollection':
        """
        Find all elements on this page matching selector.

        Args:
            selector: CSS-like selector string
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True)
            regex: Whether to use regex for text search in :contains (default: False)
            case: Whether to do case-sensitive text search (default: True)
            **kwargs: Additional filter parameters
            
        Returns:
            ElementCollection with matching elements
        """
        from natural_pdf.selectors.parser import parse_selector
        selector_obj = parse_selector(selector)
        
        # Pass regex and case flags to selector function
        kwargs['regex'] = regex
        kwargs['case'] = case
        
        # First get all matching elements without applying exclusions initially within _apply_selector
        results_collection = self._apply_selector(selector_obj, **kwargs) # _apply_selector doesn't filter
        
        # Filter the results based on exclusions if requested
        if apply_exclusions and self._exclusions and results_collection:
            filtered_elements = self._filter_elements_by_exclusions(results_collection.elements)
            return ElementCollection(filtered_elements)
        else:
            # Return the unfiltered collection
            return results_collection
    
    def _apply_selector(self, selector_obj: Dict, **kwargs) -> 'ElementCollection': # Removed apply_exclusions arg
        """
        Apply selector to page elements.
        Exclusions are now handled by the calling methods (find, find_all) if requested.
        
        Args:
            selector_obj: Parsed selector dictionary
            **kwargs: Additional filter parameters including 'regex' and 'case'
            
        Returns:
            ElementCollection of matching elements (unfiltered by exclusions)
        """
        from natural_pdf.selectors.parser import selector_to_filter_func
        
        # Get element type to filter
        element_type = selector_obj.get('type', 'any').lower()
        
        # Determine which elements to search based on element type
        elements_to_search = []
        if element_type == 'any':
            elements_to_search = self._element_mgr.get_all_elements()
        elif element_type == 'text':
            elements_to_search = self._element_mgr.words
        elif element_type == 'char':
            elements_to_search = self._element_mgr.chars
        elif element_type == 'word':
            elements_to_search = self._element_mgr.words
        elif element_type == 'rect' or element_type == 'rectangle':
            elements_to_search = self._element_mgr.rects
        elif element_type == 'line':
            elements_to_search = self._element_mgr.lines
        elif element_type == 'region':
            elements_to_search = self._element_mgr.regions
        else:
            elements_to_search = self._element_mgr.get_all_elements()
        
        # Create filter function from selector, passing any additional parameters
        filter_func = selector_to_filter_func(selector_obj, **kwargs)
        
        # Apply the filter to matching elements
        matching_elements = [element for element in elements_to_search if filter_func(element)]
        
        # Handle spatial pseudo-classes that require relationship checking
        for pseudo in selector_obj.get('pseudo_classes', []):
            name = pseudo.get('name')
            args = pseudo.get('args', '')
            
            if name in ('above', 'below', 'near', 'left-of', 'right-of'):
                # Find the reference element first
                from natural_pdf.selectors.parser import parse_selector
                ref_selector = parse_selector(args) if isinstance(args, str) else args
                # Recursively call _apply_selector for reference element (exclusions handled later)
                ref_elements = self._apply_selector(ref_selector, **kwargs) 
                
                if not ref_elements:
                    return ElementCollection([])
                
                ref_element = ref_elements.first
                if not ref_element: continue
                
                # Filter elements based on spatial relationship
                if name == 'above':
                    matching_elements = [el for el in matching_elements if hasattr(el, 'bottom') and hasattr(ref_element, 'top') and el.bottom <= ref_element.top]
                elif name == 'below':
                    matching_elements = [el for el in matching_elements if hasattr(el, 'top') and hasattr(ref_element, 'bottom') and el.top >= ref_element.bottom]
                elif name == 'left-of':
                    matching_elements = [el for el in matching_elements if hasattr(el, 'x1') and hasattr(ref_element, 'x0') and el.x1 <= ref_element.x0]
                elif name == 'right-of':
                    matching_elements = [el for el in matching_elements if hasattr(el, 'x0') and hasattr(ref_element, 'x1') and el.x0 >= ref_element.x1]
                elif name == 'near':
                    def distance(el1, el2):
                         if not (hasattr(el1, 'x0') and hasattr(el1, 'x1') and hasattr(el1, 'top') and hasattr(el1, 'bottom') and
                                 hasattr(el2, 'x0') and hasattr(el2, 'x1') and hasattr(el2, 'top') and hasattr(el2, 'bottom')):
                             return float('inf') # Cannot calculate distance
                         el1_center_x = (el1.x0 + el1.x1) / 2
                         el1_center_y = (el1.top + el1.bottom) / 2
                         el2_center_x = (el2.x0 + el2.x1) / 2
                         el2_center_y = (el2.top + el2.bottom) / 2
                         return ((el1_center_x - el2_center_x) ** 2 + (el1_center_y - el2_center_y) ** 2) ** 0.5
                    
                    threshold = kwargs.get('near_threshold', 50)
                    matching_elements = [el for el in matching_elements if distance(el, ref_element) <= threshold]
        
        # Sort elements in reading order if requested
        if kwargs.get('reading_order', True):
            if all(hasattr(el, 'top') and hasattr(el, 'x0') for el in matching_elements):
                 matching_elements.sort(key=lambda el: (el.top, el.x0))
            else:
                 logger.warning("Cannot sort elements in reading order: Missing required attributes (top, x0).")
        
        # Create result collection - exclusions are handled by the calling methods (find, find_all)
        result = ElementCollection(matching_elements)
                
        return result

    def create_region(self, x0: float, top: float, x1: float, bottom: float) -> Any:
        """
        Create a region on this page with the specified coordinates.
        
        Args:
            x0: Left x-coordinate
            top: Top y-coordinate
            x1: Right x-coordinate
            bottom: Bottom y-coordinate
            
        Returns:
            Region object for the specified coordinates
        """
        from natural_pdf.elements.region import Region
        return Region(self, (x0, top, x1, bottom))
        
    def region(self, left: float = None, top: float = None, right: float = None, bottom: float = None, 
              width: str = "full") -> Any:
        """
        Create a region on this page with more intuitive named parameters.
        
        Args:
            left: Left x-coordinate (default: 0)
            top: Top y-coordinate (default: 0)
            right: Right x-coordinate (default: page width)
            bottom: Bottom y-coordinate (default: page height)
            width: Width mode - "full" for full page width or "element" for element width
            
        Returns:
            Region object for the specified coordinates
            
        Examples:
            >>> page.region(top=100, bottom=200)  # Full width from y=100 to y=200
            >>> page.region(left=50, right=150, top=100, bottom=200)  # Specific rectangle
        """
        # Handle defaults
        left = 0 if left is None else left
        top = 0 if top is None else top
        right = self.width if right is None else right
        bottom = self.height if bottom is None else bottom
        
        # Handle width parameter
        if width == "full":
            left = 0
            right = self.width
        elif width != "element":
            raise ValueError("Width must be 'full' or 'element'")
            
        from natural_pdf.elements.region import Region
        region = Region(self, (left, top, right, bottom))
        return region
        
    def get_elements(self, apply_exclusions=True, debug_exclusions: bool = False) -> List['Element']:
        """
        Get all elements on this page.
        
        Args:
            apply_exclusions: Whether to apply exclusion regions (default: True).
            debug_exclusions: Whether to output detailed exclusion debugging info (default: False).
            
        Returns:
            List of all elements on the page, potentially filtered by exclusions.
        """
        # Get all elements from the element manager
        all_elements = self._element_mgr.get_all_elements()
        
        # Apply exclusions if requested
        if apply_exclusions and self._exclusions:
            return self._filter_elements_by_exclusions(all_elements, debug_exclusions=debug_exclusions)
        else:
            if debug_exclusions:
                 print(f"Page {self.index}: get_elements returning all {len(all_elements)} elements (exclusions not applied).")
            return all_elements
        
    def filter_elements(self, elements: List['Element'], selector: str, **kwargs) -> List['Element']:
        """
        Filter a list of elements based on a selector.
        
        Args:
            elements: List of elements to filter
            selector: CSS-like selector string
            **kwargs: Additional filter parameters
            
        Returns:
            List of elements that match the selector
        """
        from natural_pdf.selectors.parser import parse_selector, selector_to_filter_func
        
        # Parse the selector
        selector_obj = parse_selector(selector)
        
        # Create filter function from selector
        filter_func = selector_to_filter_func(selector_obj, **kwargs)
        
        # Apply the filter to the elements
        matching_elements = [element for element in elements if filter_func(element)]
        
        # Sort elements in reading order if requested
        if kwargs.get('reading_order', True):
            if all(hasattr(el, 'top') and hasattr(el, 'x0') for el in matching_elements):
                 matching_elements.sort(key=lambda el: (el.top, el.x0))
            else:
                 logger.warning("Cannot sort elements in reading order: Missing required attributes (top, x0).")
        
        return matching_elements
    
    def until(self, selector: str, include_endpoint: bool = True, **kwargs) -> Any:
        """
        Select content from the top of the page until matching selector.

        Args:
            selector: CSS-like selector string
            include_endpoint: Whether to include the endpoint element in the region
            **kwargs: Additional selection parameters
            
        Returns:
            Region object representing the selected content
            
        Examples:
            >>> page.until('text:contains("Conclusion")')  # Select from top to conclusion
            >>> page.until('line[width>=2]', include_endpoint=False)  # Select up to thick line
        """
        # Find the target element 
        target = self.find(selector, **kwargs)
        if not target:
            # If target not found, return a default region (full page)
            from natural_pdf.elements.region import Region
            return Region(self, (0, 0, self.width, self.height))
            
        # Create a region from the top of the page to the target
        from natural_pdf.elements.region import Region
        # Ensure target has positional attributes before using them
        target_top = getattr(target, 'top', 0)
        target_bottom = getattr(target, 'bottom', self.height)

        if include_endpoint:
            # Include the target element
            region = Region(self, (0, 0, self.width, target_bottom))
        else:
            # Up to the target element
            region = Region(self, (0, 0, self.width, target_top))
            
        region.end_element = target
        return region

    
    def crop(self, bbox=None, **kwargs) -> Any:
        """
        Crop the page to the specified bounding box.

        This is a direct wrapper around pdfplumber's crop method.
        
        Args:
            bbox: Bounding box (x0, top, x1, bottom) or None
            **kwargs: Additional parameters (top, bottom, left, right)

        Returns:
            Cropped page object (pdfplumber.Page)
        """
        # Returns the pdfplumber page object, not a natural-pdf Page
        return self._page.crop(bbox, **kwargs)

    def extract_text(self, 
                  preserve_whitespace=True,
                  use_exclusions=True,
                  debug_exclusions=False, **kwargs) -> str:
        """
        Extract text from this page, respecting any exclusion regions.
        
        Args:
            preserve_whitespace: Whether to keep blank characters (default: True)
            use_exclusions: Whether to apply exclusion regions (default: True)
            debug_exclusions: Whether to output detailed exclusion debugging info (default: False)
            **kwargs: Additional extraction parameters passed to pdfplumber
            
        Returns:
            Extracted text as string
        """
        if not use_exclusions or not self._exclusions:
            # If no exclusions or exclusions disabled, use regular extraction
            if debug_exclusions:
                print(f"Page {self.index}: Extracting text via pdfplumber (exclusions not applied).")
            # Note: pdfplumber still uses keep_blank_chars parameter
            return self._page.extract_text(keep_blank_chars=preserve_whitespace, **kwargs)
        
        # --- Exclusion Logic ---
        # 1. Get all potentially relevant text elements (words)
        all_text_elements = self.words # Use the words property
        if debug_exclusions:
            print(f"Page {self.index}: Starting text extraction with {len(all_text_elements)} words before exclusion.")

        # 2. Filter elements using the centralized method
        filtered_elements = self._filter_elements_by_exclusions(all_text_elements, debug_exclusions=debug_exclusions)

        # 3. Extract text from the filtered elements
        collection = ElementCollection(filtered_elements)
        # Ensure elements are sorted for logical text flow (might be redundant if self.words is sorted)
        if all(hasattr(el, 'top') and hasattr(el, 'x0') for el in collection.elements):
             collection.sort(key=lambda el: (el.top, el.x0))
        
        # Join text, handling potential missing text attributes gracefully
        result = " ".join(getattr(el, 'text', '') for el in collection.elements)
                
        if debug_exclusions:
            print(f"Page {self.index}: Extracted {len(result)} characters of text with exclusions applied.")
            
        return result

    def extract_table(self, table_settings={}) -> List[Any]:
        """
        Extract the largest table from this page.
        
        Args:
            table_settings: Additional extraction parameters
            
        Returns:
            List of extracted tables (or None if no table found)
        """
        # pdfplumber returns None if no table found
        return self._page.extract_table(table_settings)

    def extract_tables(self, table_settings={}) -> List[Any]:
        """
        Extract tables from this page.
        
        Args:
            table_settings: Additional extraction parameters
            
        Returns:
            List of extracted tables
        """
        # pdfplumber returns list of tables
        return self._page.extract_tables(table_settings)

    def _load_elements(self):
        """Load all elements from the page via ElementManager."""
        self._element_mgr.load_elements()
    
    def _create_char_elements(self):
        """DEPRECATED: Use self._element_mgr.chars"""
        logger.warning("_create_char_elements is deprecated. Access via self._element_mgr.chars.")
        return self._element_mgr.chars # Delegate

    def _process_font_information(self, char_dict):
         """DEPRECATED: Handled by ElementManager"""
         logger.warning("_process_font_information is deprecated. Handled by ElementManager.")
         # ElementManager handles this internally
         pass 

    def _group_chars_into_words(self, keep_spaces=True, font_attrs=None):
        """DEPRECATED: Use self._element_mgr.words"""
        logger.warning("_group_chars_into_words is deprecated. Access via self._element_mgr.words.")
        return self._element_mgr.words # Delegate

    def _process_line_into_words(self, line_chars, keep_spaces, font_attrs):
        """DEPRECATED: Handled by ElementManager"""
        logger.warning("_process_line_into_words is deprecated. Handled by ElementManager.")
        pass
    
    def _check_font_attributes_match(self, char, prev_char, font_attrs):
        """DEPRECATED: Handled by ElementManager"""
        logger.warning("_check_font_attributes_match is deprecated. Handled by ElementManager.")
        pass
    
    def _create_word_element(self, chars, font_attrs):
        """DEPRECATED: Handled by ElementManager"""
        logger.warning("_create_word_element is deprecated. Handled by ElementManager.")
        pass

    @property
    def chars(self) -> List[Any]:
        """Get all character elements on this page."""
        return self._element_mgr.chars
    
    @property
    def words(self) -> List[Any]:
        """Get all word elements on this page."""
        return self._element_mgr.words
    
    @property
    def rects(self) -> List[Any]:
        """Get all rectangle elements on this page."""
        return self._element_mgr.rects
    
    @property
    def lines(self) -> List[Any]:
        """Get all line elements on this page."""
        return self._element_mgr.lines
    
    def highlight(self, 
                 bbox: Optional[Tuple[float, float, float, float]] = None, 
                 color: Optional[Union[Tuple, str]] = None, 
                 label: Optional[str] = None,
                 use_color_cycling: bool = False,
                 element: Optional[Any] = None,
                 include_attrs: Optional[List[str]] = None,
                 existing: str = 'append') -> 'Page':
        """
        Highlight a bounding box or the entire page.
        Delegates to the central HighlightingService.
        
        Args:
            bbox: Bounding box (x0, top, x1, bottom). If None, highlight entire page.
            color: RGBA color tuple/string for the highlight.
            label: Optional label for the highlight.
            use_color_cycling: If True and no label/color, use next cycle color.
            element: Optional original element being highlighted (for attribute extraction).
            include_attrs: List of attribute names from 'element' to display.
            existing: How to handle existing highlights ('append' or 'replace').

        Returns:
            Self for method chaining.
        """
        target_bbox = bbox if bbox is not None else (0, 0, self.width, self.height)
        self._highlighter.add(
            page_index=self.index,
            bbox=target_bbox,
            color=color,
            label=label,
            use_color_cycling=use_color_cycling,
            element=element,
            include_attrs=include_attrs,
            existing=existing
        )
        return self

    def highlight_polygon(
        self, 
        polygon: List[Tuple[float, float]],
        color: Optional[Union[Tuple, str]] = None, 
        label: Optional[str] = None,
        use_color_cycling: bool = False,
        element: Optional[Any] = None,
        include_attrs: Optional[List[str]] = None,
        existing: str = 'append') -> 'Page':
        """
        Highlight a polygon shape on the page.
        Delegates to the central HighlightingService.
        
        Args:
            polygon: List of (x, y) points defining the polygon.
            color: RGBA color tuple/string for the highlight.
            label: Optional label for the highlight.
            use_color_cycling: If True and no label/color, use next cycle color.
            element: Optional original element being highlighted (for attribute extraction).
            include_attrs: List of attribute names from 'element' to display.
            existing: How to handle existing highlights ('append' or 'replace').

        Returns:
            Self for method chaining.
        """
        self._highlighter.add_polygon(
            page_index=self.index,
            polygon=polygon,
            color=color,
            label=label,
            use_color_cycling=use_color_cycling,
            element=element,
            include_attrs=include_attrs,
            existing=existing
        )
        return self
    
    def show(self, 
            scale: float = 2.0,
            width: Optional[int] = None,
            labels: bool = True,
            legend_position: str = 'right',
            render_ocr: bool = False) -> Optional[Image.Image]:
        """
        Generates and returns an image of the page with persistent highlights rendered.
        
        Args:
            scale: Scale factor for rendering.
            width: Optional width for the output image.
            labels: Whether to include a legend for labels.
            legend_position: Position of the legend.
            render_ocr: Whether to render OCR text.
            
        Returns:
            PIL Image object of the page with highlights, or None if rendering fails.
        """
        return self.to_image(
            scale=scale,
            width=width,
            labels=labels, 
            legend_position=legend_position, 
            render_ocr=render_ocr,
            include_highlights=True # Ensure highlights are requested
        )
        
    def save_image(self, 
            filename: str, 
            scale: float = 2.0,
            width: Optional[int] = None,
            labels: bool = True,
            legend_position: str = 'right',
            render_ocr: bool = False,
            include_highlights: bool = True, # Allow saving without highlights
            resolution: Optional[float] = None,
            **kwargs) -> 'Page':
        """
        Save the page image to a file, rendering highlights via HighlightingService.
        
        Args:
            filename: Path to save the image to.
            scale: Scale factor for rendering highlights.
            width: Optional width for the output image.
            labels: Whether to include a legend.
            legend_position: Position of the legend.
            render_ocr: Whether to render OCR text.
            include_highlights: Whether to render highlights.
            resolution: Resolution for base image rendering.
            **kwargs: Additional args for pdfplumber's to_image.
            
        Returns:
            Self for method chaining.
        """
        # Use to_image to generate and save the image
        self.to_image(
            path=filename,
            scale=scale,
            width=width,
            labels=labels, 
            legend_position=legend_position,
            render_ocr=render_ocr,
            include_highlights=include_highlights,
            resolution=resolution,
            **kwargs
        )
        return self
        
    def clear_highlights(self) -> 'Page':
        """
        Clear all highlights *from this specific page* via HighlightingService.
        
        Returns:
            Self for method chaining
        """
        self._highlighter.clear_page(self.index)
        return self
        
    def analyze_text_styles(self, options: Optional[TextStyleOptions] = None) -> ElementCollection:
        """
        Analyze text elements by style, adding attributes directly to elements.

        This method uses TextStyleAnalyzer to process text elements (typically words)
        on the page. It adds the following attributes to each processed element:
        - style_label: A descriptive or numeric label for the style group.
        - style_key: A hashable tuple representing the style properties used for grouping.
        - style_properties: A dictionary containing the extracted style properties.

        Args:
            options: Optional TextStyleOptions to configure the analysis.
                     If None, the analyzer's default options are used.

        Returns:
            ElementCollection containing all processed text elements with added style attributes.
        """
        # Create analyzer (optionally pass default options from PDF config here)
        # For now, it uses its own defaults if options=None
        analyzer = TextStyleAnalyzer()

        # Analyze the page. The analyzer now modifies elements directly
        # and returns the collection of processed elements.
        processed_elements_collection = analyzer.analyze(self, options=options)

        # Return the collection of elements which now have style attributes
        return processed_elements_collection

    def to_image(self,
            path: Optional[str] = None,
            scale: float = 2.0,
            width: Optional[int] = None,
            labels: bool = True,
            legend_position: str = 'right',
            render_ocr: bool = False,
            resolution: Optional[float] = None,
            include_highlights: bool = True,
            **kwargs) -> Optional[Image.Image]:
        """
        Generate a PIL image of the page, using HighlightingService if needed.
        
        Args:
            path: Optional path to save the image to.
            scale: Scale factor for rendering highlights.
            width: Optional width for the output image.
            labels: Whether to include a legend for highlights.
            legend_position: Position of the legend.
            render_ocr: Whether to render OCR text on highlights.
            resolution: Resolution in DPI for base page image (default: scale * 72).
            include_highlights: Whether to render highlights.
            **kwargs: Additional parameters for pdfplumber.to_image.
            
        Returns:
            PIL Image of the page, or None if rendering fails.
        """
        image = None
        try:
            if include_highlights:
                # Delegate rendering to the central service
                image = self._highlighter.render_page(
                    page_index=self.index,
                    scale=scale,
                    labels=labels,
                    legend_position=legend_position,
                    render_ocr=render_ocr,
                    resolution=resolution,
                    **kwargs
                )
            else:
                # Get the base page image directly from pdfplumber if no highlights needed
                render_resolution = resolution if resolution is not None else scale * 72
                # Use the underlying pdfplumber page object
                img_object = self._page.to_image(resolution=render_resolution, **kwargs)
                # Access the PIL image directly (assuming pdfplumber structure)
                image = img_object.annotated if hasattr(img_object, 'annotated') else img_object._repr_png_()
                if isinstance(image, bytes): # Handle cases where it returns bytes
                     from io import BytesIO
                     image = Image.open(BytesIO(image)).convert('RGB') # Convert to RGB for consistency
        
        except Exception as e:
            logger.error(f"Error rendering page {self.index}: {e}", exc_info=True)
            return None # Return None on error

        if image is None: return None

        # Resize the final image if width is provided
        if width is not None and width > 0 and image.width > 0:
            aspect_ratio = image.height / image.width
            height = int(width * aspect_ratio)
            try:
                image = image.resize((width, height), Image.Resampling.LANCZOS) # Use modern resampling
            except Exception as resize_error:
                 logger.warning(f"Could not resize image: {resize_error}")
        
        # Save the image if path is provided
        if path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(path), exist_ok=True)
                image.save(path)
                logger.debug(f"Saved page image to: {path}")
            except Exception as save_error:
                 logger.error(f"Failed to save image to {path}: {save_error}")
            
        return image
        
    def _create_text_elements_from_ocr(self, ocr_results: List[Dict[str, Any]], image_width=None, image_height=None) -> List[TextElement]:
        """DEPRECATED: Use self._element_mgr.create_text_elements_from_ocr"""
        logger.warning("_create_text_elements_from_ocr is deprecated. Use self._element_mgr version.")
        return self._element_mgr.create_text_elements_from_ocr(ocr_results, image_width, image_height)
        
    def apply_ocr(
        self,
        engine: Optional[str] = None,
        options: Optional[OCROptions] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
    ) -> List[TextElement]:
        """
        Apply OCR to THIS page and add results to page elements via PDF.apply_ocr_to_pages.
        
        Returns:
            List of created TextElements derived from OCR results for this page.
        """
        if not hasattr(self._parent, 'apply_ocr_to_pages'):
             logger.error(f"Page {self.number}: Parent PDF missing 'apply_ocr_to_pages'. Cannot apply OCR.")
             return []

        logger.info(f"Page {self.number}: Delegating apply_ocr to PDF.apply_ocr_to_pages.")
        try:
            # Delegate to parent PDF, targeting only this page's index
            self._parent.apply_ocr_to_pages(
                pages=[self.index],
                engine=engine, options=options, languages=languages,
                min_confidence=min_confidence, device=device
            )
        except Exception as e:
             logger.error(f"Page {self.number}: Error during delegated OCR call: {e}", exc_info=True)
             return []

        # Return the OCR elements specifically added to this page
        # Use element manager to retrieve them
        ocr_elements = [el for el in self.words if getattr(el, 'source', None) == 'ocr']
        logger.debug(f"Page {self.number}: apply_ocr completed. Found {len(ocr_elements)} OCR elements.")
        return ocr_elements
        
    def extract_ocr_elements(
        self,
        engine: Optional[str] = None,
        options: Optional[OCROptions] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
    ) -> List[TextElement]:
        """
        Extract text elements using OCR *without* adding them to the page's elements.
        Uses the shared OCRManager instance.
        """
        if not self._ocr_manager:
             logger.error(f"Page {self.number}: OCRManager not available. Cannot extract OCR elements.")
             return []
        
        logger.info(f"Page {self.number}: Extracting OCR elements (extract only)...")
        try:
            ocr_scale = getattr(self._parent, '_config', {}).get('ocr_image_scale', 2.0)
            # Get base image without highlights
            image = self.to_image(scale=ocr_scale, include_highlights=False)
            if not image:
                 logger.error(f"  Failed to render page {self.number} to image for OCR extraction.")
                 return []
            logger.debug(f"  Rendered image size: {image.width}x{image.height}")
        except Exception as e:
            logger.error(f"  Failed to render page {self.number} to image: {e}", exc_info=True)
            return []
        
        manager_args = {'images': image, 'options': options, 'engine': engine}
        if languages is not None: manager_args['languages'] = languages
        if min_confidence is not None: manager_args['min_confidence'] = min_confidence
        if device is not None: manager_args['device'] = device
        
        logger.debug(f"  Calling OCR Manager (extract only) with args: { {k:v for k,v in manager_args.items() if k != 'images'} }")
        try:
            # apply_ocr now returns List[List[Dict]] or List[Dict]
            results_list = self._ocr_manager.apply_ocr(**manager_args)
            # If it returned a list of lists (batch mode), take the first list
            results = results_list[0] if isinstance(results_list, list) and results_list and isinstance(results_list[0], list) else results_list
            
            if not isinstance(results, list):
                 logger.error(f"  OCR Manager returned unexpected type: {type(results)}")
                 results = []
            logger.info(f"  OCR Manager returned {len(results)} results for extraction.")
        except Exception as e:
             logger.error(f"  OCR processing failed during extraction: {e}", exc_info=True)
             return []
        
        # Convert results but DO NOT add to ElementManager
        logger.debug(f"  Converting OCR results to TextElements (extract only)...")
        # Use a temporary method to create elements without adding them globally
        temp_elements = []
        scale_x = self.width / image.width if image.width else 1
        scale_y = self.height / image.height if image.height else 1
        for result in results:
            x0, top, x1, bottom = [float(c) for c in result['bbox']]
            elem_data = {
                'text': result['text'], 'confidence': result['confidence'],
                'x0': x0 * scale_x, 'top': top * scale_y,
                'x1': x1 * scale_x, 'bottom': bottom * scale_y,
                'width': (x1 - x0) * scale_x, 'height': (bottom - top) * scale_y,
                'object_type': 'text', 'source': 'ocr',
                'fontname': 'OCR-temp', 'size': 10.0, 'page_number': self.number
            }
            temp_elements.append(TextElement(elem_data, self))

        logger.info(f"  Created {len(temp_elements)} TextElements from OCR (extract only).")
        return temp_elements
        
    @property
    def layout_analyzer(self) -> LayoutAnalyzer:
        """Get or create the layout analyzer for this page."""
        if self._layout_analyzer is None: 
             if not self._layout_manager:
                  logger.warning("LayoutManager not available, cannot create LayoutAnalyzer.")
                  return None 
             self._layout_analyzer = LayoutAnalyzer(self) 
        return self._layout_analyzer 

    def analyze_layout(
        self,
        engine: Optional[str] = None,
        options: Optional[LayoutOptions] = None,
        confidence: Optional[float] = None,
        classes: Optional[List[str]] = None,
        exclude_classes: Optional[List[str]] = None,
        device: Optional[str] = None,
        existing: str = "replace"
    ) -> ElementCollection[Region]:
        """
        Analyze the page layout using the configured LayoutManager.
        Adds detected Region objects to the page's element manager.

        Returns:
            ElementCollection containing the detected Region objects.
        """
        analyzer = self.layout_analyzer
        if not analyzer:
             logger.error("Layout analysis failed: LayoutAnalyzer not initialized (is LayoutManager available?).")
             return ElementCollection([]) # Return empty collection

        # The analyzer's analyze_layout method already adds regions to the page
        # and its element manager. We just need to retrieve them.
        analyzer.analyze_layout(
            engine=engine,
            options=options,
            confidence=confidence,
            classes=classes,
            exclude_classes=exclude_classes,
            device=device,
            existing=existing
        )

        # Retrieve the detected regions from the element manager
        # Filter regions based on source='detected' and potentially the model used if available
        detected_regions = [r for r in self._element_mgr.regions
                            if r.source == 'detected' and (not engine or getattr(r, 'model', None) == engine)]

        return ElementCollection(detected_regions)

    def get_section_between(self, start_element=None, end_element=None, boundary_inclusion='both') -> Optional[Region]: # Return Optional
        """
        Get a section between two elements on this page.
        """
        # Create a full-page region to operate within
        page_region = self.create_region(0, 0, self.width, self.height)
        
        # Delegate to the region's method
        try:
            return page_region.get_section_between(
                start_element=start_element,
                end_element=end_element,
                boundary_inclusion=boundary_inclusion
            )
        except Exception as e:
             logger.error(f"Error getting section between elements on page {self.index}: {e}", exc_info=True)
             return None
    
    def get_sections(self, 
                  start_elements=None, 
                  end_elements=None,
                  boundary_inclusion='both',
                  y_threshold=5.0,
                  bounding_box=None) -> List[Region]:
        """
        Get sections of a page defined by start/end elements.
        Uses the page-level implementation.
        """
        # Helper function to get bounds from bounding_box parameter
        def get_bounds():
            if bounding_box:
                x0, top, x1, bottom = bounding_box
                # Clamp to page boundaries
                return max(0, x0), max(0, top), min(self.width, x1), min(self.height, bottom)
            else:
                return 0, 0, self.width, self.height
                
        regions = []
        
        # Handle cases where elements are provided as strings (selectors)
        if isinstance(start_elements, str):
            start_elements = self.find_all(start_elements).elements # Get list of elements
        elif hasattr(start_elements, 'elements'): # Handle ElementCollection input
             start_elements = start_elements.elements
            
        if isinstance(end_elements, str):
            end_elements = self.find_all(end_elements).elements
        elif hasattr(end_elements, 'elements'):
             end_elements = end_elements.elements

        # Ensure start_elements is a list
        if start_elements is None: start_elements = []
        if end_elements is None: end_elements = []

        valid_inclusions = ['start', 'end', 'both', 'none']
        if boundary_inclusion not in valid_inclusions:
            raise ValueError(f"boundary_inclusion must be one of {valid_inclusions}")
        
        if not start_elements:
            return regions
            
        # Combine start and end elements with their type
        all_boundaries = []
        for el in start_elements: all_boundaries.append((el, 'start'))
        for el in end_elements: all_boundaries.append((el, 'end'))
                
        # Sort all boundary elements primarily by top, then x0
        try:
             all_boundaries.sort(key=lambda x: (x[0].top, x[0].x0))
        except AttributeError as e:
             logger.error(f"Error sorting boundaries: Element missing top/x0 attribute? {e}")
             return [] # Cannot proceed if elements lack position

        # Process sorted boundaries to find sections
        current_start_element = None
        active_section_started = False

        for element, element_type in all_boundaries:
            if element_type == 'start':
                # If we have an active section, this start implicitly ends it
                if active_section_started:
                    end_boundary_el = element # Use this start as the end boundary
                    # Determine region boundaries
                    sec_top = current_start_element.top if boundary_inclusion in ['start', 'both'] else current_start_element.bottom
                    sec_bottom = end_boundary_el.top if boundary_inclusion not in ['end', 'both'] else end_boundary_el.bottom
                    
                    if sec_top < sec_bottom: # Ensure valid region
                        x0, _, x1, _ = get_bounds()
                        region = self.create_region(x0, sec_top, x1, sec_bottom)
                        region.start_element = current_start_element
                        region.end_element = end_boundary_el # Mark the element that ended it
                        region.is_end_next_start = True # Mark how it ended
                        regions.append(region)
                    active_section_started = False # Reset for the new start
                
                # Set this as the potential start of the next section
                current_start_element = element
                active_section_started = True

            elif element_type == 'end' and active_section_started:
                # We found an explicit end for the current section
                end_boundary_el = element
                sec_top = current_start_element.top if boundary_inclusion in ['start', 'both'] else current_start_element.bottom
                sec_bottom = end_boundary_el.bottom if boundary_inclusion in ['end', 'both'] else end_boundary_el.top
                
                if sec_top < sec_bottom: # Ensure valid region
                    x0, _, x1, _ = get_bounds()
                    region = self.create_region(x0, sec_top, x1, sec_bottom)
                    region.start_element = current_start_element
                    region.end_element = end_boundary_el
                    region.is_end_next_start = False
                    regions.append(region)
                
                # Reset: section ended explicitly
                current_start_element = None
                active_section_started = False
        
        # Handle the last section if it was started but never explicitly ended
        if active_section_started:
            sec_top = current_start_element.top if boundary_inclusion in ['start', 'both'] else current_start_element.bottom
            x0, _, x1, page_bottom = get_bounds()
            if sec_top < page_bottom:
                 region = self.create_region(x0, sec_top, x1, page_bottom)
                 region.start_element = current_start_element
                 region.end_element = None # Ended by page end
                 region.is_end_next_start = False
                 regions.append(region)
            
        return regions
            
    def __repr__(self) -> str:
        """String representation of the page."""
        return f"<Page number={self.number} index={self.index}>"
        
    def ask(self, question: str, min_confidence: float = 0.1, model: str = None, debug: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Ask a question about the page content using document QA.
        """
        try:
             from natural_pdf.qa.document_qa import get_qa_engine
             # Get or initialize QA engine with specified model
             qa_engine = get_qa_engine(model_name=model) if model else get_qa_engine()
             # Ask the question using the QA engine
             return qa_engine.ask_pdf_page(self, question, min_confidence=min_confidence, debug=debug, **kwargs)
        except ImportError:
             logger.error("Question answering requires the 'natural_pdf.qa' module. Please install necessary dependencies.")
             return {"answer": None, "confidence": 0.0, "found": False, "page_num": self.number, "source_elements": []}
        except Exception as e:
             logger.error(f"Error during page.ask: {e}", exc_info=True)
             return {"answer": None, "confidence": 0.0, "found": False, "page_num": self.number, "source_elements": []}

    def show_preview(self, 
                     temporary_highlights: List[Dict], 
                     scale: float = 2.0, 
                     width: Optional[int] = None,
                     labels: bool = True,
                     legend_position: str = 'right',
                     render_ocr: bool = False) -> Optional[Image.Image]:
        """
        Generates and returns a non-stateful preview image containing only
        the provided temporary highlights.

        Args:
            temporary_highlights: List of highlight data dictionaries (as prepared by
                                  ElementCollection._prepare_highlight_data).
            scale: Scale factor for rendering.
            width: Optional width for the output image.
            labels: Whether to include a legend.
            legend_position: Position of the legend.
            render_ocr: Whether to render OCR text.

        Returns:
            PIL Image object of the preview, or None if rendering fails.
        """
        try:
            # Delegate rendering to the highlighter service's preview method
            img = self._highlighter.render_preview(
                page_index=self.index,
                temporary_highlights=temporary_highlights,
                scale=scale,
                labels=labels,
                legend_position=legend_position,
                render_ocr=render_ocr
            )
        except AttributeError:
            logger.error(f"HighlightingService does not have the required 'render_preview' method.")
            return None
        except Exception as e:
            logger.error(f"Error calling highlighter.render_preview for page {self.index}: {e}", exc_info=True)
            return None

        # Return the rendered image directly
        return img

    @property
    def text_style_labels(self) -> List[str]:
        """ 
        Get a sorted list of unique text style labels found on the page.

        Runs text style analysis with default options if it hasn't been run yet.
        To use custom options, call `analyze_text_styles(options=...)` explicitly first.

        Returns:
            A sorted list of unique style label strings.
        """
        # Check if the summary attribute exists from a previous run
        if not hasattr(self, '_text_styles_summary') or not self._text_styles_summary:
            # If not, run the analysis with default options
            logger.debug(f"Page {self.number}: Running default text style analysis to get labels.")
            self.analyze_text_styles() # Use default options

        # Extract labels from the summary dictionary
        if hasattr(self, '_text_styles_summary') and self._text_styles_summary:
            # The summary maps style_key -> {'label': ..., 'properties': ...}
            labels = {style_info['label'] for style_info in self._text_styles_summary.values()}
            return sorted(list(labels))
        else:
            # Fallback if summary wasn't created for some reason (e.g., no text elements)
             logger.warning(f"Page {self.number}: Text style summary not found after analysis.")
             return []

    def viewer(self,
                           # elements_to_render: Optional[List['Element']] = None, # No longer needed, from_page handles it
                           # include_element_types: List[str] = ['word', 'line', 'rect', 'region'] # No longer needed
                          ) -> 'SimpleInteractiveViewerWidget': # Return type hint updated
        """
        Creates and returns an interactive ipywidget for exploring elements on this page.

        Uses SimpleInteractiveViewerWidget.from_page() to create the viewer.

        Returns:
            A SimpleInteractiveViewerWidget instance ready for display in Jupyter.

        Raises:
            RuntimeError: If required dependencies (ipywidgets) are missing.
            ValueError: If image rendering or data preparation fails within from_page.
        """
        # Import the widget class (might need to be moved to top if used elsewhere)
        from natural_pdf.widgets.viewer import SimpleInteractiveViewerWidget

        logger.info(f"Generating interactive viewer for Page {self.number} using SimpleInteractiveViewerWidget.from_page...")

        try:
            # Delegate creation entirely to the from_page class method
            viewer_widget = SimpleInteractiveViewerWidget.from_page(self)
            if viewer_widget is None:
                 # This case might happen if from_page had error handling to return None, though we removed most.
                 # Keeping a check here just in case.
                 raise RuntimeError("SimpleInteractiveViewerWidget.from_page returned None, indicating an issue during widget creation.")

            logger.info("Interactive viewer widget created successfully.")
            return viewer_widget
        except ImportError as e:
            logger.error("Failed to import SimpleInteractiveViewerWidget. Ensure natural_pdf.widgets and ipywidgets are installed.")
            raise RuntimeError("Widget class not found. ipywidgets or natural_pdf.widgets might be missing or setup incorrect.") from e
        except Exception as e:
            logger.error(f"Failed to create interactive viewer: {e}", exc_info=True)
            # Re-raise the exception to make it visible to the user
            raise RuntimeError(f"Failed to create interactive viewer: {e}") from e
