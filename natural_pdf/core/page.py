import pdfplumber
import os
import tempfile
from typing import List, Optional, Union, Any, Dict, Callable, TYPE_CHECKING, Tuple
from PIL import Image

if TYPE_CHECKING:
    import pdfplumber
    from natural_pdf.core.pdf import PDF
    from natural_pdf.elements.collections import ElementCollection
    from natural_pdf.utils.highlighting import HighlightManager
    from natural_pdf.elements.base import Element

from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement
from natural_pdf.analyzers.document_layout import (
    YOLODocLayoutDetector, 
    TableTransformerDetector,
    PaddleLayoutDetector,
    DoclingLayoutDetector,
    convert_to_regions
)
from natural_pdf.utils.ocr import OCRManager

# Import OCR engines
try:
    from natural_pdf.ocr import OCREngine, EasyOCREngine, PaddleOCREngine
    HAS_OCR_ENGINES = True
except ImportError:
    # Fallback if the OCR engines are not available
    HAS_OCR_ENGINES = False


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
                       Default: ['fontname', 'size'] (Group by font name and size)
                       None: Only consider spatial relationships
                       List: Custom attributes to consider (e.g., ['fontname', 'size', 'color'])
        """
        self._page = page
        self._parent = parent
        self._index = index
        self._elements = None  # Lazy-loaded
        self._highlight_manager = None  # Lazy-loaded
        self._text_styles = None  # Lazy-loaded text style analyzer results
        self._exclusions = []  # List to store exclusion functions/regions
        
        # Region management
        self._regions = {
            'detected': [],  # Layout detection results
            'named': {},     # Named regions (name -> region)
        }
        
        # Default to grouping by fontname and size if not specified
        self._font_attrs = ['fontname', 'size'] if font_attrs is None else font_attrs
        
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
            
        # Make sure regions is in _elements for selectors
        if self._elements is not None and 'regions' not in self._elements:
            self._elements['regions'] = []
            
        # Add to elements for selector queries
        if self._elements is not None:
            if region not in self._elements['regions']:
                self._elements['regions'].append(region)
                
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
        
        # First get all matching elements without applying exclusions
        results = self._apply_selector(selector_obj, **kwargs)
        
        # Then filter by exclusions if requested
        if apply_exclusions and self._exclusions and results:
            # Get all exclusion regions, including those from lambda functions
            exclusion_regions = self._get_exclusion_regions(include_callable=True)
            
            # Apply exclusion regions if any
            if exclusion_regions:
                results = results.exclude_regions(exclusion_regions)
        
        return results.first if results else None

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
        
        # First get all matching elements without applying exclusions
        results = self._apply_selector(selector_obj, **kwargs)
        
        # Then filter by exclusions if requested
        if apply_exclusions and self._exclusions and results:
            # Get all exclusion regions, including those from lambda functions
            exclusion_regions = self._get_exclusion_regions(include_callable=True)
            
            # Apply exclusion regions if any
            if exclusion_regions:
                results = results.exclude_regions(exclusion_regions)
        
        return results
    
    def _apply_selector(self, selector_obj: Dict, apply_exclusions=True, **kwargs) -> 'ElementCollection':
        """
        Apply selector to page elements.
        
        Args:
            selector_obj: Parsed selector dictionary
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True)
            **kwargs: Additional filter parameters
            
        Returns:
            ElementCollection of matching elements
        """
        from natural_pdf.elements.collections import ElementCollection
        from natural_pdf.selectors.parser import selector_to_filter_func
        
        # Load all elements if not already loaded
        self._load_elements()
        
        # Get element type to filter
        element_type = selector_obj.get('type', 'any').lower()
        
        # Determine which elements to search based on element type
        elements_to_search = []
        if element_type == 'any':
            # Search all element types
            for key, elements_list in self._elements.items():
                # Skip chars if we have words for text search (avoid duplication)
                if key == 'chars' and 'words' in self._elements:
                    continue
                elements_to_search.extend(elements_list)
        elif element_type == 'text':
            # Prefer word elements over character elements for text
            if 'words' in self._elements:
                elements_to_search = self._elements.get('words', [])
            else:
                elements_to_search = self._elements.get('chars', [])
        elif element_type == 'char':
            elements_to_search = self._elements.get('chars', [])
        elif element_type == 'word':
            elements_to_search = self._elements.get('words', [])
        elif element_type == 'rect' or element_type == 'rectangle':
            elements_to_search = self._elements.get('rects', [])
        elif element_type == 'line':
            elements_to_search = self._elements.get('lines', [])
        elif element_type == 'region':
            # Start with an empty list
            elements_to_search = []
            
            # Add regions from _elements if available
            if 'regions' in self._elements and self._elements['regions']:
                elements_to_search.extend(self._elements['regions'])
                
            # If no regions in _elements, look in _regions
            if not elements_to_search:
                # Add detected regions
                elements_to_search.extend(self._regions['detected'])
                
                # Add named regions
                elements_to_search.extend(self._regions['named'].values())
        else:
            # If type doesn't match a specific category, look in all categories
            for key, elements_list in self._elements.items():
                # Skip chars if we have words for text search (avoid duplication)
                if key == 'chars' and 'words' in self._elements:
                    continue
                elements_to_search.extend(elements_list)
        
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
                ref_elements = self._apply_selector(ref_selector)
                
                if not ref_elements:
                    # No reference elements found, so no matches
                    return ElementCollection([])
                
                # Use the first reference element for now
                # TODO: Improve this to consider all reference elements
                ref_element = ref_elements.first()
                
                # Filter elements based on spatial relationship
                if name == 'above':
                    matching_elements = [el for el in matching_elements if el.bottom <= ref_element.top]
                elif name == 'below':
                    matching_elements = [el for el in matching_elements if el.top >= ref_element.bottom]
                elif name == 'left-of':
                    matching_elements = [el for el in matching_elements if el.x1 <= ref_element.x0]
                elif name == 'right-of':
                    matching_elements = [el for el in matching_elements if el.x0 >= ref_element.x1]
                elif name == 'near':
                    # Calculate distance between centers
                    def distance(el1, el2):
                        el1_center_x = (el1.x0 + el1.x1) / 2
                        el1_center_y = (el1.top + el1.bottom) / 2
                        el2_center_x = (el2.x0 + el2.x1) / 2
                        el2_center_y = (el2.top + el2.bottom) / 2
                        return ((el1_center_x - el2_center_x) ** 2 + (el1_center_y - el2_center_y) ** 2) ** 0.5
                    
                    # Get distance threshold from kwargs or use default
                    threshold = kwargs.get('near_threshold', 50)  # Default 50 points
                    matching_elements = [el for el in matching_elements if distance(el, ref_element) <= threshold]
        
        # Sort elements in reading order if requested
        if kwargs.get('reading_order', True):
            # TODO: Implement proper reading order sorting
            # For now, simple top-to-bottom, left-to-right ordering
            matching_elements.sort(key=lambda el: (el.top, el.x0))
        
        # Create result collection
        result = ElementCollection(matching_elements)
        
        # Apply exclusions if requested and if there are exclusions defined
        # Note: We don't apply exclusions here as that would cause recursion
        # Exclusions are applied at the higher level via exclude_regions
        
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
        
    def get_elements(self, apply_exclusions=True) -> List['Element']:
        """
        Get all elements on this page.
        
        Args:
            apply_exclusions: Whether to apply exclusion regions
            
        Returns:
            List of all elements on the page
        """
        # Load elements if not already loaded
        self._load_elements()
        
        # Combine all element types
        all_elements = []
        all_elements.extend(self.words)
        all_elements.extend(self.rects)
        all_elements.extend(self.lines)
        # Add other element types as needed
        
        # Apply exclusions if requested
        if apply_exclusions and self._exclusions:
            exclusion_regions = self._get_exclusion_regions(include_callable=True)
            if exclusion_regions:
                # Keep elements that are not in any exclusion region
                filtered_elements = []
                for element in all_elements:
                    in_exclusion = False
                    for region in exclusion_regions:
                        if region._is_element_in_region(element):
                            in_exclusion = True
                            break
                    if not in_exclusion:
                        filtered_elements.append(element)
                return filtered_elements
        
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
        filter_func = selector_to_filter_func(selector_obj)
        
        # Apply the filter to the elements
        matching_elements = [element for element in elements if filter_func(element)]
        
        # Sort elements in reading order if requested
        if kwargs.get('reading_order', True):
            matching_elements.sort(key=lambda el: (el.top, el.x0))
        
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
            # If target not found, return a default region
            from natural_pdf.elements.region import Region
            return Region(self, (0, 0, self.width, self.height))
            
        # Create a region from the top of the page to the target
        from natural_pdf.elements.region import Region
        if include_endpoint:
            # Include the target element
            region = Region(self, (0, 0, self.width, target.bottom))
        else:
            # Up to the target element
            region = Region(self, (0, 0, self.width, target.top))
            
        region.end_element = target
        return region
        
    # Alias for backward compatibility
    def select_until(self, selector: str, include_target: bool = True, **kwargs) -> Any:
        """
        DEPRECATED: Use until() instead.
        Select content from this point until matching selector.

        Args:
            selector: CSS-like selector string
            include_target: Whether to include the target element in the region
            **kwargs: Additional selection parameters
            
        Returns:
            Region object representing the selected content
        """
        import warnings
        warnings.warn(
            "select_until() is deprecated and will be removed in a future version. Use until() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.until(selector, include_endpoint=include_target, **kwargs)
    
    def crop(self, bbox=None, **kwargs) -> Any:
        """
        Crop the page to the specified bounding box.

        This is a direct wrapper around pdfplumber's crop method.
        
        Args:
            bbox: Bounding box (x0, top, x1, bottom) or None
            **kwargs: Additional parameters (top, bottom, left, right)

        Returns:
            Cropped page object
        """
        # TODO: Create proper wrapper for cropped page
        return self._page.crop(bbox, **kwargs)
    
    def extract_text(self, 
                  preserve_whitespace=True,
                  use_exclusions=True,
                  debug_exclusions=False, ocr=None, **kwargs) -> str:
        """
        Extract text from this page, respecting any exclusion regions.
        
        Args:
            preserve_whitespace: Whether to keep blank characters (default: True)
            use_exclusions: Whether to apply exclusion regions (default: True)
            debug_exclusions: Whether to output detailed exclusion debugging info (default: False)
            ocr: OCR configuration. If None, uses PDF settings
            **kwargs: Additional extraction parameters
            
        Returns:
            Extracted text as string
        """
        if not self._exclusions or not use_exclusions:
            # If no exclusions or exclusions disabled, use regular extraction
            if debug_exclusions:
                print(f"Page {self.index}: No exclusions to apply or use_exclusions=False")
            # Note: pdfplumber still uses keep_blank_chars parameter
            return self._page.extract_text(keep_blank_chars=preserve_whitespace, **kwargs)
        
        # Get all exclusion regions
        if debug_exclusions:
            print(f"Page {self.index}: Getting exclusion regions with debugging enabled")
        
        # Important: We need to evaluate lambda functions from PDF level
        # These functions are stored directly in _exclusions and not as tuples
        exclusion_regions = self._get_exclusion_regions(include_callable=True, debug=debug_exclusions)
        
        if not exclusion_regions:
            if debug_exclusions:
                print(f"Page {self.index}: No valid exclusion regions were found")
            # Note: pdfplumber still uses keep_blank_chars parameter
            return self._page.extract_text(keep_blank_chars=preserve_whitespace, **kwargs)
        
        if debug_exclusions:
            print(f"Page {self.index}: Found {len(exclusion_regions)} exclusion regions to apply")
        
        # Find all text elements
        all_text = self.find_all('text')
        
        if debug_exclusions:
            print(f"Page {self.index}: Found {len(all_text)} text elements before exclusion filtering")
        
        # Filter out elements in excluded regions
        filtered_elements = []
        excluded_count = 0
        
        for element in all_text:
            exclude = False
            for region in exclusion_regions:
                if region._is_element_in_region(element):
                    exclude = True
                    excluded_count += 1
                    break
            if not exclude:
                filtered_elements.append(element)
        
        if debug_exclusions:
            print(f"Page {self.index}: Excluded {excluded_count} elements, keeping {len(filtered_elements)}")
        
        # Extract text from filtered elements
        from natural_pdf.elements.collections import ElementCollection
        collection = ElementCollection(filtered_elements)
        result = collection.extract_text(preserve_whitespace=preserve_whitespace, **kwargs)
        
        # Apply OCR if explicitly requested
        use_ocr = ocr is True or (ocr is not None and isinstance(ocr, dict) and ocr.get('enabled', False))
        if use_ocr:
            # Process OCR parameter into normalized config
            ocr_config = self._get_ocr_config(ocr)
            
            # Apply OCR if explicitly enabled or in auto mode and no text found
            if ocr_config.get('enabled') is True or ocr is True or (
                ocr_config.get('enabled') == 'auto' and not result.strip()
            ):
                print(f"Using OCR for extract_text")
                # Get existing OCR elements or run OCR
                if any(elem.source == 'ocr' for elem in filtered_elements):
                    # We already have OCR elements, just re-extract from them
                    ocr_elements = [elem for elem in filtered_elements if elem.source == 'ocr']
                    ocr_collection = ElementCollection(ocr_elements)
                    ocr_text = ocr_collection.extract_text(preserve_whitespace=preserve_whitespace, **kwargs)
                    
                    if ocr_text.strip():
                        result = ocr_text
                else:
                    # Run OCR and get text from OCR elements
                    ocr_elements = self.apply_ocr(**ocr_config)
                    
                    if ocr_elements:
                        # Filter OCR elements by exclusions
                        if use_exclusions:
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
                            
                        ocr_collection = ElementCollection(filtered_ocr)
                        ocr_text = ocr_collection.extract_text(preserve_whitespace=preserve_whitespace, **kwargs)
                        
                        # Use OCR text if it's not empty
                        if ocr_text.strip():
                            result = ocr_text
        
        if debug_exclusions:
            print(f"Page {self.index}: Extracted {len(result)} characters of text with exclusions applied")
            
        return result

    def extract_table(self, table_settings={}) -> List[Any]:
        """
        Extract the largest table from this page.
        
        Args:
            table_settings: Additional extraction parameters
            
        Returns:
            List of extracted tables
        """
        # For now, directly use pdfplumber's extraction
        return self._page.extract_table(table_settings)

    def extract_tables(self, table_settings={}) -> List[Any]:
        """
        Extract tables from this page.
        
        Args:
            table_settings: Additional extraction parameters
            
        Returns:
            List of extracted tables
        """
        # For now, directly use pdfplumber's extraction
        return self._page.extract_tables(table_settings)
    
    def _load_elements(self, include_ocr=None):
        """
        Load all elements from the page (lazy loading).
        
        Args:
            include_ocr: Whether to include OCR text elements. If None, uses PDF settings.
        """
        if self._elements is None:
            from natural_pdf.elements.text import TextElement
            from natural_pdf.elements.rect import RectangleElement
            from natural_pdf.elements.line import LineElement
            
            # Get the font attributes to use for word grouping
            font_attrs = self._font_attrs
            
            # Get keep_spaces setting from PDF config or default to True (new behavior)
            keep_spaces = self._parent._config.get('keep_spaces', True)
            
            # Process characters, annotating with font information
            chars = []
            for c in self._page.chars:
                # Check for font references (F0, F1, etc.) and map to actual fonts
                if c.get('fontname', '').startswith('F') and len(c['fontname']) <= 3:
                    # Access the PDF resource info to get actual font name
                    font_ref = c['fontname']
                    try:
                        # Try to get font info from resources
                        if self._page.page_obj.get('Resources', {}).get('Font', {}):
                            fonts = self._page.page_obj['Resources']['Font']
                            if font_ref in fonts:
                                font_obj = fonts[font_ref]
                                if font_obj.get('BaseFont'):
                                    c['real_fontname'] = font_obj['BaseFont']
                    except (KeyError, AttributeError, TypeError):
                        pass
                
                # Add source attribute for native text elements
                c['source'] = 'native'
                chars.append(TextElement(c, self))
            
            # Create word-level text elements by grouping chars
            from itertools import groupby
            from operator import itemgetter
            
            # Sort chars by y-position (line) and then x-position
            sorted_chars = sorted(self._page.chars, key=lambda c: (round(c['top']), c['x0']))
            
            # Group chars by line (similar y-position)
            line_groups = []
            for _, line_chars in groupby(sorted_chars, key=lambda c: round(c['top'])):
                line_chars = list(line_chars)
                
                # Now group chars into words based on x-distance and font attributes
                words = []
                current_word = []
                
                for i, char in enumerate(line_chars):
                    # Handle whitespace characters differently based on keep_spaces setting
                    if char['text'].isspace():
                        if keep_spaces:
                            # Include spaces in words when keep_spaces is enabled
                            if current_word:
                                current_word.append(char)
                            else:
                                # Skip leading spaces at the start of a line
                                continue
                        else:
                            # Original behavior: Skip whitespace and close current word
                            if current_word:
                                # Combine text from characters and normalize spaces
                                text = ''.join(c['text'] for c in current_word)
                                
                                # Collapse multiple consecutive spaces into a single space
                                import re
                                text = re.sub(r'\s+', ' ', text)
                                
                                # Create a combined word object
                                word_obj = {
                                    'text': text,
                                    'x0': min(c['x0'] for c in current_word),
                                    'x1': max(c['x1'] for c in current_word),
                                    'top': min(c['top'] for c in current_word),
                                    'bottom': max(c['bottom'] for c in current_word),
                                    'fontname': current_word[0].get('fontname', ''),
                                    'size': current_word[0].get('size', 0),
                                    'object_type': 'word',
                                    'page_number': current_word[0]['page_number']
                                }
                                
                                # Handle real fontname if available
                                if 'real_fontname' in current_word[0]:
                                    word_obj['real_fontname'] = current_word[0]['real_fontname']
                                    
                                # Handle color - use the first char's color
                                if 'non_stroking_color' in current_word[0]:
                                    word_obj['non_stroking_color'] = current_word[0]['non_stroking_color']
                                
                                # Copy any additional font attributes
                                for attr in font_attrs:
                                    if attr in current_word[0]:
                                        word_obj[attr] = current_word[0][attr]
                                    
                                # Add source attribute for native text elements
                                word_obj['source'] = 'native'
                                words.append(TextElement(word_obj, self))
                                current_word = []
                            continue
                    
                    # If this is a new word, start it
                    if not current_word:
                        current_word.append(char)
                    else:
                        # Check if this char is part of the current word or a new word
                        prev_char = current_word[-1]
                        
                        # Check if font attributes match for this character
                        font_attrs_match = True
                        if font_attrs:
                            for attr in font_attrs:
                                # If attribute doesn't match or isn't present in both chars, break word
                                if attr not in char or attr not in prev_char or char[attr] != prev_char[attr]:
                                    font_attrs_match = False
                                    break
                        
                        # If font attributes don't match, it's a new word
                        if not font_attrs_match:
                            # Combine text from characters and normalize spaces
                            text = ''.join(c['text'] for c in current_word)
                            
                            # Collapse multiple consecutive spaces into a single space
                            import re
                            text = re.sub(r'\s+', ' ', text)
                            
                            # Finish current word
                            word_obj = {
                                'text': text,
                                'x0': min(c['x0'] for c in current_word),
                                'x1': max(c['x1'] for c in current_word),
                                'top': min(c['top'] for c in current_word),
                                'bottom': max(c['bottom'] for c in current_word),
                                'fontname': current_word[0].get('fontname', ''),
                                'size': current_word[0].get('size', 0),
                                'object_type': 'word',
                                'page_number': current_word[0]['page_number']
                            }
                            
                            # Handle real fontname if available
                            if 'real_fontname' in current_word[0]:
                                word_obj['real_fontname'] = current_word[0]['real_fontname']
                                
                            # Handle color - use the first char's color
                            if 'non_stroking_color' in current_word[0]:
                                word_obj['non_stroking_color'] = current_word[0]['non_stroking_color']
                            
                            # Copy any additional font attributes
                            for attr in font_attrs:
                                if attr in current_word[0]:
                                    word_obj[attr] = current_word[0][attr]
                                
                            # Add source attribute for native text elements
                            word_obj['source'] = 'native'
                            words.append(TextElement(word_obj, self))
                            current_word = [char]
                        # If the gap between chars is larger than a threshold, it's a new word
                        # Use a wider threshold when keep_spaces is enabled to allow for natural spaces
                        elif char['x0'] - prev_char['x1'] > prev_char['width'] * (1.5 if keep_spaces else 0.5):
                            # Combine text from characters and normalize spaces
                            text = ''.join(c['text'] for c in current_word)
                            
                            # Collapse multiple consecutive spaces into a single space
                            import re
                            text = re.sub(r'\s+', ' ', text)
                            
                            # Finish current word
                            word_obj = {
                                'text': text,
                                'x0': min(c['x0'] for c in current_word),
                                'x1': max(c['x1'] for c in current_word),
                                'top': min(c['top'] for c in current_word),
                                'bottom': max(c['bottom'] for c in current_word),
                                'fontname': current_word[0].get('fontname', ''),
                                'size': current_word[0].get('size', 0),
                                'object_type': 'word',
                                'page_number': current_word[0]['page_number']
                            }
                            
                            # Handle real fontname if available
                            if 'real_fontname' in current_word[0]:
                                word_obj['real_fontname'] = current_word[0]['real_fontname']
                                
                            # Handle color - use the first char's color
                            if 'non_stroking_color' in current_word[0]:
                                word_obj['non_stroking_color'] = current_word[0]['non_stroking_color']
                            
                            # Copy any additional font attributes
                            for attr in font_attrs:
                                if attr in current_word[0]:
                                    word_obj[attr] = current_word[0][attr]
                                
                            # Add source attribute for native text elements
                            word_obj['source'] = 'native'
                            words.append(TextElement(word_obj, self))
                            current_word = [char]
                        else:
                            # Continue current word
                            current_word.append(char)
                    
                # Handle the last word if there is one
                if current_word:
                    # Combine text from characters and normalize spaces
                    text = ''.join(c['text'] for c in current_word)
                    
                    # Collapse multiple consecutive spaces into a single space
                    import re
                    text = re.sub(r'\s+', ' ', text)
                    
                    word_obj = {
                        'text': text,
                        'x0': min(c['x0'] for c in current_word),
                        'x1': max(c['x1'] for c in current_word),
                        'top': min(c['top'] for c in current_word),
                        'bottom': max(c['bottom'] for c in current_word),
                        'fontname': current_word[0].get('fontname', ''),
                        'size': current_word[0].get('size', 0),
                        'object_type': 'word',
                        'page_number': current_word[0]['page_number']
                    }
                    
                    # Handle real fontname if available
                    if 'real_fontname' in current_word[0]:
                        word_obj['real_fontname'] = current_word[0]['real_fontname']
                        
                    # Handle color - use the first char's color
                    if 'non_stroking_color' in current_word[0]:
                        word_obj['non_stroking_color'] = current_word[0]['non_stroking_color']
                    
                    # Copy any additional font attributes
                    for attr in font_attrs:
                        if attr in current_word[0]:
                            word_obj[attr] = current_word[0][attr]
                        
                    # Add source attribute for native text elements
                    word_obj['source'] = 'native'
                    words.append(TextElement(word_obj, self))
            
                line_groups.extend(words)
            
            self._elements = {
                'chars': chars,
                'words': line_groups,
                'rects': [RectangleElement(r, self) for r in self._page.rects],
                'lines': [LineElement(l, self) for l in self._page.lines],
                # Add other element types as needed
            }
            
            # Check if we should run OCR
            apply_ocr = False
            
            # Check if OCR is explicitly requested
            if include_ocr is True:
                apply_ocr = True
            # Otherwise, check PDF-level settings for auto mode
            elif include_ocr is None and self._parent._ocr_config.get('enabled') == 'auto':
                # In auto mode, apply OCR if few or no text elements found
                if len(line_groups) < 5:  # Arbitrary threshold
                    apply_ocr = True
            
            # Apply OCR if needed
            if apply_ocr:
                ocr_elements = self.apply_ocr()
                # OCR elements are already added to self._elements in apply_ocr()
    
    @property
    def chars(self) -> List[Any]:
        """Get all character elements on this page."""
        self._load_elements()
        return self._elements['chars']
    
    @property
    def words(self) -> List[Any]:
        """Get all word elements on this page."""
        self._load_elements()
        return self._elements['words']
    
    @property
    def rects(self) -> List[Any]:
        """Get all rectangle elements on this page."""
        self._load_elements()
        return self._elements['rects']
    
    @property
    def lines(self) -> List[Any]:
        """Get all line elements on this page."""
        self._load_elements()
        return self._elements['lines']
    
    @property
    def _highlight_mgr(self) -> 'HighlightManager':
        """Get the highlight manager for this page."""
        if self._highlight_manager is None:
            from natural_pdf.utils.highlighting import HighlightManager
            self._highlight_manager = HighlightManager(self)
        return self._highlight_manager
    
    def highlight(self, 
                 color: Optional[Tuple[int, int, int, int]] = None, 
                 label: Optional[str] = None) -> 'Page':
        """
        Highlight the entire page.
        
        Args:
            color: RGBA color tuple for the highlight, or None to use the next color
            label: Optional label for the highlight
            
        Returns:
            Self for method chaining
        """
        # Add a highlight for the entire page
        self._highlight_mgr.add_highlight(
            (0, 0, self.width, self.height), color, label
        )
        return self
    
    def show(self, 
            scale: float = 2.0,
            width: Optional[int] = None,
            labels: bool = True,
            legend_position: str = 'right',
            render_ocr: bool = False) -> Image.Image:
        """
        Show the page with any highlights.
        
        Args:
            scale: Scale factor for rendering
            width: Optional width for the output image in pixels
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            render_ocr: Whether to render OCR text with white background boxes
            
        Returns:
            PIL Image of the page with highlights
        """
        # Use to_image to get the image
        return self.to_image(
            scale=scale,
            width=width,
            labels=labels, 
            legend_position=legend_position, 
            render_ocr=render_ocr
        )
        
        
        
    def save_image(self, 
            filename: str, 
            scale: float = 2.0,
            width: Optional[int] = None,
            labels: bool = True,
            legend_position: str = 'right',
            render_ocr: bool = False) -> 'Page':
        """
        Save the page with any highlights to an image file.
        
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
        return debug_ocr_to_html([self], output_path)
        
    def save(self, 
            filename: str, 
            scale: float = 2.0,
            width: Optional[int] = None,
            labels: bool = False,
            legend_position: str = 'right') -> 'Page':
        """
        DEPRECATED: Use to_image() instead.
        Save the page with any highlights to an image file.
        """
        import warnings
        warnings.warn(
            "save() is deprecated and will be removed in a future version. Use to_image() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.to_image(
            path=filename,
            scale=scale,
            width=width,
            show_labels=labels,
            legend_position=legend_position
        )
        return self
        
    def clear_highlights(self) -> 'Page':
        """
        Clear all highlights from the page.
        
        Returns:
            Self for method chaining
        """
        self._highlight_mgr.clear_highlights()
        return self
        
    def analyze_text_styles(self) -> Dict[str, 'ElementCollection']:
        """
        Analyze and group text elements by their style properties.
        
        Returns:
            Dictionary mapping style labels to element collections
        """
        # Import the analyzer
        from natural_pdf.analyzers.text_structure import TextStyleAnalyzer
        
        # Create analyzer
        analyzer = TextStyleAnalyzer()
        
        # Analyze the page and store the results
        self._text_styles = analyzer.analyze(self)
        
        # Return the analyzed styles
        return self._text_styles
        
    def highlight_text_styles(self) -> 'Page':
        """
        Highlight text elements grouped by their style properties.
        
        This automatically analyzes the styles if they haven't been analyzed yet.
        
        Returns:
            Self for method chaining
        """
        # Analyze styles if not already done
        if self._text_styles is None:
            self.analyze_text_styles()
            
        # Highlight each style group with its own color
        for label, elements in self._text_styles.items():
            elements.highlight(label=label)
            
        return self
    
    def highlight_all(self, 
                     include_types: Optional[List[str]] = None, 
                     include_text_styles: bool = False,
                     include_layout_regions: bool = False,
                     apply_exclusions: bool = True,
                     use_color_cycling: bool = True,
                     layout_confidence: float = 0.2) -> 'Page':
        """
        Highlight all elements on the page, grouped by type or style.
        
        Each element type or style gets its own color and label in the legend.
        
        Args:
            include_types: Optional list of element types to include
                           (e.g., ['text', 'line', 'rect'])
                           If None, all available types will be included
            include_text_styles: Whether to highlight text by style groups
                                (font, size, etc.) instead of as a single group
            include_layout_regions: Whether to include detected layout regions
                                   (will run layout detection if not already done)
                                   Layout regions will be grouped by model and type
            apply_exclusions: Whether to respect exclusion zones (default: True)
            use_color_cycling: Whether to use different colors for each type (default: True)
            layout_confidence: Confidence threshold for layout regions (default: 0.2)
                               If True is passed, all regions will be included regardless of confidence
                           
        Returns:
            Self for method chaining
        """
        # Load all elements if not already loaded
        self._load_elements()
        
        # Get exclusion regions if we're applying exclusions
        exclusion_regions = []
        if apply_exclusions and self._exclusions:
            # Get exclusion regions using callable functions when appropriate
            exclusion_regions = self._get_exclusion_regions(include_callable=True)
        
        # Define all available element types
        all_types = {
            'text': self.words,
            'char': self.chars,
            'rect': self.rects,
            'line': self.lines,
            # Add other types as they become available
        }
        
        # Highlight by text styles if requested
        # This takes precedence over normal text highlighting
        if include_text_styles:
            # Analyze text styles 
            styles = self.analyze_text_styles()
            
            # Apply exclusions to each style group if needed
            if apply_exclusions and exclusion_regions:
                for label, elements in styles.items():
                    # Filter out excluded elements
                    filtered_elements = elements.exclude_regions(exclusion_regions)
                    # Highlight with appropriate label
                    filtered_elements.highlight(label=label, use_color_cycling=use_color_cycling)
            else:
                # Highlight without exclusions
                for label, elements in styles.items():
                    elements.highlight(label=label, use_color_cycling=use_color_cycling)
            
            # Highlight non-text elements normally
            if include_types:
                # Filter to only include non-text types
                non_text_types = [t for t in include_types if t != 'text']
                
                # Highlight each non-text type
                for element_type in non_text_types:
                    if element_type in all_types and all_types[element_type]:
                        label = f"{element_type.capitalize()} Elements"
                        elements = all_types[element_type]
                        
                        # Skip empty collections
                        if not elements:
                            continue
                            
                        # Create an ElementCollection if needed
                        from natural_pdf.elements.collections import ElementCollection
                        if not isinstance(elements, ElementCollection):
                            elements = ElementCollection(elements)
                        
                        # Apply exclusions if needed
                        if apply_exclusions and exclusion_regions:
                            elements = elements.exclude_regions(exclusion_regions)
                            
                        # Highlight with appropriate label
                        elements.highlight(label=label, cycle_colors=cycle_colors)
            else:
                # Highlight all non-text elements
                for element_type in all_types.keys():
                    if element_type != 'text' and element_type != 'char':
                        if all_types[element_type]:
                            label = f"{element_type.capitalize()} Elements"
                            elements = all_types[element_type]
                            
                            # Skip empty collections
                            if not elements:
                                continue
                                
                            # Create an ElementCollection if needed
                            from natural_pdf.elements.collections import ElementCollection
                            if not isinstance(elements, ElementCollection):
                                elements = ElementCollection(elements)
                            
                            # Apply exclusions if needed
                            if apply_exclusions and exclusion_regions:
                                elements = elements.exclude_regions(exclusion_regions)
                                
                            # Highlight with appropriate label
                            elements.highlight(label=label, use_color_cycling=use_color_cycling)
            
            return self
            
        # Normal highlight_all behavior (by element type)
        # Determine which types to highlight
        types_to_highlight = include_types if include_types else all_types.keys()
        
        # Highlight each type of element with its own color/label
        for element_type in types_to_highlight:
            if element_type in all_types and all_types[element_type]:
                # Format label (e.g., "text" -> "Text Elements")
                label = f"{element_type.capitalize()} Elements"
                
                # Get the elements and highlight them
                elements = all_types[element_type]
                
                # Skip empty collections
                if not elements:
                    continue
                    
                # Create an ElementCollection if needed
                from natural_pdf.elements.collections import ElementCollection
                if not isinstance(elements, ElementCollection):
                    elements = ElementCollection(elements)
                
                # Apply exclusions if needed
                if apply_exclusions and exclusion_regions:
                    elements = elements.exclude_regions(exclusion_regions)
                    
                # Highlight with appropriate label
                elements.highlight(label=label, use_color_cycling=use_color_cycling)
                
        # Include layout regions if requested
        if include_layout_regions:
            # Run layout detection if not already done
            if (not hasattr(self, 'detected_layout_regions') or not self.detected_layout_regions) and \
               ('detected' not in self._regions or not self._regions['detected']):
                # Make sure to run analyze_layout with include_highlights=False
                self.analyze_layout(confidence=layout_confidence)
            
            # Get layout regions from either detected_layout_regions or _regions['detected']
            layout_regions = []
            if hasattr(self, 'detected_layout_regions') and self.detected_layout_regions:
                layout_regions = self.detected_layout_regions
            elif 'detected' in self._regions and self._regions['detected']:
                layout_regions = self._regions['detected']
            
            # Filter regions by confidence (handle case where layout_confidence=True)
            if isinstance(layout_confidence, bool):
                # If True is passed, don't filter by confidence
                filtered_regions = layout_regions
            else:
                # Filter by confidence threshold
                filtered_regions = [r for r in layout_regions if hasattr(r, 'confidence') and r.confidence >= layout_confidence]
            layout_regions = filtered_regions
            
            # Group regions by model and type for better visualization
            models = set(r.model for r in layout_regions if hasattr(r, 'model'))
            
            for model in models:
                # Get regions for this model
                model_regions = [r for r in layout_regions if hasattr(r, 'model') and r.model == model]
                
                # Group by type within model
                types = set(r.region_type for r in model_regions if hasattr(r, 'region_type'))
                
                for region_type in types:
                    # Get regions of this type
                    type_regions = [r for r in model_regions if hasattr(r, 'region_type') and r.region_type == region_type]
                    
                    # Create a collection and highlight
                    from natural_pdf.elements.collections import ElementCollection
                    collection = ElementCollection(type_regions)
                    
                    # Determine color based on type (similar to highlight_layout logic)
                    color = None
                    if model == 'tatr':
                        if region_type == 'table':
                            color = (1, 0, 0, 0.3)  # Red for tables
                        elif region_type == 'table row':
                            color = (0, 1, 0, 0.3)  # Green for rows
                        elif region_type == 'table column':
                            color = (0, 0, 1, 0.3)  # Blue for columns
                        elif region_type == 'table column header':
                            color = (0, 1, 1, 0.3)  # Cyan for column headers
                    
                    # Don't use ElementCollection for this case since we want individual confidence scores
                    # Instead, highlight each region individually with its own confidence
                    for region in type_regions:
                        # Create a label with model and type
                        label = f"Layout ({model}): {region_type}"
                        
                        # Highlight with the same color scheme but don't automatically include attributes
                        region.highlight(
                            label=label, 
                            color=color, 
                            use_color_cycling=use_color_cycling
                            # No include_attrs by default - user must explicitly request it
                        )
                
        return self
        
    def to_image(self,
            path: Optional[str] = None,
            scale: float = 2.0,
            width: Optional[int] = None,
            labels: bool = True,
            legend_position: str = 'right',
            render_ocr: bool = False,
            resolution: float = None,
            include_highlights: bool = True,
            **kwargs) -> Image.Image:
        """
        Generate a PIL image of the page, optionally with highlights, and optionally save it to a file.
        
        Args:
            path: Optional path to save the image to
            scale: Scale factor for rendering highlights (default: 2.0)
            width: Optional width for the output image in pixels (height calculated to maintain aspect ratio)
            labels: Whether to include a legend for labels (default: True)
            legend_position: Position of the legend (default: 'right')
            render_ocr: Whether to render OCR text with white background boxes (default: False)
            resolution: Resolution in DPI for base page image (default: scale * 72)
            include_highlights: Whether to include highlights (default: True)
            **kwargs: Additional parameters for pdfplumber.to_image
            
        Returns:
            PIL Image of the page
            
        Examples:
            >>> # Get base page image without highlights
            >>> img = page.to_image(include_highlights=False)
            >>> 
            >>> # Get image with highlights and no labels
            >>> img = page.to_image(labels=False)
            >>> 
            >>> # Save image with specific width
            >>> page.to_image(path="output.png", width=800)
        """
        # Use resolution based on scale if not provided
        if resolution is None:
            resolution = scale * 72  # Convert scale to DPI (72 is base DPI)
            
        if include_highlights and hasattr(self, '_highlight_mgr'):
            # Get the highlighted image
            image = self._highlight_mgr.get_highlighted_image(scale, labels, legend_position, render_ocr)
        else:
            # Get the base page image from pdfplumber
            image = self._page.to_image(resolution=resolution, **kwargs).annotated
        
        # Resize the image if width is provided
        if width is not None and width > 0:
            # Calculate height to maintain aspect ratio
            aspect_ratio = image.height / image.width
            height = int(width * aspect_ratio)
            # Resize the image
            image = image.resize((width, height), Image.LANCZOS)
        
        # Save the image if path is provided
        if path:
            image.save(path)
            
        return image
        
    def _get_ocr_config(self, ocr_params: Optional[Union[bool, str, List, Dict]] = None) -> Dict[str, Any]:
        """
        Get the OCR configuration by merging defaults, PDF settings, and provided params.
        
        Args:
            ocr_params: OCR parameters to override defaults
            
        Returns:
            Merged OCR configuration
        """
        if HAS_OCR_ENGINES and hasattr(self._parent, '_ocr_engine') and self._parent._ocr_engine:
            # Use new OCR engine system
            engine = self._parent._ocr_engine
            
            # Get normalized PDF-level config
            pdf_config = self._parent._ocr_config
            
            # Special case: If ocr_params is boolean True, convert to config with enabled=True
            if ocr_params is True:
                ocr_params = {"enabled": True}
            
            # Normalize provided config
            if ocr_params is not None:
                provided_config = engine.normalize_config(ocr_params)
                
                # If provided config explicitly sets enabled, respect that
                if "enabled" in provided_config:
                    # Always merge configs to get language settings etc. from PDF-level config
                    result_config = engine.merge_configs(pdf_config, provided_config)
                    # Only print status if verbose mode is not explicitly disabled
                    if provided_config.get('verbose', True):
                        print(f"OCR enabled status from provided params: {provided_config.get('enabled')}")
                    return result_config
                else:
                    # Merge configs and keep PDF-level enabled status
                    result_config = engine.merge_configs(pdf_config, provided_config)
                    # Only print status if verbose mode is not explicitly disabled
                    if provided_config.get('verbose', True):
                        print(f"OCR enabled status from PDF config: {pdf_config.get('enabled')}")
                    return result_config
            else:
                # Use PDF-level config
                # Only print status if verbose mode is not explicitly disabled
                if ocr_params is None or not isinstance(ocr_params, dict) or ocr_params.get('verbose', True):
                    print(f"Using PDF-level OCR config: {pdf_config}")
                return pdf_config
        else:
            # Fallback to legacy OCR manager
            ocr_manager = OCRManager.get_instance()
            
            # Get normalized PDF-level config
            pdf_config = self._parent._ocr_config
            
            # Special case: If ocr_params is boolean True, convert to config with enabled=True
            if ocr_params is True:
                ocr_params = {"enabled": True}
            
            # Normalize provided config
            if ocr_params is not None:
                provided_config = ocr_manager.normalize_config(ocr_params)
                
                # If provided config explicitly sets enabled, respect that
                if "enabled" in provided_config:
                    # Always merge configs to get language settings etc. from PDF-level config
                    result_config = ocr_manager.merge_configs(pdf_config, provided_config)
                    print(f"OCR enabled status from provided params: {provided_config.get('enabled')}")
                    return result_config
                else:
                    # Merge configs and keep PDF-level enabled status
                    result_config = ocr_manager.merge_configs(pdf_config, provided_config)
                    print(f"OCR enabled status from PDF config: {pdf_config.get('enabled')}")
                    return result_config
            else:
                # Use PDF-level config
                print(f"Using PDF-level OCR config: {pdf_config}")
                return pdf_config
            
    def _create_text_elements_from_ocr(self, ocr_results: List[Dict[str, Any]], image_width=None, image_height=None) -> List[TextElement]:
        """
        Convert OCR results to TextElement objects.
        
        Args:
            ocr_results: List of OCR results with text, bbox, and confidence
            image_width: Width of the source image (for coordinate scaling)
            image_height: Height of the source image (for coordinate scaling)
            
        Returns:
            List of created TextElement objects
        """
        elements = []
        
        # Calculate scale factors to convert from image coordinates to PDF coordinates
        # Default to 1.0 if not provided (assume coordinates are already in PDF space)
        scale_x = 1.0
        scale_y = 1.0
        
        if image_width and image_height:
            scale_x = self.width / image_width
            scale_y = self.height / image_height
        
        for result in ocr_results:
            # Convert numpy int32 to float if needed and scale to PDF coordinates
            x0 = float(result['bbox'][0]) * scale_x
            top = float(result['bbox'][1]) * scale_y
            x1 = float(result['bbox'][2]) * scale_x
            bottom = float(result['bbox'][3]) * scale_y
            
            # Create a TextElement object with additional required fields for highlighting
            element_data = {
                'text': result['text'],
                'x0': x0,
                'top': top,
                'x1': x1,
                'bottom': bottom,
                'width': x1 - x0,
                'height': bottom - top,
                'object_type': 'text',
                'source': 'ocr',
                'confidence': result['confidence'],
                # Add default font information to work with existing expectations
                'fontname': 'OCR-detected',
                'size': 10.0,
                'page_number': self.number
            }
            
            elem = TextElement(element_data, self)
            elements.append(elem)
            
            # Add to page's elements
            if hasattr(self, '_elements') and self._elements is not None:
                # Add to words list to make it accessible via standard API
                if 'words' in self._elements:
                    self._elements['words'].append(elem)
                else:
                    self._elements['words'] = [elem]
                
        return elements
        
    def apply_ocr(self, **ocr_params) -> List[TextElement]:
        """
        Apply OCR to this page and register results as text elements.
        
        Args:
            **ocr_params: OCR parameters to override defaults
            
        Returns:
            List of created text elements
        """
        # Get OCR config (merge defaults, PDF settings, and provided params)
        # Ensure OCR is enabled for this explicit OCR call
        if isinstance(ocr_params, dict):
            ocr_params["enabled"] = True
        else:
            ocr_params = {"enabled": True}
            
        config = self._get_ocr_config(ocr_params)
        
        # Skip if OCR is still disabled (should not happen after the above override)
        if not config.get('enabled'):
            print(f"OCR is disabled in config despite override - forcing enabled=True")
            config["enabled"] = True
            
        # Render page to image
        print(f"Rendering page {self.number} to image for OCR...")
        image = self.to_image()
        print(f"Image size: {image.width}x{image.height}")
        
        # Save image for debugging if needed
        try:
            import os
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"page_{self.number}_for_ocr.png")
            image.save(debug_path)
            print(f"Saved page image for debugging to {debug_path}")
        except Exception as e:
            print(f"Could not save debug image: {e}")
        
        # Process the image with the appropriate OCR engine
        print(f"Processing image with OCR engine...")
        if HAS_OCR_ENGINES and hasattr(self._parent, '_ocr_engine') and self._parent._ocr_engine:
            # Use new OCR engine system
            print(f"Using OCR engine: {self._parent._ocr_engine.__class__.__name__}")
            engine = self._parent._ocr_engine
            results = engine.process_image(image, config)
        else:
            # Fallback to legacy OCR manager
            print(f"Using legacy OCR manager")
            ocr_mgr = OCRManager.get_instance()
            results = ocr_mgr.detect_and_recognize(image, config)
            
        print(f"OCR returned {len(results)} results")
        
        # Convert results to elements and add to page, with image dimensions for scaling
        elements = self._create_text_elements_from_ocr(results, image.width, image.height)
        
        return elements
        
    def extract_ocr_elements(self, **ocr_params) -> List[TextElement]:
        """
        Extract text elements using OCR.
        
        This method applies OCR to the page and returns the resulting text elements
        without modifying the page's elements list.
        
        Args:
            **ocr_params: OCR parameters to override defaults
            
        Returns:
            List of text elements created from OCR
        """
        print("=" * 40)
        print(f"Page.extract_ocr_elements called with params: {ocr_params}")
        
        # Get OCR config
        # Ensure OCR is enabled for this explicit OCR call
        if isinstance(ocr_params, dict):
            ocr_params["enabled"] = True
        else:
            ocr_params = {"enabled": True}
            
        config = self._get_ocr_config(ocr_params)
        print(f"OCR config after normalization: {config}")
        
        # Skip if OCR is still disabled (should not happen after the above override)
        if not config.get('enabled'):
            print(f"OCR is disabled in config despite override - forcing enabled=True")
            config["enabled"] = True
        
        # Try direct OCR test for debugging
        import os
        try:
            print("Trying direct OCR test for debugging...")
            
            # Save image to temp file for debugging
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
            os.makedirs(output_dir, exist_ok=True)
            temp_image_path = os.path.join(output_dir, "direct_ocr_debug.png")
            
            # Get the image using the direct method
            print("Generating page image...")
            from PIL import Image
            image = self.to_image()
            image.save(temp_image_path)
            print(f"Saved image to {temp_image_path}")
            
            try:
                import easyocr
                print("Testing direct EasyOCR...")
                reader = easyocr.Reader(['en'])
                import numpy as np
                result = reader.readtext(np.array(image))
                print(f"Direct EasyOCR test got {len(result)} results")
            except ImportError:
                print("EasyOCR not available for direct test")
            except Exception as e:
                print(f"Error in direct EasyOCR test: {e}")
                
            try:
                import paddleocr
                print("Testing direct PaddleOCR...")
                reader = paddleocr.PaddleOCR(lang='en')
                import numpy as np
                result = reader.ocr(np.array(image), cls=False)
                if result is not None and len(result) > 0:
                    page_result = result[0] if isinstance(result[0], list) else result
                    print(f"Direct PaddleOCR test got {len(page_result)} results")
                else:
                    print(f"Direct PaddleOCR test got no results: {result}")
            except ImportError:
                print("PaddleOCR not available for direct test")
            except Exception as e:
                print(f"Error in direct PaddleOCR test: {e}")
        except Exception as e:
            print(f"Error in direct OCR test: {e}")
            
        # Now try the normal process
        print("Proceeding with normal OCR process...")
            
        # Render page to image
        print(f"Rendering page {self.number} to image for OCR...")
        image = self.to_image()
        print(f"Image size: {image.width}x{image.height}")
        
        # Process the image with the appropriate OCR engine
        print(f"Processing image with OCR engine...")
        results = []
        
        try:
            if HAS_OCR_ENGINES and hasattr(self._parent, '_ocr_engine') and self._parent._ocr_engine:
                # Use new OCR engine system
                print(f"Using OCR engine: {self._parent._ocr_engine.__class__.__name__}")
                engine = self._parent._ocr_engine
                
                # Directly test the engine
                print(f"Direct test of {engine.__class__.__name__}.process_image")
                results = engine.process_image(image, config)
                print(f"Engine returned {len(results)} results")
            else:
                # Fallback to legacy OCR manager
                print(f"Using legacy OCR manager")
                ocr_mgr = OCRManager.get_instance()
                results = ocr_mgr.detect_and_recognize(image, config)
                print(f"OCR manager returned {len(results)} results")
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        print(f"OCR returned {len(results)} results")
        if len(results) > 0:
            print(f"First result: {results[0]}")
        
        # Create a copy of the original _elements so we can restore it later
        original_elements = None
        if hasattr(self, '_elements'):
            original_elements = self._elements
            # Temporarily set _elements to None so they aren't added to the page
            self._elements = None
        
        # Create elements with proper scaling (but don't add to page)
        print(f"Creating text elements from {len(results)} OCR results...")
        elements = self._create_text_elements_from_ocr(results, image.width, image.height)
        print(f"Created {len(elements)} text elements")
        
        # Restore original elements
        if original_elements is not None:
            self._elements = original_elements
            
        print(f"Returning {len(elements)} OCR elements")
        print("=" * 40)
        return elements
        
    def analyze_layout(self, 
                      model: str = "docling",
                      confidence: float = 0.2,
                      classes: Optional[List[str]] = None,
                      exclude_classes: Optional[List[str]] = None,
                      device: str = "cpu",
                      existing: str = "replace",
                      model_params: Optional[Dict[str, Any]] = None,
                      # Legacy parameters for backward compatibility
                      model_path: Optional[str] = None,
                      image_size: int = 1024,
                      create_cells: bool = False) -> 'Page':
        """
        Analyze the page layout using a machine learning model.
        
        Args:
            model: Model type to use ('yolo', 'tatr', 'paddle', or 'docling')
            confidence: Minimum confidence threshold for detections
            classes: Specific classes to detect (None for all supported classes)
            exclude_classes: Classes to exclude from detection
            device: Device to use for inference ('cpu' or 'cuda:0'/'gpu')
            existing: How to handle existing regions: 'replace' (default) or 'append'
            model_params: Dictionary of model-specific parameters:
                - YOLO: {"model_path": "...", "image_size": 1024}
                - TATR: {"model_path": "...", "create_cells": False}
                - Paddle: {"lang": "en", "use_angle_cls": False, "enable_table": True}
                - Docling: {"model_name": "ds4sd/SmolDocling-256M-preview", "prompt_text": "...", "verbose": False}
            model_path: (Legacy) Optional path to custom model file
            image_size: (Legacy) Size to resize the image to before detection (YOLO only)
            create_cells: (Legacy) Whether to create cell regions for TATR table regions
            
        Returns:
            Self for method chaining
        """
        # Initialize model_params if None
        if model_params is None:
            model_params = {}
            
        # Handle legacy parameters by adding them to model_params
        if model_path is not None:
            model_params['model_path'] = model_path
        if model.lower() == "yolo" and image_size != 1024:
            model_params['image_size'] = image_size
        if model.lower() == "tatr" and create_cells:
            model_params['create_cells'] = create_cells
            
        # Create a temporary directory to store the page image
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, f"page_{self.index}.png")
        
        try:
            # Render the page as an image and save to temp file
            # Explicitly set include_highlights=False to ensure we get the original page image
            page_image = self.to_image(resolution=150.0, include_highlights=False)
            page_image.save(temp_image_path)
            
            # Initialize the appropriate detector based on the model type
            if model.lower() == "yolo":
                # Extract YOLO-specific parameters
                model_file = model_params.get('model_path', "doclayout_yolo_docstructbench_imgsz1024.pt")
                yolo_image_size = model_params.get('image_size', 1024)
                
                detector = YOLODocLayoutDetector(
                    model_file=model_file,
                    device=device
                )
                # Run detection
                detections = detector.detect(
                    temp_image_path,
                    confidence=confidence,
                    classes=classes,
                    exclude_classes=exclude_classes,
                    image_size=yolo_image_size
                )
                
            elif model.lower() == "tatr" or model.lower() == "table-transformer":
                # Extract TATR-specific parameters
                tatr_model_path = model_params.get('model_path')
                
                detector = TableTransformerDetector(
                    detection_model="microsoft/table-transformer-detection" if tatr_model_path is None else tatr_model_path,
                    device=device
                )
                # Run detection
                detections = detector.detect(
                    temp_image_path,
                    confidence=confidence,
                    classes=classes,
                    exclude_classes=exclude_classes
                )
                
            elif model.lower() == "paddle":
                # Extract PaddlePaddle-specific parameters
                paddle_lang = model_params.get('lang', 'en')
                use_angle_cls = model_params.get('use_angle_cls', False)
                enable_table = model_params.get('enable_table', True)
                show_log = model_params.get('show_log', False)
                
                # Convert device format
                paddle_device = 'gpu' if device.startswith('cuda') else device
                
                # Initialize PaddleLayoutDetector
                detector = PaddleLayoutDetector(
                    lang=paddle_lang,
                    use_angle_cls=use_angle_cls,
                    device=paddle_device,
                    enable_table=enable_table,
                    show_log=show_log
                )
                
                # Run detection
                detections = detector.detect(
                    temp_image_path,
                    confidence=confidence,
                    classes=classes,
                    exclude_classes=exclude_classes
                )
                
            elif model.lower() == "docling":
                # Extract Docling-specific parameters
                verbose = model_params.get('verbose', False)
                
                # Pass all other model_params directly to DocumentConverter
                detector_kwargs = {k: v for k, v in model_params.items() if k != 'verbose'}
                
                # Initialize DoclingLayoutDetector
                detector = DoclingLayoutDetector(
                    verbose=verbose,
                    **detector_kwargs
                )
                
                # Run detection
                detections = detector.detect(
                    temp_image_path,
                    confidence=confidence,
                    classes=classes,
                    exclude_classes=exclude_classes
                )
                
                # Store the original Docling document for advanced usage
                self.docling_document = detector.get_docling_document()
                
            else:
                raise ValueError(f"Unsupported model type: {model}. Currently supported: 'yolo', 'tatr', 'paddle', 'docling'")
            
            # Calculate the scale factor to convert from image to PDF coordinates
            # Note: This assumes the image resolution is 150 DPI
            scale_x = self.width / page_image.width
            scale_y = self.height / page_image.height
            
            # Create a list to store layout regions
            layout_regions = []
            
            # Convert detections to regions
            # First create all regions and track by docling_id if available
            docling_id_to_region = {}
            
            for detection in detections:
                x_min, y_min, x_max, y_max = detection['bbox']
                
                # Convert coordinates from image to PDF space
                pdf_x0 = x_min * scale_x
                pdf_y0 = y_min * scale_y
                pdf_x1 = x_max * scale_x
                pdf_y1 = y_max * scale_y
                
                # Create a region
                region = Region(self, (pdf_x0, pdf_y0, pdf_x1, pdf_y1))
                region.region_type = detection['class']
                region.normalized_type = detection['normalized_class']
                region.confidence = detection['confidence']
                region.model = model  # Store which model detected this region
                region.source = 'detected'  # Set the source for selectors
                
                # If this is a Docling detection, include text content
                if model.lower() == 'docling':
                    if 'text' in detection:
                        region.text_content = detection.get('text')
                        
                    # Track by docling_id for building hierarchy later
                    if 'docling_id' in detection:
                        region.docling_id = detection['docling_id']
                        docling_id_to_region[detection['docling_id']] = region
                        
                    # Store parent ID for hierarchy building
                    if 'parent_id' in detection:
                        region.parent_id = detection.get('parent_id')
                
                layout_regions.append(region)
                
            # If using Docling model, build parent-child relationships
            if model.lower() == 'docling':
                # Second pass to establish parent-child relationships
                for region in layout_regions:
                    if hasattr(region, 'parent_id') and region.parent_id:
                        parent_region = docling_id_to_region.get(region.parent_id)
                        if parent_region:
                            parent_region.add_child(region)
            
            # Handle existing regions based on mode
            if existing.lower() == 'append':
                # Append to existing detected regions
                self._regions['detected'].extend(layout_regions)
            else:
                # Replace existing detected regions
                self._regions['detected'] = layout_regions
            
            # Make sure elements is initialized
            self._load_elements()
            
            # Update elements collection for selectors
            if 'regions' not in self._elements:
                self._elements['regions'] = []
                
            # Update elements collection based on existing mode
            if existing.lower() == 'append':
                # Only add new regions that aren't already in the collection
                for region in layout_regions:
                    if region not in self._elements['regions']:
                        self._elements['regions'].append(region)
            else:
                # Replace existing regions in _elements with detected regions, keep named regions
                # First get all named regions from _elements['regions']
                named_regions = [r for r in self._elements['regions'] if r.source == 'named']
                # Then create a new list with named regions and layout regions
                self._elements['regions'] = named_regions + layout_regions
            
            # Create cells for table regions if requested and using TATR
            create_cells_flag = model_params.get('create_cells', create_cells)
            if model.lower() == 'tatr' and create_cells_flag:
                # Debug log
                print(f"Creating cells for {len([r for r in layout_regions if r.region_type == 'table'])} table regions")
                
                cell_count = 0
                for region in layout_regions:
                    # Check if it's a table region
                    if region.region_type == 'table':
                        try:
                            # Create cells for the table
                            cells = region.create_cells()
                            cell_count += len(cells)
                            
                            # Add cell regions to our tracking structures
                            layout_regions.extend(cells)
                            
                            # Also add to _elements for selectors
                            if 'regions' in self._elements:
                                self._elements['regions'].extend(cells)
                                
                            # And to _regions['detected']
                            self._regions['detected'].extend(cells)
                            
                        except Exception as e:
                            print(f"Error creating cells for table: {e}")
                            
                # Debug log
                print(f"Created {cell_count} cells in total")
                        
            # Store layout regions in an instance variable so they can be accessed after the method returns
            self.detected_layout_regions = layout_regions
            return self
            
        finally:
            # Clean up temporary file and directory
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            os.rmdir(temp_dir)
    
    def highlight_layout(self, 
                        layout_regions: Optional[List[Region]] = None,
                        confidence: float = 0.2,
                        label_format: str = "{type} ({conf:.2f}){model}") -> 'Page':
        """
        Highlight detected layout regions on the page.
        
        Args:
            layout_regions: List of regions to highlight (runs analyze_layout if None)
            confidence: Minimum confidence threshold for highlighting regions
            label_format: Format string for region labels
            
        Returns:
            Self for method chaining
        """
        # If no regions provided, use detected_layout_regions, detected regions, or run layout detection
        if layout_regions:
            regions = layout_regions
        elif hasattr(self, 'detected_layout_regions') and self.detected_layout_regions:
            regions = self.detected_layout_regions
        elif 'detected' in self._regions and self._regions['detected']:
            regions = self._regions['detected']
        else:
            # Call analyze_layout with include_highlights=False and use the result directly
            self.analyze_layout(confidence=confidence)
            regions = self.detected_layout_regions
            
        # Highlight each region with its type as the label
        for region in regions:
            # Skip regions below confidence threshold
            if region.confidence < confidence:
                continue
                
            # No model filtering here - use selectors for that
                
            # Format label
            model_suffix = f" ({region.model})" if hasattr(region, 'model') else ""
            label = label_format.format(
                type=region.region_type,
                conf=region.confidence,
                model=model_suffix
            )
            
            # Highlight region with appropriate color based on model
            if hasattr(region, 'model') and region.model == 'tatr':
                # Use different colors for table structure elements
                if region.region_type == 'table':
                    color = (1, 0, 0, 0.3)  # Red for tables
                elif region.region_type == 'table row':
                    color = (0, 1, 0, 0.3)  # Green for rows
                elif region.region_type == 'table column':
                    color = (0, 0, 1, 0.3)  # Blue for columns
                elif region.region_type == 'table column header':
                    color = (0, 1, 1, 0.3)  # Cyan for column headers
                else:
                    color = None  # Default color cycling
                region.highlight(label=label, color=color)
            else:
                region.highlight(label=label)
            
        return self
        
    def get_section_between(self, start_element=None, end_element=None, boundary_inclusion='both') -> Region:
        """
        Get a section between two elements on this page.
        
        Args:
            start_element: Element marking the start of the section
            end_element: Element marking the end of the section
            boundary_inclusion: How to include boundary elements: 'start', 'end', 'both', or 'none'
            
        Returns:
            Region representing the section between elements
        """
        # Create a full-page region
        page_region = self.create_region(0, 0, self.width, self.height)
        
        # Get the section from the region
        return page_region.get_section_between(
            start_element=start_element,
            end_element=end_element,
            boundary_inclusion=boundary_inclusion
        )
    
    def get_sections(self, 
                  start_elements=None, 
                  end_elements=None,
                  boundary_inclusion='both',
                  y_threshold=5.0,
                  bounding_box=None):
        """
        Get sections of a page defined by start/end elements.
        
        Args:
            start_elements: Elements or selector string that mark the start of sections
            end_elements: Elements or selector string that mark the end of sections
            boundary_inclusion: How to include boundary elements: 'start', 'end', 'both', or 'none'
            y_threshold: Maximum vertical difference to consider elements on same line
            bounding_box: Optional tuple (x0, top, x1, bottom) to limit the section area
            
        Returns:
            List of Region objects representing the sections
        """
        # Helper function to get bounds from bounding_box parameter
        def get_bounds():
            if bounding_box:
                return bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
            else:
                return 0, 0, self.width, self.height
                
        regions = []
        
        # Handle cases where elements are provided as strings (selectors)
        if isinstance(start_elements, str):
            start_elements = self.find_all(start_elements)
            
        if isinstance(end_elements, str):
            end_elements = self.find_all(end_elements)

        # Validate boundary_inclusion parameter
        valid_inclusions = ['start', 'end', 'both', 'none']
        if boundary_inclusion not in valid_inclusions:
            raise ValueError(f"boundary_inclusion must be one of {valid_inclusions}")
        
        # If no start elements, can't do anything
        if not start_elements:
            return regions
            
        # Sort elements by position (top-to-bottom, left-to-right)
        all_elements = []
        
        for element in start_elements:
            all_elements.append((element, 'start'))
            
        if end_elements:
            for element in end_elements:
                all_elements.append((element, 'end'))
                
        # Group elements with similar Y coordinates
        # Consider elements on the same line if they're within the threshold
        
        # First sort all elements by Y position
        all_elements.sort(key=lambda x: x[0].top)
        
        # Group elements on the same line
        grouped_elements = []
        current_group = []
        current_group_type = None
        current_y = None
        
        for element, element_type in all_elements:
            if current_y is None or abs(element.top - current_y) <= y_threshold:
                # Element is on the same line as current group
                if current_group and element_type != current_group_type:
                    # If we have a mixed group, prioritize start elements over end elements
                    if element_type == 'start':
                        current_group_type = 'start'
                elif not current_group:
                    current_group_type = element_type
                    
                current_group.append(element)
                current_y = element.top  # Update reference Y
            else:
                # Element is on a new line, close current group and start a new one
                if current_group:
                    # Find the leftmost element in the group
                    leftmost = min(current_group, key=lambda e: e.x0)
                    grouped_elements.append((leftmost, current_group_type))
                
                # Start a new group
                current_group = [element]
                current_group_type = element_type
                current_y = element.top
        
        # Add the last group
        if current_group:
            # Find the leftmost element in the group
            leftmost = min(current_group, key=lambda e: e.x0)
            grouped_elements.append((leftmost, current_group_type))
        
        # Use the grouped elements for sectioning
        all_elements = grouped_elements
        
        # Find sections
        current_start = None
        
        for i, (element, element_type) in enumerate(all_elements):
            if element_type == 'start':
                # If we already have a start without an end, create a section until this start
                if current_start is not None:
                    # Create a region from current_start to this start
                    start_element = current_start
                    end_element = element
                    
                    # Determine region boundaries based on inclusion parameter
                    if boundary_inclusion in ['start', 'both']:
                        top = start_element.top
                    else:
                        top = start_element.bottom
                        
                    if boundary_inclusion in ['end', 'both']:
                        bottom = end_element.bottom
                    else:
                        bottom = end_element.top
                        
                    # Create the region
                    x0, _, x1, _ = get_bounds()
                    region = self.create_region(x0, top, x1, bottom)
                    region.start_element = start_element
                    region.end_element = end_element
                    region.is_end_next_start = True
                    regions.append(region)
                
                # Save this element as the current start
                current_start = element
                
            elif element_type == 'end' and current_start is not None:
                # We found an end for the current start
                start_element = current_start
                end_element = element
                
                # Determine region boundaries based on inclusion parameter
                if boundary_inclusion in ['start', 'both']:
                    top = start_element.top
                else:
                    top = start_element.bottom
                    
                if boundary_inclusion in ['end', 'both']:
                    bottom = end_element.bottom
                else:
                    bottom = end_element.top
                    
                # Create the region
                x0, _, x1, _ = get_bounds()
                region = self.create_region(x0, top, x1, bottom)
                region.start_element = start_element
                region.end_element = end_element
                region.is_end_next_start = False
                regions.append(region)
                
                # Reset current start so we don't use it again
                current_start = None
        
        # If we have a start without an end at the end, create a section to the page bottom
        if current_start is not None:
            # Determine region top boundary based on inclusion parameter
            if boundary_inclusion in ['start', 'both']:
                top = current_start.top
            else:
                top = current_start.bottom
                
            # Create the region to the bottom of the page
            x0, _, x1, page_bottom = get_bounds()
            region = self.create_region(x0, top, x1, page_bottom)
            region.start_element = current_start
            region.end_element = None
            region.is_end_next_start = False
            regions.append(region)
            
        return regions
            
    def __repr__(self) -> str:
        """String representation of the page."""
        return f"<Page number={self.number} index={self.index}>"
        
    def ask(self, question: str, min_confidence: float = 0.1, model: str = None, debug: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Ask a question about the page content using document QA.
        
        This method uses a document question answering model to extract answers from the page content.
        It leverages both textual content and layout information for better understanding.
        
        Args:
            question: The question to ask about the page content
            min_confidence: Minimum confidence threshold for answers (0.0-1.0)
            model: Optional model name to use for QA (if None, uses default model)
            **kwargs: Additional parameters to pass to the QA engine
            
        Returns:
            Dictionary with answer details: {
                "answer": extracted text,
                "confidence": confidence score,
                "found": whether an answer was found,
                "page_num": page number,
                "source_elements": list of elements that contain the answer (if found)
            }
        """
        from natural_pdf.qa.document_qa import get_qa_engine
        
        # Get or initialize QA engine with specified model
        qa_engine = get_qa_engine(model_name=model) if model else get_qa_engine()
        
        # Ask the question using the QA engine
        return qa_engine.ask_pdf_page(self, question, min_confidence=min_confidence, debug=debug, **kwargs)
