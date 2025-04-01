import pdfplumber
import logging
import tempfile
import os
import re
import urllib.request
from typing import List, Optional, Union, Any, Dict, Callable, Tuple, Type, Iterable # Added Iterable
from PIL import Image

from natural_pdf.core.page import Page
from natural_pdf.selectors.parser import parse_selector
from natural_pdf.elements.collections import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.ocr import OCRManager, OCROptions
from natural_pdf.analyzers.layout.layout_manager import LayoutManager # Import the new LayoutManager
from natural_pdf.core.highlighting_service import HighlightingService # <-- Import the new service

# Set up module logger
logger = logging.getLogger("natural_pdf.core.pdf")


class PDF:
    """
    Enhanced PDF wrapper built on top of pdfplumber.
    
    This class provides a fluent interface for working with PDF documents,
    with improved selection, navigation, and extraction capabilities.
    """
    
    def __init__(self, path_or_url: str, reading_order: bool = True, 
                 font_attrs: Optional[List[str]] = None,
                 keep_spaces: bool = True):
        """
        Initialize the enhanced PDF object.
        
        Args:
            path_or_url: Path to the PDF file or a URL to a PDF
            reading_order: Whether to use natural reading order
            font_attrs: Font attributes to consider when grouping characters into words.
                       Default: ['fontname', 'size'] (Group by font name and size)
                       None: Only consider spatial relationships
                       List: Custom attributes to consider (e.g., ['fontname', 'size', 'color'])
            keep_spaces: Whether to include spaces in word elements (default: True).
                       True: Spaces are part of words, better for multi-word searching
                       False: Break text at spaces, each word is separate (legacy behavior)
        """
        # Check if the input is a URL
        is_url = path_or_url.startswith('http://') or path_or_url.startswith('https://')
        
        # Initialize path-related attributes
        self._original_path = path_or_url
        self._temp_file = None
        
        if is_url:
            logger.info(f"Downloading PDF from URL: {path_or_url}")
            try:
                # Create a temporary file to store the downloaded PDF
                self._temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                
                # Download the PDF
                with urllib.request.urlopen(path_or_url) as response:
                    self._temp_file.write(response.read())
                    self._temp_file.flush()
                    self._temp_file.close()
                
                # Use the temporary file path
                path = self._temp_file.name
                logger.info(f"PDF downloaded to temporary file: {path}")
            except Exception as e:
                if self._temp_file and hasattr(self._temp_file, 'name'):
                    try:
                        os.unlink(self._temp_file.name)
                    except:
                        pass
                logger.error(f"Failed to download PDF from URL: {e}")
                raise ValueError(f"Failed to download PDF from URL: {e}")
        else:
            # Use the provided path directly
            path = path_or_url
            
        logger.info(f"Initializing PDF from {path}")
        logger.debug(f"Parameters: reading_order={reading_order}, font_attrs={font_attrs}, keep_spaces={keep_spaces}")
        
        self._pdf = pdfplumber.open(path)
        self._path = path
        self._reading_order = reading_order
        self._config = {
            'keep_spaces': keep_spaces
        }
                
        self._font_attrs = font_attrs  # Store the font attribute configuration

        if OCRManager:
            self._ocr_manager = OCRManager()
            logger.info(f"Initialized OCRManager. Available engines: {self._ocr_manager.get_available_engines()}")
        else:
            self._ocr_manager = None
            logger.warning("OCRManager could not be imported. OCR functionality disabled.")

        if LayoutManager:
            self._layout_manager = LayoutManager()
            logger.info(f"Initialized LayoutManager. Available engines: {self._layout_manager.get_available_engines()}")
        else:
            self._layout_manager = None
            logger.warning("LayoutManager could not be imported. Layout analysis disabled.")

        self._pages = [Page(p, parent=self, index=i, font_attrs=font_attrs) for i, p in enumerate(self._pdf.pages)]
        self._element_cache = {}
        self._exclusions = []  # List to store exclusion functions/regions
        self._regions = []  # List to store region functions/definitions
        
        # Initialize the Highlighting Service
        self.highlighter = HighlightingService(self) 
        logger.info("Initialized HighlightingService.")

    @property
    def pages(self) -> 'PageCollection':
        """Access pages as a PageCollection object."""
        from natural_pdf.elements.collections import PageCollection
        return PageCollection(self._pages)
                
    def add_exclusion(self, exclusion_func: Callable[[Page], Region], label: str = None) -> 'PDF':
        """
        Add an exclusion function to the PDF. Text from these regions will be excluded from extraction.
        
        Args:
            exclusion_func: A function that takes a Page and returns a Region to exclude
            label: Optional label for this exclusion
            
        Returns:
            Self for method chaining
        """
        # Store exclusion with its label at PDF level
        exclusion_data = (exclusion_func, label)
        self._exclusions.append(exclusion_data)
        
        # Create a wrapper function that properly evaluates on each page
        def exclusion_wrapper(page):
            try:
                region = exclusion_func(page)
                return region
            except Exception as e:
                print(f"Error in PDF-level exclusion for page {page.index}: {e}")
                return None
        
        # Apply this exclusion to all pages using the wrapper
        for page in self._pages:
            page.add_exclusion(exclusion_wrapper)
            
        return self

    def apply_ocr_to_pages(
        self,
        pages: Optional[Union[Iterable[int], range, slice]] = None,
        engine: Optional[str] = None,
        options: Optional['OCROptions'] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        # Add other simple mode args if needed
    ) -> 'PDF':
        """
        Applies OCR to specified pages (or all pages) of the PDF using batch processing.

        This method renders the specified pages to images, sends them as a batch
        to the OCRManager, and adds the resulting TextElements to each respective page.

        Args:
            pages: An iterable of 0-based page indices (list, range, tuple),
                   a slice object, or None to process all pages.
            engine: Name of the engine (e.g., 'easyocr', 'paddleocr', 'surya').
                    Uses manager's default if None. Ignored if 'options' is provided.
            options: An specific Options object (e.g., EasyOCROptions) for
                     advanced configuration. Overrides simple arguments.
            languages: List of language codes for simple mode.
            min_confidence: Minimum confidence threshold for simple mode.
            device: Device string ('cpu', 'cuda', etc.) for simple mode.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If page indices are invalid or the engine name is invalid.
            TypeError: If unexpected keyword arguments are provided in simple mode.
            RuntimeError: If the OCRManager or selected engine is not available.
        """
        if not self._ocr_manager:
             logger.error("OCRManager not available. Cannot apply OCR.")
             # Or raise RuntimeError("OCRManager not initialized.")
             return self

        # --- Determine Target Pages ---
        target_pages: List[Page] = []
        if pages is None:
            target_pages = self._pages
        elif isinstance(pages, slice):
            target_pages = self._pages[pages]
        elif hasattr(pages, '__iter__'): # Check if it's iterable (list, range, tuple, etc.)
            try:
                target_pages = [self._pages[i] for i in pages]
            except IndexError:
                raise ValueError("Invalid page index provided in 'pages' iterable.")
            except TypeError:
                 raise TypeError("'pages' must be None, a slice, or an iterable of page indices (int).")
        else:
             raise TypeError("'pages' must be None, a slice, or an iterable of page indices (int).")

        if not target_pages:
            logger.warning("No pages selected for OCR processing.")
            return self

        page_numbers = [p.number for p in target_pages]
        logger.info(f"Applying batch OCR to pages: {page_numbers}...")

        # --- Render Images for Batch ---
        images_pil: List[Image.Image] = []
        page_image_map: List[Tuple[Page, Image.Image]] = [] # Store page and its image
        logger.info(f"Rendering {len(target_pages)} pages to images...")
        try:
            ocr_scale = getattr(self, '_config', {}).get('ocr_image_scale', 2.0)
            for i, page in enumerate(target_pages):
                logger.debug(f"  Rendering page {page.number} (index {page.index})...")
                # Use page.to_image but ensure highlights are off for OCR base image
                img = page.to_image(scale=ocr_scale, include_highlights=False)
                images_pil.append(img)
                page_image_map.append((page, img)) # Store pair
        except Exception as e:
            logger.error(f"Failed to render one or more pages for batch OCR: {e}", exc_info=True)
            # Decide whether to continue with successfully rendered pages or fail completely
            # For now, let's fail if any page rendering fails.
            raise RuntimeError(f"Failed to render page {page.number} for OCR.") from e

        if not images_pil:
             logger.error("No images were successfully rendered for batch OCR.")
             return self

        # --- Prepare Arguments for Manager ---
        manager_args = {'images': images_pil, 'options': options, 'engine': engine}
        if languages is not None: manager_args['languages'] = languages
        if min_confidence is not None: manager_args['min_confidence'] = min_confidence
        if device is not None: manager_args['device'] = device

        # --- Call OCR Manager for Batch Processing ---
        logger.info(f"Calling OCR Manager for batch processing {len(images_pil)} images...")
        try:
            # The manager's apply_ocr handles the batch input and returns List[List[Dict]]
            batch_results = self._ocr_manager.apply_ocr(**manager_args)

            if not isinstance(batch_results, list) or len(batch_results) != len(images_pil):
                logger.error(f"OCR Manager returned unexpected result format or length for batch processing. "
                             f"Expected list of length {len(images_pil)}, got {type(batch_results)} "
                             f"with length {len(batch_results) if isinstance(batch_results, list) else 'N/A'}.")
                # Handle error - maybe return early or try processing valid parts?
                return self # Return self without adding elements

            logger.info("OCR Manager batch processing complete.")

        except Exception as e:
             logger.error(f"Batch OCR processing failed: {e}", exc_info=True)
             return self # Return self without adding elements

        # --- Distribute Results and Add Elements to Pages ---
        logger.info("Adding OCR results to respective pages...")
        total_elements_added = 0
        for i, (page, img) in enumerate(page_image_map):
            results_for_page = batch_results[i]
            if not isinstance(results_for_page, list):
                 logger.warning(f"Skipping results for page {page.number}: Expected list, got {type(results_for_page)}")
                 continue

            logger.debug(f"  Processing {len(results_for_page)} results for page {page.number}...")
            # Use the page's element manager to create elements from its results
            # Changed from page._create_text_elements_from_ocr to use element_mgr
            elements = page._element_mgr.create_text_elements_from_ocr(results_for_page, img.width, img.height)

            if elements:
                 # Note: element_mgr.create_text_elements_from_ocr already adds them
                 total_elements_added += len(elements)
                 logger.debug(f"  Added {len(elements)} OCR TextElements to page {page.number}.")
            else:
                 logger.debug(f"  No valid TextElements created for page {page.number}.")

        logger.info(f"Finished adding OCR results. Total elements added across {len(target_pages)} pages: {total_elements_added}")
        return self
  
    def add_region(self, region_func: Callable[[Page], Region], name: str = None) -> 'PDF':
        """
        Add a region function to the PDF. This creates regions on all pages using the provided function.
        
        Args:
            region_func: A function that takes a Page and returns a Region
            name: Optional name for the region
            
        Returns:
            Self for method chaining
        """
        # Store region with its name at PDF level
        region_data = (region_func, name)
        self._regions.append(region_data)
        
        # Create a wrapper function that properly evaluates on each page
        def region_wrapper(page):
            try:
                region = region_func(page)
                if region:
                    # Apply name if provided
                    if name:
                        region.name = name
                    region.source = 'named'
                return region
            except Exception as e:
                print(f"Error in PDF-level region for page {page.index}: {e}")
                return None
        
        # Apply this region to all pages
        for page in self._pages:
            try:
                region = region_wrapper(page)
                if region:
                    page.add_region(region, name=name)
            except Exception as e:
                print(f"Error adding region to page {page.index}: {e}")
            
        return self
        
    def find(self, selector: str, apply_exclusions=True, regex=False, case=True, **kwargs) -> Any:
        """
        Find the first element matching the selector.
        
        Args:
            selector: CSS-like selector string (e.g., 'text:contains("Annual Report")')
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True)
            regex: Whether to use regex for text search in :contains (default: False)
            case: Whether to do case-sensitive text search (default: True)
            **kwargs: Additional filter parameters
            
        Returns:
            Element object or None if not found
        """
        selector_obj = parse_selector(selector)
        
        # Pass regex and case flags to selector function
        kwargs['regex'] = regex
        kwargs['case'] = case
        
        results = self._apply_selector(selector_obj, apply_exclusions=apply_exclusions, **kwargs)
        return results.first if results else None
    
    def find_all(self, selector: str, apply_exclusions=True, regex=False, case=True, **kwargs) -> ElementCollection:
        """
        Find all elements matching the selector.
        
        Args:
            selector: CSS-like selector string (e.g., 'text[color=(1,0,0)]')
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True)
            regex: Whether to use regex for text search in :contains (default: False)
            case: Whether to do case-sensitive text search (default: True)
            **kwargs: Additional filter parameters
            
        Returns:
            ElementCollection with matching elements
        """
        selector_obj = parse_selector(selector)
        
        # Pass regex and case flags to selector function
        kwargs['regex'] = regex
        kwargs['case'] = case
        
        results = self._apply_selector(selector_obj, apply_exclusions=apply_exclusions, **kwargs)
        return results
    
    def _apply_selector(self, selector_obj: Dict, apply_exclusions=True, **kwargs) -> ElementCollection:
        """
        Apply selector to PDF elements across all pages.
        
        Args:
            selector_obj: Parsed selector dictionary
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True)
            **kwargs: Additional filter parameters
            
        Returns:
            ElementCollection of matching elements
        """
        from natural_pdf.elements.collections import ElementCollection
        
        # Determine page range to search
        page_range = kwargs.get('pages', range(len(self.pages)))
        if isinstance(page_range, (int, slice)):
            # Convert int or slice to range
            if isinstance(page_range, int):
                page_range = [page_range]
            elif isinstance(page_range, slice):
                start = page_range.start or 0
                stop = page_range.stop or len(self.pages)
                step = page_range.step or 1
                page_range = range(start, stop, step)
        
        # Check for cross-page pseudo-classes
        cross_page = False
        for pseudo in selector_obj.get('pseudo_classes', []):
            if pseudo.get('name') in ('spans', 'continues'):
                cross_page = True
                break
        
        # If searching across pages, handle specially
        if cross_page:
            # TODO: Implement cross-page element matching
            return ElementCollection([])
        
        # Regular case: collect elements from each page
        all_elements = []
        for page_idx in page_range:
            if 0 <= page_idx < len(self.pages):
                page = self.pages[page_idx]
                page_elements = page._apply_selector(selector_obj, apply_exclusions=apply_exclusions, **kwargs)
                all_elements.extend(page_elements.elements)
        
        # Create a combined collection
        combined = ElementCollection(all_elements)
        
        # Sort in document order if requested
        if kwargs.get('document_order', True):
            # Check if elements have page, top, x0 before sorting
            if all(hasattr(el, 'page') and hasattr(el, 'top') and hasattr(el, 'x0') for el in combined.elements):
                 combined.sort(key=lambda el: (el.page.index, el.top, el.x0))
            else:
                 logger.warning("Cannot sort elements in document order: Missing required attributes (page, top, x0).")
            
        return combined
    
    def extract_text(self, selector: Optional[str] = None, preserve_whitespace=True, 
                  use_exclusions=True, debug_exclusions=False, **kwargs) -> str:
        """
        Extract text from the entire document or matching elements.
        
        Args:
            selector: Optional selector to filter elements
            preserve_whitespace: Whether to keep blank characters (default: True)
            use_exclusions: Whether to apply exclusion regions (default: True)
            debug_exclusions: Whether to output detailed debugging for exclusions (default: False)
            **kwargs: Additional extraction parameters
            
        Returns:
            Extracted text as string
        """
        # If selector is provided, find elements first
        if selector:
            elements = self.find_all(selector)
            return elements.extract_text(preserve_whitespace=preserve_whitespace, **kwargs)
        
        # Otherwise extract from all pages
        if debug_exclusions:
            print(f"PDF: Extracting text with exclusions from {len(self.pages)} pages")
            print(f"PDF: Found {len(self._exclusions)} document-level exclusions")
        
        texts = []
        for page in self.pages:
            texts.append(page.extract_text(
                preserve_whitespace=preserve_whitespace, 
                use_exclusions=use_exclusions,
                debug_exclusions=debug_exclusions,
                **kwargs
            ))
        
        if debug_exclusions:
            print(f"PDF: Combined {len(texts)} pages of text")
            
        return "\n".join(texts)
    
    # Note: extract_text_compat method removed
    
    def extract(self, selector: str, preserve_whitespace=True, **kwargs) -> str:
        """
        Shorthand for finding elements and extracting their text.
        
        Args:
            selector: CSS-like selector string
            preserve_whitespace: Whether to keep blank characters (default: True)
            **kwargs: Additional extraction parameters
            
        Returns:
            Extracted text from matching elements
        """
        return self.extract_text(selector, preserve_whitespace=preserve_whitespace, **kwargs)
        
    # def debug_ocr(self, output_path, pages=None):
    #     """
    #     Generate an interactive HTML debug report for OCR results.
        
    #     This creates a single-file HTML report with:
    #     - Side-by-side view of image regions and OCR text
    #     - Confidence scores with color coding
    #     - Editable correction fields
    #     - Filtering and sorting options
    #     - Export functionality for corrected text
        
    #     Args:
    #         output_path: Path to save the HTML report
    #         pages: Pages to include in the report (default: all pages)
    #               Can be a page index, slice, or list of page indices
            
    #     Returns:
    #         Self for method chaining
    #     """
    #     from natural_pdf.utils.ocr import debug_ocr_to_html
        
    #     if pages is None:
    #         # Include all pages
    #         target_pages = self.pages
    #     elif isinstance(pages, int):
    #         # Single page index
    #         target_pages = [self.pages[pages]]
    #     elif isinstance(pages, slice):
    #         # Slice of pages
    #         target_pages = self.pages[pages]
    #     else:
    #         # Assume it's an iterable of page indices
    #         target_pages = [self.pages[i] for i in pages]
            
    #     debug_ocr_to_html(target_pages, output_path)
    #     return self
    
    def extract_tables(self, selector: Optional[str] = None, merge_across_pages: bool = False, **kwargs) -> List[Any]:
        """
        Extract tables from the document or matching elements.
        
        Args:
            selector: Optional selector to filter tables
            merge_across_pages: Whether to merge tables that span across pages
            **kwargs: Additional extraction parameters
            
        Returns:
            List of extracted tables
        """
        # TODO: Implement table extraction
        return []  # Placeholder
    
    def ask(self, question: str, 
           mode: str = "extractive", 
           pages: Union[int, List[int], range] = None,
           min_confidence: float = 0.1,
           model: str = None,
           **kwargs) -> Dict[str, Any]:
        """
        Ask a question about the document content.
        
        Args:
            question: Question to ask about the document
            mode: "extractive" to extract answer from document, "generative" to generate
            pages: Specific pages to query (default: all pages)
            min_confidence: Minimum confidence threshold for answers
            model: Optional model name for question answering
            **kwargs: Additional parameters passed to the QA engine
            
        Returns:
            A dictionary containing the answer, confidence, and other metadata.
            Result will have an 'answer' key containing the answer text.
        """
        from natural_pdf.qa import get_qa_engine
        
        # Initialize or get QA engine
        qa_engine = get_qa_engine() if model is None else get_qa_engine(model_name=model)
        
        # Determine which pages to query
        if pages is None:
            target_pages = list(range(len(self.pages)))
        elif isinstance(pages, int):
            # Single page
            target_pages = [pages]
        elif isinstance(pages, (list, range)):
            # List or range of pages
            target_pages = pages
        else:
            raise ValueError(f"Invalid pages parameter: {pages}")
        
        # Actually query each page and gather results
        results = []
        for page_idx in target_pages:
            if 0 <= page_idx < len(self.pages):
                page = self.pages[page_idx]
                page_result = qa_engine.ask_pdf_page(
                    page=page,
                    question=question,
                    min_confidence=min_confidence,
                    **kwargs
                )
                
                # Add to results if it found an answer
                if page_result and page_result.get("found", False):
                    results.append(page_result)
        
        # Sort results by confidence
        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Return the best result, or a default result if none found
        if results:
            return results[0]
        else:
            # Return a structure indicating no answer found
            return {
                 "answer": None,
                 "confidence": 0.0,
                 "found": False,
                 "page_num": None, # Or maybe the pages searched?
                 "source_elements": []
            }
                
    def __len__(self) -> int:
        """Return the number of pages in the PDF."""
        return len(self.pages)
    
    def __getitem__(self, key) -> Union[Page, List[Page]]:
        """Access pages by index or slice."""
        # Check if self._pages has been initialized
        if not hasattr(self, '_pages'):
             raise AttributeError("PDF pages not initialized yet.")
        if isinstance(key, slice):
             # Return a PageCollection slice
             from natural_pdf.elements.collections import PageCollection
             return PageCollection(self._pages[key])
        # Return a single Page object
        return self._pages[key]
        
    def close(self):
        """Close the underlying PDF file and clean up any temporary files."""
        if hasattr(self, '_pdf') and self._pdf is not None:
            self._pdf.close()
            self._pdf = None
            
        # Clean up temporary file if it exists
        if hasattr(self, '_temp_file') and self._temp_file is not None:
            try:
                if hasattr(self._temp_file, 'name') and self._temp_file.name and os.path.exists(self._temp_file.name):
                    os.unlink(self._temp_file.name)
                    logger.debug(f"Removed temporary PDF file: {self._temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary PDF file: {e}")
            finally:
                self._temp_file = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()