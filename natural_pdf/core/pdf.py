import pdfplumber
import logging
import tempfile
import os
import re
import urllib.request
from typing import List, Optional, Union, Any, Dict, Callable, Tuple, Type, Iterable, TYPE_CHECKING # Added Iterable and TYPE_CHECKING
from pathlib import Path # Added Path
import copy # Add import for deepcopy
from PIL import Image

from natural_pdf.core.page import Page
from natural_pdf.selectors.parser import parse_selector
from natural_pdf.elements.collections import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.ocr import OCRManager, OCROptions
from natural_pdf.analyzers.layout.layout_manager import LayoutManager # Import the new LayoutManager
from natural_pdf.core.highlighting_service import HighlightingService # <-- Import the new service

# Import the flag directly - this should always work

# --- Add Search Service Imports (needed for new methods) ---
try:
    from natural_pdf.search import (
        get_search_service,
        SearchServiceProtocol,
        SearchOptions,
        TextSearchOptions, # Keep for ask default
        BaseSearchOptions
    )
    from typing import Any as TypingAny # Import Any if not already
except ImportError:
    # Define dummies if needed for type hints within the class
    SearchServiceProtocol = object
    SearchOptions, TextSearchOptions, BaseSearchOptions = object, object, object
    TypingAny = object
    # Dummy factory needed for default arg in methods
    def get_search_service(**kwargs) -> SearchServiceProtocol:
        raise ImportError("Search dependencies are not installed. Install with: pip install natural-pdf[search]")

# --- End Search Service Imports ---

# Set up logger early
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
        self._resolved_path = None # Store the actual path used by pdfplumber

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
                self._resolved_path = self._temp_file.name
                logger.info(f"PDF downloaded to temporary file: {self._resolved_path}")
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
            self._resolved_path = path_or_url

        logger.info(f"Initializing PDF from {self._resolved_path}")
        logger.debug(f"Parameters: reading_order={reading_order}, font_attrs={font_attrs}, keep_spaces={keep_spaces}")
        
        try:
            self._pdf = pdfplumber.open(self._resolved_path)
        except Exception as e:
             logger.error(f"Failed to open PDF with pdfplumber: {self._resolved_path}. Error: {e}", exc_info=True)
             # Clean up temp file if creation failed
             self.close()
             raise IOError(f"Failed to open PDF file/URL: {path_or_url}") from e

        self._path = self._resolved_path # Keep original path too?
        self.path = self._resolved_path # Public attribute for the resolved path
        self.source_path = self._original_path # Public attribute for the user-provided path/URL

        self._reading_order = reading_order
        self._config = {
            'keep_spaces': keep_spaces
        }

        self._font_attrs = font_attrs  # Store the font attribute configuration

        # Initialize Managers and Services (conditionally available)
        self._ocr_manager = OCRManager() if OCRManager else None
        self._layout_manager = LayoutManager() if LayoutManager else None
        self.highlighter = HighlightingService(self) 

        # Initialize pages last, passing necessary refs
        self._pages = [Page(p, parent=self, index=i, font_attrs=font_attrs) for i, p in enumerate(self._pdf.pages)]

        # Other state
        self._element_cache = {}
        self._exclusions = []  # List to store exclusion functions/regions
        self._regions = []  # List to store region functions/definitions
        
        logger.info("Initialized HighlightingService.")
        logger.info(f"PDF '{self.source_path}' initialized with {len(self._pages)} pages.")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Access metadata as a dictionary."""
        return self._pdf.metadata

    @property
    def pages(self) -> 'PageCollection':
        """Access pages as a PageCollection object."""
        from natural_pdf.elements.collections import PageCollection
        # Ensure _pages is initialized
        if not hasattr(self, '_pages'):
             raise AttributeError("PDF pages not yet initialized.")
        return PageCollection(self._pages)
                
    def clear_exclusions(self) -> 'PDF':
        """
        Clear all exclusion functions from the PDF.
        
        Returns:
            Self for method chaining
        """
        # Ensure _pages is initialized
        if not hasattr(self, '_pages'):
             raise AttributeError("PDF pages not yet initialized.")

        self._exclusions = []
        # Also clear from pages
        for page in self._pages:
            page.clear_exclusions()
        return self
    
    def add_exclusion(self, exclusion_func: Callable[['Page'], Optional[Region]], label: str = None) -> 'PDF':
        """
        Add an exclusion function to the PDF. Text from these regions will be excluded from extraction.
        
        Args:
            exclusion_func: A function that takes a Page and returns a Region to exclude, or None.
            label: Optional label for this exclusion
            
        Returns:
            Self for method chaining
        """
        # Ensure _pages is initialized
        if not hasattr(self, '_pages'):
             raise AttributeError("PDF pages not yet initialized.")

        # Store exclusion with its label at PDF level
        exclusion_data = (exclusion_func, label)
        self._exclusions.append(exclusion_data)

        # Apply this exclusion to all pages
        for page in self._pages:
            # We pass the original function, Page.add_exclusion handles calling it
            page.add_exclusion(exclusion_func, label=label)

        return self

    def apply_ocr(
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
        failed_page_num = 'unknown' # Keep track of potentially failing page
        try:
            ocr_scale = getattr(self, '_config', {}).get('ocr_image_scale', 2.0)
            for i, page in enumerate(target_pages):
                failed_page_num = page.number # Update current page number in case of error
                logger.debug(f"  Rendering page {page.number} (index {page.index})...")
                # Use page.to_image but ensure highlights are off for OCR base image
                img = page.to_image(scale=ocr_scale, include_highlights=False)
                images_pil.append(img)
                page_image_map.append((page, img)) # Store pair
        except Exception as e:
            logger.error(f"Failed to render one or more pages for batch OCR: {e}", exc_info=True)
            raise RuntimeError(f"Failed to render page {failed_page_num} for OCR.") from e

        if not images_pil:
             logger.error("No images were successfully rendered for batch OCR.")
             return self

        # --- Prepare Arguments for Manager ---
        manager_args = {'images': images_pil, 'options': options, 'engine': engine}
        simple_args = {}
        if languages is not None: simple_args['languages'] = languages
        if min_confidence is not None: simple_args['min_confidence'] = min_confidence
        if device is not None: simple_args['device'] = device
        manager_args.update(simple_args) # Add simple args if options not provided

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
            try:
                # Calculate scale factors based on rendered image vs page dims
                 img_scale_x = page.width / img.width if img.width > 0 else 1
                 img_scale_y = page.height / img.height if img.height > 0 else 1
                 elements = page._element_mgr.create_text_elements_from_ocr(results_for_page, img_scale_x, img_scale_y)

                 if elements:
                      # Note: element_mgr.create_text_elements_from_ocr already adds them
                      total_elements_added += len(elements)
                      logger.debug(f"  Added {len(elements)} OCR TextElements to page {page.number}.")
                 else:
                      logger.debug(f"  No valid TextElements created for page {page.number}.")
            except Exception as e:
                 logger.error(f"  Error adding OCR elements to page {page.number}: {e}", exc_info=True)
                 # Continue to next page

        logger.info(f"Finished adding OCR results. Total elements added across {len(target_pages)} pages: {total_elements_added}")
        return self
  
    def add_region(self, region_func: Callable[['Page'], Optional[Region]], name: str = None) -> 'PDF':
        """
        Add a region function to the PDF. This creates regions on all pages using the provided function.
        
        Args:
            region_func: A function that takes a Page and returns a Region, or None.
            name: Optional name for the region
            
        Returns:
            Self for method chaining
        """
        # Ensure _pages is initialized
        if not hasattr(self, '_pages'):
             raise AttributeError("PDF pages not yet initialized.")

        # Store region with its name at PDF level
        region_data = (region_func, name)
        self._regions.append(region_data)

        # Apply this region to all pages
        for page in self._pages:
            try:
                # Call the function to get the region for this specific page
                region_instance = region_func(page)
                if region_instance and isinstance(region_instance, Region):
                    # If a valid region is returned, add it to the page
                    page.add_region(region_instance, name=name, source='named')
                elif region_instance is not None:
                     logger.warning(f"Region function did not return a valid Region object for page {page.number}. Got: {type(region_instance)}")
            except Exception as e:
                logger.error(f"Error executing or adding region function for page {page.number}: {e}", exc_info=True)

        return self
        
    def find(self, selector: str, apply_exclusions=True, regex=False, case=True, **kwargs) -> Optional[Any]:
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
        # Ensure _pages is initialized
        if not hasattr(self, '_pages'):
             raise AttributeError("PDF pages not yet initialized.")

        selector_obj = parse_selector(selector)
        
        # Pass regex and case flags to selector function
        kwargs['regex'] = regex
        kwargs['case'] = case
        
        results = self._apply_selector(selector_obj, apply_exclusions=apply_exclusions, first_only=True, **kwargs)
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
        # Ensure _pages is initialized
        if not hasattr(self, '_pages'):
             raise AttributeError("PDF pages not yet initialized.")

        selector_obj = parse_selector(selector)
        
        # Pass regex and case flags to selector function
        kwargs['regex'] = regex
        kwargs['case'] = case
        
        results = self._apply_selector(selector_obj, apply_exclusions=apply_exclusions, first_only=False, **kwargs)
        return results
    
    def _apply_selector(self, selector_obj: Dict, apply_exclusions=True, first_only=False, **kwargs) -> ElementCollection:
        """
        Apply selector to PDF elements across all pages.
        
        Args:
            selector_obj: Parsed selector dictionary
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True)
            first_only: If True, stop searching after the first match is found.
            **kwargs: Additional filter parameters
            
        Returns:
            ElementCollection of matching elements
        """
        from natural_pdf.elements.collections import ElementCollection
        
        # Determine page range to search
        page_indices = kwargs.get('pages', range(len(self._pages)))
        if isinstance(page_indices, int):
            page_indices = [page_indices]
        elif isinstance(page_indices, slice):
            page_indices = range(*page_indices.indices(len(self._pages)))

        # Check for cross-page pseudo-classes (currently not supported)
        for pseudo in selector_obj.get('pseudo_classes', []):
            if pseudo.get('name') in ('spans', 'continues'):
                logger.warning("Cross-page selectors ('spans', 'continues') are not yet supported.")
                return ElementCollection([])

        # Regular case: collect elements from each page
        all_elements = []
        for page_idx in page_indices:
            if 0 <= page_idx < len(self._pages):
                page = self._pages[page_idx]
                # Pass first_only down to page._apply_selector
                page_elements_collection = page._apply_selector(
                    selector_obj,
                    apply_exclusions=apply_exclusions,
                    first_only=first_only,
                    **kwargs
                )
                if page_elements_collection:
                     page_elements = page_elements_collection.elements
                     all_elements.extend(page_elements)
                     # If we only need the first match overall, and we found one on this page, stop
                     if first_only and page_elements:
                          break # Stop iterating through pages
            else:
                 logger.warning(f"Page index {page_idx} out of range (0-{len(self._pages)-1}).")

        # Create a combined collection
        combined = ElementCollection(all_elements)

        # Sort in document order if requested and not first_only (already sorted by page)
        if not first_only and kwargs.get('document_order', True):
            # Check if elements have page, top, x0 before sorting
            if all(hasattr(el, 'page') and hasattr(el, 'top') and hasattr(el, 'x0') for el in combined.elements):
                 combined.sort(key=lambda el: (el.page.index, el.top, el.x0))
            else:
                 # Elements might be Regions without inherent sorting order yet
                 # Attempt sorting by page index if possible
                 try:
                     combined.sort(key=lambda el: el.page.index)
                 except AttributeError:
                     logger.warning("Cannot sort elements in document order: Missing required attributes (e.g., page).")

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
        # Ensure _pages is initialized
        if not hasattr(self, '_pages'):
             raise AttributeError("PDF pages not yet initialized.")

        # If selector is provided, find elements first
        if selector:
            elements = self.find_all(selector, apply_exclusions=use_exclusions, **kwargs)
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
        # Ensure _pages is initialized
        if not hasattr(self, '_pages'):
             raise AttributeError("PDF pages not yet initialized.")
        return self.extract_text(selector, preserve_whitespace=preserve_whitespace, use_exclusions=True, **kwargs) # apply_exclusions is handled by find_all in extract_text
        
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
        # Ensure _pages is initialized
        if not hasattr(self, '_pages'):
             raise AttributeError("PDF pages not yet initialized.")
        # TODO: Implement table extraction
        logger.warning("PDF.extract_tables is not fully implemented yet.")
        all_tables = []
        for page in self.pages:
             # Assuming page.extract_tables(**kwargs) exists or is added
             if hasattr(page, 'extract_tables'):
                  all_tables.extend(page.extract_tables(**kwargs))
             else:
                  logger.debug(f"Page {page.number} does not have extract_tables method.")
        # Placeholder filtering
        if selector:
            logger.warning("Filtering extracted tables by selector is not implemented.")
            # Would need to parse selector and filter the list `all_tables`
        # Placeholder merging
        if merge_across_pages:
             logger.warning("Merging tables across pages is not implemented.")
             # Would need logic to detect and merge related tables
        return all_tables
    
    # --- New Method: save_searchable ---
    def save_searchable(self, output_path: Union[str, 'Path'], dpi: int = 300, **kwargs):
        """
        Saves the PDF with an OCR text layer, making content searchable.

        Requires optional dependencies. Install with: pip install "natural-pdf[ocr-save]"

        Note: OCR must have been applied to the pages beforehand
              (e.g., using pdf.apply_ocr()).

        Args:
            output_path: Path to save the searchable PDF.
            dpi: Resolution for rendering and OCR overlay (default 300).
            **kwargs: Additional keyword arguments passed to the exporter.
        """
        # Import moved here, assuming it's always available now
        from natural_pdf.exporters.searchable_pdf import create_searchable_pdf

        # TODO: Need a reliable way for Page to signal if it has OCR elements.
        # This requires adding a method/attribute to the Page class, e.g., page.has_ocr_elements()
        # or checking if page.get_elements(source='ocr') returns anything.
        # For now, we pass through and let the exporter handle pages without OCR elements.
        # if not any(page.get_elements(source='ocr') for page in self.pages):
        #      logger.warning("No OCR elements found on pages. "
        #                   "Ensure apply_ocr() was called. "
        #                   "Output PDF might not be searchable.")

        # Convert pathlib.Path to string if necessary
        output_path_str = str(output_path)

        create_searchable_pdf(self, output_path_str, dpi=dpi, **kwargs)
        logger.info(f"Searchable PDF saved to: {output_path_str}")

    # --- End New Method ---

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

    def search_within_index(
        self,
        query: Union[str, Path, Image.Image, Region],
        search_service: SearchServiceProtocol, # Now required
        options: Optional[SearchOptions] = None,
    ) -> List[Dict[str, Any]]:
        """
        Finds relevant documents specifically originating from THIS PDF document
        within a search index managed by the provided SearchService.

        This method uses a pre-configured SearchService instance and adds
        a filter to the search query to scope results only to pages from
        this specific PDF object (based on its resolved path).

        Args:
            query: The search query (text, image path, PIL Image, Region).
            search_service: A pre-configured SearchService instance pointing to the
                            index where this PDF's content (or related content)
                            is expected to be found.
            options: Optional SearchOptions to configure the query (top_k, filters, etc.).
                     Any existing filters in `options` will be combined with the
                     PDF-scoping filter using an 'AND' condition.

        Returns:
            A list of result dictionaries, sorted by relevance, containing only
            results originating from this PDF's pages.

        Raises:
            ImportError: If search dependencies are not installed.
            ValueError: If search_service is None.
            TypeError: If search_service does not conform to the protocol.
            FileNotFoundError: If the collection managed by the service does not exist.
            RuntimeError: For other search failures.
        """
        if not search_service:
             raise ValueError("A configured SearchServiceProtocol instance must be provided.")
        # Optional stricter check:
        # if not isinstance(search_service, SearchServiceProtocol):
        #     raise TypeError("Provided search_service does not conform to SearchServiceProtocol.")

        # Get collection name from service for logging
        collection_name = getattr(search_service, 'collection_name', '<Unknown Collection>')
        logger.info(f"Searching within index '{collection_name}' (via provided service) for content from PDF '{self.path}'. Query type: {type(query).__name__}.")

        # --- 1. Get Search Service Instance --- (REMOVED - provided directly)
        # service: SearchServiceProtocol
        # if search_service:
        #     service = search_service
        # else:
        #     logger.debug(f"Getting SearchService instance via factory (persist={persist}, collection={collection_name})...")
        #     factory_args = {**kwargs, 'collection_name': collection_name, 'persist': persist}
        #     # TODO: Pass embedding model from options/pdf config if needed?
        #     service = get_search_service(**factory_args)
        service = search_service # Use validated provided service

        # --- 2. Prepare Query and Options ---
        query_input = query
        # Resolve options (use default TextSearch if none provided)
        effective_options = copy.deepcopy(options) if options is not None else TextSearchOptions()

        # Handle Region query - extract text for now
        if isinstance(query, Region):
             logger.debug("Query is a Region object. Extracting text.")
             if not isinstance(effective_options, TextSearchOptions):
                 logger.warning("Querying with Region image requires MultiModalSearchOptions (Not fully implemented). Falling back to text extraction.")
             query_input = query.extract_text()
             if not query_input or query_input.isspace():
                   logger.error("Region has no extractable text for query.")
                   return []

        # --- 3. Add Filter to Scope Search to THIS PDF ---
        # Assume metadata field 'pdf_path' stores the resolved path used during indexing
        pdf_scope_filter = {
             "field": "pdf_path", # Or potentially "source_path" depending on indexing metadata
             "operator": "eq",
             "value": self.path # Use the resolved path of this PDF instance
        }
        logger.debug(f"Applying filter to scope search to PDF: {pdf_scope_filter}")

        # Combine with existing filters in options (if any)
        if effective_options.filters:
            logger.debug(f"Combining PDF scope filter with existing filters: {effective_options.filters}")
            # Assume filters are compatible with the underlying search service
            # If existing filters aren't already in an AND block, wrap them
            if isinstance(effective_options.filters, dict) and effective_options.filters.get("operator") == "AND":
                 # Already an AND block, just append the condition
                 effective_options.filters["conditions"].append(pdf_scope_filter)
            elif isinstance(effective_options.filters, list):
                 # Assume list represents implicit AND conditions
                 effective_options.filters = {"operator": "AND", "conditions": effective_options.filters + [pdf_scope_filter]}
            elif isinstance(effective_options.filters, dict): # Single filter dict
                 effective_options.filters = {"operator": "AND", "conditions": [effective_options.filters, pdf_scope_filter]}
            else:
                 logger.warning(f"Unsupported format for existing filters: {type(effective_options.filters)}. Overwriting with PDF scope filter.")
                 effective_options.filters = pdf_scope_filter
        else:
            effective_options.filters = pdf_scope_filter

        logger.debug(f"Final filters for service search: {effective_options.filters}")

        # --- 4. Call SearchService --- 
        try:
            # Call the service's search method (no collection_name needed)
            results = service.search(
                query=query_input,
                options=effective_options,
            )
            logger.info(f"SearchService returned {len(results)} results scoped to PDF '{self.path}' within collection '{collection_name}'.")
            return results
        except FileNotFoundError as fnf:
             logger.error(f"Search failed: Collection '{collection_name}' not found by service. Error: {fnf}")
             raise # Re-raise specific error
        except Exception as e:
             logger.error(f"SearchService search failed for PDF '{self.path}' in collection '{collection_name}': {e}", exc_info=True)
             raise RuntimeError(f"Search within index failed for PDF '{self.path}'. See logs for details.") from e

    def __len__(self) -> int:
        """Return the number of pages in the PDF."""
        # Ensure _pages is initialized
        if not hasattr(self, '_pages'):
            # Return 0 or raise error if not fully initialized? Let's return 0.
             return 0
        return len(self._pages)
    
    def __getitem__(self, key) -> Union[Page, 'PageCollection']: # Return PageCollection for slice
        """Access pages by index or slice."""
        # Check if self._pages has been initialized
        if not hasattr(self, '_pages'):
             raise AttributeError("PDF pages not initialized yet.")
        if isinstance(key, slice):
             # Return a PageCollection slice
             from natural_pdf.elements.collections import PageCollection
             return PageCollection(self._pages[key])
        # Check index bounds before accessing
        if isinstance(key, int):
            if 0 <= key < len(self._pages):
                 return self._pages[key]
            else:
                 raise IndexError(f"Page index {key} out of range (0-{len(self._pages)-1}).")
        else:
             raise TypeError(f"Page indices must be integers or slices, not {type(key)}.")
        
    def close(self):
        """Close the underlying PDF file and clean up any temporary files."""
        if hasattr(self, '_pdf') and self._pdf is not None:
             try:
                  self._pdf.close()
                  logger.debug(f"Closed underlying pdfplumber PDF object for {self.source_path}")
             except Exception as e:
                  logger.warning(f"Error closing pdfplumber object: {e}")
             finally:
                  self._pdf = None

        # Clean up temporary file if it exists
        if hasattr(self, '_temp_file') and self._temp_file is not None:
            temp_file_path = None
            try:
                 if hasattr(self._temp_file, 'name') and self._temp_file.name:
                      temp_file_path = self._temp_file.name
                      if os.path.exists(temp_file_path):
                           os.unlink(temp_file_path)
                           logger.debug(f"Removed temporary PDF file: {temp_file_path}")
            except Exception as e:
                 logger.warning(f"Failed to clean up temporary PDF file '{temp_file_path}': {e}")
            finally:
                 self._temp_file = None

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# --- Added TYPE_CHECKING import (if not already present) ---
if TYPE_CHECKING:
    from pathlib import Path # Assuming Path is used for type hint