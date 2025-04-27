import copy
import logging
import os
import re
import tempfile
import urllib.request
import time
import threading
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from natural_pdf.utils.tqdm_utils import get_tqdm

import pdfplumber
from PIL import Image

from natural_pdf.analyzers.layout.layout_manager import LayoutManager
from natural_pdf.core.highlighting_service import HighlightingService
from natural_pdf.core.page import Page
from natural_pdf.elements.collections import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.ocr import OCRManager, OCROptions
from natural_pdf.selectors.parser import parse_selector

from natural_pdf.classification.manager import ClassificationManager
from natural_pdf.classification.manager import ClassificationError
from natural_pdf.classification.results import ClassificationResult
from natural_pdf.extraction.manager import StructuredDataManager

from natural_pdf.utils.locks import pdf_render_lock
from natural_pdf.elements.base import Element
from natural_pdf.classification.mixin import ClassificationMixin
from natural_pdf.extraction.mixin import ExtractionMixin

try:
    from typing import Any as TypingAny

    from natural_pdf.search import TextSearchOptions
    from natural_pdf.search import (
        BaseSearchOptions,
        SearchOptions,
        SearchServiceProtocol,
        get_search_service,
    )
except ImportError:
    SearchServiceProtocol = object
    SearchOptions, TextSearchOptions, BaseSearchOptions = object, object, object
    TypingAny = object

    def get_search_service(**kwargs) -> SearchServiceProtocol:
        raise ImportError(
            "Search dependencies are not installed. Install with: pip install natural-pdf[search]"
        )

logger = logging.getLogger("natural_pdf.core.pdf")
tqdm = get_tqdm()

DEFAULT_MANAGERS = {
    "classification": ClassificationManager,
    "structured_data": StructuredDataManager,
}

class PDF(ExtractionMixin):
    """
    Enhanced PDF wrapper built on top of pdfplumber.

    This class provides a fluent interface for working with PDF documents,
    with improved selection, navigation, and extraction capabilities.
    """

    def __init__(
        self,
        path_or_url: str,
        reading_order: bool = True,
        font_attrs: Optional[List[str]] = None,
        keep_spaces: bool = True,
    ):
        """
        Initialize the enhanced PDF object.

        Args:
            path_or_url: Path to the PDF file or a URL to a PDF
            reading_order: Whether to use natural reading order
            font_attrs: Font attributes for grouping characters into words
            keep_spaces: Whether to include spaces in word elements
        """
        is_url = path_or_url.startswith("http://") or path_or_url.startswith("https://")

        self._original_path = path_or_url
        self._temp_file = None
        self._resolved_path = None

        if is_url:
            logger.info(f"Downloading PDF from URL: {path_or_url}")
            try:
                self._temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                with urllib.request.urlopen(path_or_url) as response:
                    self._temp_file.write(response.read())
                    self._temp_file.flush()
                    self._temp_file.close()
                self._resolved_path = self._temp_file.name
                logger.info(f"PDF downloaded to temporary file: {self._resolved_path}")
            except Exception as e:
                if self._temp_file and hasattr(self._temp_file, "name"):
                    try:
                        os.unlink(self._temp_file.name)
                    except:
                        pass
                logger.error(f"Failed to download PDF from URL: {e}")
                raise ValueError(f"Failed to download PDF from URL: {e}")
        else:
            self._resolved_path = path_or_url

        logger.info(f"Initializing PDF from {self._resolved_path}")
        logger.debug(
            f"Parameters: reading_order={reading_order}, font_attrs={font_attrs}, keep_spaces={keep_spaces}"
        )

        try:
            self._pdf = pdfplumber.open(self._resolved_path)
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}", exc_info=True)
            self.close()
            raise IOError(f"Failed to open PDF file/URL: {path_or_url}") from e

        self._path = self._resolved_path
        self.path = self._resolved_path
        self.source_path = self._original_path

        self._reading_order = reading_order
        self._config = {"keep_spaces": keep_spaces}
        self._font_attrs = font_attrs

        self._ocr_manager = OCRManager() if OCRManager else None
        self._layout_manager = LayoutManager() if LayoutManager else None
        self.highlighter = HighlightingService(self)
        self._classification_manager_instance = ClassificationManager()
        self._manager_registry = {}

        self._pages = [
            Page(p, parent=self, index=i, font_attrs=font_attrs)
            for i, p in enumerate(self._pdf.pages)
        ]

        self._element_cache = {}
        self._exclusions = []
        self._regions = []

        logger.info(f"PDF '{self.source_path}' initialized with {len(self._pages)} pages.")

        self._initialize_managers()
        self._initialize_highlighter()

    def _initialize_managers(self):
        """Initialize manager instances based on DEFAULT_MANAGERS."""
        self._managers = {}
        for key, manager_class in DEFAULT_MANAGERS.items():
            try:
                self._managers[key] = manager_class()
                logger.debug(f"Initialized manager for key '{key}': {manager_class.__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize manager {manager_class.__name__}: {e}")
                self._managers[key] = None

    def get_manager(self, key: str) -> Any:
        """Retrieve a manager instance by its key."""
        if key not in self._managers:
            raise KeyError(f"No manager registered for key '{key}'. Available: {list(self._managers.keys())}")
        
        manager_instance = self._managers.get(key)
        
        if manager_instance is None:
             manager_class = DEFAULT_MANAGERS.get(key)
             if manager_class:
                  raise RuntimeError(f"Manager '{key}' ({manager_class.__name__}) failed to initialize previously.")
             else:
                  raise RuntimeError(f"Manager '{key}' failed to initialize (class not found).")

        return manager_instance

    def _initialize_highlighter(self):
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        """Access metadata as a dictionary."""
        return self._pdf.metadata

    @property
    def pages(self) -> "PageCollection":
        """Access pages as a PageCollection object."""
        from natural_pdf.elements.collections import PageCollection

        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")
        return PageCollection(self._pages)

    def clear_exclusions(self) -> "PDF":
        """
        Clear all exclusion functions from the PDF.

        Returns:
            Self for method chaining
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        self._exclusions = []
        for page in self._pages:
            page.clear_exclusions()
        return self

    def add_exclusion(
        self, exclusion_func: Callable[["Page"], Optional[Region]], label: str = None
    ) -> "PDF":
        """
        Add an exclusion function to the PDF. Text from these regions will be excluded from extraction.

        Args:
            exclusion_func: A function that takes a Page and returns a Region to exclude, or None
            label: Optional label for this exclusion

        Returns:
            Self for method chaining
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        exclusion_data = (exclusion_func, label)
        self._exclusions.append(exclusion_data)

        for page in self._pages:
            page.add_exclusion(exclusion_func, label=label)

        return self

    def apply_ocr(
        self,
        engine: Optional[str] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        apply_exclusions: bool = True,
        detect_only: bool = False,
        replace: bool = True,
        options: Optional[Any] = None,
        pages: Optional[Union[Iterable[int], range, slice]] = None,
    ) -> "PDF":
        """
        Applies OCR to specified pages of the PDF using batch processing.

        Args:
            engine: Name of the OCR engine
            languages: List of language codes
            min_confidence: Minimum confidence threshold 
            device: Device to run OCR on
            resolution: DPI resolution for page images
            apply_exclusions: Whether to mask excluded areas
            detect_only: If True, only detect text boxes
            replace: Whether to replace existing OCR elements
            options: Engine-specific options
            pages: Page indices to process or None for all pages

        Returns:
            Self for method chaining
        """
        if not self._ocr_manager:
            logger.error("OCRManager not available. Cannot apply OCR.")
            return self

        thread_id = threading.current_thread().name
        logger.debug(f"[{thread_id}] PDF.apply_ocr starting for {self.path}")
        
        target_pages = []
        if pages is None:
            target_pages = self._pages
        elif isinstance(pages, slice):
            target_pages = self._pages[pages]
        elif hasattr(pages, "__iter__"):
            try:
                target_pages = [self._pages[i] for i in pages]
            except IndexError:
                raise ValueError("Invalid page index provided in 'pages' iterable.")
            except TypeError:
                raise TypeError("'pages' must be None, a slice, or an iterable of page indices.")
        else:
            raise TypeError("'pages' must be None, a slice, or an iterable of page indices.")

        if not target_pages:
            logger.warning("No pages selected for OCR processing.")
            return self

        page_numbers = [p.number for p in target_pages]
        logger.info(f"Applying batch OCR to pages: {page_numbers}...")
        
        final_resolution = resolution or getattr(self, "_config", {}).get("resolution", 150)
        logger.debug(f"Using OCR image resolution: {final_resolution} DPI")

        images_pil = []
        page_image_map = []
        logger.info(f"[{thread_id}] Rendering {len(target_pages)} pages...")
        failed_page_num = "unknown"
        render_start_time = time.monotonic()
        
        try:
            for i, page in enumerate(tqdm(target_pages, desc="Rendering pages", leave=False)):
                failed_page_num = page.number
                logger.debug(f"  Rendering page {page.number} (index {page.index})...")
                to_image_kwargs = {
                    "resolution": final_resolution,
                    "include_highlights": False,
                    "exclusions": "mask" if apply_exclusions else None,
                }
                img = page.to_image(**to_image_kwargs)
                if img is None:
                    logger.error(f"  Failed to render page {page.number} to image.")
                    continue
                images_pil.append(img)
                page_image_map.append((page, img))
        except Exception as e:
            logger.error(f"Failed to render pages for batch OCR: {e}")
            raise RuntimeError(f"Failed to render page {failed_page_num} for OCR.") from e
            
        render_end_time = time.monotonic()
        logger.debug(f"[{thread_id}] Finished rendering {len(images_pil)} images (Duration: {render_end_time - render_start_time:.2f}s)")

        if not images_pil or not page_image_map:
            logger.error("No images were successfully rendered for batch OCR.")
            return self

        manager_args = {
            "images": images_pil,
            "engine": engine,
            "languages": languages,
            "min_confidence": min_confidence,
            "device": device,
            "options": options,
            "detect_only": detect_only,
        }
        manager_args = {k: v for k, v in manager_args.items() if v is not None}

        ocr_call_args = {k:v for k,v in manager_args.items() if k!='images'}
        logger.info(f"[{thread_id}] Calling OCR Manager with args: {ocr_call_args}...")
        ocr_start_time = time.monotonic()
        
        try:
            batch_results = self._ocr_manager.apply_ocr(**manager_args)

            if not isinstance(batch_results, list) or len(batch_results) != len(images_pil):
                logger.error(f"OCR Manager returned unexpected result format or length.")
                return self

            logger.info("OCR Manager batch processing complete.")
        except Exception as e:
            logger.error(f"Batch OCR processing failed: {e}")
            return self
            
        ocr_end_time = time.monotonic()
        logger.debug(f"[{thread_id}] OCR processing finished (Duration: {ocr_end_time - ocr_start_time:.2f}s)")

        logger.info("Adding OCR results to respective pages...")
        total_elements_added = 0
        
        for i, (page, img) in enumerate(page_image_map):
            results_for_page = batch_results[i]
            if not isinstance(results_for_page, list):
                logger.warning(f"Skipping results for page {page.number}: Expected list, got {type(results_for_page)}")
                continue

            logger.debug(f"  Processing {len(results_for_page)} results for page {page.number}...")
            try:
                if manager_args.get("replace", True) and hasattr(page, "_element_mgr"):
                    page._element_mgr.remove_ocr_elements()
                
                img_scale_x = page.width / img.width if img.width > 0 else 1
                img_scale_y = page.height / img.height if img.height > 0 else 1
                elements = page._element_mgr.create_text_elements_from_ocr(
                    results_for_page, img_scale_x, img_scale_y
                )

                if elements:
                    total_elements_added += len(elements)
                    logger.debug(f"  Added {len(elements)} OCR TextElements to page {page.number}.")
                else:
                    logger.debug(f"  No valid TextElements created for page {page.number}.")
            except Exception as e:
                logger.error(f"  Error adding OCR elements to page {page.number}: {e}")

        logger.info(f"Finished adding OCR results. Total elements added: {total_elements_added}")
        return self

    def add_region(
        self, region_func: Callable[["Page"], Optional[Region]], name: str = None
    ) -> "PDF":
        """
        Add a region function to the PDF.

        Args:
            region_func: A function that takes a Page and returns a Region, or None
            name: Optional name for the region

        Returns:
            Self for method chaining
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        region_data = (region_func, name)
        self._regions.append(region_data)

        for page in self._pages:
            try:
                region_instance = region_func(page)
                if region_instance and isinstance(region_instance, Region):
                    page.add_region(region_instance, name=name, source="named")
                elif region_instance is not None:
                    logger.warning(f"Region function did not return a valid Region for page {page.number}")
            except Exception as e:
                logger.error(f"Error adding region for page {page.number}: {e}")

        return self

    def find(
        self, selector: str, apply_exclusions=True, regex=False, case=True, **kwargs
    ) -> Optional[Any]:
        """
        Find the first element matching the selector.

        Args:
            selector: CSS-like selector string
            apply_exclusions: Whether to exclude elements in exclusion regions
            regex: Whether to use regex for text search
            case: Whether to do case-sensitive text search
            **kwargs: Additional filter parameters

        Returns:
            Element object or None if not found
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        selector_obj = parse_selector(selector)
        kwargs["regex"] = regex
        kwargs["case"] = case

        results = self._apply_selector(
            selector_obj, apply_exclusions=apply_exclusions, first_only=True, **kwargs
        )
        return results.first if results else None

    def find_all(
        self, selector: str, apply_exclusions=True, regex=False, case=True, **kwargs
    ) -> ElementCollection:
        """
        Find all elements matching the selector.

        Args:
            selector: CSS-like selector string
            apply_exclusions: Whether to exclude elements in exclusion regions
            regex: Whether to use regex for text search
            case: Whether to do case-sensitive text search
            **kwargs: Additional filter parameters

        Returns:
            ElementCollection with matching elements
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        selector_obj = parse_selector(selector)
        kwargs["regex"] = regex
        kwargs["case"] = case

        results = self._apply_selector(
            selector_obj, apply_exclusions=apply_exclusions, first_only=False, **kwargs
        )
        return results

    def _apply_selector(
        self, selector_obj: Dict, apply_exclusions=True, first_only=False, **kwargs
    ) -> ElementCollection:
        """
        Apply selector to PDF elements across all pages.

        Args:
            selector_obj: Parsed selector dictionary
            apply_exclusions: Whether to exclude elements in exclusion regions
            first_only: If True, stop searching after the first match is found
            **kwargs: Additional filter parameters

        Returns:
            ElementCollection of matching elements
        """
        from natural_pdf.elements.collections import ElementCollection

        page_indices = kwargs.get("pages", range(len(self._pages)))
        if isinstance(page_indices, int):
            page_indices = [page_indices]
        elif isinstance(page_indices, slice):
            page_indices = range(*page_indices.indices(len(self._pages)))

        for pseudo in selector_obj.get("pseudo_classes", []):
            if pseudo.get("name") in ("spans", "continues"):
                logger.warning("Cross-page selectors ('spans', 'continues') are not yet supported.")
                return ElementCollection([])

        all_elements = []
        for page_idx in page_indices:
            if 0 <= page_idx < len(self._pages):
                page = self._pages[page_idx]
                page_elements_collection = page._apply_selector(
                    selector_obj, apply_exclusions=apply_exclusions, first_only=first_only, **kwargs
                )
                if page_elements_collection:
                    page_elements = page_elements_collection.elements
                    all_elements.extend(page_elements)
                    if first_only and page_elements:
                        break
            else:
                logger.warning(f"Page index {page_idx} out of range (0-{len(self._pages)-1}).")

        combined = ElementCollection(all_elements)

        if not first_only and kwargs.get("document_order", True):
            if all(
                hasattr(el, "page") and hasattr(el, "top") and hasattr(el, "x0")
                for el in combined.elements
            ):
                combined.sort(key=lambda el: (el.page.index, el.top, el.x0))
            else:
                try:
                    combined.sort(key=lambda el: el.page.index)
                except AttributeError:
                    logger.warning("Cannot sort elements in document order: Missing required attributes.")

        return combined

    def extract_text(
        self,
        selector: Optional[str] = None,
        preserve_whitespace=True,
        use_exclusions=True,
        debug_exclusions=False,
        **kwargs,
    ) -> str:
        """
        Extract text from the entire document or matching elements.

        Args:
            selector: Optional selector to filter elements
            preserve_whitespace: Whether to keep blank characters
            use_exclusions: Whether to apply exclusion regions
            debug_exclusions: Whether to output detailed debugging for exclusions
            **kwargs: Additional extraction parameters

        Returns:
            Extracted text as string
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        if selector:
            elements = self.find_all(selector, apply_exclusions=use_exclusions, **kwargs)
            return elements.extract_text(preserve_whitespace=preserve_whitespace, **kwargs)

        if debug_exclusions:
            print(f"PDF: Extracting text with exclusions from {len(self.pages)} pages")
            print(f"PDF: Found {len(self._exclusions)} document-level exclusions")

        texts = []
        for page in self.pages:
            texts.append(
                page.extract_text(
                    preserve_whitespace=preserve_whitespace,
                    use_exclusions=use_exclusions,
                    debug_exclusions=debug_exclusions,
                    **kwargs,
                )
            )

        if debug_exclusions:
            print(f"PDF: Combined {len(texts)} pages of text")

        return "\n".join(texts)

    def extract_tables(
        self, selector: Optional[str] = None, merge_across_pages: bool = False, **kwargs
    ) -> List[Any]:
        """
        Extract tables from the document or matching elements.

        Args:
            selector: Optional selector to filter tables
            merge_across_pages: Whether to merge tables that span across pages
            **kwargs: Additional extraction parameters

        Returns:
            List of extracted tables
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")
            
        logger.warning("PDF.extract_tables is not fully implemented yet.")
        all_tables = []
        
        for page in self.pages:
            if hasattr(page, "extract_tables"):
                all_tables.extend(page.extract_tables(**kwargs))
            else:
                logger.debug(f"Page {page.number} does not have extract_tables method.")
                
        if selector:
            logger.warning("Filtering extracted tables by selector is not implemented.")
            
        if merge_across_pages:
            logger.warning("Merging tables across pages is not implemented.")
            
        return all_tables

    def save_searchable(self, output_path: Union[str, "Path"], dpi: int = 300, **kwargs):
        """
        Saves the PDF with an OCR text layer, making content searchable.

        Requires optional dependencies. Install with: pip install "natural-pdf[ocr-save]"

        Args:
            output_path: Path to save the searchable PDF
            dpi: Resolution for rendering and OCR overlay
            **kwargs: Additional keyword arguments passed to the exporter
        """
        from natural_pdf.exporters.searchable_pdf import create_searchable_pdf

        output_path_str = str(output_path)
        create_searchable_pdf(self, output_path_str, dpi=dpi, **kwargs)
        logger.info(f"Searchable PDF saved to: {output_path_str}")

    def ask(
        self,
        question: str,
        mode: str = "extractive",
        pages: Union[int, List[int], range] = None,
        min_confidence: float = 0.1,
        model: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
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
            A dictionary containing the answer, confidence, and other metadata
        """
        from natural_pdf.qa import get_qa_engine

        qa_engine = get_qa_engine() if model is None else get_qa_engine(model_name=model)

        if pages is None:
            target_pages = list(range(len(self.pages)))
        elif isinstance(pages, int):
            target_pages = [pages]
        elif isinstance(pages, (list, range)):
            target_pages = pages
        else:
            raise ValueError(f"Invalid pages parameter: {pages}")

        results = []
        for page_idx in target_pages:
            if 0 <= page_idx < len(self.pages):
                page = self.pages[page_idx]
                page_result = qa_engine.ask_pdf_page(
                    page=page, question=question, min_confidence=min_confidence, **kwargs
                )

                if page_result and page_result.get("found", False):
                    results.append(page_result)

        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        if results:
            return results[0]
        else:
            return {
                "answer": None,
                "confidence": 0.0,
                "found": False,
                "page_num": None,
                "source_elements": [],
            }

    def search_within_index(
        self,
        query: Union[str, Path, Image.Image, Region],
        search_service: SearchServiceProtocol,
        options: Optional[SearchOptions] = None,
    ) -> List[Dict[str, Any]]:
        """
        Finds relevant documents from this PDF within a search index.

        Args:
            query: The search query (text, image path, PIL Image, Region)
            search_service: A pre-configured SearchService instance
            options: Optional SearchOptions to configure the query

        Returns:
            A list of result dictionaries, sorted by relevance

        Raises:
            ImportError: If search dependencies are not installed
            ValueError: If search_service is None
            TypeError: If search_service does not conform to the protocol
            FileNotFoundError: If the collection managed by the service does not exist
            RuntimeError: For other search failures
        """
        if not search_service:
            raise ValueError("A configured SearchServiceProtocol instance must be provided.")

        collection_name = getattr(search_service, "collection_name", "<Unknown Collection>")
        logger.info(f"Searching within index '{collection_name}' for content from PDF '{self.path}'")

        service = search_service

        query_input = query
        effective_options = copy.deepcopy(options) if options is not None else TextSearchOptions()

        if isinstance(query, Region):
            logger.debug("Query is a Region object. Extracting text.")
            if not isinstance(effective_options, TextSearchOptions):
                logger.warning("Querying with Region image requires MultiModalSearchOptions. Falling back to text extraction.")
            query_input = query.extract_text()
            if not query_input or query_input.isspace():
                logger.error("Region has no extractable text for query.")
                return []

        # Add filter to scope search to THIS PDF
        pdf_scope_filter = {
            "field": "pdf_path",
            "operator": "eq",
            "value": self.path,
        }
        logger.debug(f"Applying filter to scope search to PDF: {pdf_scope_filter}")

        # Combine with existing filters in options (if any)
        if effective_options.filters:
            logger.debug(f"Combining PDF scope filter with existing filters")
            if isinstance(effective_options.filters, dict) and effective_options.filters.get("operator") == "AND":
                effective_options.filters["conditions"].append(pdf_scope_filter)
            elif isinstance(effective_options.filters, list):
                effective_options.filters = {
                    "operator": "AND",
                    "conditions": effective_options.filters + [pdf_scope_filter],
                }
            elif isinstance(effective_options.filters, dict):
                effective_options.filters = {
                    "operator": "AND",
                    "conditions": [effective_options.filters, pdf_scope_filter],
                }
            else:
                logger.warning(f"Unsupported format for existing filters. Overwriting with PDF scope filter.")
                effective_options.filters = pdf_scope_filter
        else:
            effective_options.filters = pdf_scope_filter

        logger.debug(f"Final filters for service search: {effective_options.filters}")

        try:
            results = service.search(
                query=query_input,
                options=effective_options,
            )
            logger.info(f"SearchService returned {len(results)} results from PDF '{self.path}'")
            return results
        except FileNotFoundError as fnf:
            logger.error(f"Search failed: Collection not found. Error: {fnf}")
            raise
        except Exception as e:
            logger.error(f"SearchService search failed: {e}")
            raise RuntimeError(f"Search within index failed. See logs for details.") from e

    def export_ocr_correction_task(self, output_zip_path: str, **kwargs):
        """
        Exports OCR results from this PDF into a correction task package.

        Args:
            output_zip_path: The path to save the output zip file
            **kwargs: Additional arguments passed to create_correction_task_package
        """
        try:
            from natural_pdf.utils.packaging import create_correction_task_package
            create_correction_task_package(source=self, output_zip_path=output_zip_path, **kwargs)
        except ImportError:
            logger.error("Failed to import 'create_correction_task_package'. Packaging utility might be missing.")
        except Exception as e:
            logger.error(f"Failed to export correction task: {e}")
            raise

    def correct_ocr(
        self,
        correction_callback: Callable[[Any], Optional[str]],
        pages: Optional[Union[Iterable[int], range, slice]] = None,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> "PDF":
        """
        Applies corrections to OCR text elements using a callback function.

        Args:
            correction_callback: Function that takes an element and returns corrected text or None
            pages: Optional page indices/slice to limit the scope of correction
            max_workers: Maximum number of threads to use for parallel execution
            progress_callback: Optional callback function for progress updates

        Returns:
            Self for method chaining
        """
        target_page_indices = []
        if pages is None:
            target_page_indices = list(range(len(self._pages)))
        elif isinstance(pages, slice):
            target_page_indices = list(range(*pages.indices(len(self._pages))))
        elif hasattr(pages, "__iter__"):
            try:
                target_page_indices = [int(i) for i in pages]
                for idx in target_page_indices:
                    if not (0 <= idx < len(self._pages)):
                        raise IndexError(f"Page index {idx} out of range (0-{len(self._pages)-1}).")
            except (IndexError, TypeError, ValueError) as e:
                raise ValueError(f"Invalid page index in 'pages': {pages}. Error: {e}") from e
        else:
            raise TypeError("'pages' must be None, a slice, or an iterable of page indices.")

        if not target_page_indices:
            logger.warning("No pages selected for OCR correction.")
            return self

        logger.info(f"Starting OCR correction for pages: {target_page_indices}")

        for page_idx in target_page_indices:
            page = self._pages[page_idx]
            try:
                page.correct_ocr(
                    correction_callback=correction_callback,
                    max_workers=max_workers,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                logger.error(f"Error during correct_ocr on page {page_idx}: {e}")

        logger.info("OCR correction process finished.")
        return self

    def __len__(self) -> int:
        """Return the number of pages in the PDF."""
        if not hasattr(self, "_pages"):
            return 0
        return len(self._pages)

    def __getitem__(self, key) -> Union[Page, "PageCollection"]:
        """Access pages by index or slice."""
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not initialized yet.")
            
        if isinstance(key, slice):
            from natural_pdf.elements.collections import PageCollection
            return PageCollection(self._pages[key])
            
        if isinstance(key, int):
            if 0 <= key < len(self._pages):
                return self._pages[key]
            else:
                raise IndexError(f"Page index {key} out of range (0-{len(self._pages)-1}).")
        else:
            raise TypeError(f"Page indices must be integers or slices, not {type(key)}.")

    def close(self):
        """Close the underlying PDF file and clean up any temporary files."""
        if hasattr(self, "_pdf") and self._pdf is not None:
            try:
                self._pdf.close()
                logger.debug(f"Closed pdfplumber PDF object for {self.source_path}")
            except Exception as e:
                logger.warning(f"Error closing pdfplumber object: {e}")
            finally:
                self._pdf = None

        if hasattr(self, "_temp_file") and self._temp_file is not None:
            temp_file_path = None
            try:
                if hasattr(self._temp_file, "name") and self._temp_file.name:
                    temp_file_path = self._temp_file.name
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        logger.debug(f"Removed temporary PDF file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file '{temp_file_path}': {e}")
            finally:
                self._temp_file = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def get_id(self) -> str:
        """Get unique identifier for this PDF."""
        return self.path

    # --- Classification Methods --- #

    def classify_pages(
        self,
        categories: List[str],
        model: Optional[str] = None,
        pages: Optional[Union[Iterable[int], range, slice]] = None,
        analysis_key: str = "classification",
        using: Optional[str] = None,
        **kwargs,
    ) -> "PDF":
        """
        Classifies specified pages of the PDF.

        Args:
            categories: List of category names
            model: Model identifier ('text', 'vision', or specific HF ID)
            pages: Page indices, slice, or None for all pages
            analysis_key: Key to store results in page's analyses dict
            using: Processing mode ('text' or 'vision')
            **kwargs: Additional arguments for the ClassificationManager

        Returns:
            Self for method chaining
        """
        if not categories:
            raise ValueError("Categories list cannot be empty.")

        try:
            manager = self.get_manager('classification')
        except (ValueError, RuntimeError) as e:
            raise ClassificationError(f"Cannot get ClassificationManager: {e}") from e

        if not manager or not manager.is_available():
            try:
                from natural_pdf.classification.manager import _CLASSIFICATION_AVAILABLE
                if not _CLASSIFICATION_AVAILABLE:
                    raise ImportError("Classification dependencies missing.")
            except ImportError:
                raise ImportError(
                    "Classification dependencies missing. "
                    "Install with: pip install \"natural-pdf[classification]\""
                )
            raise ClassificationError("ClassificationManager not available.")

        target_pages = []
        if pages is None:
            target_pages = self._pages
        elif isinstance(pages, slice):
            target_pages = self._pages[pages]
        elif hasattr(pages, "__iter__"):
            try:
                target_pages = [self._pages[i] for i in pages]
            except IndexError:
                raise ValueError("Invalid page index provided.")
            except TypeError:
                raise TypeError("'pages' must be None, a slice, or an iterable of page indices.")
        else:
            raise TypeError("'pages' must be None, a slice, or an iterable of page indices.")

        if not target_pages:
            logger.warning("No pages selected for classification.")
            return self

        inferred_using = manager.infer_using(model if model else manager.DEFAULT_TEXT_MODEL, using)
        logger.info(f"Classifying {len(target_pages)} pages using model '{model or '(default)'}' (mode: {inferred_using})")

        page_contents = []
        pages_to_classify = []
        logger.debug(f"Gathering content for {len(target_pages)} pages...")
        
        for page in target_pages:
            try:
                content = page._get_classification_content(model_type=inferred_using, **kwargs)
                page_contents.append(content)
                pages_to_classify.append(page)
            except ValueError as e:
                logger.warning(f"Skipping page {page.number}: Cannot get content - {e}")
            except Exception as e:
                logger.warning(f"Skipping page {page.number}: Error getting content - {e}")

        if not page_contents:
            logger.warning("No content could be gathered for batch classification.")
            return self
            
        logger.debug(f"Gathered content for {len(pages_to_classify)} pages.")

        try:
            batch_results = manager.classify_batch(
                item_contents=page_contents,
                categories=categories,
                model_id=model,
                using=inferred_using,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            raise ClassificationError(f"Batch classification failed: {e}") from e

        if len(batch_results) != len(pages_to_classify):
            logger.error(f"Mismatch between number of results ({len(batch_results)}) and pages ({len(pages_to_classify)})")
            return self

        logger.debug(f"Distributing {len(batch_results)} results to pages under key '{analysis_key}'...")
        for page, result_obj in zip(pages_to_classify, batch_results):
            try:
                if not hasattr(page, 'analyses') or page.analyses is None:
                    page.analyses = {}
                page.analyses[analysis_key] = result_obj
            except Exception as e:
                logger.warning(f"Failed to store classification results for page {page.number}: {e}")

        logger.info(f"Finished classifying PDF pages.")
        return self

    # --- End Classification Methods --- #

    # --- Extraction Support --- #
    def _get_extraction_content(self, using: str = 'text', **kwargs) -> Any:
        """
        Retrieves the content for the entire PDF.

        Args:
            using: 'text' or 'vision'
            **kwargs: Additional arguments passed to extract_text or page.to_image

        Returns:
            str: Extracted text if using='text'
            List[PIL.Image.Image]: List of page images if using='vision'
            None: If content cannot be retrieved
        """
        if using == 'text':
            try:
                layout = kwargs.pop('layout', True)
                return self.extract_text(layout=layout, **kwargs)
            except Exception as e:
                logger.error(f"Error extracting text from PDF: {e}")
                return None
        elif using == 'vision':
            page_images = []
            logger.info(f"Rendering {len(self.pages)} pages to images...")
            
            resolution = kwargs.pop('resolution', 72)
            include_highlights = kwargs.pop('include_highlights', False)
            labels = kwargs.pop('labels', False)
            
            try:
                for page in tqdm(self.pages, desc="Rendering Pages"):
                    img = page.to_image(
                        resolution=resolution,
                        include_highlights=include_highlights,
                        labels=labels,
                        **kwargs
                    )
                    if img:
                        page_images.append(img)
                    else:
                        logger.warning(f"Failed to render page {page.number}, skipping.")
                if not page_images:
                    logger.error("Failed to render any pages.")
                    return None
                return page_images
            except Exception as e:
                logger.error(f"Error rendering pages: {e}")
                return None
        else:
            logger.error(f"Unsupported value for 'using': {using}")
            return None
    # --- End Extraction Support --- #
