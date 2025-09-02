import copy
import io
import logging
import os
import re
import tempfile
import threading
import time
import urllib.request
import weakref
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)

import pdfplumber
from tqdm.auto import tqdm

from natural_pdf.analyzers.checkbox.mixin import CheckboxDetectionMixin
from natural_pdf.analyzers.layout.layout_manager import LayoutManager
from natural_pdf.classification.manager import ClassificationError
from natural_pdf.classification.mixin import ClassificationMixin
from natural_pdf.classification.results import ClassificationResult
from natural_pdf.core.highlighting_service import HighlightingService
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.elements.base import Element
from natural_pdf.elements.region import Region
from natural_pdf.export.mixin import ExportMixin
from natural_pdf.extraction.manager import StructuredDataManager
from natural_pdf.extraction.mixin import ExtractionMixin
from natural_pdf.ocr import OCRManager, OCROptions
from natural_pdf.selectors.parser import parse_selector
from natural_pdf.text_mixin import TextMixin
from natural_pdf.utils.locks import pdf_render_lock
from natural_pdf.vision.mixin import VisualSearchMixin

if TYPE_CHECKING:
    from natural_pdf.elements.element_collection import ElementCollection

try:
    from typing import Any as TypingAny

    from natural_pdf.search import (
        BaseSearchOptions,
        SearchOptions,
        SearchServiceProtocol,
        TextSearchOptions,
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


try:
    from natural_pdf.exporters.searchable_pdf import create_searchable_pdf
except ImportError:
    create_searchable_pdf = None
try:
    from natural_pdf.exporters.original_pdf import create_original_pdf
except ImportError:
    create_original_pdf = None

logger = logging.getLogger("natural_pdf.core.pdf")


def _get_classification_manager_class():
    """Lazy import for ClassificationManager."""
    from natural_pdf.classification.manager import ClassificationManager

    return ClassificationManager


DEFAULT_MANAGERS = {
    "classification": _get_classification_manager_class,
    "structured_data": StructuredDataManager,
}

# Deskew Imports (Conditional)
import numpy as np
from PIL import Image

try:
    import img2pdf
    from deskew import determine_skew

    DESKEW_AVAILABLE = True
except ImportError:
    DESKEW_AVAILABLE = False
    img2pdf = None
# End Deskew Imports

# --- Lazy Page List Helper --- #
from collections.abc import Sequence


class _LazyPageList(Sequence):
    """A lightweight, list-like object that lazily instantiates natural-pdf Page objects.

    This class implements the Sequence protocol to provide list-like access to PDF pages
    while minimizing memory usage. Pages are only created when accessed, and once created,
    they are cached for subsequent access. This design allows efficient handling of large
    PDF documents without loading all pages into memory immediately.

    The sequence holds `None` placeholders until an index is accessed, at which point
    a real `Page` object is created, cached, and returned. Slices and iteration are
    also supported and will materialize pages on demand.

    Attributes:
        _parent_pdf: Reference to the parent PDF object.
        _plumber_pdf: Underlying pdfplumber PDF object.
        _font_attrs: Font attributes to use when creating pages.
        _cache: List of cached Page objects (None until accessed).
        _load_text: Whether to load text layer when creating pages.
        _indices: Optional range of indices this list represents (for slices).

    Example:
        ```python
        # Access is transparent - pages created on demand
        pdf = npdf.PDF("document.pdf")
        first_page = pdf.pages[0]  # Creates Page object here
        last_page = pdf.pages[-1]  # Creates another Page object

        # Slicing works too
        first_three = pdf.pages[0:3]  # Returns another lazy list

        # Iteration creates all pages
        for page in pdf.pages:  # Each page created as needed
            print(f"Page {page.index}")
        ```
    """

    def __init__(
        self,
        parent_pdf: "PDF",
        plumber_pdf: "pdfplumber.PDF",
        font_attrs=None,
        load_text=True,
        indices: Optional[List[int]] = None,
    ):
        self._parent_pdf = parent_pdf
        self._plumber_pdf = plumber_pdf
        self._font_attrs = font_attrs
        self._load_text = load_text

        # If indices is provided, this is a sliced view
        if indices is not None:
            self._indices = indices
            self._cache = [None] * len(indices)
        else:
            # Full PDF - one slot per pdfplumber page
            self._indices = list(range(len(plumber_pdf.pages)))
            self._cache = [None] * len(plumber_pdf.pages)

    # Internal helper -----------------------------------------------------
    def _create_page(self, index: int) -> "Page":
        """Create and cache a page at the given index within this list."""
        cached = self._cache[index]
        if cached is None:
            # Get the actual page index in the full PDF
            actual_page_index = self._indices[index]

            # First check if this page is already cached in the parent PDF's main page list
            if (
                hasattr(self._parent_pdf, "_pages")
                and hasattr(self._parent_pdf._pages, "_cache")
                and actual_page_index < len(self._parent_pdf._pages._cache)
                and self._parent_pdf._pages._cache[actual_page_index] is not None
            ):
                # Reuse the already-cached page from the parent PDF
                # This ensures we get any exclusions that were already applied
                cached = self._parent_pdf._pages._cache[actual_page_index]
                self._cache[index] = cached
                return cached

            # Import here to avoid circular import problems
            from natural_pdf.core.page import Page

            # Create new page
            plumber_page = self._plumber_pdf.pages[actual_page_index]
            cached = Page(
                plumber_page,
                parent=self._parent_pdf,
                index=actual_page_index,
                font_attrs=self._font_attrs,
                load_text=self._load_text,
            )

            # Apply any stored exclusions to the newly created page
            if hasattr(self._parent_pdf, "_exclusions"):
                for exclusion_data in self._parent_pdf._exclusions:
                    exclusion_func, label = exclusion_data
                    try:
                        cached.add_exclusion(exclusion_func, label=label)
                    except Exception as e:
                        logger.warning(f"Failed to apply exclusion to page {cached.number}: {e}")

            # Check if the parent PDF already has a cached page with page-specific exclusions
            if hasattr(self._parent_pdf, "_pages") and hasattr(self._parent_pdf._pages, "_cache"):
                parent_cache = self._parent_pdf._pages._cache
                if (
                    actual_page_index < len(parent_cache)
                    and parent_cache[actual_page_index] is not None
                ):
                    existing_page = parent_cache[actual_page_index]
                    # Copy over any page-specific exclusions from the existing page
                    # Only copy non-callable exclusions (regions/elements) to avoid duplicating PDF-level exclusions
                    if hasattr(existing_page, "_exclusions") and existing_page._exclusions:
                        for exclusion_data in existing_page._exclusions:
                            exclusion_item = exclusion_data[0]
                            # Skip callable exclusions as they're PDF-level and already applied above
                            if not callable(exclusion_item):
                                try:
                                    cached.add_exclusion(
                                        *exclusion_data[:2]
                                    )  # exclusion_item and label
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to copy page-specific exclusion to page {cached.number}: {e}"
                                    )

            # Apply any stored regions to the newly created page
            if hasattr(self._parent_pdf, "_regions"):
                for region_data in self._parent_pdf._regions:
                    region_func, name = region_data
                    try:
                        region_instance = region_func(cached)
                        if region_instance and hasattr(region_instance, "__class__"):
                            # Check if it's a Region-like object (avoid importing Region here)
                            cached.add_region(region_instance, name=name, source="named")
                        elif region_instance is not None:
                            logger.warning(
                                f"Region function did not return a valid Region for page {cached.number}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to apply region to page {cached.number}: {e}")

            self._cache[index] = cached

            # Also cache in the parent PDF's main page list if this is a slice
            if (
                hasattr(self._parent_pdf, "_pages")
                and hasattr(self._parent_pdf._pages, "_cache")
                and actual_page_index < len(self._parent_pdf._pages._cache)
                and self._parent_pdf._pages._cache[actual_page_index] is None
            ):
                self._parent_pdf._pages._cache[actual_page_index] = cached

        return cached

    # Sequence protocol ---------------------------------------------------
    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the slice of our current indices
            slice_indices = range(*key.indices(len(self)))
            # Extract the actual page indices for this slice
            actual_indices = [self._indices[i] for i in slice_indices]
            # Return a new lazy list for the slice
            return _LazyPageList(
                self._parent_pdf,
                self._plumber_pdf,
                font_attrs=self._font_attrs,
                load_text=self._load_text,
                indices=actual_indices,
            )
        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("Page index out of range")
            return self._create_page(key)
        else:
            raise TypeError("Page indices must be integers or slices")

    def __iter__(self):
        for i in range(len(self)):
            yield self._create_page(i)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<_LazyPageList(len={len(self)})>"


# --- End Lazy Page List Helper --- #


class PDF(
    TextMixin,
    ExtractionMixin,
    ExportMixin,
    ClassificationMixin,
    CheckboxDetectionMixin,
    VisualSearchMixin,
    Visualizable,
):
    """Enhanced PDF wrapper built on top of pdfplumber.

    This class provides a fluent interface for working with PDF documents,
    with improved selection, navigation, and extraction capabilities. It integrates
    OCR, layout analysis, and AI-powered data extraction features while maintaining
    compatibility with the underlying pdfplumber API.

    The PDF class supports loading from files, URLs, or streams, and provides
    spatial navigation, element selection with CSS-like selectors, and advanced
    document processing workflows including multi-page content flows.

    Attributes:
        pages: Lazy-loaded list of Page objects for document pages.
        path: Resolved path to the PDF file or source identifier.
        source_path: Original path, URL, or stream identifier provided during initialization.
        highlighter: Service for rendering highlighted visualizations of document content.

    Example:
        Basic usage:
        ```python
        import natural_pdf as npdf

        pdf = npdf.PDF("document.pdf")
        page = pdf.pages[0]
        text_elements = page.find_all('text:contains("Summary")')
        ```

        Advanced usage with OCR:
        ```python
        pdf = npdf.PDF("scanned_document.pdf")
        pdf.apply_ocr(engine="easyocr", resolution=144)
        tables = pdf.pages[0].find_all('table')
        ```
    """

    def __init__(
        self,
        path_or_url_or_stream,
        reading_order: bool = True,
        font_attrs: Optional[List[str]] = None,
        keep_spaces: bool = True,
        text_tolerance: Optional[dict] = None,
        auto_text_tolerance: bool = True,
        text_layer: bool = True,
    ):
        """Initialize the enhanced PDF object.

        Args:
            path_or_url_or_stream: Path to the PDF file (str/Path), a URL (str),
                or a file-like object (stream). URLs must start with 'http://' or 'https://'.
            reading_order: If True, use natural reading order for text extraction.
                Defaults to True.
            font_attrs: List of font attributes for grouping characters into words.
                Common attributes include ['fontname', 'size']. Defaults to None.
            keep_spaces: If True, include spaces in word elements during text extraction.
                Defaults to True.
            text_tolerance: PDFplumber-style tolerance settings for text grouping.
                Dictionary with keys like 'x_tolerance', 'y_tolerance'. Defaults to None.
            auto_text_tolerance: If True, automatically scale text tolerance based on
                font size and document characteristics. Defaults to True.
            text_layer: If True, preserve existing text layer from the PDF. If False,
                removes all existing text elements during initialization, useful for
                OCR-only workflows. Defaults to True.

        Raises:
            TypeError: If path_or_url_or_stream is not a valid type.
            IOError: If the PDF file cannot be opened or read.
            ValueError: If URL download fails.

        Example:
            ```python
            # From file path
            pdf = npdf.PDF("document.pdf")

            # From URL
            pdf = npdf.PDF("https://example.com/document.pdf")

            # From stream
            with open("document.pdf", "rb") as f:
                pdf = npdf.PDF(f)

            # With custom settings
            pdf = npdf.PDF("document.pdf",
                          reading_order=False,
                          text_layer=False,  # For OCR-only processing
                          font_attrs=['fontname', 'size', 'flags'])
            ```
        """
        self._original_path_or_stream = path_or_url_or_stream
        self._temp_file = None
        self._resolved_path = None
        self._is_stream = False
        self._text_layer = text_layer
        stream_to_open = None

        if hasattr(path_or_url_or_stream, "read"):  # Check if it's file-like
            logger.info("Initializing PDF from in-memory stream.")
            self._is_stream = True
            self._resolved_path = None  # No resolved file path for streams
            self.source_path = "<stream>"  # Identifier for source
            self.path = self.source_path  # Use source identifier as path for streams
            stream_to_open = path_or_url_or_stream
            try:
                if hasattr(path_or_url_or_stream, "read"):
                    # If caller provided an in-memory binary stream, capture bytes for potential re-export
                    current_pos = path_or_url_or_stream.tell()
                    path_or_url_or_stream.seek(0)
                    self._original_bytes = path_or_url_or_stream.read()
                    path_or_url_or_stream.seek(current_pos)
            except Exception:
                pass
        elif isinstance(path_or_url_or_stream, (str, Path)):
            path_or_url = str(path_or_url_or_stream)
            self.source_path = path_or_url  # Store original path/URL as source
            is_url = path_or_url.startswith("http://") or path_or_url.startswith("https://")

            if is_url:
                logger.info(f"Downloading PDF from URL: {path_or_url}")
                try:
                    with urllib.request.urlopen(path_or_url) as response:
                        data = response.read()
                    # Load directly into an in-memory buffer — no temp file needed
                    buffer = io.BytesIO(data)
                    buffer.seek(0)
                    self._temp_file = None  # No on-disk temp file
                    self._resolved_path = path_or_url  # For repr / get_id purposes
                    stream_to_open = buffer  # pdfplumber accepts file-like objects
                except Exception as e:
                    logger.error(f"Failed to download PDF from URL: {e}")
                    raise ValueError(f"Failed to download PDF from URL: {e}")
            else:
                self._resolved_path = str(Path(path_or_url).resolve())  # Resolve local paths
                stream_to_open = self._resolved_path
            self.path = self._resolved_path  # Use resolved path for file-based PDFs
        else:
            raise TypeError(
                f"Invalid input type: {type(path_or_url_or_stream)}. "
                f"Expected path (str/Path), URL (str), or file-like object."
            )

        logger.info(f"Opening PDF source: {self.source_path}")
        logger.debug(
            f"Parameters: reading_order={reading_order}, font_attrs={font_attrs}, keep_spaces={keep_spaces}"
        )

        try:
            self._pdf = pdfplumber.open(stream_to_open)
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}", exc_info=True)
            self.close()  # Attempt cleanup if opening fails
            raise IOError(f"Failed to open PDF source: {self.source_path}") from e

        # Store configuration used for initialization
        self._reading_order = reading_order
        self._config = {"keep_spaces": keep_spaces}
        self._font_attrs = font_attrs

        self._ocr_manager = OCRManager() if OCRManager else None
        self._layout_manager = LayoutManager() if LayoutManager else None
        self.highlighter = HighlightingService(self)
        # self._classification_manager_instance = ClassificationManager() # Removed this line
        self._manager_registry = {}

        # Lazily instantiate pages only when accessed
        self._pages = _LazyPageList(
            self, self._pdf, font_attrs=font_attrs, load_text=self._text_layer
        )

        self._element_cache = {}
        self._exclusions = []
        self._regions = []

        logger.info(f"PDF '{self.source_path}' initialized with {len(self._pages)} pages.")

        self._initialize_managers()
        self._initialize_highlighter()

        # Remove text layer if requested
        if not self._text_layer:
            logger.info("Removing text layer as requested (text_layer=False)")
            # Text layer is not loaded when text_layer=False, so no need to remove
            pass

        # Analysis results accessed via self.analyses property (see below)

        # --- Automatic cleanup when object is garbage-collected ---
        self._finalizer = weakref.finalize(
            self,
            PDF._finalize_cleanup,
            self._pdf,
            getattr(self, "_temp_file", None),
            getattr(self, "_is_stream", False),
        )

        # --- Text tolerance settings ------------------------------------
        # Users can pass pdfplumber-style keys (x_tolerance, x_tolerance_ratio,
        # y_tolerance, etc.) via *text_tolerance*.  We also keep a flag that
        # enables automatic tolerance scaling when explicit values are not
        # supplied.
        self._config["auto_text_tolerance"] = bool(auto_text_tolerance)
        if text_tolerance:
            # Only copy recognised primitives (numbers / None); ignore junk.
            allowed = {
                "x_tolerance",
                "x_tolerance_ratio",
                "y_tolerance",
                "keep_blank_chars",  # passthrough convenience
            }
            for k, v in text_tolerance.items():
                if k in allowed:
                    self._config[k] = v

    def _initialize_managers(self):
        """Set up manager factories for lazy instantiation."""
        # Store factories/classes for each manager key
        self._manager_factories = dict(DEFAULT_MANAGERS)
        self._managers = {}  # Will hold instantiated managers

    def get_manager(self, key: str) -> Any:
        """Retrieve a manager instance by its key, instantiating it lazily if needed.

        Managers are specialized components that handle specific functionality like
        classification, structured data extraction, or OCR processing. They are
        instantiated on-demand to minimize memory usage and startup time.

        Args:
            key: The manager key to retrieve. Common keys include 'classification'
                and 'structured_data'.

        Returns:
            The manager instance for the specified key.

        Raises:
            KeyError: If no manager is registered for the given key.
            RuntimeError: If the manager failed to initialize.

        Example:
            ```python
            pdf = npdf.PDF("document.pdf")
            classification_mgr = pdf.get_manager('classification')
            structured_data_mgr = pdf.get_manager('structured_data')
            ```
        """
        # Check if already instantiated
        if key in self._managers:
            manager_instance = self._managers[key]
            if manager_instance is None:
                raise RuntimeError(f"Manager '{key}' failed to initialize previously.")
            return manager_instance

        # Not instantiated yet: get factory/class
        if not hasattr(self, "_manager_factories") or key not in self._manager_factories:
            raise KeyError(
                f"No manager registered for key '{key}'. Available: {list(getattr(self, '_manager_factories', {}).keys())}"
            )
        factory_or_class = self._manager_factories[key]
        try:
            resolved = factory_or_class
            # If it's a callable that's not a class, call it to get the class/instance
            if not isinstance(resolved, type) and callable(resolved):
                resolved = resolved()
            # If it's a class, instantiate it
            if isinstance(resolved, type):
                instance = resolved()
            else:
                instance = resolved  # Already an instance
            self._managers[key] = instance
            return instance
        except Exception as e:
            logger.error(f"Failed to initialize manager for key '{key}': {e}")
            self._managers[key] = None
            raise RuntimeError(f"Manager '{key}' failed to initialize: {e}") from e

    def _initialize_highlighter(self):
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        """Access PDF metadata as a dictionary.

        Returns document metadata such as title, author, creation date, and other
        properties embedded in the PDF file. The exact keys available depend on
        what metadata was included when the PDF was created.

        Returns:
            Dictionary containing PDF metadata. Common keys include 'Title',
            'Author', 'Subject', 'Creator', 'Producer', 'CreationDate', and
            'ModDate'. May be empty if no metadata is available.

        Example:
            ```python
            pdf = npdf.PDF("document.pdf")
            print(pdf.metadata.get('Title', 'No title'))
            print(f"Created: {pdf.metadata.get('CreationDate')}")
            ```
        """
        return self._pdf.metadata

    @property
    def pages(self) -> "PageCollection":
        """Access pages as a PageCollection object.

        Provides access to individual pages of the PDF document through a
        collection interface that supports indexing, slicing, and iteration.
        Pages are lazy-loaded to minimize memory usage.

        Returns:
            PageCollection object that provides list-like access to PDF pages.

        Raises:
            AttributeError: If PDF pages are not yet initialized.

        Example:
            ```python
            pdf = npdf.PDF("document.pdf")

            # Access individual pages
            first_page = pdf.pages[0]
            last_page = pdf.pages[-1]

            # Slice pages
            first_three = pdf.pages[0:3]

            # Iterate over pages
            for page in pdf.pages:
                print(f"Page {page.index} has {len(page.chars)} characters")
            ```
        """
        from natural_pdf.core.page_collection import PageCollection

        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")
        return PageCollection(self._pages)

    def clear_exclusions(self) -> "PDF":
        """Clear all exclusion functions from the PDF.

        Removes all previously added exclusion functions that were used to filter
        out unwanted content (like headers, footers, or administrative text) from
        text extraction and analysis operations.

        Returns:
            Self for method chaining.

        Raises:
            AttributeError: If PDF pages are not yet initialized.

        Example:
            ```python
            pdf = npdf.PDF("document.pdf")
            pdf.add_exclusion(lambda page: page.find('text:contains("CONFIDENTIAL")').above())

            # Later, remove all exclusions
            pdf.clear_exclusions()
            ```
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        self._exclusions = []

        # Clear exclusions only from already-created (cached) pages to avoid forcing page creation
        for i in range(len(self._pages)):
            if self._pages._cache[i] is not None:  # Only clear from existing pages
                try:
                    self._pages._cache[i].clear_exclusions()
                except Exception as e:
                    logger.warning(f"Failed to clear exclusions from existing page {i}: {e}")
        return self

    def add_exclusion(self, exclusion_func, label: str = None) -> "PDF":
        """Add an exclusion function to the PDF.

        Exclusion functions define regions of each page that should be ignored during
        text extraction and analysis operations. This is useful for filtering out headers,
        footers, watermarks, or other administrative content that shouldn't be included
        in the main document processing.

        Args:
            exclusion_func: A function that takes a Page object and returns a Region
                to exclude from processing, or None if no exclusion should be applied
                to that page. The function is called once per page.
            label: Optional descriptive label for this exclusion rule, useful for
                debugging and identification.

        Returns:
            Self for method chaining.

        Raises:
            AttributeError: If PDF pages are not yet initialized.

        Example:
            ```python
            pdf = npdf.PDF("document.pdf")

            # Exclude headers (top 50 points of each page)
            pdf.add_exclusion(
                lambda page: page.region(0, 0, page.width, 50),
                label="header_exclusion"
            )

            # Exclude any text containing "CONFIDENTIAL"
            pdf.add_exclusion(
                lambda page: page.find('text:contains("CONFIDENTIAL")').above(include_source=True)
                if page.find('text:contains("CONFIDENTIAL")') else None,
                label="confidential_exclusion"
            )

            # Chain multiple exclusions
            pdf.add_exclusion(header_func).add_exclusion(footer_func)
            ```
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        # ------------------------------------------------------------------
        # Support selector strings and ElementCollection objects directly.
        # Store exclusion and apply only to already-created pages.
        # ------------------------------------------------------------------
        from natural_pdf.elements.element_collection import ElementCollection  # local import

        if isinstance(exclusion_func, str) or isinstance(exclusion_func, ElementCollection):
            # Store for bookkeeping and lazy application
            self._exclusions.append((exclusion_func, label))

            # Don't modify already-cached pages - they will get PDF-level exclusions
            # dynamically through _get_exclusion_regions()
            return self

        # Fallback to original callable / Region behaviour ------------------
        exclusion_data = (exclusion_func, label)
        self._exclusions.append(exclusion_data)

        # Don't modify already-cached pages - they will get PDF-level exclusions
        # dynamically through _get_exclusion_regions()

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
        """Apply OCR to specified pages of the PDF using batch processing.

        Performs optical character recognition on the specified pages, converting
        image-based text into searchable and extractable text elements. This method
        supports multiple OCR engines and provides batch processing for efficiency.

        Args:
            engine: Name of the OCR engine to use. Supported engines include
                'easyocr' (default), 'surya', 'paddle', and 'doctr'. If None,
                uses the global default from natural_pdf.options.ocr.engine.
            languages: List of language codes for OCR recognition (e.g., ['en', 'es']).
                If None, uses the global default from natural_pdf.options.ocr.languages.
            min_confidence: Minimum confidence threshold (0.0-1.0) for accepting
                OCR results. Text with lower confidence will be filtered out.
                If None, uses the global default.
            device: Device to run OCR on ('cpu', 'cuda', 'mps'). Engine-specific
                availability varies. If None, uses engine defaults.
            resolution: DPI resolution for rendering pages to images before OCR.
                Higher values improve accuracy but increase processing time and memory.
                Typical values: 150 (fast), 300 (balanced), 600 (high quality).
            apply_exclusions: If True, mask excluded regions before OCR to prevent
                processing of headers, footers, or other unwanted content.
            detect_only: If True, only detect text bounding boxes without performing
                character recognition. Useful for layout analysis workflows.
            replace: If True, replace any existing OCR elements on the pages.
                If False, append new OCR results to existing elements.
            options: Engine-specific options object (e.g., EasyOCROptions, SuryaOptions).
                Allows fine-tuning of engine behavior beyond common parameters.
            pages: Page indices to process. Can be:
                - None: Process all pages
                - slice: Process a range of pages (e.g., slice(0, 10))
                - Iterable[int]: Process specific page indices (e.g., [0, 2, 5])

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If invalid page index is provided.
            TypeError: If pages parameter has invalid type.
            RuntimeError: If OCR engine is not available or fails.

        Example:
            ```python
            pdf = npdf.PDF("scanned_document.pdf")

            # Basic OCR on all pages
            pdf.apply_ocr()

            # High-quality OCR with specific settings
            pdf.apply_ocr(
                engine='easyocr',
                languages=['en', 'es'],
                resolution=300,
                min_confidence=0.8
            )

            # OCR specific pages only
            pdf.apply_ocr(pages=[0, 1, 2])  # First 3 pages
            pdf.apply_ocr(pages=slice(5, 10))  # Pages 5-9

            # Detection-only workflow for layout analysis
            pdf.apply_ocr(detect_only=True, resolution=150)
            ```

        Note:
            OCR processing can be time and memory intensive, especially at high
            resolutions. Consider using exclusions to mask unwanted regions and
            processing pages in batches for large documents.
        """
        if not self._ocr_manager:
            logger.error("OCRManager not available. Cannot apply OCR.")
            return self

        # Apply global options as defaults, but allow explicit parameters to override
        import natural_pdf

        # Use global OCR options if parameters are not explicitly set
        if engine is None:
            engine = natural_pdf.options.ocr.engine
        if languages is None:
            languages = natural_pdf.options.ocr.languages
        if min_confidence is None:
            min_confidence = natural_pdf.options.ocr.min_confidence
        if device is None:
            pass  # No default device in options.ocr anymore

        thread_id = threading.current_thread().name
        logger.debug(f"[{thread_id}] PDF.apply_ocr starting for {self.path}")

        target_pages = []

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
                # Use render() for clean image without highlights
                img = page.render(resolution=final_resolution)
                if img is None:
                    logger.error(f"  Failed to render page {page.number} to image.")
                    continue
                images_pil.append(img)
                page_image_map.append((page, img))
        except Exception as e:
            logger.error(f"Failed to render pages for batch OCR: {e}")
            logger.error(f"Failed to render pages for batch OCR: {e}")
            raise RuntimeError(f"Failed to render page {failed_page_num} for OCR.") from e

        render_end_time = time.monotonic()
        logger.debug(
            f"[{thread_id}] Finished rendering {len(images_pil)} images (Duration: {render_end_time - render_start_time:.2f}s)"
        )
        logger.debug(
            f"[{thread_id}] Finished rendering {len(images_pil)} images (Duration: {render_end_time - render_start_time:.2f}s)"
        )

        if not images_pil or not page_image_map:
            logger.error("No images were successfully rendered for batch OCR.")
            return self

        manager_args = {
            "images": images_pil,
            "engine": engine,
            "languages": languages,
            "min_confidence": min_confidence,
            "min_confidence": min_confidence,
            "device": device,
            "options": options,
            "detect_only": detect_only,
        }
        manager_args = {k: v for k, v in manager_args.items() if v is not None}

        ocr_call_args = {k: v for k, v in manager_args.items() if k != "images"}
        logger.info(f"[{thread_id}] Calling OCR Manager with args: {ocr_call_args}...")
        logger.info(f"[{thread_id}] Calling OCR Manager with args: {ocr_call_args}...")
        ocr_start_time = time.monotonic()

        batch_results = self._ocr_manager.apply_ocr(**manager_args)

        if not isinstance(batch_results, list) or len(batch_results) != len(images_pil):
            logger.error(f"OCR Manager returned unexpected result format or length.")
            return self

        logger.info("OCR Manager batch processing complete.")

        ocr_end_time = time.monotonic()
        logger.debug(
            f"[{thread_id}] OCR processing finished (Duration: {ocr_end_time - ocr_start_time:.2f}s)"
        )

        logger.info("Adding OCR results to respective pages...")
        total_elements_added = 0

        for i, (page, img) in enumerate(page_image_map):
            results_for_page = batch_results[i]
            if not isinstance(results_for_page, list):
                logger.warning(
                    f"Skipping results for page {page.number}: Expected list, got {type(results_for_page)}"
                )
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
        self, region_func: Callable[["Page"], Optional["Region"]], name: str = None
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

        # Apply only to already-created (cached) pages to avoid forcing page creation
        for i in range(len(self._pages)):
            if self._pages._cache[i] is not None:  # Only apply to existing pages
                page = self._pages._cache[i]
                try:
                    region_instance = region_func(page)
                    if region_instance and isinstance(region_instance, Region):
                        page.add_region(region_instance, name=name, source="named")
                    elif region_instance is not None:
                        logger.warning(
                            f"Region function did not return a valid Region for page {page.number}"
                        )
                except Exception as e:
                    logger.error(f"Error adding region for page {page.number}: {e}")

        return self

    @overload
    def find(
        self,
        *,
        text: str,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> Optional[Any]: ...

    @overload
    def find(
        self,
        selector: str,
        *,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> Optional[Any]: ...

    def find(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> Optional[Any]:
        """
        Find the first element matching the selector OR text content across all pages.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for (equivalent to 'text:contains(...)').
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            **kwargs: Additional filter parameters.

        Returns:
            Element object or None if not found.
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        if selector is not None and text is not None:
            raise ValueError("Provide either 'selector' or 'text', not both.")
        if selector is None and text is None:
            raise ValueError("Provide either 'selector' or 'text'.")

        # Construct selector if 'text' is provided
        effective_selector = ""
        if text is not None:
            escaped_text = text.replace('"', '\\"').replace("'", "\\'")
            effective_selector = f'text:contains("{escaped_text}")'
            logger.debug(
                f"Using text shortcut: find(text='{text}') -> find('{effective_selector}')"
            )
        elif selector is not None:
            effective_selector = selector
        else:
            raise ValueError("Internal error: No selector or text provided.")

        selector_obj = parse_selector(effective_selector)

        # Search page by page
        for page in self.pages:
            # Note: _apply_selector is on Page, so we call find directly here
            # We pass the constructed/validated effective_selector
            element = page.find(
                selector=effective_selector,  # Use the processed selector
                apply_exclusions=apply_exclusions,
                regex=regex,  # Pass down flags
                case=case,
                **kwargs,
            )
            if element:
                return element
        return None  # Not found on any page

    @overload
    def find_all(
        self,
        *,
        text: str,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> "ElementCollection": ...

    @overload
    def find_all(
        self,
        selector: str,
        *,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> "ElementCollection": ...

    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> "ElementCollection":
        """
        Find all elements matching the selector OR text content across all pages.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for (equivalent to 'text:contains(...)').
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            **kwargs: Additional filter parameters.

        Returns:
            ElementCollection with matching elements.
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        if selector is not None and text is not None:
            raise ValueError("Provide either 'selector' or 'text', not both.")
        if selector is None and text is None:
            raise ValueError("Provide either 'selector' or 'text'.")

        # Construct selector if 'text' is provided
        effective_selector = ""
        if text is not None:
            escaped_text = text.replace('"', '\\"').replace("'", "\\'")
            effective_selector = f'text:contains("{escaped_text}")'
            logger.debug(
                f"Using text shortcut: find_all(text='{text}') -> find_all('{effective_selector}')"
            )
        elif selector is not None:
            effective_selector = selector
        else:
            raise ValueError("Internal error: No selector or text provided.")

        # Instead of parsing here, let each page parse and apply
        # This avoids parsing the same selector multiple times if not needed
        # selector_obj = parse_selector(effective_selector)

        # kwargs["regex"] = regex # Removed: Already passed explicitly
        # kwargs["case"] = case   # Removed: Already passed explicitly

        all_elements = []
        for page in self.pages:
            # Call page.find_all with the effective selector and flags
            page_elements = page.find_all(
                selector=effective_selector,
                apply_exclusions=apply_exclusions,
                regex=regex,
                case=case,
                **kwargs,
            )
            if page_elements:
                all_elements.extend(page_elements.elements)

        from natural_pdf.elements.element_collection import ElementCollection

        return ElementCollection(all_elements)

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

    def get_sections(
        self,
        start_elements=None,
        end_elements=None,
        new_section_on_page_break=False,
        include_boundaries="both",
        orientation="vertical",
    ) -> "ElementCollection":
        """
        Extract sections from the entire PDF based on start/end elements.

        This method delegates to the PageCollection.get_sections() method,
        providing a convenient way to extract document sections across all pages.

        Args:
            start_elements: Elements or selector string that mark the start of sections (optional)
            end_elements: Elements or selector string that mark the end of sections (optional)
            new_section_on_page_break: Whether to start a new section at page boundaries (default: False)
            include_boundaries: How to include boundary elements: 'start', 'end', 'both', or 'none' (default: 'both')
            orientation: 'vertical' (default) or 'horizontal' - determines section direction

        Returns:
            ElementCollection of Region objects representing the extracted sections

        Example:
            Extract sections between headers:
            ```python
            pdf = npdf.PDF("document.pdf")

            # Get sections between headers
            sections = pdf.get_sections(
                start_elements='text[size>14]:bold',
                end_elements='text[size>14]:bold'
            )

            # Get sections that break at page boundaries
            sections = pdf.get_sections(
                start_elements='text:contains("Chapter")',
                new_section_on_page_break=True
            )
            ```

        Note:
            You can provide only start_elements, only end_elements, or both.
            - With only start_elements: sections go from each start to the next start (or end of document)
            - With only end_elements: sections go from beginning of document to each end
            - With both: sections go from each start to the corresponding end
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        return self.pages.get_sections(
            start_elements=start_elements,
            end_elements=end_elements,
            new_section_on_page_break=new_section_on_page_break,
            include_boundaries=include_boundaries,
            orientation=orientation,
        )

    def split(self, divider, **kwargs) -> "ElementCollection":
        """
        Divide the PDF into sections based on the provided divider elements.

        Args:
            divider: Elements or selector string that mark section boundaries
            **kwargs: Additional parameters passed to get_sections()
                - include_boundaries: How to include boundary elements (default: 'start')
                - orientation: 'vertical' or 'horizontal' (default: 'vertical')
                - new_section_on_page_break: Whether to split at page boundaries (default: False)

        Returns:
            ElementCollection of Region objects representing the sections

        Example:
            # Split a PDF by chapter titles
            chapters = pdf.split("text[size>20]:contains('Chapter')")

            # Export each chapter to a separate file
            for i, chapter in enumerate(chapters):
                chapter_text = chapter.extract_text()
                with open(f"chapter_{i+1}.txt", "w") as f:
                    f.write(chapter_text)

            # Split by horizontal rules/lines
            sections = pdf.split("line[orientation=horizontal]")

            # Split only by page breaks (no divider elements)
            pages = pdf.split(None, new_section_on_page_break=True)
        """
        # Delegate to pages collection
        return self.pages.split(divider, **kwargs)

    def save_searchable(self, output_path: Union[str, "Path"], dpi: int = 300, **kwargs):
        """
        DEPRECATED: Use save_pdf(..., ocr=True) instead.
        Saves the PDF with an OCR text layer, making content searchable.

        Requires optional dependencies. Install with: pip install \"natural-pdf[ocr-export]\"

        Args:
            output_path: Path to save the searchable PDF
            dpi: Resolution for rendering and OCR overlay
            **kwargs: Additional keyword arguments passed to the exporter
        """
        logger.warning(
            "PDF.save_searchable() is deprecated. Use PDF.save_pdf(..., ocr=True) instead."
        )
        if create_searchable_pdf is None:
            raise ImportError(
                "Saving searchable PDF requires 'pikepdf'. "
                'Install with: pip install "natural-pdf[ocr-export]"'
            )
        output_path_str = str(output_path)
        # Call the exporter directly, passing self (the PDF instance)
        create_searchable_pdf(self, output_path_str, dpi=dpi, **kwargs)
        # Logger info is handled within the exporter now
        # logger.info(f"Searchable PDF saved to: {output_path_str}")

    def save_pdf(
        self,
        output_path: Union[str, Path],
        ocr: bool = False,
        original: bool = False,
        dpi: int = 300,
    ):
        """
        Saves the PDF object (all its pages) to a new file.

        Choose one saving mode:
        - `ocr=True`: Creates a new, image-based PDF using OCR results from all pages.
          Text generated during the natural-pdf session becomes searchable,
          but original vector content is lost. Requires 'ocr-export' extras.
        - `original=True`: Saves a copy of the original PDF file this object represents.
          Any OCR results or analyses from the natural-pdf session are NOT included.
          If the PDF was opened from an in-memory buffer, this mode may not be suitable.
          Requires 'ocr-export' extras.

        Args:
            output_path: Path to save the new PDF file.
            ocr: If True, save as a searchable, image-based PDF using OCR data.
            original: If True, save the original source PDF content.
            dpi: Resolution (dots per inch) used only when ocr=True.

        Raises:
            ValueError: If the PDF has no pages, if neither or both 'ocr'
                        and 'original' are True.
            ImportError: If required libraries are not installed for the chosen mode.
            RuntimeError: If an unexpected error occurs during saving.
        """
        if not self.pages:
            raise ValueError("Cannot save an empty PDF object.")

        if not (ocr ^ original):  # XOR: exactly one must be true
            raise ValueError("Exactly one of 'ocr' or 'original' must be True.")

        output_path_obj = Path(output_path)
        output_path_str = str(output_path_obj)

        if ocr:
            has_vector_elements = False
            for page in self.pages:
                if (
                    hasattr(page, "rects")
                    and page.rects
                    or hasattr(page, "lines")
                    and page.lines
                    or hasattr(page, "curves")
                    and page.curves
                    or (
                        hasattr(page, "chars")
                        and any(getattr(el, "source", None) != "ocr" for el in page.chars)
                    )
                    or (
                        hasattr(page, "words")
                        and any(getattr(el, "source", None) != "ocr" for el in page.words)
                    )
                ):
                    has_vector_elements = True
                    break
            if has_vector_elements:
                logger.warning(
                    "Warning: Saving with ocr=True creates an image-based PDF. "
                    "Original vector elements (rects, lines, non-OCR text/chars) "
                    "will not be preserved in the output file."
                )

            logger.info(f"Saving searchable PDF (OCR text layer) to: {output_path_str}")
            try:
                # Delegate to the searchable PDF exporter, passing self (PDF instance)
                create_searchable_pdf(self, output_path_str, dpi=dpi)
            except Exception as e:
                raise RuntimeError(f"Failed to create searchable PDF: {e}") from e

        elif original:
            if create_original_pdf is None:
                raise ImportError(
                    "Saving with original=True requires 'pikepdf'. "
                    'Install with: pip install "natural-pdf[ocr-export]"'
                )

            # Optional: Add warning about losing OCR data similar to PageCollection
            has_ocr_elements = False
            for page in self.pages:
                if hasattr(page, "find_all"):
                    ocr_text_elements = page.find_all("text[source=ocr]")
                    if ocr_text_elements:
                        has_ocr_elements = True
                        break
                elif hasattr(page, "words"):  # Fallback
                    if any(getattr(el, "source", None) == "ocr" for el in page.words):
                        has_ocr_elements = True
                        break
            if has_ocr_elements:
                logger.warning(
                    "Warning: Saving with original=True preserves original page content. "
                    "OCR text generated in this session will not be included in the saved file."
                )

            logger.info(f"Saving original PDF content to: {output_path_str}")
            try:
                # Delegate to the original PDF exporter, passing self (PDF instance)
                create_original_pdf(self, output_path_str)
            except Exception as e:
                # Re-raise exception from exporter
                raise e

    def _get_render_specs(
        self,
        mode: Literal["show", "render"] = "show",
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        highlights: Optional[List[Dict[str, Any]]] = None,
        crop: Union[bool, Literal["content"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> List[RenderSpec]:
        """Get render specifications for this PDF.

        For PDF objects, this delegates to the pages collection to handle
        multi-page rendering.

        Args:
            mode: Rendering mode - 'show' includes highlights, 'render' is clean
            color: Color for highlighting pages in show mode
            highlights: Additional highlight groups to show
            crop: Whether to crop pages
            crop_bbox: Explicit crop bounds
            **kwargs: Additional parameters

        Returns:
            List of RenderSpec objects, one per page
        """
        # Delegate to pages collection
        return self.pages._get_render_specs(
            mode=mode, color=color, highlights=highlights, crop=crop, crop_bbox=crop_bbox, **kwargs
        )

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
        Ask a single question about the document content.

        Args:
            question: Question string to ask about the document
            mode: "extractive" to extract answer from document, "generative" to generate
            pages: Specific pages to query (default: all pages)
            min_confidence: Minimum confidence threshold for answers
            model: Optional model name for question answering
            **kwargs: Additional parameters passed to the QA engine

        Returns:
            Dict containing: answer, confidence, found, page_num, source_elements, etc.
        """
        # Delegate to ask_batch and return the first result
        results = self.ask_batch(
            [question], mode=mode, pages=pages, min_confidence=min_confidence, model=model, **kwargs
        )
        return (
            results[0]
            if results
            else {
                "answer": None,
                "confidence": 0.0,
                "found": False,
                "page_num": None,
                "source_elements": [],
            }
        )

    def ask_batch(
        self,
        questions: List[str],
        mode: str = "extractive",
        pages: Union[int, List[int], range] = None,
        min_confidence: float = 0.1,
        model: str = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Ask multiple questions about the document content using batch processing.

        This method processes multiple questions efficiently in a single batch,
        avoiding the multiprocessing resource accumulation that can occur with
        sequential individual question calls.

        Args:
            questions: List of question strings to ask about the document
            mode: "extractive" to extract answer from document, "generative" to generate
            pages: Specific pages to query (default: all pages)
            min_confidence: Minimum confidence threshold for answers
            model: Optional model name for question answering
            **kwargs: Additional parameters passed to the QA engine

        Returns:
            List of Dicts, each containing: answer, confidence, found, page_num, source_elements, etc.
        """
        from natural_pdf.qa import get_qa_engine

        if not questions:
            return []

        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            raise TypeError("'questions' must be a list of strings")

        qa_engine = get_qa_engine() if model is None else get_qa_engine(model_name=model)

        # Resolve target pages
        if pages is None:
            target_pages = self.pages
        elif isinstance(pages, int):
            if 0 <= pages < len(self.pages):
                target_pages = [self.pages[pages]]
            else:
                raise IndexError(f"Page index {pages} out of range (0-{len(self.pages)-1})")
        elif isinstance(pages, (list, range)):
            target_pages = []
            for page_idx in pages:
                if 0 <= page_idx < len(self.pages):
                    target_pages.append(self.pages[page_idx])
                else:
                    logger.warning(f"Page index {page_idx} out of range, skipping")
        else:
            raise ValueError(f"Invalid pages parameter: {pages}")

        if not target_pages:
            logger.warning("No valid pages found for QA processing.")
            return [
                {
                    "answer": None,
                    "confidence": 0.0,
                    "found": False,
                    "page_num": None,
                    "source_elements": [],
                }
                for _ in questions
            ]

        logger.info(
            f"Processing {len(questions)} question(s) across {len(target_pages)} page(s) using batch QA..."
        )

        # Collect all page images and metadata for batch processing
        page_images = []
        page_word_boxes = []
        page_metadata = []

        for page in target_pages:
            # Get page image
            try:
                # Use render() for clean image without highlights
                page_image = page.render(resolution=150)
                if page_image is None:
                    logger.warning(f"Failed to render image for page {page.number}, skipping")
                    continue

                # Get text elements for word boxes
                elements = page.find_all("text")
                if not elements:
                    logger.warning(f"No text elements found on page {page.number}")
                    word_boxes = []
                else:
                    word_boxes = qa_engine._get_word_boxes_from_elements(
                        elements, offset_x=0, offset_y=0
                    )

                page_images.append(page_image)
                page_word_boxes.append(word_boxes)
                page_metadata.append({"page_number": page.number, "page_object": page})

            except Exception as e:
                logger.warning(f"Error processing page {page.number}: {e}")
                continue

        if not page_images:
            logger.warning("No page images could be processed for QA.")
            return [
                {
                    "answer": None,
                    "confidence": 0.0,
                    "found": False,
                    "page_num": None,
                    "source_elements": [],
                }
                for _ in questions
            ]

        # Process all questions against all pages in batch
        all_results = []

        for question_text in questions:
            question_results = []

            # Ask this question against each page (but in batch per page)
            for i, (page_image, word_boxes, page_meta) in enumerate(
                zip(page_images, page_word_boxes, page_metadata)
            ):
                try:
                    # Use the DocumentQA batch interface
                    page_result = qa_engine.ask(
                        image=page_image,
                        question=question_text,
                        word_boxes=word_boxes,
                        min_confidence=min_confidence,
                        **kwargs,
                    )

                    if page_result and page_result.found:
                        # Add page metadata to result
                        page_result_dict = {
                            "answer": page_result.answer,
                            "confidence": page_result.confidence,
                            "found": page_result.found,
                            "page_num": page_meta["page_number"],
                            "source_elements": getattr(page_result, "source_elements", []),
                            "start": getattr(page_result, "start", -1),
                            "end": getattr(page_result, "end", -1),
                        }
                        question_results.append(page_result_dict)

                except Exception as e:
                    logger.warning(
                        f"Error processing question '{question_text}' on page {page_meta['page_number']}: {e}"
                    )
                    continue

            # Sort results by confidence and take the best one for this question
            question_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)

            if question_results:
                all_results.append(question_results[0])
            else:
                # No results found for this question
                all_results.append(
                    {
                        "answer": None,
                        "confidence": 0.0,
                        "found": False,
                        "page_num": None,
                        "source_elements": [],
                    }
                )

        return all_results

    def search_within_index(
        self,
        query: Union[str, Path, Image.Image, "Region"],
        search_service: "SearchServiceProtocol",
        options: Optional["SearchOptions"] = None,
    ) -> List[Dict[str, Any]]:
        """
        Finds relevant documents from this PDF within a search index.
        Finds relevant documents from this PDF within a search index.

        Args:
            query: The search query (text, image path, PIL Image, Region)
            search_service: A pre-configured SearchService instance
            options: Optional SearchOptions to configure the query
            query: The search query (text, image path, PIL Image, Region)
            search_service: A pre-configured SearchService instance
            options: Optional SearchOptions to configure the query

        Returns:
            A list of result dictionaries, sorted by relevance
            A list of result dictionaries, sorted by relevance

        Raises:
            ImportError: If search dependencies are not installed
            ValueError: If search_service is None
            TypeError: If search_service does not conform to the protocol
            FileNotFoundError: If the collection managed by the service does not exist
            RuntimeError: For other search failures
            ImportError: If search dependencies are not installed
            ValueError: If search_service is None
            TypeError: If search_service does not conform to the protocol
            FileNotFoundError: If the collection managed by the service does not exist
            RuntimeError: For other search failures
        """
        if not search_service:
            raise ValueError("A configured SearchServiceProtocol instance must be provided.")

        collection_name = getattr(search_service, "collection_name", "<Unknown Collection>")
        logger.info(
            f"Searching within index '{collection_name}' for content from PDF '{self.path}'"
        )

        service = search_service

        query_input = query
        effective_options = copy.deepcopy(options) if options is not None else TextSearchOptions()

        if isinstance(query, Region):
            logger.debug("Query is a Region object. Extracting text.")
            if not isinstance(effective_options, TextSearchOptions):
                logger.warning(
                    "Querying with Region image requires MultiModalSearchOptions. Falling back to text extraction."
                )
            query_input = query.extract_text()
            if not query_input or query_input.isspace():
                logger.error("Region has no extractable text for query.")
                return []

        # Add filter to scope search to THIS PDF
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
            if (
                isinstance(effective_options.filters, dict)
                and effective_options.filters.get("operator") == "AND"
            ):
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
                logger.warning(
                    f"Unsupported format for existing filters. Overwriting with PDF scope filter."
                )
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
            logger.error(f"Search failed: Collection not found. Error: {fnf}")
            raise
        except Exception as e:
            logger.error(f"SearchService search failed: {e}")
            raise RuntimeError(f"Search within index failed. See logs for details.") from e
            logger.error(f"SearchService search failed: {e}")
            raise RuntimeError(f"Search within index failed. See logs for details.") from e

    def export_ocr_correction_task(self, output_zip_path: str, **kwargs):
        """
        Exports OCR results from this PDF into a correction task package.
        Exports OCR results from this PDF into a correction task package.

        Args:
            output_zip_path: The path to save the output zip file
            output_zip_path: The path to save the output zip file
            **kwargs: Additional arguments passed to create_correction_task_package
        """
        try:
            from natural_pdf.utils.packaging import create_correction_task_package

            create_correction_task_package(source=self, output_zip_path=output_zip_path, **kwargs)
        except ImportError:
            logger.error(
                "Failed to import 'create_correction_task_package'. Packaging utility might be missing."
            )
            logger.error(
                "Failed to import 'create_correction_task_package'. Packaging utility might be missing."
            )
        except Exception as e:
            logger.error(f"Failed to export correction task: {e}")
            raise
            logger.error(f"Failed to export correction task: {e}")
            raise

    def update_text(
        self,
        transform: Callable[[Any], Optional[str]],
        pages: Optional[Union[Iterable[int], range, slice]] = None,
        selector: str = "text",
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> "PDF":
        """
        Applies corrections to text elements using a callback function.

        Args:
            correction_callback: Function that takes an element and returns corrected text or None
            pages: Optional page indices/slice to limit the scope of correction
            selector: Selector to apply corrections to (default: "text")
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
            logger.warning("No pages selected for text update.")
            return self

        logger.info(
            f"Starting text update for pages: {target_page_indices} with selector='{selector}'"
        )

        for page_idx in target_page_indices:
            page = self._pages[page_idx]
            try:
                page.update_text(
                    transform=transform,
                    selector=selector,
                    max_workers=max_workers,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                logger.error(f"Error during text update on page {page_idx}: {e}")
                logger.error(f"Error during text update on page {page_idx}: {e}")

        logger.info("Text update process finished.")
        return self

    def __len__(self) -> int:
        """Return the number of pages in the PDF."""
        if not hasattr(self, "_pages"):
            return 0
        return len(self._pages)

    def __getitem__(self, key) -> Union["Page", "PageCollection"]:
        """Access pages by index or slice."""
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not initialized yet.")

        if isinstance(key, slice):
            from natural_pdf.core.page_collection import PageCollection

            # Use the lazy page list's slicing which returns another _LazyPageList
            lazy_slice = self._pages[key]
            # Wrap in PageCollection for compatibility
            return PageCollection(lazy_slice)
        elif isinstance(key, int):
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
                    # Only unlink if it exists and _is_stream is False (meaning WE created it)
                    if not self._is_stream and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        logger.debug(f"Removed temporary PDF file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file '{temp_file_path}': {e}")

        # Cancels the weakref finalizer so we don't double-clean
        if hasattr(self, "_finalizer") and self._finalizer.alive:
            self._finalizer()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """Return a string representation of the PDF object."""
        if not hasattr(self, "_pages"):
            page_count_str = "uninitialized"
        else:
            page_count_str = str(len(self._pages))

        source_info = getattr(self, "source_path", "unknown source")
        return f"<PDF source='{source_info}' pages={page_count_str}>"

    def get_id(self) -> str:
        """Get unique identifier for this PDF."""
        """Get unique identifier for this PDF."""
        return self.path

    # --- Deskew Method --- #

    def deskew(
        self,
        pages: Optional[Union[Iterable[int], range, slice]] = None,
        resolution: int = 300,
        angle: Optional[float] = None,
        detection_resolution: int = 72,
        force_overwrite: bool = False,
        **deskew_kwargs,
    ) -> "PDF":
        """
        Creates a new, in-memory PDF object containing deskewed versions of the
        specified pages from the original PDF.

        This method renders each selected page, detects and corrects skew using the 'deskew'
        library, and then combines the resulting images into a new PDF using 'img2pdf'.
        The new PDF object is returned directly.

        Important: The returned PDF is image-based. Any existing text, OCR results,
        annotations, or other elements from the original pages will *not* be carried over.

        Args:
            pages: Page indices/slice to include (0-based). If None, processes all pages.
            resolution: DPI resolution for rendering the output deskewed pages.
            angle: The specific angle (in degrees) to rotate by. If None, detects automatically.
            detection_resolution: DPI resolution used for skew detection if angles are not
                                  already cached on the page objects.
            force_overwrite: If False (default), raises a ValueError if any target page
                             already contains processed elements (text, OCR, regions) to
                             prevent accidental data loss. Set to True to proceed anyway.
            **deskew_kwargs: Additional keyword arguments passed to `deskew.determine_skew`
                             during automatic detection (e.g., `max_angle`, `num_peaks`).

        Returns:
            A new PDF object representing the deskewed document.

        Raises:
            ImportError: If 'deskew' or 'img2pdf' libraries are not installed.
            ValueError: If `force_overwrite` is False and target pages contain elements.
            FileNotFoundError: If the source PDF cannot be read (if file-based).
            IOError: If creating the in-memory PDF fails.
            RuntimeError: If rendering or deskewing individual pages fails.
        """
        if not DESKEW_AVAILABLE:
            raise ImportError(
                "Deskew/img2pdf libraries missing. Install with: pip install natural-pdf[deskew]"
            )

        target_pages = self._get_target_pages(pages)  # Use helper to resolve pages

        # --- Safety Check --- #
        if not force_overwrite:
            for page in target_pages:
                # Check if the element manager has been initialized and contains any elements
                if (
                    hasattr(page, "_element_mgr")
                    and page._element_mgr
                    and page._element_mgr.has_elements()
                ):
                    raise ValueError(
                        f"Page {page.number} contains existing elements (text, OCR, etc.). "
                        f"Deskewing creates an image-only PDF, discarding these elements. "
                        f"Set force_overwrite=True to proceed."
                    )

        # --- Process Pages --- #
        deskewed_images_bytes = []
        logger.info(f"Deskewing {len(target_pages)} pages (output resolution={resolution} DPI)...")

        for page in tqdm(target_pages, desc="Deskewing Pages", leave=False):
            try:
                # Use page.deskew to get the corrected PIL image
                # Pass down resolutions and kwargs
                deskewed_img = page.deskew(
                    resolution=resolution,
                    angle=angle,  # Let page.deskew handle detection/caching
                    detection_resolution=detection_resolution,
                    **deskew_kwargs,
                )

                if not deskewed_img:
                    logger.warning(
                        f"Page {page.number}: Failed to generate deskewed image, skipping."
                    )
                    continue

                # Convert image to bytes for img2pdf (use PNG for lossless quality)
                with io.BytesIO() as buf:
                    deskewed_img.save(buf, format="PNG")
                    deskewed_images_bytes.append(buf.getvalue())

            except Exception as e:
                logger.error(
                    f"Page {page.number}: Failed during deskewing process: {e}", exc_info=True
                )
                # Option: Raise a runtime error, or continue and skip the page?
                # Raising makes the whole operation fail if one page fails.
                raise RuntimeError(f"Failed to process page {page.number} during deskewing.") from e

        # --- Create PDF --- #
        if not deskewed_images_bytes:
            raise RuntimeError("No pages were successfully processed to create the deskewed PDF.")

        logger.info(f"Combining {len(deskewed_images_bytes)} deskewed images into in-memory PDF...")
        try:
            # Use img2pdf to combine image bytes into PDF bytes
            pdf_bytes = img2pdf.convert(deskewed_images_bytes)

            # Wrap bytes in a stream
            pdf_stream = io.BytesIO(pdf_bytes)

            # Create a new PDF object from the stream using original config
            logger.info("Creating new PDF object from deskewed stream...")
            new_pdf = PDF(
                pdf_stream,
                reading_order=self._reading_order,
                font_attrs=self._font_attrs,
                keep_spaces=self._config.get("keep_spaces", True),
                text_layer=self._text_layer,
            )
            return new_pdf
        except Exception as e:
            logger.error(f"Failed to create in-memory PDF using img2pdf/PDF init: {e}")
            raise IOError("Failed to create deskewed PDF object from image stream.") from e

    # --- End Deskew Method --- #

    # --- Classification Methods --- #

    def classify_pages(
        self,
        labels: List[str],
        model: Optional[str] = None,
        pages: Optional[Union[Iterable[int], range, slice]] = None,
        analysis_key: str = "classification",
        using: Optional[str] = None,
        **kwargs,
    ) -> "PDF":
        """
        Classifies specified pages of the PDF.

        Args:
            labels: List of category names
            model: Model identifier ('text', 'vision', or specific HF ID)
            pages: Page indices, slice, or None for all pages
            analysis_key: Key to store results in page's analyses dict
            using: Processing mode ('text' or 'vision')
            **kwargs: Additional arguments for the ClassificationManager

        Returns:
            Self for method chaining
        """
        if not labels:
            raise ValueError("Labels list cannot be empty.")

        try:
            manager = self.get_manager("classification")
        except (ValueError, RuntimeError) as e:
            raise ClassificationError(f"Cannot get ClassificationManager: {e}") from e

        if not manager or not manager.is_available():
            from natural_pdf.classification.manager import is_classification_available

            if not is_classification_available():
                raise ImportError(
                    "Classification dependencies missing. "
                    'Install with: pip install "natural-pdf[ai]"'
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
        logger.info(
            f"Classifying {len(target_pages)} pages using model '{model or '(default)'}' (mode: {inferred_using})"
        )

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
                labels=labels,
                model_id=model,
                using=inferred_using,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            raise ClassificationError(f"Batch classification failed: {e}") from e

        if len(batch_results) != len(pages_to_classify):
            logger.error(
                f"Mismatch between number of results ({len(batch_results)}) and pages ({len(pages_to_classify)})"
            )
            return self

        logger.debug(
            f"Distributing {len(batch_results)} results to pages under key '{analysis_key}'..."
        )
        for page, result_obj in zip(pages_to_classify, batch_results):
            try:
                if not hasattr(page, "analyses") or page.analyses is None:
                    page.analyses = {}
                page.analyses[analysis_key] = result_obj
            except Exception as e:
                logger.warning(
                    f"Failed to store classification results for page {page.number}: {e}"
                )

        logger.info(f"Finished classifying PDF pages.")
        return self

    # --- End Classification Methods --- #

    # --- Extraction Support --- #
    def _get_extraction_content(self, using: str = "text", **kwargs) -> Any:
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
        if using == "text":
            try:
                layout = kwargs.pop("layout", True)
                return self.extract_text(layout=layout, **kwargs)
            except Exception as e:
                logger.error(f"Error extracting text from PDF: {e}")
                return None
        elif using == "vision":
            page_images = []
            logger.info(f"Rendering {len(self.pages)} pages to images...")

            resolution = kwargs.pop("resolution", 72)
            include_highlights = kwargs.pop("include_highlights", False)
            labels = kwargs.pop("labels", False)

            try:
                for page in tqdm(self.pages, desc="Rendering Pages"):
                    # Use render() for clean images
                    img = page.render(
                        resolution=resolution,
                        **kwargs,
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

    def _gather_analysis_data(
        self,
        analysis_keys: List[str],
        include_content: bool,
        include_images: bool,
        image_dir: Optional[Path],
        image_format: str,
        image_resolution: int,
    ) -> List[Dict[str, Any]]:
        """
        Gather analysis data from all pages in the PDF.

        Args:
            analysis_keys: Keys in the analyses dictionary to export
            include_content: Whether to include extracted text
            include_images: Whether to export images
            image_dir: Directory to save images
            image_format: Format to save images
            image_resolution: Resolution for exported images

        Returns:
            List of dictionaries containing analysis data
        """
        if not hasattr(self, "_pages") or not self._pages:
            logger.warning(f"No pages found in PDF {self.path}")
            return []

        all_data = []

        for page in tqdm(self._pages, desc="Gathering page data", leave=False):
            # Basic page information
            page_data = {
                "pdf_path": self.path,
                "page_number": page.number,
                "page_index": page.index,
            }

            # Include extracted text if requested
            if include_content:
                try:
                    page_data["content"] = page.extract_text(preserve_whitespace=True)
                except Exception as e:
                    logger.error(f"Error extracting text from page {page.number}: {e}")
                    page_data["content"] = ""

            # Save image if requested
            if include_images:
                try:
                    # Create image filename
                    image_filename = f"pdf_{Path(self.path).stem}_page_{page.number}.{image_format}"
                    image_path = image_dir / image_filename

                    # Save image
                    page.save_image(
                        str(image_path), resolution=image_resolution, include_highlights=True
                    )

                    # Add relative path to data
                    page_data["image_path"] = str(Path(image_path).relative_to(image_dir.parent))
                except Exception as e:
                    logger.error(f"Error saving image for page {page.number}: {e}")
                    page_data["image_path"] = None

            # Add analyses data
            for key in analysis_keys:
                if not hasattr(page, "analyses") or not page.analyses:
                    raise ValueError(f"Page {page.number} does not have analyses data")

                if key not in page.analyses:
                    raise KeyError(f"Analysis key '{key}' not found in page {page.number}")

                # Get the analysis result
                analysis_result = page.analyses[key]

                # If the result has a to_dict method, use it
                if hasattr(analysis_result, "to_dict"):
                    analysis_data = analysis_result.to_dict()
                else:
                    # Otherwise, use the result directly if it's dict-like
                    try:
                        analysis_data = dict(analysis_result)
                    except (TypeError, ValueError):
                        # Last resort: convert to string
                        analysis_data = {"raw_result": str(analysis_result)}

                # Add analysis data to page data with the key as prefix
                for k, v in analysis_data.items():
                    page_data[f"{key}.{k}"] = v

            all_data.append(page_data)

        return all_data

    def _get_target_pages(
        self, pages: Optional[Union[Iterable[int], range, slice]] = None
    ) -> List["Page"]:
        """
        Helper method to get a list of Page objects based on the input pages.

        Args:
            pages: Page indices, slice, or None for all pages

        Returns:
            List of Page objects
        """
        if pages is None:
            return self._pages
        elif isinstance(pages, slice):
            return self._pages[pages]
        elif hasattr(pages, "__iter__"):
            try:
                return [self._pages[i] for i in pages]
            except IndexError:
                raise ValueError("Invalid page index provided in 'pages' iterable.")
            except TypeError:
                raise TypeError("'pages' must be None, a slice, or an iterable of page indices.")
        else:
            raise TypeError("'pages' must be None, a slice, or an iterable of page indices.")

    # --- Classification Mixin Implementation --- #

    def _get_classification_manager(self) -> "ClassificationManager":
        """Returns the ClassificationManager instance for this PDF."""
        try:
            return self.get_manager("classification")
        except (KeyError, RuntimeError) as e:
            raise AttributeError(f"Could not retrieve ClassificationManager: {e}") from e

    def _get_classification_content(self, model_type: str, **kwargs) -> Union[str, Image.Image]:
        """
        Provides the content for classifying the entire PDF.

        Args:
            model_type: 'text' or 'vision'.
            **kwargs: Additional arguments (e.g., for text extraction or image rendering).

        Returns:
            Extracted text (str) or the first page's image (PIL.Image).

        Raises:
            ValueError: If model_type is 'vision' and PDF has != 1 page,
                      or if model_type is unsupported, or if content cannot be generated.
        """
        if model_type == "text":
            try:
                # Extract text from the whole document
                text = self.extract_text(**kwargs)  # Pass relevant kwargs
                if not text or text.isspace():
                    raise ValueError("PDF contains no extractable text for classification.")
                return text
            except Exception as e:
                logger.error(f"Error extracting text for PDF classification: {e}")
                raise ValueError("Failed to extract text for classification.") from e

        elif model_type == "vision":
            if len(self.pages) == 1:
                # Use the single page's content method
                try:
                    return self.pages[0]._get_classification_content(model_type="vision", **kwargs)
                except Exception as e:
                    logger.error(f"Error getting image from single page for classification: {e}")
                    raise ValueError("Failed to get image from single page.") from e
            elif len(self.pages) == 0:
                raise ValueError("Cannot classify empty PDF using vision model.")
            else:
                raise ValueError(
                    f"Vision classification for a PDF object is only supported for single-page PDFs. "
                    f"This PDF has {len(self.pages)} pages. Use pdf.pages[0].classify() or pdf.classify_pages()."
                )
        else:
            raise ValueError(f"Unsupported model_type for PDF classification: {model_type}")

    # --- End Classification Mixin Implementation ---

    # ------------------------------------------------------------------
    # Unified analysis storage (maps to metadata["analysis"])
    # ------------------------------------------------------------------

    @property
    def analyses(self) -> Dict[str, Any]:
        if not hasattr(self, "metadata") or self.metadata is None:
            # For PDF, metadata property returns self._pdf.metadata which may be None
            self._pdf.metadata = self._pdf.metadata or {}
        if self.metadata is None:
            # Fallback safeguard
            self._pdf.metadata = {}
        return self.metadata.setdefault("analysis", {})  # type: ignore[attr-defined]

    @analyses.setter
    def analyses(self, value: Dict[str, Any]):
        if not hasattr(self, "metadata") or self.metadata is None:
            self._pdf.metadata = self._pdf.metadata or {}
        self.metadata["analysis"] = value  # type: ignore[attr-defined]

    # Static helper for weakref.finalize to avoid capturing 'self'
    @staticmethod
    def _finalize_cleanup(plumber_pdf, temp_file_obj, is_stream):
        try:
            if plumber_pdf is not None:
                plumber_pdf.close()
        except Exception:
            pass

        if temp_file_obj and not is_stream:
            try:
                path = temp_file_obj.name if hasattr(temp_file_obj, "name") else None
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file '{path}': {e}")

    def analyze_layout(self, *args, **kwargs) -> "ElementCollection[Region]":
        """
        Analyzes the layout of all pages in the PDF.

        This is a convenience method that calls analyze_layout on the PDF's
        page collection.

        Args:
            *args: Positional arguments passed to pages.analyze_layout().
            **kwargs: Keyword arguments passed to pages.analyze_layout().

        Returns:
            An ElementCollection of all detected Region objects.
        """
        return self.pages.analyze_layout(*args, **kwargs)

    def detect_checkboxes(self, *args, **kwargs) -> "ElementCollection[Region]":
        """
        Detects checkboxes on all pages in the PDF.

        This is a convenience method that calls detect_checkboxes on the PDF's
        page collection.

        Args:
            *args: Positional arguments passed to pages.detect_checkboxes().
            **kwargs: Keyword arguments passed to pages.detect_checkboxes().

        Returns:
            An ElementCollection of all detected checkbox Region objects.
        """
        return self.pages.detect_checkboxes(*args, **kwargs)

    def highlights(self, show: bool = False) -> "HighlightContext":
        """
        Create a highlight context for accumulating highlights.

        This allows for clean syntax to show multiple highlight groups:

        Example:
            with pdf.highlights() as h:
                h.add(pdf.find_all('table'), label='tables', color='blue')
                h.add(pdf.find_all('text:bold'), label='bold text', color='red')
                h.show()

        Or with automatic display:
            with pdf.highlights(show=True) as h:
                h.add(pdf.find_all('table'), label='tables')
                h.add(pdf.find_all('text:bold'), label='bold')
                # Automatically shows when exiting the context

        Args:
            show: If True, automatically show highlights when exiting context

        Returns:
            HighlightContext for accumulating highlights
        """
        from natural_pdf.core.highlighting_service import HighlightContext

        return HighlightContext(self, show_on_exit=show)
