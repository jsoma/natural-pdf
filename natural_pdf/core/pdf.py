import copy
import io
import json
import logging
import os
import ssl
import urllib.request
import weakref
from collections.abc import Iterator, Sequence
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
    Union,
    cast,
    overload,
)

import pdfplumber
from PIL import Image
from pydantic import Field, create_model

if TYPE_CHECKING:
    from typing import Any as _Any

    from pdfplumber.pdf import PDF as PdfPlumberPDF

    def tqdm(*args: _Any, **kwargs: _Any) -> _Any: ...

else:
    from tqdm.auto import tqdm

    PdfPlumberPDF = Any  # type: ignore[assignment]

from natural_pdf.classification.classification_provider import (
    get_classification_engine,
    run_classification_batch,
)
from natural_pdf.classification.manager import ClassificationError
from natural_pdf.core.context import PDFContext
from natural_pdf.core.highlighting_service import HighlightingService
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.elements.region import Region
from natural_pdf.export.mixin import ExportMixin
from natural_pdf.extraction.manager import StructuredDataManager
from natural_pdf.ocr.ocr_manager import (
    normalize_ocr_options,
    resolve_ocr_device,
    resolve_ocr_engine_name,
    resolve_ocr_languages,
    resolve_ocr_min_confidence,
)
from natural_pdf.qa.qa_result import QAResult
from natural_pdf.search import (
    BaseSearchOptions,
    SearchOptions,
    TextSearchOptions,
    get_search_service,
)
from natural_pdf.search.search_service_protocol import SearchServiceProtocol
from natural_pdf.services.base import ServiceHostMixin, resolve_service
from natural_pdf.services.delegates import attach_capability

if TYPE_CHECKING:
    from natural_pdf.core.highlighting_service import HighlightContext
    from natural_pdf.core.page import Page
    from natural_pdf.core.page_collection import PageCollection
    from natural_pdf.elements.element_collection import ElementCollection

try:
    import certifi  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    certifi = None  # type: ignore[assignment]

CreateSearchablePdfFn = Callable[..., None]
CreateOriginalPdfFn = Callable[..., None]

try:
    from natural_pdf.exporters.searchable_pdf import (
        create_searchable_pdf as _create_searchable_pdf_impl,
    )
except ImportError:
    _create_searchable_pdf_impl = None  # type: ignore[assignment]

try:
    from natural_pdf.exporters.original_pdf import create_original_pdf as _create_original_pdf_impl
except ImportError:
    _create_original_pdf_impl = None  # type: ignore[assignment]

create_searchable_pdf: Optional[CreateSearchablePdfFn] = cast(
    Optional[CreateSearchablePdfFn], _create_searchable_pdf_impl
)
create_original_pdf: Optional[CreateOriginalPdfFn] = cast(
    Optional[CreateOriginalPdfFn], _create_original_pdf_impl
)

logger = logging.getLogger("natural_pdf.core.pdf")


ManagerFactory = Union[Callable[[], Any], type[Any]]
ManagerFactories = Dict[str, ManagerFactory]
ManagerCache = Dict[str, Any]
ExclusionSpec = Tuple[Any, Optional[str]]
RegionFactory = Callable[["Page"], Optional["Region"]]
RegionRegistry = List[Tuple[RegionFactory, Optional[str]]]

DEFAULT_MANAGERS: ManagerFactories = {
    "structured_data": StructuredDataManager,
}

DEFAULT_GENERATIVE_QA_PROMPT = (
    "You answer questions about the supplied document content. "
    "Respond using JSON matching the schema. Populate the 'answer' field with a concise reply; "
    "if the answer cannot be determined, set 'answer' to an empty string."
)

# Deskew Imports (Conditional)
try:
    import img2pdf  # type: ignore[import]

    DESKEW_AVAILABLE = True
except ImportError:
    DESKEW_AVAILABLE = False
    img2pdf = None
# End Deskew Imports

# --- Lazy Page List Helper --- #


class _LazyPageList(Sequence["Page"]):
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
        plumber_pdf: "PdfPlumberPDF",
        font_attrs: Optional[List[str]] = None,
        load_text: bool = True,
        indices: Optional[List[int]] = None,
    ):
        self._parent_pdf: "PDF" = parent_pdf
        self._plumber_pdf: "PdfPlumberPDF" = plumber_pdf
        self._font_attrs: Optional[List[str]] = font_attrs
        self._load_text: bool = load_text

        if indices is not None:
            self._indices: List[int] = indices
            self._cache: List[Optional["Page"]] = [None] * len(indices)
        else:
            self._indices = list(range(len(plumber_pdf.pages)))
            self._cache = [None] * len(plumber_pdf.pages)

    # Internal helper -----------------------------------------------------
    def _create_page(self, index: int) -> "Page":
        """Create and cache a page at the given index within this list."""
        cached: Optional["Page"] = self._cache[index]
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
                return cast("Page", cached)

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
                context=self._parent_pdf._context,
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
                if actual_page_index < len(parent_cache):
                    existing_page = parent_cache[actual_page_index]
                    if existing_page is not None and getattr(existing_page, "_exclusions", None):
                        for exclusion_data in existing_page._exclusions:
                            exclusion_item = exclusion_data[0]
                            if not callable(exclusion_item):
                                try:
                                    cached.add_exclusion(*exclusion_data[:2])
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

        return cast("Page", cached)

    # Sequence protocol ---------------------------------------------------
    def __len__(self) -> int:
        return len(self._cache)

    @overload
    def __getitem__(self, key: int) -> "Page": ...

    @overload
    def __getitem__(self, key: slice) -> "_LazyPageList": ...

    def __getitem__(self, key: Union[int, slice]) -> Union["Page", "_LazyPageList"]:
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

    def __iter__(self) -> Iterator["Page"]:
        for i in range(len(self)):
            yield self._create_page(i)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<_LazyPageList(len={len(self)})>"


# --- End Lazy Page List Helper --- #


class PDF(
    ServiceHostMixin,
    ExportMixin,
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

    @classmethod
    def from_images(
        cls,
        images: Union["Image.Image", List["Image.Image"], str, List[str], Path, List[Path]],
        resolution: int = 300,
        apply_ocr: bool = True,
        ocr_engine: Optional[str] = None,
        **pdf_options,
    ) -> "PDF":
        """Create a PDF from image(s).

        Args:
            images: Single image, list of images, or path(s)/URL(s) to image files
            resolution: DPI for the PDF (default: 300, good for OCR and viewing)
            apply_ocr: Apply OCR to make searchable (default: True)
            ocr_engine: OCR engine to use (default: auto-detect)
            **pdf_options: Options passed to PDF constructor

        Returns:
            PDF object containing the images as pages

        Example:
            ```python
            # Simple scan to searchable PDF
            pdf = PDF.from_images("scan.jpg")

            # From URL
            pdf = PDF.from_images("https://example.com/image.png")

            # Multiple pages (mix of local and URLs)
            pdf = PDF.from_images(["page1.png", "https://example.com/page2.jpg"])

            # Without OCR
            pdf = PDF.from_images(images, apply_ocr=False)

            # With specific engine
            pdf = PDF.from_images(images, ocr_engine='surya')
            ```
        """
        import urllib.request

        from PIL import ImageOps

        def _open_image(source: Union[Image.Image, str, Path]) -> Image.Image:
            """Open an image from file path, URL, or return PIL Image as-is."""
            if isinstance(source, Image.Image):
                return source

            source_str = str(source)
            if source_str.startswith(("http://", "https://")):
                # Download from URL
                with urllib.request.urlopen(source_str) as response:
                    img_data = response.read()
                return Image.open(io.BytesIO(img_data))
            else:
                # Local file path
                return Image.open(source)

        # Normalize inputs to list of PIL Images
        if isinstance(images, (str, Path)):
            image_list = [_open_image(images)]
        elif isinstance(images, Image.Image):
            image_list = [images]
        elif isinstance(images, list):
            image_list = [_open_image(item) for item in images]
        else:
            raise TypeError(
                "images must be a path, PIL Image, or a list of paths/PIL Image instances."
            )

        # Process images
        processed_images: List[Image.Image] = []
        for img in image_list:
            # Fix EXIF rotation
            img = ImageOps.exif_transpose(img) or img

            # Convert RGBA to RGB (PDF doesn't handle transparency well)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, "white")
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode not in ["RGB", "L", "1", "CMYK"]:
                img = img.convert("RGB")

            processed_images.append(img)

        # Create PDF at specified resolution
        # Use BytesIO to keep in memory
        pdf_buffer = io.BytesIO()
        processed_images[0].save(
            pdf_buffer,
            "PDF",
            save_all=True,
            append_images=processed_images[1:] if len(processed_images) > 1 else [],
            resolution=resolution,
        )
        pdf_buffer.seek(0)

        # Create PDF object
        pdf = cls(pdf_buffer, **pdf_options)

        # Store metadata about source
        pdf._from_images = True
        pdf._source_metadata = {
            "type": "images",
            "count": len(processed_images),
            "resolution": resolution,
        }

        # Apply OCR if requested
        if apply_ocr:
            pdf.apply_ocr(engine=ocr_engine, resolution=resolution)

        return pdf

    def __init__(
        self,
        path_or_url_or_stream,
        reading_order: bool = True,
        font_attrs: Optional[List[str]] = None,
        keep_spaces: bool = True,
        text_tolerance: Optional[dict] = None,
        auto_text_tolerance: bool = True,
        text_layer: bool = True,
        context: Optional[PDFContext] = None,
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
        self._context = context or PDFContext.with_defaults()
        self._init_service_host(self._context)

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
                    ssl_context = None
                    try:
                        if certifi is not None:
                            ssl_context = ssl.create_default_context(cafile=certifi.where())
                        else:
                            ssl_context = ssl.create_default_context()
                    except Exception:
                        ssl_context = ssl.create_default_context()

                    with urllib.request.urlopen(path_or_url, context=ssl_context) as response:
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

        # Deprecated managers remain for backwards compatibility but are no longer instantiated.
        self._layout_manager = None

        self.highlighter: HighlightingService = HighlightingService(self)
        self._manager_factories: ManagerFactories = {}
        self._managers: ManagerCache = {}

        # Lazily instantiate pages only when accessed
        self._pages: _LazyPageList = _LazyPageList(
            self, self._pdf, font_attrs=font_attrs, load_text=self._text_layer
        )

        self._element_cache: Dict[str, Any] = {}
        self._exclusions: List[ExclusionSpec] = []
        self._regions: RegionRegistry = []
        self._from_images: bool = False
        self._source_metadata: Optional[Dict[str, Any]] = None

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
                "y_tolerance_ratio",
                "keep_blank_chars",  # passthrough convenience
            }
            for k, v in text_tolerance.items():
                if k in allowed:
                    self._config[k] = v

    def _initialize_managers(self) -> None:
        """Set up manager factories for lazy instantiation."""
        # Store factories/classes for each manager key
        self._manager_factories = dict(DEFAULT_MANAGERS)
        self._managers = {}

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
        if not hasattr(self, "_pdf") or self._pdf is None:
            return {}
        meta = getattr(self._pdf, "metadata", None)
        if meta is None:
            return {}
        if isinstance(meta, dict):
            return cast(Dict[str, Any], meta)
        # pdfplumber stores metadata as a custom object exposing dict-like interface.
        try:
            return dict(meta)
        except TypeError:
            logger.debug("Unable to coerce PDF metadata to dict; returning empty metadata.")
            return {}

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
        return PageCollection(cast(Sequence["Page"], self._pages), context=self._context)

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
            cached_page = self._pages._cache[i]
            if cached_page is None:
                continue
            try:
                cached_page.clear_exclusions()
            except Exception as e:
                logger.warning(f"Failed to clear exclusions from existing page {i}: {e}")
        return self

    def add_exclusion(self, exclusion_func, label: Optional[str] = None) -> "PDF":
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
        normalized_options = normalize_ocr_options(options)
        engine_name = resolve_ocr_engine_name(
            context=self,
            requested=engine,
            options=normalized_options,
            scope="pdf",
        )
        resolved_languages = resolve_ocr_languages(self, languages, scope="pdf")
        resolved_min_confidence = resolve_ocr_min_confidence(self, min_confidence, scope="pdf")
        resolved_device = resolve_ocr_device(self, device, scope="pdf")

        target_pages = self._get_target_pages(pages)
        if not target_pages:
            logger.warning("No pages selected for OCR processing.")
            return self

        final_resolution = resolution or self._config.get("resolution", 150)
        logger.info(
            "Applying OCR to %d page(s) with engine '%s' at %s DPI.",
            len(target_pages),
            engine_name,
            final_resolution,
        )

        for page in tqdm(target_pages, desc="Applying OCR", leave=False):
            page.apply_ocr(
                engine=engine_name,
                options=normalized_options,
                languages=resolved_languages,
                min_confidence=resolved_min_confidence,
                device=resolved_device,
                resolution=final_resolution,
                detect_only=detect_only,
                apply_exclusions=apply_exclusions,
                replace=replace,
            )

        return self

    def add_region(
        self, region_func: Callable[["Page"], Optional["Region"]], name: Optional[str] = None
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
            cached_page = self._pages._cache[i]
            if cached_page is None:
                continue
            try:
                region_instance = region_func(cached_page)
                if region_instance and isinstance(region_instance, Region):
                    cached_page.add_region(region_instance, name=name, source="named")
                elif region_instance is not None:
                    logger.warning(
                        f"Region function did not return a valid Region for page {cached_page.number}"
                    )
            except Exception as e:
                logger.error(f"Error adding region for page {cached_page.number}: {e}")

        return self

    def find(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        overlap: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
        engine: Optional[str] = None,
    ) -> Optional[Any]:
        """Return the first element across all pages that matches selector/text filters.

        Args:
            selector: CSS-like selector string.
            text: Text shortcut equivalent to ``text:contains(...)``.
            overlap: Present for API parity; ignored for full-document searches.
            apply_exclusions: Whether exclusion regions should be honoured.
            regex: Whether selector/text filters use regular expressions.
            case: Whether text comparisons are case-sensitive.
            text_tolerance: Optional pdfplumber-style tolerance overrides applied temporarily.
            auto_text_tolerance: Optional overrides for automatic tolerance behaviour.
            reading_order: Whether matches use document reading order.
            near_threshold: Maximum distance (in points) used by ``:near`` selectors.
            engine: Optional selector engine registered with the provider.

        Returns:
            The first matching :class:`natural_pdf.elements.base.Element`, if any.
        """
        return resolve_service(self, "selector").find(
            self,
            selector=selector,
            text=text,
            overlap=overlap,
            apply_exclusions=apply_exclusions,
            regex=regex,
            case=case,
            text_tolerance=text_tolerance,
            auto_text_tolerance=auto_text_tolerance,
            reading_order=reading_order,
            near_threshold=near_threshold,
            engine=engine,
        )

    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        overlap: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
        engine: Optional[str] = None,
    ) -> "ElementCollection":
        """Return every element across all pages that satisfies selector/text filters.

        Args:
            selector: CSS-like selector string.
            text: Text shortcut equivalent to ``text:contains(...)``.
            overlap: Present for API parity; ignored for full-document searches.
            apply_exclusions: Whether exclusion regions should be honoured.
            regex: Whether selector/text filters use regular expressions.
            case: Whether text comparisons are case-sensitive.
            text_tolerance: Optional pdfplumber-style tolerance overrides applied temporarily.
            auto_text_tolerance: Optional overrides for automatic tolerance behaviour.
            reading_order: Whether matches use document reading order.
            near_threshold: Maximum distance (in points) used by ``:near`` selectors.
            engine: Optional selector engine registered with the provider.

        Returns:
            :class:`natural_pdf.elements.element_collection.ElementCollection` of matches.
        """
        return resolve_service(self, "selector").find_all(
            self,
            selector=selector,
            text=text,
            overlap=overlap,
            apply_exclusions=apply_exclusions,
            regex=regex,
            case=case,
            text_tolerance=text_tolerance,
            auto_text_tolerance=auto_text_tolerance,
            reading_order=reading_order,
            near_threshold=near_threshold,
            engine=engine,
        )

    def extract_text(
        self,
        selector: Optional[str] = None,
        preserve_whitespace: bool = True,
        preserve_line_breaks: bool = True,
        page_separator: Optional[str] = "\n",
        use_exclusions: bool = True,
        debug_exclusions: bool = False,
        *,
        layout: bool = True,
        x_density: Optional[float] = None,
        y_density: Optional[float] = None,
        x_tolerance: Optional[float] = None,
        y_tolerance: Optional[float] = None,
        line_dir: Optional[str] = None,
        char_dir: Optional[str] = None,
        strip_final: bool = False,
        strip_empty: bool = False,
    ) -> str:
        """
        Extract text from the entire document or matching elements.

        Args:
            selector: Optional selector to filter elements
            preserve_whitespace: Whether to keep blank characters
            preserve_line_breaks: When False, collapse newlines in each page's text.
            page_separator: String inserted between page texts when combining results.
            use_exclusions: Whether to apply exclusion regions
            debug_exclusions: Whether to output detailed debugging for exclusions
            preserve_whitespace: Whether to keep blank characters
            use_exclusions: Whether to apply exclusion regions
            debug_exclusions: Whether to output detailed debugging for exclusions
            layout: Whether to enable layout-aware spacing (default: True).
            x_density: Horizontal character density override.
            y_density: Vertical line density override.
            x_tolerance: Horizontal clustering tolerance.
            y_tolerance: Vertical clustering tolerance.
            line_dir: Line reading direction override.
            char_dir: Character reading direction override.
            strip_final: When True, strip trailing whitespace from the combined text.
            strip_empty: When True, drop empty lines from the output.

        Returns:
            Extracted text as string
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        if selector:
            elements = self.find_all(
                selector,
                apply_exclusions=use_exclusions,
                layout=layout,
                x_density=x_density,
                y_density=y_density,
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
                line_dir=line_dir,
                char_dir=char_dir,
                strip_final=strip_final,
                strip_empty=strip_empty,
            )
            return elements.extract_text(
                preserve_whitespace=preserve_whitespace,
                preserve_line_breaks=preserve_line_breaks,
                layout=layout,
                x_density=x_density,
                y_density=y_density,
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
                line_dir=line_dir,
                char_dir=char_dir,
                strip_final=strip_final,
                strip_empty=strip_empty,
            )

        if debug_exclusions:
            print(f"PDF: Extracting text with exclusions from {len(self.pages)} pages")
            print(f"PDF: Found {len(self._exclusions)} document-level exclusions")

        texts = []
        for page in self.pages:
            texts.append(
                page.extract_text(
                    preserve_whitespace=preserve_whitespace,
                    preserve_line_breaks=preserve_line_breaks,
                    use_exclusions=use_exclusions,
                    debug_exclusions=debug_exclusions,
                    layout=layout,
                    x_density=x_density,
                    y_density=y_density,
                    x_tolerance=x_tolerance,
                    y_tolerance=y_tolerance,
                    line_dir=line_dir,
                    char_dir=char_dir,
                    strip_final=strip_final,
                    strip_empty=strip_empty,
                )
            )

        if debug_exclusions:
            print(f"PDF: Combined {len(texts)} pages of text")

        separator = "" if page_separator is None else page_separator
        return separator.join(texts)

    def extract_tables(
        self,
        selector: Optional[str] = None,
        merge_across_pages: bool = False,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        check_tatr: bool = True,
    ) -> List[Any]:
        """
        Extract tables from the document or matching elements.

        Args:
            selector: Optional selector to filter tables (not yet implemented).
            merge_across_pages: Whether to merge tables that span across pages (not yet implemented).
            method: Extraction strategy to prefer. Mirrors ``Page.extract_tables``.
            table_settings: Per-method configuration forwarded to ``Page.extract_tables``.
            check_tatr: When True, visit TATR table regions before falling back to page extraction.

        Returns:
            List of extracted tables
        """
        if not hasattr(self, "_pages"):
            raise AttributeError("PDF pages not yet initialized.")

        logger.warning("PDF.extract_tables is not fully implemented yet.")
        all_tables = []

        for page in self.pages:
            if hasattr(page, "extract_tables"):
                all_tables.extend(
                    page.extract_tables(
                        method=method,
                        table_settings=table_settings,
                        check_tatr=check_tatr,
                    )
                )
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

    def split(
        self,
        divider,
        *,
        include_boundaries: str = "start",
        orientation: str = "vertical",
        new_section_on_page_break: bool = False,
    ) -> "ElementCollection":
        """
        Divide the PDF into sections based on the provided divider elements.

        Args:
            divider: Elements or selector string that mark section boundaries
            include_boundaries: How to include boundary elements (default: 'start').
            orientation: 'vertical' or 'horizontal' (default: 'vertical').
            new_section_on_page_break: Whether to split at page boundaries (default: False).

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
        return self.pages.split(
            divider,
            include_boundaries=include_boundaries,
            orientation=orientation,
            new_section_on_page_break=new_section_on_page_break,
        )

    def save_searchable(self, output_path: Union[str, "Path"], dpi: int = 300):
        """
        DEPRECATED: Use save_pdf(..., ocr=True) instead.
        Saves the PDF with an OCR text layer, making content searchable.

        Requires optional dependencies. Install with: pip install \"natural-pdf[ocr-export]\"

        Args:
            output_path: Path to save the searchable PDF
            dpi: Resolution for rendering and OCR overlay.
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
        create_searchable_pdf(self, output_path_str, dpi=dpi)
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
            if create_searchable_pdf is None:
                raise ImportError(
                    "Saving with ocr=True requires the OCR export dependencies. "
                    'Install with: pip install "natural-pdf[ocr-export]"'
                )
            has_vector_elements = False
            for page in self.pages:
                rects = getattr(page, "rects", None)
                lines = getattr(page, "lines", None)
                curves = getattr(page, "curves", None)
                chars = getattr(page, "chars", None)
                words = getattr(page, "words", None)
                if (
                    (rects and len(rects) > 0)  # type: ignore[arg-type]
                    or (lines and len(lines) > 0)  # type: ignore[arg-type]
                    or (curves and len(curves) > 0)  # type: ignore[arg-type]
                    or (
                        chars
                        and any(
                            getattr(el, "source", None) != "ocr"
                            for el in cast(Iterable[Any], chars)
                        )
                    )
                    or (
                        words
                        and any(
                            getattr(el, "source", None) != "ocr"
                            for el in cast(Iterable[Any], words)
                        )
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

    def _resolve_qa_pages(self, pages: Optional[Union[int, Iterable[int], range]]) -> List["Page"]:
        """
        Normalize the ``pages`` argument for QA operations into a list of Page objects.

        Args:
            pages: None for all pages, an integer page index, or an iterable of indices.

        Returns:
            List of resolved Page objects (may be empty if no valid indices were supplied).
        """
        if pages is None:
            return list(self.pages)

        total_pages = len(self.pages)

        if isinstance(pages, int):
            if 0 <= pages < total_pages:
                return [cast("Page", self.pages[pages])]
            raise IndexError(f"Page index {pages} out of range (0-{total_pages-1})")

        if isinstance(pages, range) or isinstance(pages, list) or isinstance(pages, tuple):
            resolved: List["Page"] = []
            for page_idx in pages:
                if isinstance(page_idx, int) and 0 <= page_idx < total_pages:
                    resolved.append(cast("Page", self.pages[page_idx]))
                else:
                    logger.warning(f"Page index {page_idx} out of range, skipping")
            return resolved

        raise ValueError(f"Invalid pages parameter: {pages}")

    def ask(
        self,
        question: str,
        *,
        mode: Literal["extractive", "generative"] = "extractive",
        pages: Optional[Union[int, Iterable[int], range]] = None,
        min_confidence: float = 0.1,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
    ) -> QAResult:
        """
        Ask a single question about the document content.

        Args:
            question: Question string to ask about the document
            mode: "extractive" to extract answer from document, "generative" to generate
            pages: Specific pages to query (default: all pages)
            min_confidence: Minimum confidence threshold for answers (extractive mode).
            model: Optional model name for the QA engine or LLM.
            temperature: Optional sampling temperature for LLM-backed QA.
            top_p: Optional nucleus sampling parameter for LLM-backed QA.
            llm_client: Client instance to use when ``mode="generative"``.
            prompt: Optional system prompt override for generative QA.

        Returns:
            :class:`QAResult` containing answer metadata. Confidence may be ``None`` in generative
            mode; ``source_elements`` may be empty if no span is available.
        """
        # Delegate to ask_batch and return the first result
        results = self.ask_batch(
            [question],
            mode=mode,
            pages=pages,
            min_confidence=min_confidence,
            model=model,
            temperature=temperature,
            top_p=top_p,
            llm_client=llm_client,
            prompt=prompt,
        )
        if results:
            return results[0]
        return QAResult(
            {
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
        *,
        mode: Literal["extractive", "generative"] = "extractive",
        pages: Optional[Union[int, Iterable[int], range]] = None,
        min_confidence: float = 0.1,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
    ) -> List[QAResult]:
        """
        Ask multiple questions about the document content using batch processing.

        This method processes multiple questions efficiently in a single batch,
        avoiding the multiprocessing resource accumulation that can occur with
        sequential individual question calls.

        Args:
            questions: List of question strings to ask about the document
            mode: "extractive" to extract answer from document, "generative" to generate
            pages: Specific pages to query (default: all pages)
            min_confidence: Minimum confidence threshold for extractive answers.
            model: Optional model name for the QA engine or LLM.
            temperature: Optional sampling temperature for LLM-backed QA.
            top_p: Optional nucleus sampling parameter for LLM-backed QA.
            llm_client: Client instance to use when ``mode="generative"``.
            prompt: Optional system prompt override for generative QA.

        Returns:
            List of :class:`QAResult` objects containing answer metadata. Confidence may be
            ``None`` in generative mode; ``source_elements`` may be empty if no span is available.
        """
        if not questions:
            return []

        if mode not in ("extractive", "generative"):
            raise ValueError("mode must be either 'extractive' or 'generative'")

        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            raise TypeError("'questions' must be a list of strings")

        target_pages = self._resolve_qa_pages(pages)

        def _empty_result() -> QAResult:
            return QAResult(
                {
                    "answer": None,
                    "confidence": 0.0,
                    "found": False,
                    "page_num": None,
                    "source_elements": [],
                }
            )

        if not target_pages:
            logger.warning("No valid pages found for QA processing.")
            return [_empty_result() for _ in questions]

        if mode == "generative":
            return self._ask_batch_generative(
                questions,
                target_pages=target_pages,
                llm_client=llm_client,
                prompt=prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
            )

        if temperature is not None or top_p is not None:
            logger.info(
                "temperature/top_p parameters are only honored in generative mode; ignoring them for extractive QA."
            )

        from natural_pdf.core.page_collection import PageCollection

        page_collection = PageCollection(target_pages, context=self._context)
        qa_result = page_collection.ask(
            questions,
            min_confidence=min_confidence,
            model=model,
        )

        normalized = qa_result if isinstance(qa_result, list) else [qa_result]
        return [
            QAResult(result) if not isinstance(result, QAResult) else result
            for result in normalized
        ]

    def _ask_batch_generative(
        self,
        questions: List[str],
        *,
        target_pages: List["Page"],
        llm_client: Optional[Any],
        prompt: Optional[str],
        model: Optional[str],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> List[QAResult]:
        """Handle the ``mode='generative'`` logic for :meth:`ask_batch`."""

        def _empty_result() -> QAResult:
            return QAResult(
                {
                    "answer": None,
                    "confidence": 0.0,
                    "found": False,
                    "page_num": None,
                    "source_elements": [],
                }
            )

        if llm_client is None:
            raise ValueError("Generative QA requires 'llm_client' to be provided.")

        try:
            manager = self.get_manager("structured_data")
        except (KeyError, RuntimeError) as exc:
            raise RuntimeError(
                "StructuredDataManager is not available; cannot run generative QA."
            ) from exc

        if manager is None or not manager.is_available():
            raise RuntimeError("StructuredDataManager is not available for generative QA.")

        # Compile text content from selected pages
        page_sections: List[str] = []
        for page in target_pages:
            page_text = ""
            try:
                page_text = page.extract_text(layout=True) or ""
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Failed to extract layout text from page %s: %s",
                    getattr(page, "number", "?"),
                    exc,
                )

            if not page_text.strip():
                try:
                    page_text = page.extract_text(layout=False) or ""
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Failed to extract text (layout=False) from page %s: %s",
                        getattr(page, "number", "?"),
                        exc,
                    )

            page_text = page_text.strip()
            if page_text:
                page_sections.append(f"Page {page.number}:\n{page_text}")

        combined_text = "\n\n".join(page_sections).strip()
        if not combined_text:
            logger.warning(
                "Generative QA could not extract text from the selected pages. "
                "Consider running apply_ocr() before using mode='generative'."
            )
            return [_empty_result() for _ in questions]

        max_chars = 20000
        if len(combined_text) > max_chars:
            logger.info(
                "Truncating combined page text for generative QA from %d to %d characters",
                len(combined_text),
                max_chars,
            )
            combined_text = combined_text[:max_chars]

        GenerativeQAResponse = create_model(
            "GenerativeQAResponse",
            answer=(str, Field("", description="Answer to the question; leave empty if unknown.")),
        )

        llm_kwargs: Dict[str, Any] = {}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if top_p is not None:
            llm_kwargs["top_p"] = top_p

        prompt_text = prompt or DEFAULT_GENERATIVE_QA_PROMPT
        results: List[QAResult] = []

        for question_text in questions:
            question_text = question_text.strip()
            if not question_text:
                results.append(_empty_result())
                continue

            payload = json.dumps(
                {
                    "question": question_text,
                    "document": combined_text,
                }
            )

            try:
                extraction_result = manager.extract(
                    content=payload,
                    schema=GenerativeQAResponse,
                    client=llm_client,
                    prompt=prompt_text,
                    using="text",
                    model=model,
                    **llm_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Generative QA failed for question %r: %s", question_text, exc)
                results.append(_empty_result())
                continue

            parsed = getattr(extraction_result, "data", None)
            if not extraction_result.success or parsed is None:
                logger.debug(
                    "Generative QA returned no parsed data for question %r (success=%s, error=%s)",
                    question_text,
                    getattr(extraction_result, "success", None),
                    getattr(extraction_result, "error_message", None),
                )
                results.append(_empty_result())
                continue

            answer = getattr(parsed, "answer", "")
            normalized_answer = answer.strip() if isinstance(answer, str) else ""

            found = bool(normalized_answer)

            if not found:
                results.append(_empty_result())
                continue

            results.append(
                QAResult(
                    {
                        "answer": normalized_answer,
                        "confidence": None,
                        "found": True,
                        "page_num": None,
                        "source_elements": [],
                    }
                )
            )

        return results

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
            logger.debug("Combining PDF scope filter with existing filters")
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
                    "Unsupported format for existing filters. Overwriting with PDF scope filter."
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
            raise RuntimeError("Search within index failed. See logs for details.") from e
            logger.error(f"SearchService search failed: {e}")
            raise RuntimeError("Search within index failed. See logs for details.") from e

    def export_ocr_correction_task(
        self,
        output_zip_path: str,
        *,
        overwrite: bool = False,
        suggest=None,
        resolution: int = 300,
    ):
        """
        Exports OCR results from this PDF into a correction task package.
        Exports OCR results from this PDF into a correction task package.

        Args:
            output_zip_path: The path to save the output zip file.
            overwrite: When True, replace any existing archive at ``output_zip_path``.
            suggest: Optional callable that can provide OCR suggestions per region.
            resolution: DPI used when rendering page images for the package.
        """
        try:
            from natural_pdf.utils.packaging import create_correction_task_package

            create_correction_task_package(
                source=self,
                output_zip_path=output_zip_path,
                overwrite=overwrite,
                suggest=suggest,
                resolution=resolution,
            )
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
        *,
        selector: str = "text",
        apply_exclusions: bool = False,
        pages: Optional[Union[Iterable[int], range, slice]] = None,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> "PDF":
        """
        Applies corrections to text elements using a callback function.

        Args:
            transform: Function that takes an element and returns corrected text or None
            selector: Selector to apply corrections to (default: "text")
            apply_exclusions: Whether to honour exclusion regions while selecting text.
            pages: Optional page indices/slice to limit the scope of correction
            max_workers: Maximum number of threads to use for parallel execution
            progress_callback: Optional callback function for progress updates

        Returns:
            Self for method chaining
        """
        pages_to_update = self._get_target_pages(pages)

        if not pages_to_update:
            logger.warning("No pages selected for text update.")
            return self

        page_identities = [page.index for page in pages_to_update]
        logger.info(f"Starting text update for pages: {page_identities} with selector='{selector}'")

        for page in pages_to_update:
            try:
                page.update_text(
                    transform=transform,
                    selector=selector,
                    apply_exclusions=apply_exclusions,
                    max_workers=max_workers,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                logger.error(f"Error during text update on page {page.index}: {e}")
                logger.error(f"Error during text update on page {page.index}: {e}")

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
            **deskew_kwargs: Additional keyword arguments forwarded to the deskew engine.
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
                if hasattr(page, "has_element_cache") and page.has_element_cache():
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
            if img2pdf is None:
                raise RuntimeError(
                    "img2pdf library is not available despite DESKEW_AVAILABLE flag being set."
                )
            pdf_bytes = img2pdf.convert(deskewed_images_bytes)
            if pdf_bytes is None:
                raise RuntimeError("img2pdf.convert returned no data.")

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
        min_confidence: float = 0.0,
        multi_label: bool = False,
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

        target_pages = self._get_target_pages(pages)

        if not target_pages:
            logger.warning("No pages selected for classification.")
            return self

        engine_name = kwargs.pop("classification_engine", None)
        engine_obj = get_classification_engine(self, engine_name)
        inferred_using = engine_obj.infer_using(model or engine_obj.default_model("text"), using)
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
            batch_results = run_classification_batch(
                context=self,
                contents=page_contents,
                labels=labels,
                model_id=model or engine_obj.default_model(inferred_using),
                using=inferred_using,
                min_confidence=min_confidence,
                multi_label=multi_label,
                batch_size=8,
                progress_bar=True,
                engine_name=engine_name,
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

        logger.info("Finished classifying PDF pages.")
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
            kwargs.pop("include_highlights", False)
            kwargs.pop("labels", False)

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
        image_dir_path = Path(image_dir) if image_dir is not None else None
        if include_images and image_dir_path is None:
            raise ValueError("'image_dir' must be provided when include_images=True.")

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
            if include_images and image_dir_path is not None:
                try:
                    # Create image filename
                    source_stem = Path(self.path).stem if isinstance(self.path, str) else "pdf"
                    image_filename = f"pdf_{source_stem}_page_{page.number}.{image_format}"
                    image_path = image_dir_path / image_filename

                    # Save image
                    page.save_image(
                        str(image_path), resolution=image_resolution, include_highlights=True
                    )

                    # Add relative path to data
                    page_data["image_path"] = str(
                        Path(image_path).relative_to(image_dir_path.parent)
                    )
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
        self, pages: Optional[Union[int, Iterable[int], range, slice]] = None
    ) -> List["Page"]:
        """
        Helper method to get a list of Page objects based on the input pages.

        Args:
            pages: Page index/int, iterable of indices, slice, or None for all pages

        Returns:
            List of Page objects
        """
        if pages is None:
            return list(self._pages)
        if isinstance(pages, int):
            total = len(self._pages)
            idx = pages if pages >= 0 else pages + total
            if not (0 <= idx < total):
                raise IndexError(f"Page index {pages} out of range (0-{total-1}).")
            return [cast("Page", self._pages[idx])]
        if isinstance(pages, slice):
            return list(self._pages[pages])
        if isinstance(pages, range):
            indices = list(pages)
        elif isinstance(pages, Iterable):
            try:
                indices = [int(i) for i in pages]
            except (TypeError, ValueError) as exc:
                raise TypeError("'pages' iterable must contain integer indices.") from exc
        else:
            raise TypeError(
                "'pages' must be None, an int, a slice, or an iterable of page indices."
            )

        resolved: List["Page"] = []
        total = len(self._pages)
        for idx in indices:
            actual_idx = idx if idx >= 0 else idx + total
            if not (0 <= actual_idx < total):
                raise IndexError(f"Page index {idx} out of range (0-{total-1}).")
            resolved.append(cast("Page", self._pages[actual_idx]))
        return resolved

    # --- Classification Mixin Implementation --- #

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
                    single_page = cast("Page", self.pages[0])
                    return single_page._get_classification_content(model_type="vision", **kwargs)
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
        plumber_pdf = getattr(self, "_pdf", None)
        if plumber_pdf is None:
            raise RuntimeError("Underlying pdfplumber PDF is not initialized.")

        metadata = getattr(plumber_pdf, "metadata", None)
        if metadata is None:
            metadata = {}
            plumber_pdf.metadata = metadata

        analysis_entry = metadata.setdefault("analysis", {})
        if not isinstance(analysis_entry, dict):
            raise TypeError("PDF metadata 'analysis' entry must be a dictionary")

        return cast(Dict[str, Any], analysis_entry)

    @analyses.setter
    def analyses(self, value: Dict[str, Any]):
        plumber_pdf = getattr(self, "_pdf", None)
        if plumber_pdf is None:
            raise RuntimeError("Underlying pdfplumber PDF is not initialized.")

        metadata = getattr(plumber_pdf, "metadata", None)
        if metadata is None:
            metadata = {}
            plumber_pdf.metadata = metadata
        metadata["analysis"] = value

    # Static helper for weakref.finalize to avoid capturing 'self'
    @staticmethod
    def _finalize_cleanup(plumber_pdf, temp_file_obj, is_stream):
        path: Optional[str] = None
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
        return resolve_service(self, "layout").analyze_layout(self, *args, **kwargs)

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


attach_capability(PDF, "text")
attach_capability(PDF, "vision")
attach_capability(PDF, "shapes")
attach_capability(PDF, "checkbox")
attach_capability(PDF, "classification")
attach_capability(PDF, "extraction")
