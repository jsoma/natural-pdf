import concurrent.futures  # Import concurrent.futures
import copy  # Added for copying options
import glob as py_glob
import logging
import os
import re  # Added for safe path generation
import threading  # Import threading for logging thread information
import time  # Import time for logging timestamps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    overload,
)

from PIL import Image
from tqdm import tqdm
from tqdm.auto import tqdm as auto_tqdm
from tqdm.notebook import tqdm as notebook_tqdm

from natural_pdf.utils.tqdm_utils import get_tqdm

# Get the appropriate tqdm class once
tqdm = get_tqdm()

# Set up logger early
# Configure logging to include thread information
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from natural_pdf.core.pdf import PDF
from natural_pdf.elements.region import Region
from natural_pdf.export.mixin import ExportMixin

# --- Search Imports ---
try:
    from natural_pdf.search.search_service_protocol import (
        Indexable,
        SearchOptions,
        SearchServiceProtocol,
    )
    from natural_pdf.search.searchable_mixin import SearchableMixin
except ImportError as e:
    logger_init = logging.getLogger(__name__)
    logger_init.warning(
        f"Failed to import Haystack components. Semantic search functionality disabled.",
    )

    # Dummy definitions
    class SearchableMixin:
        pass

    SearchServiceProtocol, SearchOptions, Indexable = object, object, object

# Import the ApplyMixin
from natural_pdf.collections.mixins import ApplyMixin
from natural_pdf.search.searchable_mixin import SearchableMixin  # Import the new mixin


class PDFCollection(SearchableMixin, ApplyMixin, ExportMixin):  # Add ExportMixin
    def __init__(
        self,
        source: Union[str, Iterable[Union[str, "PDF"]]],
        recursive: bool = True,
        **pdf_options: Any,
    ):
        """
        Initializes a collection of PDF documents from various sources.

        Args:
            source: The source of PDF documents. Can be:
                - An iterable (e.g., list) of existing PDF objects.
                - An iterable (e.g., list) of file paths/URLs/globs (strings).
                - A single file path/URL/directory/glob string.
            recursive: If source involves directories or glob patterns,
                       whether to search recursively (default: True).
            **pdf_options: Keyword arguments passed to the PDF constructor.
        """
        self._pdfs: List["PDF"] = []
        self._pdf_options = pdf_options  # Store options for potential slicing later
        self._recursive = recursive  # Store setting for potential slicing

        # Dynamically import PDF class within methods to avoid circular import at module load time
        PDF = self._get_pdf_class()

        if hasattr(source, "__iter__") and not isinstance(source, str):
            source_list = list(source)
            if not source_list:
                return  # Empty list source
            if isinstance(source_list[0], PDF):
                if all(isinstance(item, PDF) for item in source_list):
                    self._pdfs = source_list  # Direct assignment
                    # Don't adopt search context anymore
                    return
                else:
                    raise TypeError("Iterable source has mixed PDF/non-PDF objects.")
            # If it's an iterable but not PDFs, fall through to resolve sources

        # Resolve string, iterable of strings, or single string source to paths/URLs
        resolved_paths_or_urls = self._resolve_sources_to_paths(source)
        self._initialize_pdfs(resolved_paths_or_urls, PDF)  # Pass PDF class

        self._iter_index = 0

        # Initialize internal search service reference
        self._search_service: Optional[SearchServiceProtocol] = None

    @staticmethod
    def _get_pdf_class():
        """Helper method to dynamically import the PDF class."""
        try:
            # Import needs to resolve path correctly
            from natural_pdf.core.pdf import PDF

            return PDF
        except ImportError as e:
            logger.error(
                "Could not import PDF class from natural_pdf.core.pdf. Ensure it exists and there are no circular imports at runtime."
            )
            raise ImportError("PDF class is required but could not be imported.") from e

    # --- Internal Helpers ---

    def _is_url(self, s: str) -> bool:
        return s.startswith(("http://", "https://"))

    def _has_glob_magic(self, s: str) -> bool:
        return py_glob.has_magic(s)

    def _execute_glob(self, pattern: str) -> Set[str]:
        """Glob for paths and return a set of valid PDF paths."""
        found_paths = set()
        try:
            # Use iglob for potentially large directories/matches
            paths_iter = py_glob.iglob(pattern, recursive=self._recursive)
            for path_str in paths_iter:
                # Use Path object for easier checking
                p = Path(path_str)
                if p.is_file() and p.suffix.lower() == ".pdf":
                    found_paths.add(str(p.resolve()))  # Store resolved absolute path
        except Exception as e:
            logger.error(f"Error processing glob pattern '{pattern}': {e}")
        return found_paths

    def _resolve_sources_to_paths(self, source: Union[str, Iterable[str]]) -> List[str]:
        """Resolves various source types into a list of unique PDF paths/URLs."""
        final_paths = set()
        sources_to_process = []

        if isinstance(source, str):
            sources_to_process.append(source)
        elif hasattr(source, "__iter__"):
            sources_to_process.extend(list(source))
        else:  # Should not happen based on __init__ checks, but safeguard
            raise TypeError(f"Unexpected source type in _resolve_sources_to_paths: {type(source)}")

        for item in sources_to_process:
            if not isinstance(item, str):
                logger.warning(f"Skipping non-string item in source list: {type(item)}")
                continue

            item_path = Path(item)

            if self._is_url(item):
                final_paths.add(item)  # Add URL directly
            elif self._has_glob_magic(item):
                glob_results = self._execute_glob(item)
                final_paths.update(glob_results)
            elif item_path.is_dir():
                # Use glob to find PDFs in directory, respecting recursive flag
                dir_pattern = (
                    str(item_path / "**" / "*.pdf") if self._recursive else str(item_path / "*.pdf")
                )
                dir_glob_results = self._execute_glob(dir_pattern)
                final_paths.update(dir_glob_results)
            elif item_path.is_file() and item_path.suffix.lower() == ".pdf":
                final_paths.add(str(item_path.resolve()))  # Add resolved file path
            else:
                logger.warning(
                    f"Source item ignored (not a valid URL, directory, file, or glob): {item}"
                )

        return sorted(list(final_paths))

    def _initialize_pdfs(self, paths_or_urls: List[str], PDF_cls: Type):
        """Initializes PDF objects from a list of paths/URLs."""
        logger.info(f"Initializing {len(paths_or_urls)} PDF objects...")
        failed_count = 0
        for path_or_url in tqdm(paths_or_urls, desc="Loading PDFs"):
            try:
                pdf_instance = PDF_cls(path_or_url, **self._pdf_options)
                self._pdfs.append(pdf_instance)
            except Exception as e:
                logger.error(
                    f"Failed to load PDF: {path_or_url}. Error: {e}", exc_info=False
                )  # Keep log concise
                failed_count += 1
        logger.info(f"Successfully initialized {len(self._pdfs)} PDFs. Failed: {failed_count}")

    # --- Public Factory Class Methods (Simplified) ---

    @classmethod
    def from_paths(cls, paths_or_urls: List[str], **pdf_options: Any) -> "PDFCollection":
        """Creates a PDFCollection explicitly from a list of file paths or URLs."""
        # __init__ can handle List[str] directly now
        return cls(paths_or_urls, **pdf_options)

    @classmethod
    def from_glob(cls, pattern: str, recursive: bool = True, **pdf_options: Any) -> "PDFCollection":
        """Creates a PDFCollection explicitly from a single glob pattern."""
        # __init__ can handle single glob string directly
        return cls(pattern, recursive=recursive, **pdf_options)

    @classmethod
    def from_globs(
        cls, patterns: List[str], recursive: bool = True, **pdf_options: Any
    ) -> "PDFCollection":
        """Creates a PDFCollection explicitly from a list of glob patterns."""
        # __init__ can handle List[str] containing globs directly
        return cls(patterns, recursive=recursive, **pdf_options)

    @classmethod
    def from_directory(
        cls, directory_path: str, recursive: bool = True, **pdf_options: Any
    ) -> "PDFCollection":
        """Creates a PDFCollection explicitly from PDF files within a directory."""
        # __init__ can handle single directory string directly
        return cls(directory_path, recursive=recursive, **pdf_options)

    # --- Core Collection Methods ---
    def __len__(self) -> int:
        return len(self._pdfs)

    def __getitem__(self, key) -> Union["PDF", "PDFCollection"]:
        # Use dynamic import here as well
        PDF = self._get_pdf_class()
        if isinstance(key, slice):
            # Create a new collection with the sliced PDFs and original options
            new_collection = PDFCollection.__new__(PDFCollection)  # Create blank instance
            new_collection._pdfs = self._pdfs[key]
            new_collection._pdf_options = self._pdf_options
            new_collection._recursive = self._recursive
            # Search context is not copied/inherited anymore
            return new_collection
        elif isinstance(key, int):
            # Check bounds
            if 0 <= key < len(self._pdfs):
                return self._pdfs[key]
            else:
                raise IndexError(f"PDF index {key} out of range (0-{len(self._pdfs)-1}).")
        else:
            raise TypeError(f"PDF indices must be integers or slices, not {type(key)}.")

    def __iter__(self):
        return iter(self._pdfs)

    def __repr__(self) -> str:
        # Removed search status
        return f"<PDFCollection(count={len(self._pdfs)})>"
        return f"<PDFCollection(count={len(self._pdfs)})>"

    @property
    def pdfs(self) -> List["PDF"]:
        """Returns the list of PDF objects held by the collection."""
        return self._pdfs

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
        selector: Optional[str] = None,  # Now optional
        *,
        text: Optional[str] = None,  # New text parameter
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> "ElementCollection":
        """
        Find all elements matching the selector OR text across all PDFs in the collection.

        Provide EITHER `selector` OR `text`, but not both.

        This creates an ElementCollection that can span multiple PDFs. Note that
        some ElementCollection methods have limitations when spanning PDFs.

        Args:
            selector: CSS-like selector string to query elements.
            text: Text content to search for (equivalent to 'text:contains(...)').
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            **kwargs: Additional keyword arguments passed to the find_all method of each PDF.

        Returns:
            ElementCollection containing all matching elements across all PDFs.
        """
        # Validation happens within pdf.find_all

        # Collect elements from all PDFs
        all_elements = []
        for pdf in self._pdfs:
            try:
                # Pass the relevant arguments down to each PDF's find_all
                elements = pdf.find_all(
                    selector=selector,
                    text=text,
                    apply_exclusions=apply_exclusions,
                    regex=regex,
                    case=case,
                    **kwargs,
                )
                all_elements.extend(elements.elements)
            except Exception as e:
                logger.error(f"Error finding elements in {pdf.path}: {e}", exc_info=True)

        return ElementCollection(all_elements)

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
        pages: Optional[Union[slice, List[int]]] = None,
        max_workers: Optional[int] = None,
    ) -> "PDFCollection":
        """
        Apply OCR to all PDFs in the collection, potentially in parallel.

        Args:
            engine: OCR engine to use (e.g., 'easyocr', 'paddleocr', 'surya')
            languages: List of language codes for OCR
            min_confidence: Minimum confidence threshold for text detection
            device: Device to use for OCR (e.g., 'cpu', 'cuda')
            resolution: DPI resolution for page rendering
            apply_exclusions: Whether to apply exclusion regions
            detect_only: If True, only detect text regions without extracting text
            replace: If True, replace existing OCR elements
            options: Engine-specific options
            pages: Specific pages to process (None for all pages)
            max_workers: Maximum number of threads to process PDFs concurrently.
                         If None or 1, processing is sequential. (default: None)

        Returns:
            Self for method chaining
        """
        PDF = self._get_pdf_class()
        logger.info(
            f"Applying OCR to {len(self._pdfs)} PDFs in collection (max_workers={max_workers})..."
        )

        # Worker function takes PDF object again
        def _process_pdf(pdf: PDF):
            """Helper function to apply OCR to a single PDF, handling errors."""
            thread_id = threading.current_thread().name  # Get thread name for logging
            pdf_path = pdf.path  # Get path for logging
            logger.debug(f"[{thread_id}] Starting OCR process for: {pdf_path}")
            start_time = time.monotonic()
            try:
                pdf.apply_ocr(  # Call apply_ocr on the original PDF object
                    pages=pages,
                    engine=engine,
                    languages=languages,
                    min_confidence=min_confidence,
                    device=device,
                    resolution=resolution,
                    apply_exclusions=apply_exclusions,
                    detect_only=detect_only,
                    replace=replace,
                    options=options,
                    # Note: We might want a max_workers here too for page rendering?
                    # For now, PDF.apply_ocr doesn't have it.
                )
                end_time = time.monotonic()
                logger.debug(
                    f"[{thread_id}] Finished OCR process for: {pdf_path} (Duration: {end_time - start_time:.2f}s)"
                )
                return pdf_path, None
            except Exception as e:
                end_time = time.monotonic()
                logger.error(
                    f"[{thread_id}] Failed OCR process for {pdf_path} after {end_time - start_time:.2f}s: {e}",
                    exc_info=False,
                )
                return pdf_path, e  # Return path and error

        # Use ThreadPoolExecutor for parallel processing if max_workers > 1
        if max_workers is not None and max_workers > 1:
            futures = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="OCRWorker"
            ) as executor:
                for pdf in self._pdfs:
                    # Submit the PDF object to the worker function
                    futures.append(executor.submit(_process_pdf, pdf))

            # Use the selected tqdm class with as_completed for progress tracking
            progress_bar = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(self._pdfs),
                desc="Applying OCR (Parallel)",
                unit="pdf",
            )

            for future in progress_bar:
                pdf_path, error = future.result()  # Get result (or exception)
                if error:
                    progress_bar.set_postfix_str(f"Error: {pdf_path}", refresh=True)
                # Progress is updated automatically by tqdm

        else:  # Sequential processing (max_workers is None or 1)
            logger.info("Applying OCR sequentially...")
            # Use the selected tqdm class for sequential too for consistency
            # Iterate over PDF objects directly for sequential
            for pdf in tqdm(self._pdfs, desc="Applying OCR (Sequential)", unit="pdf"):
                _process_pdf(pdf)  # Call helper directly with PDF object

        logger.info("Finished applying OCR across the collection.")
        return self

    def correct_ocr(
        self,
        correction_callback: Callable[[Any], Optional[str]],
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> "PDFCollection":
        """
        Apply OCR correction to all relevant elements across all pages and PDFs
        in the collection using a single progress bar.

        Args:
            correction_callback: Function to apply to each OCR element.
                                 It receives the element and should return
                                 the corrected text (str) or None.
            max_workers: Max threads to use for parallel execution within each page.
            progress_callback: Optional callback function to call after processing each element.

        Returns:
            Self for method chaining.
        """
        PDF = self._get_pdf_class()  # Ensure PDF class is available
        if not callable(correction_callback):
            raise TypeError("`correction_callback` must be a callable function.")

        logger.info(f"Gathering OCR elements from {len(self._pdfs)} PDFs for correction...")

        # 1. Gather all target elements using the collection's find_all
        #    Crucially, set apply_exclusions=False to include elements in headers/footers etc.
        all_ocr_elements = self.find_all("text[source=ocr]", apply_exclusions=False).elements

        if not all_ocr_elements:
            logger.info("No OCR elements found in the collection to correct.")
            return self

        total_elements = len(all_ocr_elements)
        logger.info(
            f"Found {total_elements} OCR elements across the collection. Starting correction process..."
        )

        # 2. Initialize the progress bar
        progress_bar = tqdm(total=total_elements, desc="Correcting OCR Elements", unit="element")

        # 3. Iterate through PDFs and delegate to PDF.correct_ocr
        #    PDF.correct_ocr handles page iteration and passing the progress callback down.
        for pdf in self._pdfs:
            if not pdf.pages:
                continue
            try:
                pdf.correct_ocr(
                    correction_callback=correction_callback,
                    max_workers=max_workers,
                    progress_callback=progress_bar.update,  # Pass the bar's update method
                )
            except Exception as e:
                logger.error(
                    f"Error occurred during correction process for PDF {pdf.path}: {e}",
                    exc_info=True,
                )
                # Decide if we should stop or continue? For now, continue.

        progress_bar.close()

        return self

    def categorize(self, labels: List[str], **kwargs):
        """Categorizes PDFs in the collection based on content or features."""
        # Implementation requires integrating with classification models or logic
        raise NotImplementedError("categorize requires classification implementation.")

    def export_ocr_correction_task(self, output_zip_path: str, **kwargs):
        """
        Exports OCR results from all PDFs in this collection into a single
        correction task package (zip file).

        Args:
            output_zip_path: The path to save the output zip file.
            **kwargs: Additional arguments passed to create_correction_task_package
                      (e.g., image_render_scale, overwrite).
        """
        try:
            from natural_pdf.utils.packaging import create_correction_task_package

            # Pass the collection itself (self) as the source
            create_correction_task_package(source=self, output_zip_path=output_zip_path, **kwargs)
        except ImportError:
            logger.error(
                "Failed to import 'create_correction_task_package'. Packaging utility might be missing."
            )
            # Or raise
        except Exception as e:
            logger.error(f"Failed to export correction task for collection: {e}", exc_info=True)
            raise  # Re-raise the exception from the utility function

    # --- Mixin Required Implementation ---
    def get_indexable_items(self) -> Iterable[Indexable]:
        """Yields Page objects from the collection, conforming to Indexable."""
        if not self._pdfs:
            return  # Return empty iterator if no PDFs

        for pdf in self._pdfs:
            if not pdf.pages:  # Handle case where a PDF might have 0 pages after loading
                logger.warning(f"PDF '{pdf.path}' has no pages. Skipping.")
                continue
            for page in pdf.pages:
                # Optional: Add filtering here if needed (e.g., skip empty pages)
                # Assuming Page object conforms to Indexable
                # We might still want the empty page check here for efficiency
                # if not page.extract_text(use_exclusions=False).strip():
                #     logger.debug(f"Skipping empty page {page.page_number} from PDF '{pdf.path}'.")
                #     continue
                yield page

    # --- Classification Method --- #
    def classify_all(
        self,
        labels: List[str],
        using: Optional[str] = None,  # Default handled by PDF.classify -> manager
        model: Optional[str] = None,  # Optional model ID
        max_workers: Optional[int] = None,
        analysis_key: str = "classification",  # Key for storing result in PDF.analyses
        **kwargs,
    ) -> "PDFCollection":
        """
        Classify each PDF document in the collection, potentially in parallel.

        This method delegates classification to each PDF object's `classify` method.
        By default, uses the full extracted text of the PDF.
        If `using='vision'`, it classifies the first page's image, but ONLY if
        the PDF has a single page (raises ValueError otherwise).

        Args:
            labels: A list of string category names.
            using: Processing mode ('text', 'vision'). If None, manager infers (defaulting to text).
            model: Optional specific model identifier (e.g., HF ID). If None, manager uses default for 'using' mode.
            max_workers: Maximum number of threads to process PDFs concurrently.
                         If None or 1, processing is sequential.
            analysis_key: Key under which to store the ClassificationResult in each PDF's `analyses` dict.
            **kwargs: Additional arguments passed down to `pdf.classify` (e.g., device,
                      min_confidence, multi_label, text extraction options).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If labels list is empty, or if using='vision' on a multi-page PDF.
            ClassificationError: If classification fails for any PDF (will stop processing).
            ImportError: If classification dependencies are missing.
        """
        PDF = self._get_pdf_class()
        if not labels:
            raise ValueError("Labels list cannot be empty.")

        if not self._pdfs:
            logger.warning("PDFCollection is empty, skipping classification.")
            return self

        mode_desc = f"using='{using}'" if using else f"model='{model}'" if model else "default text"
        logger.info(
            f"Starting classification for {len(self._pdfs)} PDFs in collection ({mode_desc})..."
        )

        progress_bar = tqdm(
            total=len(self._pdfs), desc=f"Classifying PDFs ({mode_desc})", unit="pdf"
        )

        # Worker function
        def _process_pdf_classification(pdf: PDF):
            thread_id = threading.current_thread().name
            pdf_path = pdf.path
            logger.debug(f"[{thread_id}] Starting classification process for PDF: {pdf_path}")
            start_time = time.monotonic()
            try:
                # Call classify directly on the PDF object
                pdf.classify(
                    labels=labels,
                    using=using,
                    model=model,
                    analysis_key=analysis_key,
                    **kwargs,  # Pass other relevant args like min_confidence, multi_label
                )
                end_time = time.monotonic()
                logger.debug(
                    f"[{thread_id}] Finished classification for PDF: {pdf_path} (Duration: {end_time - start_time:.2f}s)"
                )
                progress_bar.update(1)  # Update progress bar upon success
                return pdf_path, None  # Return path and no error
            except ValueError as ve:
                # Catch specific error for vision on multi-page PDF
                end_time = time.monotonic()
                logger.error(
                    f"[{thread_id}] Skipped classification for {pdf_path} after {end_time - start_time:.2f}s: {ve}",
                    exc_info=False,
                )
                progress_bar.update(1)  # Still update progress bar
                return pdf_path, ve  # Return the specific ValueError
            except Exception as e:
                end_time = time.monotonic()
                logger.error(
                    f"[{thread_id}] Failed classification process for PDF {pdf_path} after {end_time - start_time:.2f}s: {e}",
                    exc_info=True,  # Log full traceback for unexpected errors
                )
                # Close progress bar immediately on critical error to avoid hanging
                if not progress_bar.disable:
                    progress_bar.close()
                # Re-raise the exception to stop the entire collection processing
                raise ClassificationError(f"Classification failed for {pdf_path}: {e}") from e

        # Use ThreadPoolExecutor for parallel processing if max_workers > 1
        processed_count = 0
        skipped_count = 0
        try:
            if max_workers is not None and max_workers > 1:
                logger.info(f"Classifying PDFs in parallel with {max_workers} workers.")
                futures = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers, thread_name_prefix="ClassifyWorker"
                ) as executor:
                    for pdf in self._pdfs:
                        futures.append(executor.submit(_process_pdf_classification, pdf))

                    # Wait for all futures to complete
                    # Progress updated within worker
                    for future in concurrent.futures.as_completed(futures):
                        processed_count += 1
                        pdf_path, error = (
                            future.result()
                        )  # Raise ClassificationError if worker failed critically
                        if isinstance(error, ValueError):
                            # Logged in worker, just count as skipped
                            skipped_count += 1

            else:  # Sequential processing
                logger.info("Classifying PDFs sequentially.")
                for pdf in self._pdfs:
                    processed_count += 1
                    pdf_path, error = _process_pdf_classification(
                        pdf
                    )  # Raise ClassificationError if worker failed critically
                    if isinstance(error, ValueError):
                        skipped_count += 1

            final_message = (
                f"Finished classification across the collection. Processed: {processed_count}"
            )
            if skipped_count > 0:
                final_message += f", Skipped (e.g., vision on multi-page): {skipped_count}"
            logger.info(final_message + ".")

        finally:
            # Ensure progress bar is closed properly
            if not progress_bar.disable and progress_bar.n < progress_bar.total:
                progress_bar.n = progress_bar.total  # Ensure it reaches 100%
            if not progress_bar.disable:
                progress_bar.close()

        return self

    # --- End Classification Method --- #

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
        Gather analysis data from all PDFs in the collection.

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
        if not self._pdfs:
            logger.warning("No PDFs found in collection")
            return []

        all_data = []

        for pdf in tqdm(self._pdfs, desc="Gathering PDF data", leave=False):
            # PDF level data
            pdf_data = {
                "pdf_path": pdf.path,
                "pdf_filename": Path(pdf.path).name,
                "total_pages": len(pdf.pages) if hasattr(pdf, "pages") else 0,
            }

            # Add metadata if available
            if hasattr(pdf, "metadata") and pdf.metadata:
                for k, v in pdf.metadata.items():
                    if v:  # Only add non-empty metadata
                        pdf_data[f"metadata.{k}"] = str(v)

            all_data.append(pdf_data)

        return all_data
