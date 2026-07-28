import concurrent.futures  # Import concurrent.futures
import glob as py_glob
import logging
import threading  # Import threading for logging thread information
import time  # Import time for logging timestamps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Type,
    Union,
    cast,
)

if TYPE_CHECKING:
    from natural_pdf.core.page_collection import PageCollection

from PIL import Image
from tqdm.auto import tqdm

from natural_pdf.classification.classification_provider import run_classification_batch
from natural_pdf.classification.pipelines import validate_classification_labels

# Set up logger early
# Configure logging to include thread information
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the ApplyMixin
from natural_pdf.collections.mixins import ApplyMixin
from natural_pdf.core.context import PDFContext
from natural_pdf.core.highlighting_service import HighlightingService
from natural_pdf.core.ocr_contracts import OCRRequest
from natural_pdf.core.ocr_mixin import PDFCollectionOCRMixin
from natural_pdf.core.pdf import PDF
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.export.mixin import ExportMixin
from natural_pdf.selectors.host_mixin import SelectorHostMixin
from natural_pdf.services.base import ServiceHostMixin
from natural_pdf.text.contracts import ContentFilter, TextLayoutOptions, WhitespaceMode
from natural_pdf.text.pipeline import prepare_text_transform, validate_layout_request


class PDFCollection(
    PDFCollectionOCRMixin, ServiceHostMixin, SelectorHostMixin, ApplyMixin, ExportMixin
):
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
        self._iter_index = 0

        # Dynamically import PDF class within methods to avoid circular import at module load time
        PDF = self._get_pdf_class()

        if hasattr(source, "__iter__") and not isinstance(source, str):
            source_list = list(source)
            if source_list and isinstance(source_list[0], PDF):
                if all(isinstance(item, PDF) for item in source_list):
                    self._pdfs = [cast("PDF", item) for item in source_list]
                    self._bind_service_context()
                    return
                else:
                    raise TypeError("Iterable source has mixed PDF/non-PDF objects.")
            # If it's an iterable but not PDFs, fall through to resolve sources
            source = source_list

        # Resolve string, iterable of strings, or single string source to paths/URLs
        resolved_paths_or_urls = self._resolve_sources_to_paths(
            cast(Union[str, Iterable[str]], source)
        )
        self._initialize_pdfs(resolved_paths_or_urls, PDF)  # Pass PDF class

        self._bind_service_context()

    @staticmethod
    def _get_pdf_class():
        """Helper method to dynamically import the PDF class."""
        from natural_pdf.core.pdf import PDF

        return PDF

    # --- Internal Helpers ---

    def _is_url(self, s: str) -> bool:
        return s.startswith(("http://", "https://"))

    def _has_glob_magic(self, s: str) -> bool:
        return py_glob.has_magic(s)

    def _execute_glob(self, pattern: str) -> Set[str]:
        """Glob for paths and return a set of valid PDF paths."""
        found_paths = set()
        # Use iglob for potentially large directories/matches
        paths_iter = py_glob.iglob(pattern, recursive=self._recursive)
        for path_str in paths_iter:
            # Use Path object for easier checking
            p = Path(path_str)
            if p.is_file() and p.suffix.lower() == ".pdf":
                found_paths.add(str(p.resolve()))  # Store resolved absolute path
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

    def _initialize_pdfs(self, paths_or_urls: List[str], PDF_cls: Type["PDF"]):
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

    # ------------------------------------------------------------------
    # Service context helpers
    # ------------------------------------------------------------------
    def _bind_service_context(self) -> None:
        context = self._resolve_service_context()
        self._init_service_host(context)

    def _resolve_service_context(self) -> PDFContext:
        for pdf in self._pdfs:
            context = getattr(pdf, "_context", None)
            if context is not None:
                return context
        context = self._pdf_options.get("context")
        if isinstance(context, PDFContext):
            return context
        return PDFContext.with_defaults()

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
            new_collection._iter_index = 0
            new_collection._init_service_host(self._context)
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
        return f"<PDFCollection(count={len(self._pdfs)})>"

    @property
    def pdfs(self) -> List["PDF"]:
        """Returns the list of PDF objects held by the collection."""
        return self._pdfs

    def extract_each_text(
        self,
        *,
        layout: bool | TextLayoutOptions = False,
        apply_exclusions: bool = True,
        newlines: bool | str = True,
        whitespace: WhitespaceMode = "preserve",
        strip: bool = True,
        bidi: bool = True,
        content_filter: ContentFilter | None = None,
    ) -> List[str]:
        """Extract one text string per PDF without imposing a collection join.

        A collection of documents has no universally meaningful document-boundary
        separator.  Callers that want a flattened representation must choose and
        apply that boundary themselves.
        """

        validate_layout_request(layout)
        if not isinstance(apply_exclusions, bool):
            raise TypeError("apply_exclusions must be a bool")
        prepare_text_transform(
            newlines=newlines,
            whitespace=whitespace,
            strip=strip,
            bidi=bidi,
            content_filter=content_filter,
        )
        return [
            pdf.extract_text(
                layout=layout,
                apply_exclusions=apply_exclusions,
                newlines=newlines,
                whitespace=whitespace,
                strip=strip,
                bidi=bidi,
                content_filter=content_filter,
            )
            for pdf in self._pdfs
        ]

    def show(self, limit: Optional[int] = 30, per_pdf_limit: Optional[int] = 10, **kwargs):
        """
        Display all PDFs in the collection with labels.

        Each PDF is shown with its pages in a grid layout (6 columns by default),
        and all PDFs are stacked vertically with labels.

        Args:
            limit: Maximum total pages to show across all PDFs (default: 30)
            per_pdf_limit: Maximum pages to show per PDF (default: 10)
            **kwargs: Additional arguments passed to each PDF's show() method
                     (e.g., columns, exclusions, resolution, etc.)

        Returns:
            Displayed image in Jupyter or None
        """
        if not self._pdfs:
            print("Empty collection")
            return None

        # Import here to avoid circular imports
        from PIL import ImageDraw, ImageFont

        # Calculate pages per PDF if total limit is set
        if limit and not per_pdf_limit:
            per_pdf_limit = max(1, limit // len(self._pdfs))

        # Collect images from each PDF
        all_images = []
        total_pages_shown = 0

        for pdf in self._pdfs:
            if limit and total_pages_shown >= limit:
                break

            # Calculate limit for this PDF
            pdf_limit = per_pdf_limit
            if limit:
                remaining = limit - total_pages_shown
                pdf_limit = min(per_pdf_limit or remaining, remaining)

            if pdf_limit is None:
                pdf_limit_value = len(pdf.pages)
            else:
                pdf_limit_value = pdf_limit

            # Get PDF identifier
            pdf_name = getattr(pdf, "filename", None) or getattr(pdf, "path", "Unknown")
            if isinstance(pdf_name, Path):
                pdf_name = pdf_name.name
            elif "/" in str(pdf_name):
                pdf_name = str(pdf_name).split("/")[-1]

            # Render this PDF
            try:
                # Get render specs from the PDF
                render_specs = pdf._get_render_specs(
                    mode="show", max_pages=pdf_limit_value, **kwargs
                )

                if not render_specs:
                    continue

                # Get the highlighter and render without displaying
                highlighter = cast("HighlightingService", pdf._get_highlighter())
                pdf_image = highlighter.unified_render(
                    specs=render_specs,
                    layout="grid" if len(render_specs) > 1 else "single",
                    columns=6,
                    **kwargs,
                )

                if pdf_image:
                    # Add label above the PDF image
                    label_height = 40
                    label_bg_color = (240, 240, 240)
                    label_text_color = (0, 0, 0)

                    # Create new image with space for label
                    width, height = pdf_image.size
                    labeled_image = Image.new("RGB", (width, height + label_height), "white")

                    # Draw label background
                    draw = ImageDraw.Draw(labeled_image)
                    draw.rectangle([0, 0, width, label_height], fill=label_bg_color)

                    # Draw label text
                    try:
                        # Try to use a nice font if available
                        font = ImageFont.truetype("Arial", 20)
                    except:
                        # Fallback to default font
                        font = ImageFont.load_default()

                    label_text = f"{pdf_name} ({len(pdf.pages)} pages)"
                    draw.text((10, 10), label_text, fill=label_text_color, font=font)

                    # Paste PDF image below label
                    labeled_image.paste(pdf_image, (0, label_height))

                    all_images.append(labeled_image)
                    if pdf_limit is None:
                        pdf_limit = len(pdf.pages)
                    total_pages_shown += min(pdf_limit_value, len(pdf.pages))

            except Exception as e:
                logger.warning(f"Failed to render PDF {pdf_name}: {e}")
                continue

        if not all_images:
            print("No PDFs could be rendered")
            return None

        # Combine all images vertically
        if len(all_images) == 1:
            combined = all_images[0]
        else:
            # Add spacing between PDFs
            spacing = 20
            total_height = sum(img.height for img in all_images) + spacing * (len(all_images) - 1)
            max_width = max(img.width for img in all_images)

            combined = Image.new("RGB", (max_width, total_height), "white")

            y_offset = 0
            for i, img in enumerate(all_images):
                # Center images if they're narrower than max width
                x_offset = (max_width - img.width) // 2
                combined.paste(img, (x_offset, y_offset))
                y_offset += img.height
                if i < len(all_images) - 1:
                    y_offset += spacing

        # Return the combined image (Jupyter will display it automatically)
        return combined

    def ask(self, *args, **kwargs):
        return self.services.qa.ask(self, *args, **kwargs)

    def describe(self, **kwargs):
        """
        Describe the PDF collection content using the describe service.
        """
        from natural_pdf.elements.element_collection import ElementCollection

        elements = []
        for pdf in self._pdfs:
            for page in pdf.pages:
                elements.extend(page._get_elements(include_chars=False))
        collection = ElementCollection(elements, context=getattr(self, "_context", None))
        return self.services.describe.describe(collection, **kwargs)

    def inspect(self, limit: int = 30, **kwargs):
        """
        Inspect the PDF collection content using the describe service.
        """
        from natural_pdf.elements.element_collection import ElementCollection

        elements = []
        for pdf in self._pdfs:
            for page in pdf.pages:
                elements.extend(page._get_elements(include_chars=False))
        collection = ElementCollection(elements, context=getattr(self, "_context", None))
        return self.services.describe.inspect(collection, limit=limit, **kwargs)

    def detect_lines(self, *args, **kwargs):
        return self.services.shapes.detect_lines(self, *args, **kwargs)

    def detect_checkboxes(self, *args, **kwargs):
        return self.services.checkbox.detect_checkboxes(self, *args, **kwargs)

    def _apply_pdf_collection_ocr_request(
        self,
        request: OCRRequest,
        *,
        pages: Optional[int | Iterable[int] | range | slice],
        max_workers: Optional[int],
        show_progress: bool,
    ) -> None:
        """Prepare every PDF before optionally dispatching workers."""

        logger.info(
            "Applying OCR to %d PDFs in collection (max_workers=%s)...",
            len(self._pdfs),
            max_workers,
        )
        # A caller may provide a one-shot iterable. Materialize it once so every
        # PDF receives the same page selection during the validation pass.
        page_selection = (
            pages if pages is None or isinstance(pages, (int, range, slice)) else tuple(pages)
        )
        prepared = [
            (pdf, *pdf._prepare_pdf_ocr_request(request, pages=page_selection))
            for pdf in self._pdfs
        ]

        def _process_pdf(item: tuple["PDF", OCRRequest, Iterable[Any]]) -> str:
            pdf, effective_request, target_pages = item
            thread_id = threading.current_thread().name
            pdf_path = getattr(pdf, "path", "<unknown>")
            logger.debug("[%s] Starting OCR process for: %s", thread_id, pdf_path)
            start_time = time.monotonic()
            pdf._execute_prepared_pdf_ocr_request(
                effective_request,
                target_pages,
                show_progress=False,
            )
            logger.debug(
                "[%s] Finished OCR process for: %s (Duration: %.2fs)",
                thread_id,
                pdf_path,
                time.monotonic() - start_time,
            )
            return pdf_path

        if max_workers is not None and max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="OCRWorker"
            ) as executor:
                futures = [executor.submit(_process_pdf, item) for item in prepared]
                completed = concurrent.futures.as_completed(futures)
                progress_iter = (
                    tqdm(
                        completed,
                        total=len(futures),
                        desc="Applying OCR (Parallel)",
                        unit="pdf",
                    )
                    if show_progress
                    else completed
                )
                for future in progress_iter:
                    future.result()
        else:
            logger.info("Applying OCR sequentially...")
            progress_iter = (
                tqdm(prepared, desc="Applying OCR (Sequential)", unit="pdf")
                if show_progress
                else prepared
            )
            for item in progress_iter:
                _process_pdf(item)

        logger.info("Finished applying OCR across the collection.")

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

        def _tick_progress() -> None:
            progress_bar.update()

        for pdf in self._pdfs:
            if not pdf.pages:
                continue
            for page in pdf.pages:
                try:
                    page.update_text(
                        transform=correction_callback,
                        selector="text[source=ocr]",
                        apply_exclusions=False,
                        max_workers=max_workers,
                        progress_callback=_tick_progress,
                    )
                except Exception as e:
                    logger.error(
                        f"Error occurred during correction process for page {getattr(page, 'number', '?')} of PDF {getattr(pdf, 'path', 'unknown')}: {e}",
                        exc_info=True,
                    )
                    continue

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
        from natural_pdf.utils.packaging import create_correction_task_package

        # Pass the collection itself (self) as the source
        create_correction_task_package(source=self, output_zip_path=output_zip_path, **kwargs)

    def export_training_data(self, output_dir: str, **kwargs) -> dict:
        """Export cropped text images and labels for OCR model training.

        Creates a HuggingFace ImageFolder-compatible directory with cropped
        text-element images and metadata (JSONL or CSV).

        Args:
            output_dir: Destination directory.
            **kwargs: Forwarded to :func:`~natural_pdf.exporters.training_data.export_training_data`.

        Returns:
            Summary dict with ``images``, ``skipped``, and ``output_dir`` keys.
        """
        from natural_pdf.exporters.training_data import export_training_data

        return export_training_data(source=self, output_dir=output_dir, **kwargs)

    # --- Semantic Search ---

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        model: Optional[str] = None,
    ) -> "PageCollection":
        """Semantic search across pages in all PDFs in this collection.

        Finds the pages most relevant to the query using sentence-transformers
        embeddings. Pages from all PDFs are ranked together.

        Args:
            query: Text to search for.
            top_k: Number of pages to return.
            model: Embedding model name (default: all-MiniLM-L6-v2).

        Returns:
            PageCollection of the most relevant pages, ordered by relevance.
            Each page has a ``_search_score`` attribute with the similarity score.
        """
        import numpy as np

        from natural_pdf.core.page_collection import PageCollection
        from natural_pdf.search.search_service import DEFAULT_MODEL, SearchService

        model_name = model or DEFAULT_MODEL

        # Validate before the empty-collection early return so bad arguments
        # raise regardless of collection size.
        SearchService.validate_query(query, top_k)

        # Gather all pages across all PDFs, reusing each PDF's cached
        # (fingerprint-invalidated) embeddings instead of re-encoding the
        # whole collection on every query.
        all_pages = []
        per_pdf_embeddings = []
        for pdf in self._pdfs:
            all_pages.extend(pdf.pages)
            per_pdf_embeddings.append(pdf._get_page_embeddings(model_name))

        if not all_pages:
            return PageCollection([])

        embeddings = np.concatenate([e for e in per_pdf_embeddings if len(e)], axis=0)
        results = SearchService.rank(
            query, embeddings, all_pages, top_k=top_k, model_name=model_name
        )

        ranked_pages = []
        for page, score in results:
            page._search_score = score
            ranked_pages.append(page)

        return PageCollection(ranked_pages)

    # --- Classification Method --- #
    def classify_all(
        self,
        labels: List[str],
        *,
        using: Optional[str] = None,
        model: Optional[str] = None,
        analysis_key: str = "classification",
        min_confidence: float = 0.0,
        multi_label: bool = False,
        batch_size: int = 8,
        progress_bar: bool = True,
        **kwargs,
    ) -> "PDFCollection":
        """
        Classify each PDF document in the collection using provider-backed batch processing.
        """
        validate_classification_labels(labels)

        if not self._pdfs:
            logger.warning("PDFCollection is empty, skipping classification.")
            return self

        mode_desc = f"using='{using}'" if using else f"model='{model}'" if model else "default text"
        logger.info(
            f"Starting batch classification for {len(self._pdfs)} PDFs in collection ({mode_desc})..."
        )

        from natural_pdf.exceptions import ClassificationError
        from natural_pdf.services.classification_service import (
            ClassificationService,
            checkout_classification_engine,
            partition_classification_kwargs,
        )

        options = partition_classification_kwargs(kwargs)

        # Check the engine out once and pass the instance straight to
        # run_classification_batch below, so exactly one engine instance is
        # created regardless of registration lifetime — and transient
        # instances are cleaned up when the call finishes.
        with checkout_classification_engine(self, options.engine_name) as engine_obj:
            inferred_using = engine_obj.infer_using(
                model or engine_obj.default_model("text"), using
            )

            # Split the kwarg stream the same way ClassificationService.classify
            # does: content-extraction options go to the content getter, everything
            # else (e.g. device=) to the engine call.
            pdf_contents: List[Any] = []
            valid_pdfs: List[Any] = []

            logger.info(
                f"Gathering content from {len(self._pdfs)} PDFs for batch classification..."
            )
            for pdf in self._pdfs:
                try:
                    content = pdf._get_classification_content(
                        model_type=inferred_using, **options.content
                    )
                    pdf_contents.append(content)
                    valid_pdfs.append(pdf)
                except ValueError as exc:
                    # Only genuinely empty documents may be skipped; anything else
                    # must surface instead of silently dropping the PDF.
                    if ClassificationService._is_empty_text_error(exc):
                        logger.warning(f"Skipping PDF {pdf.path}: no extractable content - {exc}")
                    else:
                        raise ClassificationError(
                            f"Failed to get classification content for {pdf.path}: {exc}"
                        ) from exc

            if not pdf_contents:
                logger.warning("No valid content could be gathered from PDFs for classification.")
                return self

            batch_results = run_classification_batch(
                context=self,
                contents=pdf_contents,
                labels=labels,
                model_id=model or engine_obj.default_model(inferred_using),
                using=inferred_using,
                min_confidence=min_confidence,
                multi_label=multi_label,
                batch_size=batch_size,
                progress_bar=progress_bar,
                engine=engine_obj,
                **options.engine,
            )

        if len(batch_results) != len(valid_pdfs):
            raise ClassificationError(
                f"Batch classification returned {len(batch_results)} results "
                f"for {len(valid_pdfs)} PDFs."
            )

        for pdf, result_obj in zip(valid_pdfs, batch_results):
            if not hasattr(pdf, "analyses") or pdf.analyses is None:
                pdf.analyses = {}
            pdf.analyses[analysis_key] = result_obj

        skipped_count = len(self._pdfs) - len(valid_pdfs)
        final_message = f"Finished batch classification. Processed: {len(valid_pdfs)}"
        if skipped_count > 0:
            final_message += f", Skipped: {skipped_count}"
        logger.info(final_message + ".")
        return self

    # ------------------------------------------------------------------
    # QA service hooks
    # ------------------------------------------------------------------
    def _qa_segments(self):
        segments = []
        for pdf in self._pdfs:
            pages = getattr(pdf, "pages", None)
            if not pages:
                continue
            try:
                segments.extend(list(pages))
            except Exception:
                continue
        return tuple(segments)

    def _qa_target_region(self):
        for pdf in self._pdfs:
            pages = getattr(pdf, "pages", None)
            if not pages:
                continue
            try:
                first_page = pages[0]
            except Exception:
                continue
            to_region = getattr(first_page, "to_region", None)
            if callable(to_region):
                return to_region()
        raise RuntimeError("PDFCollection has no pages available for QA.")

    def _qa_context_page_number(self) -> int:
        for pdf in self._pdfs:
            pages = getattr(pdf, "pages", None)
            if not pages:
                continue
            try:
                first_page = pages[0]
                return int(getattr(first_page, "number", -1))
            except Exception:
                continue
        return -1

    def _qa_source_elements(self) -> ElementCollection:
        return ElementCollection([])

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
