from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
    overload,
    runtime_checkable,
)

from natural_pdf.analyzers.checkbox.mixin import CheckboxDetectionMixin
from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin
from natural_pdf.collections.mixins import ApplyMixin, SectionsCollectionMixin
from natural_pdf.core.interfaces import SupportsGeometry, SupportsSections, SupportsTextElements
from natural_pdf.core.pdf import PDF
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.text_mixin import TextMixin
from natural_pdf.utils.sections import sanitize_sections

# New Imports


# Potentially lazy imports for optional dependencies needed in save_pdf
try:
    import pikepdf
except ImportError:
    pikepdf = None

try:
    from natural_pdf.exporters.searchable_pdf import create_searchable_pdf
except ImportError:
    create_searchable_pdf = None

# ---> ADDED Import for the new exporter
try:
    from natural_pdf.exporters.original_pdf import create_original_pdf
except ImportError:
    create_original_pdf = None
# <--- END ADDED

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from natural_pdf.core.highlighting_service import HighlightContext
    from natural_pdf.core.page import Page
    from natural_pdf.core.page_groupby import PageGroupBy
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.region import Region
    from natural_pdf.flows.flow import Flow
else:  # pragma: no cover - runtime typing helpers
    Page = Any  # type: ignore[assignment]


@runtime_checkable
class ElementsProvider(Protocol):
    @property
    def elements(self) -> Sequence[SupportsGeometry]: ...


BoundarySource = Union[str, ElementsProvider, Iterable[SupportsGeometry], Iterable["Element"], None]


class PageCollection(
    TextMixin,
    ApplyMixin,
    SectionsCollectionMixin,
    ShapeDetectionMixin,
    CheckboxDetectionMixin,
    Visualizable,
    Sequence["Page"],
):
    """
    Represents a collection of Page objects, often from a single PDF document.
    Provides methods for batch operations on these pages.
    """

    def __init__(self, pages: Sequence["Page"] | Iterable["Page"]):
        """
        Initialize a page collection.

        Args:
            pages: List or sequence of Page objects (can be lazy)
        """
        # Store the sequence as-is to preserve lazy behavior
        # Only convert to list if we need list-specific operations
        if isinstance(pages, Sequence):
            self.pages: Sequence["Page"] = pages
        else:
            # Fallback for non-sequence types – materialise to preserve ordering
            self.pages = list(pages)

    def __len__(self) -> int:
        """Return the number of pages in the collection."""
        return len(self.pages)

    @overload
    def __getitem__(self, idx: int) -> "Page": ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence["Page"]: ...

    def __getitem__(self, idx: Union[int, slice]) -> Union["Page", Sequence["Page"]]:
        """Support indexing and slicing."""
        if isinstance(idx, slice):
            return PageCollection(self.pages[idx])
        return self.pages[idx]

    def __iter__(self) -> Iterator["Page"]:
        """Support iteration."""
        return iter(self.pages)

    def __repr__(self) -> str:
        """Return a string representation showing the page count."""
        return f"<PageCollection(count={len(self)})>"

    @property
    def elements(self) -> Sequence["Page"]:
        """Alias to expose pages for APIs expecting an elements attribute."""
        return self.pages

    def _get_items_for_apply(self) -> Iterator["Page"]:
        """
        Override ApplyMixin's _get_items_for_apply to preserve lazy behavior.

        Returns an iterator that yields pages on-demand rather than materializing
        all pages at once, maintaining the lazy loading behavior.
        """
        return iter(self.pages)

    def _get_page_indices(self) -> List[int]:
        """
        Get page indices without forcing materialization of pages.

        Returns:
            List of page indices for the pages in this collection.
        """
        # Handle different types of page sequences efficiently
        indices = getattr(self.pages, "_indices", None)
        if indices is not None:
            return list(indices)
        else:
            return [p.index for p in self.pages]

    def _resolve_parent_pdf(self) -> "PDF":
        """
        Resolve the parent PDF instance shared by pages in this collection.

        Raises:
            RuntimeError: If the collection is empty or pages do not expose a parent PDF.
        """
        if not self.pages:
            raise RuntimeError("PageCollection is empty; cannot resolve parent PDF.")

        parent = getattr(self.pages[0], "_parent", None)
        if parent is None:
            raise RuntimeError("Pages in this collection do not expose a parent PDF.")

        return cast("PDF", parent)

    def extract_text(
        self,
        keep_blank_chars: bool = True,
        apply_exclusions: bool = True,
        strip: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """
        Extract text from all pages in the collection.

        Args:
            keep_blank_chars: Whether to keep blank characters (default: True)
            apply_exclusions: Whether to apply exclusion regions (default: True)
            strip: Whether to strip whitespace from the extracted text.
            **kwargs: Additional extraction parameters

        Returns:
            Combined text from all pages
        """
        texts: List[str] = []
        extra_kwargs = kwargs.copy()

        explicit_strip_final = extra_kwargs.pop("strip_final", None)
        explicit_strip_empty = extra_kwargs.pop("strip_empty", None)

        for page in self.pages:
            text = page.extract_text(
                preserve_whitespace=keep_blank_chars,
                use_exclusions=apply_exclusions,
                strip_final=(
                    explicit_strip_final
                    if explicit_strip_final is not None
                    else (strip if strip is not None else False)
                ),
                strip_empty=explicit_strip_empty if explicit_strip_empty is not None else False,
                **extra_kwargs,
            )
            texts.append(text)

        return "\n".join(texts)

    def apply_ocr(
        self,
        engine: Optional[str] = None,
        # --- Common OCR Parameters (Direct Arguments) ---
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,  # Min confidence threshold
        device: Optional[str] = None,
        resolution: Optional[int] = None,  # DPI for rendering
        apply_exclusions: bool = True,  # New parameter
        replace: bool = True,  # Whether to replace existing OCR elements
        # --- Engine-Specific Options ---
        options: Optional[Any] = None,  # e.g., EasyOCROptions(...)
    ) -> "PageCollection":
        """
        Applies OCR to all pages within this collection using batch processing.

        This delegates the work to the parent PDF object's `apply_ocr` method.

        Args:
            engine: Name of the OCR engine (e.g., 'easyocr', 'paddleocr').
            languages: List of language codes (e.g., ['en', 'fr'], ['en', 'ch']).
                       **Must be codes understood by the specific selected engine.**
                       No mapping is performed.
            min_confidence: Minimum confidence threshold for detected text (0.0 to 1.0).
            device: Device to run OCR on (e.g., 'cpu', 'cuda', 'mps').
            resolution: DPI resolution to render page images before OCR (e.g., 150, 300).
            apply_exclusions: If True (default), render page images for OCR with
                              excluded areas masked (whited out). If False, OCR
                              the raw page images without masking exclusions.
            replace: If True (default), remove any existing OCR elements before
                    adding new ones. If False, add new OCR elements to existing ones.
            options: An engine-specific options object (e.g., EasyOCROptions) or dict.

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If pages lack a parent PDF or parent lacks `apply_ocr`.
            (Propagates exceptions from PDF.apply_ocr)
        """
        if not self.pages:
            logger.warning("Cannot apply OCR to an empty PageCollection.")
            return self

        parent_pdf = self._resolve_parent_pdf()

        if not hasattr(parent_pdf, "apply_ocr") or not callable(parent_pdf.apply_ocr):
            raise RuntimeError("Parent PDF object does not have the required 'apply_ocr' method.")

        # Get the 0-based indices of the pages in this collection
        page_indices = self._get_page_indices()

        logger.info(f"Applying OCR via parent PDF to page indices: {page_indices} in collection.")

        # Delegate the batch call to the parent PDF object, passing direct args and apply_exclusions
        parent_pdf.apply_ocr(
            pages=page_indices,
            engine=engine,
            languages=languages,
            min_confidence=min_confidence,  # Pass the renamed parameter
            device=device,
            resolution=resolution,
            apply_exclusions=apply_exclusions,  # Pass down
            replace=replace,  # Pass the replace parameter
            options=options,
        )
        # The PDF method modifies the Page objects directly by adding elements.

        return self  # Return self for chaining

    def _iter_sections(self) -> Iterable["SupportsSections"]:
        return iter(self.pages)

    def split(
        self,
        divider: BoundarySource,
        *,
        include_boundaries: str = "start",
        orientation: str = "vertical",
        new_section_on_page_break: bool = False,
    ) -> "ElementCollection[Region]":
        """
        Divide this page collection into sections based on the provided divider elements.

        Args:
            divider: Elements or selector string that mark section boundaries
            include_boundaries: How to include boundary elements (default: 'start').
            orientation: 'vertical' or 'horizontal' (default: 'vertical').
            new_section_on_page_break: Whether to split at page boundaries (default: False).

        Returns:
            ElementCollection of Region objects representing the sections

        Example:
            # Split a PDF by chapter titles
            chapters = pdf.pages.split("text[size>20]:contains('CHAPTER')")

            # Split by page breaks
            page_sections = pdf.pages.split(None, new_section_on_page_break=True)

            # Split multi-page document by section headers
            sections = pdf.pages[10:20].split("text:bold:contains('Section')")
        """
        sections = self.get_sections(
            start_elements=divider,
            include_boundaries=include_boundaries,
            orientation=orientation,
            new_section_on_page_break=new_section_on_page_break,
        )

        # Add initial section if there's content before the first divider
        if sections and divider is not None:
            # Get all elements across all pages
            all_elements = []
            for page in self.pages:
                all_elements.extend(page.get_elements())

            if all_elements:
                # Find first divider
                if isinstance(divider, str):
                    # Search for first matching element
                    first_divider = None
                    for page in self.pages:
                        match = page.find(divider)
                        if match:
                            first_divider = match
                            break
                else:
                    # divider is already elements
                    first_divider = None
                    if isinstance(divider, Iterable):
                        first_candidate = next(iter(divider), None)
                    else:
                        first_candidate = divider
                    if isinstance(first_candidate, SupportsGeometry):
                        first_divider = first_candidate

                if first_divider and all_elements[0] != first_divider:
                    # There's content before the first divider
                    # Get section from start to first divider
                    initial_sections = self.get_sections(
                        start_elements=None,
                        end_elements=[first_divider],
                        include_boundaries="none",
                        orientation=orientation,
                    )
                    if initial_sections:
                        sections = ElementCollection([initial_sections[0]] + list(sections))

        return sections

    def get_sections(
        self,
        start_elements: BoundarySource = None,
        end_elements: BoundarySource = None,
        new_section_on_page_break: bool = False,
        include_boundaries: str = "both",
        orientation: str = "vertical",
    ) -> "ElementCollection":
        """Extract logical sections across this collection of pages.

        This delegates to :class:`natural_pdf.flows.flow.Flow`, which already
        implements the heavy lifting for cross-segment section extraction and
        returns either :class:`Region` or :class:`FlowRegion` objects as
        appropriate.  The arrangement is chosen based on the requested
        orientation so that horizontal sections continue to work for rotated
        content.
        """

        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.flows.flow import Flow

        if len(self) == 0:
            return ElementCollection([])

        arrangement = "vertical" if orientation == "vertical" else "horizontal"

        flow = Flow(self, arrangement=arrangement)
        sections = flow.get_sections(
            start_elements=start_elements,
            end_elements=end_elements,
            new_section_on_page_break=new_section_on_page_break,
            include_boundaries=include_boundaries,
            orientation=orientation,
        )
        section_list = sections.elements if hasattr(sections, "elements") else list(sections)
        cleaned = sanitize_sections(section_list, orientation=orientation)
        if len(cleaned) == len(section_list):
            return sections
        return ElementCollection(cleaned)

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
        Gather analysis data from all pages in the collection.

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
        if not self.elements:
            logger.warning("No pages found in collection")
            return []

        all_data = []

        for page in self.elements:
            # Basic page information
            page_data = {
                "page_number": page.number,
                "page_index": page.index,
                "width": page.width,
                "height": page.height,
            }

            # Add PDF information if available
            pdf_obj = getattr(page, "pdf", None)
            if pdf_obj is not None:
                pdf_path = getattr(pdf_obj, "path", None)
                if pdf_path:
                    page_data["pdf_path"] = pdf_path
                    page_data["pdf_filename"] = Path(pdf_path).name

            # Include extracted text if requested
            if include_content:
                try:
                    page_data["content"] = page.extract_text(preserve_whitespace=True)
                except Exception as e:
                    logger.error(f"Error extracting text from page {page.number}: {e}")
                    page_data["content"] = ""

            # Save image if requested
            if include_images:
                if image_dir is None:
                    logger.error("image_dir must be provided when include_images=True")
                else:
                    try:
                        # Create image filename
                        pdf_name = "unknown"
                        if pdf_obj is not None:
                            pdf_path = getattr(pdf_obj, "path", None)
                            if pdf_path:
                                pdf_name = Path(pdf_path).stem

                        image_filename = f"{pdf_name}_page_{page.number}.{image_format}"
                        image_path = image_dir / image_filename

                        # Save image
                        page.save_image(
                            str(image_path), resolution=image_resolution, include_highlights=True
                        )

                        # Add relative path to data
                        page_data["image_path"] = str(
                            Path(image_path).relative_to(image_dir.parent)
                        )
                    except Exception as e:
                        logger.error(f"Error saving image for page {page.number}: {e}")
                        page_data["image_path"] = None

            # Add analyses data
            if hasattr(page, "analyses") and page.analyses:
                for key in analysis_keys:
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

    # --- Deskew Method --- #

    def deskew(
        self,
        resolution: int = 300,
        detection_resolution: int = 72,
        force_overwrite: bool = False,
        **deskew_kwargs,
    ) -> "PDF":  # Changed return type
        """
        Creates a new, in-memory PDF object containing deskewed versions of the pages
        in this collection.

        This method delegates the actual processing to the parent PDF object's
        `deskew` method.

        Important: The returned PDF is image-based. Any existing text, OCR results,
        annotations, or other elements from the original pages will *not* be carried over.

        Args:
            resolution: DPI resolution for rendering the output deskewed pages.
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
            ImportError: If 'deskew' or 'img2pdf' libraries are not installed (raised by PDF.deskew).
            ValueError: If `force_overwrite` is False and target pages contain elements (raised by PDF.deskew),
                        or if the collection is empty.
            RuntimeError: If pages lack a parent PDF reference, or the parent PDF lacks the `deskew` method.
        """
        if not self.pages:
            logger.warning("Cannot deskew an empty PageCollection.")
            raise ValueError("Cannot deskew an empty PageCollection.")

        parent_pdf = self._resolve_parent_pdf()

        if not parent_pdf or not hasattr(parent_pdf, "deskew") or not callable(parent_pdf.deskew):
            raise RuntimeError(
                "Parent PDF reference not found or parent PDF lacks the required 'deskew' method."
            )

        # Get the 0-based indices of the pages in this collection
        page_indices = self._get_page_indices()
        logger.info(
            f"PageCollection: Delegating deskew to parent PDF for page indices: {page_indices}"
        )

        # Delegate the call to the parent PDF object for the relevant pages
        # Pass all relevant arguments through (no output_path anymore)
        return parent_pdf.deskew(
            pages=page_indices,
            resolution=resolution,
            detection_resolution=detection_resolution,
            force_overwrite=force_overwrite,
            **deskew_kwargs,
        )

    # --- End Deskew Method --- #

    def _get_render_specs(
        self,
        mode: Literal["show", "render"] = "show",
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        highlights: Optional[List[Dict[str, Any]]] = None,
        crop: Union[bool, Literal["content"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> List[RenderSpec]:
        """Get render specifications for this page collection.

        For page collections, we return specs for all pages that will be
        rendered into a grid layout.

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
        specs = []

        # Get max pages from kwargs if specified
        max_pages = kwargs.get("max_pages")
        pages_to_render = self.pages[:max_pages] if max_pages else self.pages

        for page in pages_to_render:
            if hasattr(page, "_get_render_specs"):
                # Page has the new unified rendering
                page_specs = page._get_render_specs(
                    mode=mode,
                    color=color,
                    highlights=highlights,
                    crop=crop,
                    crop_bbox=crop_bbox,
                    **kwargs,
                )
                specs.extend(page_specs)
            else:
                # Fallback for pages without unified rendering
                spec = RenderSpec(page=page)
                if crop_bbox:
                    spec.crop_bbox = crop_bbox
                specs.append(spec)

        return specs

    def save_pdf(
        self,
        output_path: Union[str, Path],
        ocr: bool = False,
        original: bool = False,
        dpi: int = 300,
    ):
        """
        Saves the pages in this collection to a new PDF file.

        Choose one saving mode:
        - `ocr=True`: Creates a new, image-based PDF using OCR results. This
          makes the text generated during the natural-pdf session searchable,
          but loses original vector content. Requires 'ocr-export' extras.
        - `original=True`: Extracts the original pages from the source PDF,
          preserving all vector content, fonts, and annotations. OCR results
          from the natural-pdf session are NOT included. Requires 'ocr-export' extras.

        Args:
            output_path: Path to save the new PDF file.
            ocr: If True, save as a searchable, image-based PDF using OCR data.
            original: If True, save the original, vector-based pages.
            dpi: Resolution (dots per inch) used only when ocr=True for
                 rendering page images and aligning the text layer.

        Raises:
            ValueError: If the collection is empty, if neither or both 'ocr'
                        and 'original' are True, or if 'original=True' and
                        pages originate from different PDFs.
            ImportError: If required libraries ('pikepdf', 'Pillow')
                         are not installed for the chosen mode.
            RuntimeError: If an unexpected error occurs during saving.
        """
        if not self.pages:
            raise ValueError("Cannot save an empty PageCollection.")

        if not (ocr ^ original):  # XOR: exactly one must be true
            raise ValueError("Exactly one of 'ocr' or 'original' must be True.")

        output_path_obj = Path(output_path)
        output_path_str = str(output_path_obj)

        if ocr:
            if create_searchable_pdf is None:
                raise ImportError(
                    "Saving with ocr=True requires 'pikepdf' and 'Pillow'. "
                    'Install with: pip install \\"natural-pdf[ocr-export]\\"'  # Escaped quotes
                )

            # Check for non-OCR vector elements (provide a warning)
            has_vector_elements = False
            for page in self.pages:
                # Simplified check for common vector types or non-OCR chars/words
                rects = getattr(page, "rects", [])
                lines = getattr(page, "lines", [])
                curves = getattr(page, "curves", [])
                chars = getattr(page, "chars", [])
                words = getattr(page, "words", [])

                if (
                    rects
                    or lines
                    or curves
                    or any(getattr(el, "source", None) != "ocr" for el in chars)
                    or any(getattr(el, "source", None) != "ocr" for el in words)
                ):
                    has_vector_elements = True
                    break
            if has_vector_elements:
                logger.warning(
                    "Warning: Saving with ocr=True creates an image-based PDF. "
                    "Original vector elements (rects, lines, non-OCR text/chars) "
                    "on selected pages will not be preserved in the output file."
                )

            logger.info(f"Saving searchable PDF (OCR text layer) to: {output_path_str}")
            try:
                # Delegate to the searchable PDF exporter function
                # Pass `self` (the PageCollection instance) as the source
                create_searchable_pdf(self, output_path_str, dpi=dpi)
                # Success log is now inside create_searchable_pdf if needed, or keep here
                # logger.info(f"Successfully saved searchable PDF to: {output_path_str}")
            except Exception as e:
                logger.error(f"Failed to create searchable PDF: {e}", exc_info=True)
                # Re-raise as RuntimeError for consistency, potentially handled in exporter too
                raise RuntimeError(f"Failed to create searchable PDF: {e}") from e

        elif original:
            # ---> MODIFIED: Call the new exporter
            if create_original_pdf is None:
                raise ImportError(
                    "Saving with original=True requires 'pikepdf'. "
                    'Install with: pip install \\"natural-pdf[ocr-export]\\"'  # Escaped quotes
                )

            # Check for OCR elements (provide a warning) - keep this check here
            has_ocr_elements = False
            for page in self.pages:
                # Use find_all which returns a collection; check if it's non-empty
                if hasattr(page, "find_all"):
                    ocr_text_elements = page.find_all("text[source=ocr]")
                    if ocr_text_elements:  # Check truthiness of collection
                        has_ocr_elements = True
                        break
                elif hasattr(page, "words"):  # Fallback check if find_all isn't present?
                    if any(getattr(el, "source", None) == "ocr" for el in page.words):
                        has_ocr_elements = True
                        break

            if has_ocr_elements:
                logger.warning(
                    "Warning: Saving with original=True preserves original page content. "
                    "OCR text generated in this session will not be included in the saved file."
                )

            logger.info(f"Saving original pages PDF to: {output_path_str}")
            try:
                # Delegate to the original PDF exporter function
                # Pass `self` (the PageCollection instance) as the source
                create_original_pdf(self, output_path_str)
                # Success log is now inside create_original_pdf
                # logger.info(f"Successfully saved original pages PDF to: {output_path_str}")
            except Exception as e:
                # Error logging is handled within create_original_pdf
                # Re-raise the exception caught from the exporter
                raise e  # Keep the original exception type (ValueError, RuntimeError, etc.)
            # <--- END MODIFIED

    def to_flow(
        self,
        arrangement: Literal["vertical", "horizontal"] = "vertical",
        alignment: Literal["start", "center", "end", "top", "left", "bottom", "right"] = "start",
        segment_gap: float = 0.0,
    ) -> "Flow":
        """
        Convert this PageCollection to a Flow for cross-page operations.

        This enables treating multiple pages as a continuous logical document
        structure, useful for multi-page tables, articles spanning columns,
        or any content requiring reading order across page boundaries.

        Args:
            arrangement: Primary flow direction ('vertical' or 'horizontal').
                        'vertical' stacks pages top-to-bottom (most common).
                        'horizontal' arranges pages left-to-right.
            alignment: Cross-axis alignment for pages of different sizes:
                      For vertical: 'left'/'start', 'center', 'right'/'end'
                      For horizontal: 'top'/'start', 'center', 'bottom'/'end'
            segment_gap: Virtual gap between pages in PDF points (default: 0.0).

        Returns:
            Flow object that can perform operations across all pages in sequence.

        Example:
            Multi-page table extraction:
            ```python
            pdf = npdf.PDF("multi_page_report.pdf")

            # Create flow for pages 2-4 containing a table
            table_flow = pdf.pages[1:4].to_flow()

            # Extract table as if it were continuous
            table_data = table_flow.extract_table()
            df = table_data.df
            ```

            Cross-page element search:
            ```python
            # Find all headers across multiple pages
            headers = pdf.pages[5:10].to_flow().find_all('text[size>12]:bold')

            # Analyze layout across pages
            regions = pdf.pages.to_flow().analyze_layout(engine='yolo')
            ```
        """
        from natural_pdf.flows.flow import Flow

        return Flow(
            segments=self,  # Flow constructor now handles PageCollection
            arrangement=arrangement,
            alignment=alignment,
            segment_gap=segment_gap,
        )

    def analyze_layout(self, *args, **kwargs) -> "ElementCollection[Region]":
        """
        Analyzes the layout of each page in the collection.

        This method iterates through each page, calls its analyze_layout method,
        and returns a single ElementCollection containing all the detected layout
        regions from all pages.

        Args:
            *args: Positional arguments to pass to each page's analyze_layout method.
            **kwargs: Keyword arguments to pass to each page's analyze_layout method.
                      A 'show_progress' kwarg can be included to show a progress bar.

        Returns:
            An ElementCollection of all detected Region objects.
        """
        all_regions = []

        show_progress = kwargs.pop("show_progress", True)

        iterator = self.pages
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(self.pages, desc="Analyzing layout")
            except ImportError:
                pass  # tqdm not installed

        for page in iterator:
            # Each page's analyze_layout method returns an ElementCollection
            regions_collection = page.analyze_layout(*args, **kwargs)
            if regions_collection:
                all_regions.extend(regions_collection.elements)

        return ElementCollection(all_regions)

    def detect_checkboxes(self, *args, **kwargs) -> "ElementCollection[Region]":
        """
        Detects checkboxes on each page in the collection.

        This method iterates through each page, calls its detect_checkboxes method,
        and returns a single ElementCollection containing all detected checkbox
        regions from all pages.

        Args:
            *args: Positional arguments to pass to each page's detect_checkboxes method.
            **kwargs: Keyword arguments to pass to each page's detect_checkboxes method.
                      A 'show_progress' kwarg can be included to show a progress bar.

        Returns:
            An ElementCollection of all detected checkbox Region objects.
        """
        all_checkboxes = []

        show_progress = kwargs.pop("show_progress", True)

        iterator = self.pages
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(self.pages, desc="Detecting checkboxes")
            except ImportError:
                pass  # tqdm not installed

        for page in iterator:
            # Each page's detect_checkboxes method returns an ElementCollection
            checkbox_collection = page.detect_checkboxes(*args, **kwargs)
            if checkbox_collection:
                all_checkboxes.extend(checkbox_collection.elements)

        return ElementCollection(all_checkboxes)

    def highlights(self, show: bool = False) -> "HighlightContext":
        """
        Create a highlight context for accumulating highlights.

        This allows for clean syntax to show multiple highlight groups:

        Example:
            with pages.highlights() as h:
                h.add(pages.find_all('table'), label='tables', color='blue')
                h.add(pages.find_all('text:bold'), label='bold text', color='red')
                h.show()

        Or with automatic display:
            with pages.highlights(show=True) as h:
                h.add(pages.find_all('table'), label='tables')
                h.add(pages.find_all('text:bold'), label='bold')
                # Automatically shows when exiting the context

        Args:
            show: If True, automatically show highlights when exiting context

        Returns:
            HighlightContext for accumulating highlights
        """
        from natural_pdf.core.highlighting_service import HighlightContext

        return HighlightContext(self, show_on_exit=show)

    def groupby(self, by: Union[str, Callable], *, show_progress: bool = True) -> "PageGroupBy":
        """
        Group pages by selector text or callable result.

        Args:
            by: CSS selector string or callable function
            show_progress: Whether to show progress bar during computation (default: True)

        Returns:
            PageGroupBy object supporting iteration and dict-like access

        Examples:
            # Group by header text
            for title, pages in pdf.pages.groupby('text[size=16]'):
                print(f"Section: {title}")

            # Group by callable
            for city, pages in pdf.pages.groupby(lambda p: p.find('text:contains("CITY")').extract_text()):
                process_city_pages(pages)

            # Quick exploration with indexing
            grouped = pdf.pages.groupby('text[size=16]')
            grouped.info()                    # Show all groups
            first_section = grouped[0]        # First group
            last_section = grouped[-1]       # Last group

            # Dict-like access by name
            madison_pages = grouped.get('CITY OF MADISON')
            madison_pages = grouped['CITY OF MADISON']  # Alternative

            # Disable progress bar for small collections
            grouped = pdf.pages.groupby('text[size=16]', show_progress=False)
        """
        from natural_pdf.core.page_groupby import PageGroupBy

        return PageGroupBy(self, by, show_progress=show_progress)
