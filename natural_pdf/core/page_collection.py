import hashlib
import logging
from collections.abc import MutableSequence, Sequence
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
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from pdfplumber.utils.geometry import objects_to_bbox

# New Imports
from pdfplumber.utils.text import TEXTMAP_KWARGS, WORD_EXTRACTOR_KWARGS, chars_to_textmap
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from natural_pdf.analyzers.checkbox.mixin import CheckboxDetectionMixin
from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin
from natural_pdf.classification.manager import ClassificationManager
from natural_pdf.classification.mixin import ClassificationMixin
from natural_pdf.collections.mixins import ApplyMixin, DirectionalCollectionMixin
from natural_pdf.core.pdf import PDF
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.describe.mixin import DescribeMixin, InspectMixin
from natural_pdf.elements.base import Element
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement
from natural_pdf.export.mixin import ExportMixin
from natural_pdf.ocr import OCROptions
from natural_pdf.ocr.utils import _apply_ocr_correction_to_elements
from natural_pdf.selectors.parser import parse_selector, selector_to_filter_func
from natural_pdf.text_mixin import TextMixin

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
    from natural_pdf.core.page import Page
    from natural_pdf.core.pdf import PDF  # ---> ADDED PDF type hint
    from natural_pdf.elements.region import Region
    from natural_pdf.elements.text import TextElement  # Ensure TextElement is imported
    from natural_pdf.flows.flow import Flow

T = TypeVar("T")
P = TypeVar("P", bound="Page")


class PageCollection(
    TextMixin, Generic[P], ApplyMixin, ShapeDetectionMixin, CheckboxDetectionMixin, Visualizable
):
    """
    Represents a collection of Page objects, often from a single PDF document.
    Provides methods for batch operations on these pages.
    """

    def __init__(self, pages: Union[List[P], Sequence[P]]):
        """
        Initialize a page collection.

        Args:
            pages: List or sequence of Page objects (can be lazy)
        """
        # Store the sequence as-is to preserve lazy behavior
        # Only convert to list if we need list-specific operations
        if hasattr(pages, "__iter__") and hasattr(pages, "__len__"):
            self.pages = pages
        else:
            # Fallback for non-sequence types
            self.pages = list(pages)

    def __len__(self) -> int:
        """Return the number of pages in the collection."""
        return len(self.pages)

    def __getitem__(self, idx) -> Union[P, "PageCollection[P]"]:
        """Support indexing and slicing."""
        if isinstance(idx, slice):
            return PageCollection(self.pages[idx])
        return self.pages[idx]

    def __iter__(self) -> Iterator[P]:
        """Support iteration."""
        return iter(self.pages)

    def __repr__(self) -> str:
        """Return a string representation showing the page count."""
        return f"<PageCollection(count={len(self)})>"

    def _get_items_for_apply(self) -> Iterator[P]:
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
        if hasattr(self.pages, "_indices"):
            # If it's a _LazyPageList (or slice), get indices directly
            return list(self.pages._indices)
        else:
            # Fallback: if pages are already materialized, get indices normally
            # This will force materialization but only if pages aren't lazy
            return [p.index for p in self.pages]

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
        texts = []
        for page in self.pages:
            text = page.extract_text(
                keep_blank_chars=keep_blank_chars,
                apply_exclusions=apply_exclusions,
                **kwargs,
            )
            texts.append(text)

        combined = "\n".join(texts)

        # Default strip behaviour: if caller picks, honour; else respect layout flag passed via kwargs.
        use_layout = kwargs.get("layout", False)
        strip_final = strip if strip is not None else (not use_layout)

        if strip_final:
            combined = "\n".join(line.rstrip() for line in combined.splitlines()).strip()

        return combined

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
    ) -> "PageCollection[P]":
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

        # Assume all pages share the same parent PDF object
        first_page = self.pages[0]
        if not hasattr(first_page, "_parent") or not first_page._parent:
            raise RuntimeError("Pages in this collection do not have a parent PDF reference.")

        parent_pdf = first_page._parent

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

    @overload
    def find(
        self,
        *,
        text: str,
        overlap: str = "full",
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> Optional[T]: ...

    @overload
    def find(
        self,
        selector: str,
        *,
        overlap: str = "full",
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> Optional[T]: ...

    def find(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[str] = None,
        overlap: str = "full",
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> Optional[T]:
        """
        Find the first element matching the selector OR text across all pages in the collection.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for (equivalent to 'text:contains(...)').
            overlap: How to determine if elements overlap: 'full' (fully inside),
                     'partial' (any overlap), or 'center' (center point inside).
                     (default: "full")
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            **kwargs: Additional filter parameters.

        Returns:
            First matching element or None.
        """
        # Input validation happens within page.find
        for page in self.pages:
            element = page.find(
                selector=selector,
                text=text,
                overlap=overlap,
                apply_exclusions=apply_exclusions,
                regex=regex,
                case=case,
                **kwargs,
            )
            if element:
                return element
        return None

    @overload
    def find_all(
        self,
        *,
        text: str,
        overlap: str = "full",
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
        overlap: str = "full",
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
        overlap: str = "full",
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> "ElementCollection":
        """
        Find all elements matching the selector OR text across all pages in the collection.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for (equivalent to 'text:contains(...)').
            overlap: How to determine if elements overlap: 'full' (fully inside),
                     'partial' (any overlap), or 'center' (center point inside).
                     (default: "full")
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            **kwargs: Additional filter parameters.

        Returns:
            ElementCollection with matching elements from all pages.
        """
        all_elements = []
        # Input validation happens within page.find_all
        for page in self.pages:
            elements = page.find_all(
                selector=selector,
                text=text,
                overlap=overlap,
                apply_exclusions=apply_exclusions,
                regex=regex,
                case=case,
                **kwargs,
            )
            if elements:
                all_elements.extend(elements.elements)

        return ElementCollection(all_elements)

    def update_text(
        self,
        transform: Callable[[Any], Optional[str]],
        selector: str = "text",
        max_workers: Optional[int] = None,
    ) -> "PageCollection[P]":
        """
        Applies corrections to text elements across all pages
        in this collection using a user-provided callback function, executed
        in parallel if `max_workers` is specified.

        This method delegates to the parent PDF's `update_text` method,
        targeting all pages within this collection.

        Args:
            transform: A function that accepts a single argument (an element
                       object) and returns `Optional[str]` (new text or None).
            selector: The attribute name to update. Default is 'text'.
            max_workers: The maximum number of worker threads to use for parallel
                         correction on each page. If None, defaults are used.

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If the collection is empty, pages lack a parent PDF reference,
                          or the parent PDF lacks the `update_text` method.
        """
        if not self.pages:
            logger.warning("Cannot update text for an empty PageCollection.")
            # Return self even if empty to maintain chaining consistency
            return self

        # Assume all pages share the same parent PDF object
        parent_pdf = self.pages[0]._parent
        if (
            not parent_pdf
            or not hasattr(parent_pdf, "update_text")
            or not callable(parent_pdf.update_text)
        ):
            raise RuntimeError(
                "Parent PDF reference not found or parent PDF lacks the required 'update_text' method."
            )

        page_indices = self._get_page_indices()
        logger.info(
            f"PageCollection: Delegating text update to parent PDF for page indices: {page_indices} with max_workers={max_workers} and selector='{selector}'."
        )

        # Delegate the call to the parent PDF object for the relevant pages
        # Pass the max_workers parameter down
        parent_pdf.update_text(
            transform=transform,
            pages=page_indices,
            selector=selector,
            max_workers=max_workers,
        )

        return self

    def get_sections(
        self,
        start_elements=None,
        end_elements=None,
        new_section_on_page_break=False,
        include_boundaries="both",
        orientation="vertical",
    ) -> "ElementCollection[Region]":
        """
        Extract sections from a page collection based on start/end elements.

        Args:
            start_elements: Elements or selector string that mark the start of sections (optional)
            end_elements: Elements or selector string that mark the end of sections (optional)
            new_section_on_page_break: Whether to start a new section at page boundaries (default: False)
            include_boundaries: How to include boundary elements: 'start', 'end', 'both', or 'none' (default: 'both')
            orientation: 'vertical' (default) or 'horizontal' - determines section direction

        Returns:
            List of Region objects representing the extracted sections

        Note:
            You can provide only start_elements, only end_elements, or both.
            - With only start_elements: sections go from each start to the next start (or end of page)
            - With only end_elements: sections go from beginning of document/page to each end
            - With both: sections go from each start to the corresponding end
        """
        # Find start and end elements across all pages
        if isinstance(start_elements, str):
            start_elements = self.find_all(start_elements).elements

        if isinstance(end_elements, str):
            end_elements = self.find_all(end_elements).elements

        # If no start elements and no end elements, return empty list
        if not start_elements and not end_elements:
            return []

        # If there are page break boundaries, we'll need to add them
        if new_section_on_page_break:
            # For each page boundary, create virtual "end" and "start" elements
            for i in range(len(self.pages) - 1):
                # Add a virtual "end" element at the bottom of the current page
                page = self.pages[i]
                # If end_elements is None, initialize it as an empty list
                if end_elements is None:
                    end_elements = []

                # Create a region at the bottom of the page as an artificial end marker
                from natural_pdf.elements.region import Region

                bottom_region = Region(page, (0, page.height - 1, page.width, page.height))
                bottom_region.is_page_boundary = True  # Mark it as a special boundary
                end_elements.append(bottom_region)

                # Add a virtual "start" element at the top of the next page
                next_page = self.pages[i + 1]
                top_region = Region(next_page, (0, 0, next_page.width, 1))
                top_region.is_page_boundary = True  # Mark it as a special boundary
                # If start_elements is None, initialize it as an empty list
                if start_elements is None:
                    start_elements = []
                start_elements.append(top_region)

        # Get all elements from all pages and sort them in document order
        all_elements = []
        for page in self.pages:
            elements = page.get_elements()
            all_elements.extend(elements)

        # Sort by page index, then vertical position, then horizontal position
        all_elements.sort(key=lambda e: (e.page.index, e.top, e.x0))

        # If we only have end_elements (no start_elements), create implicit start elements
        if not start_elements and end_elements:
            from natural_pdf.elements.region import Region

            start_elements = []

            # Add implicit start at the beginning of the first page
            first_page = self.pages[0]
            first_start = Region(first_page, (0, 0, first_page.width, 1))
            first_start.is_implicit_start = True
            # Don't mark this as created from any end element, so it can pair with any end
            start_elements.append(first_start)

            # For each end element (except the last), add an implicit start after it
            # Sort by page, then top, then bottom (for elements with same top), then x0
            sorted_end_elements = sorted(
                end_elements, key=lambda e: (e.page.index, e.top, e.bottom, e.x0)
            )
            for i, end_elem in enumerate(sorted_end_elements[:-1]):  # Exclude last end element
                # Create implicit start element right after this end element
                implicit_start = Region(
                    end_elem.page, (0, end_elem.bottom, end_elem.page.width, end_elem.bottom + 1)
                )
                implicit_start.is_implicit_start = True
                # Track which end element this implicit start was created from
                # to avoid pairing them together (which would create zero height)
                implicit_start.created_from_end = end_elem
                start_elements.append(implicit_start)

        # Mark section boundaries
        section_boundaries = []

        # Add start element boundaries
        for element in start_elements:
            if element in all_elements:
                idx = all_elements.index(element)
                section_boundaries.append(
                    {
                        "index": idx,
                        "element": element,
                        "type": "start",
                        "page_idx": element.page.index,
                    }
                )
            elif hasattr(element, "is_page_boundary") and element.is_page_boundary:
                # This is a virtual page boundary element
                section_boundaries.append(
                    {
                        "index": -1,  # Special index for page boundaries
                        "element": element,
                        "type": "start",
                        "page_idx": element.page.index,
                    }
                )
            elif hasattr(element, "is_implicit_start") and element.is_implicit_start:
                # This is an implicit start element
                section_boundaries.append(
                    {
                        "index": -2,  # Special index for implicit starts
                        "element": element,
                        "type": "start",
                        "page_idx": element.page.index,
                    }
                )

        # Add end element boundaries if provided
        if end_elements:
            for element in end_elements:
                if element in all_elements:
                    idx = all_elements.index(element)
                    section_boundaries.append(
                        {
                            "index": idx,
                            "element": element,
                            "type": "end",
                            "page_idx": element.page.index,
                        }
                    )
                elif hasattr(element, "is_page_boundary") and element.is_page_boundary:
                    # This is a virtual page boundary element
                    section_boundaries.append(
                        {
                            "index": -1,  # Special index for page boundaries
                            "element": element,
                            "type": "end",
                            "page_idx": element.page.index,
                        }
                    )

        # Sort boundaries by page index, then by actual document position
        def _sort_key(boundary):
            """Sort boundaries by (page_idx, position, priority)."""
            page_idx = boundary["page_idx"]
            element = boundary["element"]

            # Position on the page based on orientation
            if orientation == "vertical":
                pos = getattr(element, "top", 0.0)
            else:  # horizontal
                pos = getattr(element, "x0", 0.0)

            # Ensure starts come before ends at the same coordinate
            priority = 0 if boundary["type"] == "start" else 1

            return (page_idx, pos, priority)

        section_boundaries.sort(key=_sort_key)

        # Generate sections
        sections = []

        # --- Helper: build a FlowRegion spanning multiple pages ---
        def _build_flow_region(start_el, end_el, include_boundaries="both", orientation="vertical"):
            """Return a FlowRegion that covers from *start_el* to *end_el*.
            If *end_el* is None, the region continues to the bottom/right of the last
            page in this PageCollection.

            Args:
                start_el: Start element
                end_el: End element
                include_boundaries: How to include boundary elements: 'start', 'end', 'both', or 'none'
                orientation: 'vertical' or 'horizontal' - determines section direction
            """
            # Local imports to avoid top-level cycles
            from natural_pdf.elements.region import Region
            from natural_pdf.flows.element import FlowElement
            from natural_pdf.flows.flow import Flow
            from natural_pdf.flows.region import FlowRegion

            start_pg = start_el.page
            end_pg = end_el.page if end_el is not None else self.pages[-1]

            parts: list[Region] = []

            if orientation == "vertical":
                # Determine the start_top based on include_boundaries
                start_top = start_el.top
                if include_boundaries == "none" or include_boundaries == "end":
                    # Exclude start boundary
                    start_top = start_el.bottom if hasattr(start_el, "bottom") else start_el.top

                # Slice of first page beginning at *start_top*
                parts.append(Region(start_pg, (0, start_top, start_pg.width, start_pg.height)))
            else:  # horizontal
                # Determine the start_left based on include_boundaries
                start_left = start_el.x0
                if include_boundaries == "none" or include_boundaries == "end":
                    # Exclude start boundary
                    start_left = start_el.x1 if hasattr(start_el, "x1") else start_el.x0

                # Slice of first page beginning at *start_left*
                parts.append(Region(start_pg, (start_left, 0, start_pg.width, start_pg.height)))

            # Full middle pages
            for pg_idx in range(start_pg.index + 1, end_pg.index):
                mid_pg = self.pages[pg_idx]
                parts.append(Region(mid_pg, (0, 0, mid_pg.width, mid_pg.height)))

            # Slice of last page (if distinct)
            if end_pg is not start_pg:
                if orientation == "vertical":
                    # Determine the bottom based on include_boundaries
                    if end_el is not None:
                        if include_boundaries == "none" or include_boundaries == "start":
                            # Exclude end boundary
                            bottom = end_el.top if hasattr(end_el, "top") else end_el.bottom
                        else:
                            # Include end boundary
                            bottom = end_el.bottom
                    else:
                        bottom = end_pg.height
                    parts.append(Region(end_pg, (0, 0, end_pg.width, bottom)))
                else:  # horizontal
                    # Determine the right based on include_boundaries
                    if end_el is not None:
                        if include_boundaries == "none" or include_boundaries == "start":
                            # Exclude end boundary
                            right = end_el.x0 if hasattr(end_el, "x0") else end_el.x1
                        else:
                            # Include end boundary
                            right = end_el.x1
                    else:
                        right = end_pg.width
                    parts.append(Region(end_pg, (0, 0, right, end_pg.height)))

            flow = Flow(segments=parts, arrangement=orientation)
            src_fe = FlowElement(physical_object=start_el, flow=flow)
            return FlowRegion(
                flow=flow,
                constituent_regions=parts,
                source_flow_element=src_fe,
                boundary_element_found=end_el,
            )

        # ------------------------------------------------------------------

        current_start = None

        for i, boundary in enumerate(section_boundaries):
            # If it's a start boundary and we don't have a current start
            if boundary["type"] == "start" and current_start is None:
                current_start = boundary

            # If it's an end boundary and we have a current start
            elif boundary["type"] == "end" and current_start is not None:
                # Create a section from current_start to this boundary
                start_element = current_start["element"]
                end_element = boundary["element"]

                # Check if this is an implicit start created from this same end element
                # This would create a zero-height section, so skip this pairing
                if (
                    hasattr(start_element, "is_implicit_start")
                    and hasattr(start_element, "created_from_end")
                    and start_element.created_from_end is end_element
                ):
                    # Skip this pairing - keep current_start for next end element
                    continue

                # If both elements are on the same page, use the page's get_section_between
                if start_element.page == end_element.page:
                    # For implicit start elements, create a region from the top of the page
                    if hasattr(start_element, "is_implicit_start"):
                        from natural_pdf.elements.region import Region

                        # Adjust boundaries based on include_boundaries parameter and orientation
                        if orientation == "vertical":
                            top = start_element.top
                            bottom = end_element.bottom

                            if include_boundaries == "none":
                                # Exclude both boundaries - move past them
                                top = (
                                    start_element.bottom
                                    if hasattr(start_element, "bottom")
                                    else start_element.top
                                )
                                bottom = (
                                    end_element.top
                                    if hasattr(end_element, "top")
                                    else end_element.bottom
                                )
                            elif include_boundaries == "start":
                                # Include start, exclude end
                                bottom = (
                                    end_element.top
                                    if hasattr(end_element, "top")
                                    else end_element.bottom
                                )
                            elif include_boundaries == "end":
                                # Exclude start, include end
                                top = (
                                    start_element.bottom
                                    if hasattr(start_element, "bottom")
                                    else start_element.top
                                )
                            # "both" is default - no adjustment needed

                            section = Region(
                                start_element.page,
                                (0, top, start_element.page.width, bottom),
                            )
                            section._boundary_exclusions = include_boundaries
                        else:  # horizontal
                            left = start_element.x0
                            right = end_element.x1

                            if include_boundaries == "none":
                                # Exclude both boundaries - move past them
                                left = (
                                    start_element.x1
                                    if hasattr(start_element, "x1")
                                    else start_element.x0
                                )
                                right = (
                                    end_element.x0 if hasattr(end_element, "x0") else end_element.x1
                                )
                            elif include_boundaries == "start":
                                # Include start, exclude end
                                right = (
                                    end_element.x0 if hasattr(end_element, "x0") else end_element.x1
                                )
                            elif include_boundaries == "end":
                                # Exclude start, include end
                                left = (
                                    start_element.x1
                                    if hasattr(start_element, "x1")
                                    else start_element.x0
                                )
                            # "both" is default - no adjustment needed

                            section = Region(
                                start_element.page,
                                (left, 0, right, start_element.page.height),
                            )
                            section._boundary_exclusions = include_boundaries
                        section.start_element = start_element
                        section.boundary_element_found = end_element
                    else:
                        section = start_element.page.get_section_between(
                            start_element, end_element, include_boundaries, orientation
                        )
                    sections.append(section)
                else:
                    # Create FlowRegion spanning pages
                    flow_region = _build_flow_region(
                        start_element, end_element, include_boundaries, orientation
                    )
                    sections.append(flow_region)

                current_start = None

            # If it's another start boundary and we have a current start (for splitting by starts only)
            elif boundary["type"] == "start" and current_start is not None and not end_elements:
                # Create a section from current_start to just before this boundary
                start_element = current_start["element"]

                # Create section from current start to just before this new start
                if start_element.page == boundary["element"].page:
                    from natural_pdf.elements.region import Region

                    next_start = boundary["element"]

                    # Create section based on orientation
                    if orientation == "vertical":
                        # Determine vertical bounds
                        if include_boundaries in ["start", "both"]:
                            top = start_element.top
                        else:
                            top = start_element.bottom

                        # The section ends just before the next start
                        bottom = next_start.top

                        # Create the section with full page width
                        if top < bottom:
                            section = Region(
                                start_element.page, (0, top, start_element.page.width, bottom)
                            )
                            section.start_element = start_element
                            section.end_element = (
                                next_start  # The next start is the end of this section
                            )
                            section._boundary_exclusions = include_boundaries
                            sections.append(section)
                    else:  # horizontal
                        # Determine horizontal bounds
                        if include_boundaries in ["start", "both"]:
                            left = start_element.x0
                        else:
                            left = start_element.x1

                        # The section ends just before the next start
                        right = next_start.x0

                        # Create the section with full page height
                        if left < right:
                            section = Region(
                                start_element.page, (left, 0, right, start_element.page.height)
                            )
                            section.start_element = start_element
                            section.end_element = (
                                next_start  # The next start is the end of this section
                            )
                            section._boundary_exclusions = include_boundaries
                            sections.append(section)
                else:
                    # Cross-page section - create from current_start to the end of its page
                    from natural_pdf.elements.region import Region

                    start_page = start_element.page

                    # Handle implicit start elements and respect include_boundaries
                    if orientation == "vertical":
                        if include_boundaries in ["none", "end"]:
                            # Exclude start boundary
                            start_top = (
                                start_element.bottom
                                if hasattr(start_element, "bottom")
                                else start_element.top
                            )
                        else:
                            # Include start boundary
                            start_top = start_element.top

                        region = Region(
                            start_page, (0, start_top, start_page.width, start_page.height)
                        )
                    else:  # horizontal
                        if include_boundaries in ["none", "end"]:
                            # Exclude start boundary
                            start_left = (
                                start_element.x1
                                if hasattr(start_element, "x1")
                                else start_element.x0
                            )
                        else:
                            # Include start boundary
                            start_left = start_element.x0

                        region = Region(
                            start_page, (start_left, 0, start_page.width, start_page.height)
                        )
                    region.start_element = start_element
                    sections.append(region)

                current_start = boundary

        # Handle the last section if we have a current start
        if current_start is not None:
            start_element = current_start["element"]
            start_page = start_element.page

            if end_elements:
                # With end_elements, we need an explicit end - use the last element
                # on the last page of the collection
                last_page = self.pages[-1]
                last_page_elements = [e for e in all_elements if e.page == last_page]
                if orientation == "vertical":
                    last_page_elements.sort(key=lambda e: (e.top, e.x0))
                else:  # horizontal
                    last_page_elements.sort(key=lambda e: (e.x0, e.top))
                end_element = last_page_elements[-1] if last_page_elements else None

                # Create FlowRegion spanning multiple pages using helper
                flow_region = _build_flow_region(
                    start_element, end_element, include_boundaries, orientation
                )
                sections.append(flow_region)
            else:
                # With start_elements only, create a section to the end of the current page
                from natural_pdf.elements.region import Region

                # Handle implicit start elements and respect include_boundaries
                if orientation == "vertical":
                    if include_boundaries in ["none", "end"]:
                        # Exclude start boundary
                        start_top = (
                            start_element.bottom
                            if hasattr(start_element, "bottom")
                            else start_element.top
                        )
                    else:
                        # Include start boundary
                        start_top = start_element.top

                    region = Region(start_page, (0, start_top, start_page.width, start_page.height))
                else:  # horizontal
                    if include_boundaries in ["none", "end"]:
                        # Exclude start boundary
                        start_left = (
                            start_element.x1 if hasattr(start_element, "x1") else start_element.x0
                        )
                    else:
                        # Include start boundary
                        start_left = start_element.x0

                    region = Region(
                        start_page, (start_left, 0, start_page.width, start_page.height)
                    )
                region.start_element = start_element
                sections.append(region)

        return ElementCollection(sections)

    def split(self, divider, **kwargs) -> "ElementCollection[Region]":
        """
        Divide this page collection into sections based on the provided divider elements.

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
            chapters = pdf.pages.split("text[size>20]:contains('CHAPTER')")

            # Split by page breaks
            page_sections = pdf.pages.split(None, new_section_on_page_break=True)

            # Split multi-page document by section headers
            sections = pdf.pages[10:20].split("text:bold:contains('Section')")
        """
        # Default to 'start' boundaries for split (include divider at start of each section)
        if "include_boundaries" not in kwargs:
            kwargs["include_boundaries"] = "start"

        sections = self.get_sections(start_elements=divider, **kwargs)

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
                    first_divider = divider[0] if hasattr(divider, "__getitem__") else divider

                if first_divider and all_elements[0] != first_divider:
                    # There's content before the first divider
                    # Get section from start to first divider
                    initial_sections = self.get_sections(
                        start_elements=None,
                        end_elements=[first_divider],
                        include_boundaries="none",
                        orientation=kwargs.get("orientation", "vertical"),
                    )
                    if initial_sections:
                        sections = ElementCollection([initial_sections[0]] + list(sections))

        return sections

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
            if hasattr(page, "pdf") and page.pdf:
                page_data["pdf_path"] = page.pdf.path
                page_data["pdf_filename"] = Path(page.pdf.path).name

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
                    pdf_name = "unknown"
                    if hasattr(page, "pdf") and page.pdf:
                        pdf_name = Path(page.pdf.path).stem

                    image_filename = f"{pdf_name}_page_{page.number}.{image_format}"
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

        # Assume all pages share the same parent PDF object
        # Need to hint the type of _parent for type checkers
        if TYPE_CHECKING:
            parent_pdf: "natural_pdf.core.pdf.PDF" = self.pages[0]._parent
        else:
            parent_pdf = self.pages[0]._parent

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
