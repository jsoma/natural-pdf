import concurrent.futures  # Added import
import contextlib
import hashlib
import logging
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import (  # Added overload
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import pdfplumber
from PIL import Image
from tqdm.auto import tqdm  # Added tqdm import

from natural_pdf.elements.base import extract_bbox
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.selectors.host_mixin import SelectorHostMixin
from natural_pdf.selectors.parser import build_text_contains_selector, parse_selector
from natural_pdf.tables.result import TableResult

if TYPE_CHECKING:
    from pdfplumber.page import Page as PdfPlumberPage

    from natural_pdf.core.highlighting_service import HighlightContext, HighlightingService
    from natural_pdf.core.pdf import PDF
    from natural_pdf.describe.summary import InspectionSummary
    from natural_pdf.elements.base import Element
else:  # pragma: no cover - runtime typing helper
    PdfPlumberPage = Any  # type: ignore[assignment]

# # New Imports

# # Deskew Imports (Conditional)

from natural_pdf.analyzers.layout.layout_analyzer import LayoutAnalyzer
from natural_pdf.analyzers.layout.layout_options import LayoutOptions
from natural_pdf.analyzers.text_options import TextStyleOptions
from natural_pdf.analyzers.text_structure import TextStyleAnalyzer
from natural_pdf.core.capabilities import AnalysisHostMixin

# Add new import
from natural_pdf.core.crop_utils import resolve_crop_bbox
from natural_pdf.core.element_manager import ElementManager
from natural_pdf.core.interfaces import Bounds, SupportsGeometry
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.core.selector_utils import (
    _apply_relational_post_pseudos,
    execute_selector_query,
    normalize_selector_input,
)
from natural_pdf.describe.mixin import DescribeMixin  # Import describe mixin
from natural_pdf.deskew import run_deskew_apply, run_deskew_detect
from natural_pdf.elements.base import Element  # Import base element
from natural_pdf.elements.text import TextElement
from natural_pdf.extraction.mixin import ExtractionMixin  # Import extraction mixin
from natural_pdf.ocr import OCROptions
from natural_pdf.ocr.ocr_manager import (
    normalize_ocr_options,
    resolve_ocr_device,
    resolve_ocr_engine_name,
    resolve_ocr_languages,
    resolve_ocr_min_confidence,
    run_ocr_apply,
    run_ocr_extract,
)
from natural_pdf.qa.qa_result import QAResult
from natural_pdf.text_mixin import TextMixin

# # Import new utils
from natural_pdf.utils.text_extraction import filter_chars_spatially, generate_text_layout
from natural_pdf.vision.mixin import VisualSearchMixin
from natural_pdf.widgets.viewer import _IPYWIDGETS_AVAILABLE, InteractiveViewerWidget

# End Deskew Imports

logger = logging.getLogger(__name__)


def _jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.

    Args:
        s1: First string.
        s2: Second string.
        prefix_weight: Scaling factor for common prefix bonus (default 0.1).

    Returns:
        Similarity score in the range [0, 1].
    """
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions /= 2

    jaro = ((matches / len1) + (matches / len2) + (matches - transpositions) / matches) / 3.0

    prefix = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            break
        prefix += 1
        if prefix == 4:
            break

    jaro_winkler = jaro + prefix * prefix_weight * (1.0 - jaro)
    return max(0.0, min(1.0, jaro_winkler))


class Page(TextMixin, AnalysisHostMixin, Visualizable):
    """Enhanced Page wrapper built on top of pdfplumber.Page.

    This class provides a fluent interface for working with PDF pages,
    with improved selection, navigation, extraction, and question-answering capabilities.
    It integrates multiple analysis capabilities through mixins and provides spatial
    navigation with CSS-like selectors.

    The Page class serves as the primary interface for document analysis, offering:
    - Element selection and spatial navigation
    - OCR and layout analysis integration
    - Table detection and extraction
    - AI-powered classification and data extraction
    - Visual debugging with highlighting and cropping
    - Text style analysis and structure detection

    Attributes:
        index: Zero-based index of this page in the PDF.
        number: One-based page number (index + 1).
        width: Page width in points.
        height: Page height in points.
        bbox: Bounding box tuple (x0, top, x1, bottom) of the page.
        chars: Collection of character elements on the page.
        words: Collection of word elements on the page.
        lines: Collection of line elements on the page.
        rects: Collection of rectangle elements on the page.
        images: Collection of image elements on the page.
        metadata: Dictionary for storing analysis results and custom data.

    Example:
        Basic usage:
        ```python
        pdf = npdf.PDF("document.pdf")
        page = pdf.pages[0]

        # Find elements with CSS-like selectors
        headers = page.find_all('text[size>12]:bold')
        summaries = page.find('text:contains("Summary")')

        # Spatial navigation
        content_below = summaries.below(until='text[size>12]:bold')

        # Table extraction
        tables = page.extract_table()
        ```

        Advanced usage:
        ```python
        # Apply OCR if needed
        page.apply_ocr(engine='easyocr', resolution=300)

        # Layout analysis
        page.analyze_layout(engine='yolo')

        # AI-powered extraction
        data = page.extract_structured_data(MySchema)

        # Visual debugging
        page.find('text:contains("Important")').show()
        ```
    """

    def __init__(
        self,
        page: "PdfPlumberPage",
        parent: "PDF",
        index: int,
        font_attrs=None,
        load_text: bool = True,
    ):
        """Initialize a page wrapper.

        Creates an enhanced Page object that wraps a pdfplumber page with additional
        functionality for spatial navigation, analysis, and AI-powered extraction.

        Args:
            page: The underlying pdfplumber page object that provides raw PDF data.
            parent: Parent PDF object that contains this page and provides access
                to managers and global settings.
            index: Zero-based index of this page in the PDF document.
            font_attrs: List of font attributes to consider when grouping characters
                into words. Common attributes include ['fontname', 'size', 'flags'].
                If None, uses default character-to-word grouping rules.
            load_text: If True, load and process text elements from the PDF's text layer.
                If False, skip text layer processing (useful for OCR-only workflows).

        Note:
            This constructor is typically called automatically when accessing pages
            through the PDF.pages collection. Direct instantiation is rarely needed.

        Example:
            ```python
            # Pages are usually accessed through the PDF object
            pdf = npdf.PDF("document.pdf")
            page = pdf.pages[0]  # Page object created automatically

            # Direct construction (advanced usage)
            import pdfplumber
            with pdfplumber.open("document.pdf") as plumber_pdf:
                plumber_page = plumber_pdf.pages[0]
                page = Page(plumber_page, pdf, 0, load_text=True)
            ```
        """
        self._page = page
        self._parent = parent
        self._index = index
        self._load_text = load_text
        self._text_styles_summary: Dict[str, Any] = {}
        self._text_styles = None  # Lazy-loaded text style analyzer results
        self._exclusions = []  # List to store exclusion functions/regions
        self._skew_angle: Optional[float] = None  # Stores detected skew angle

        self.metadata: Dict[str, Any] = {}

        # Region management
        self._regions = {
            "detected": [],  # Layout detection results
            "named": {},  # Named regions (name -> region)
        }

        if not hasattr(self._parent, "_config"):
            raise AttributeError(
                "Parent PDF is missing _config; cannot initialise Page configuration"
            )

        # Page-scoped configuration begins as a shallow copy of the parent PDF-level
        # configuration so that auto-computed tolerances or other page-specific
        # values do not overwrite siblings.
        self._config = dict(self._parent._config)

        # Initialize ElementManager, passing font_attrs
        self._element_mgr: ElementManager = ElementManager(
            self, font_attrs=font_attrs, load_text=self._load_text
        )
        # self._highlighter = HighlightingService(self) # REMOVED - Use property accessor
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}
        self.metadata.setdefault("analysis", {})

        # Initialize the internal variable with a single underscore
        self._layout_analyzer = None

        self._load_elements()
        self._to_image_cache: Dict[tuple, Optional["Image.Image"]] = {}

        # Flag to prevent infinite recursion when computing exclusions
        self._computing_exclusions = False

    def _get_render_specs(
        self,
        mode: Literal["show", "render"] = "show",
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        highlights: Optional[List[Dict[str, Any]]] = None,
        crop: Union[bool, Literal["content"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> List[RenderSpec]:
        """Get render specifications for this page.

        Args:
            mode: Rendering mode - 'show' includes page highlights, 'render' is clean
            color: Default color for highlights in show mode
            highlights: Additional highlight groups to show
            crop: Whether to crop the page
            crop_bbox: Explicit crop bounds
            **kwargs: Additional parameters

        Returns:
            List containing a single RenderSpec for this page
        """
        spec = RenderSpec(page=self)

        elements: Optional[List[Element]] = None
        content_bbox: Optional[Tuple[float, float, float, float]] = None

        def ensure_content_bbox() -> Optional[Tuple[float, float, float, float]]:
            nonlocal elements, content_bbox
            if content_bbox is not None:
                return content_bbox
            if elements is None:
                elements = self.get_elements(apply_exclusions=False)
            if not elements:
                return None
            x_coords: List[float] = []
            y_coords: List[float] = []
            for elem in elements:
                if hasattr(elem, "bbox") and elem.bbox:
                    x0, y0, x1, y1 = elem.bbox
                    x_coords.extend([x0, x1])
                    y_coords.extend([y0, y1])
            if x_coords and y_coords:
                content_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                return content_bbox
            return None

        spec.crop_bbox = resolve_crop_bbox(
            width=self.width,
            height=self.height,
            crop=crop,
            crop_bbox=crop_bbox,
            content_bbox_fn=ensure_content_bbox,
        )

        # Add highlights in show mode
        if mode == "show":
            # Add page's persistent highlights if any
            page_highlights = self._highlighter.get_highlights_for_page(self.index)
            for highlight in page_highlights:
                spec.add_highlight(
                    bbox=highlight.bbox,
                    polygon=highlight.polygon,
                    color=highlight.color,
                    label=highlight.label,
                    element=None,  # Persistent highlights don't have element refs
                )

            # Add additional highlight groups if provided
            if highlights:
                for group in highlights:
                    raw_elements = group.get("elements")
                    if not raw_elements:
                        continue

                    if isinstance(raw_elements, ElementCollection):
                        elements_iter: Iterable[Any] = raw_elements.elements
                    elif isinstance(raw_elements, Iterable):
                        elements_iter = cast(Iterable[Any], raw_elements)
                    else:
                        elements_iter = (raw_elements,)

                    group_color = group.get("color", color)
                    group_label = group.get("label")

                    for elem in elements_iter:
                        spec.add_highlight(element=elem, color=group_color, label=group_label)

            # Handle exclusions visualization
            exclusions_param = kwargs.get("exclusions")
            if exclusions_param:
                # Get exclusion regions
                exclusion_regions = self._get_exclusion_regions(include_callable=True)

                if exclusion_regions:
                    # Determine color for exclusions
                    exclusion_color = (
                        exclusions_param if isinstance(exclusions_param, str) else "red"
                    )

                    # Add exclusion regions as highlights
                    for region in exclusion_regions:
                        spec.add_highlight(
                            element=region,
                            color=exclusion_color,
                            label=f"Exclusion: {region.label or 'unnamed'}",
                        )

        return [spec]

    def to_region(self) -> Region:
        """Return a Region covering the full page."""
        return self.region(0, 0, self.width, self.height)

    qa_target = "page"

    def _qa_context_page_number(self) -> int:
        return self.number

    def _qa_source_elements(self) -> ElementCollection:
        return ElementCollection([])

    def _qa_target_region(self) -> Region:
        return self.to_region()

    def _element_to_region(self, element: Any, label: Optional[str] = None) -> Optional[Region]:
        bbox = extract_bbox(element)
        if bbox is None:
            return None
        return Region(self, bbox, label=label)

    def _exclusion_element_manager(self) -> ElementManager:
        return self._element_mgr

    def _invalidate_exclusion_cache(self) -> None:
        if self._element_mgr:
            self._element_mgr.invalidate_cache()

    @property
    def pdf(self) -> "PDF":
        """Provides public access to the parent PDF object."""
        return self._parent

    @property
    def number(self) -> int:
        """Get page number (1-based)."""
        return self._page.page_number

    @property
    def page_number(self) -> int:
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

    @property
    def _highlighter(self) -> "HighlightingService":
        """Provides access to the parent PDF's HighlightingService."""
        if not hasattr(self._parent, "highlighter"):
            # This should ideally not happen if PDF.__init__ works correctly
            raise AttributeError("Parent PDF object does not have a 'highlighter' attribute.")
        return self._parent.highlighter

    def get_highlighter(self) -> "HighlightingService":
        """Expose the page-level HighlightingService for Visualizable consumers."""
        return self._highlighter

    def _context_page(self) -> "Page":
        return self

    def clear_exclusions(self) -> "Page":
        """
        Clear all exclusions from the page.
        """
        self._exclusions = []
        return self

    @contextlib.contextmanager
    def without_exclusions(self):
        """
        Context manager that temporarily disables exclusion processing.

        This prevents infinite recursion when exclusion callables themselves
        use find() operations. While in this context, all find operations
        will skip exclusion filtering.

        Example:
            ```python
            # This exclusion would normally cause infinite recursion:
            page.add_exclusion(lambda p: p.find("text:contains('Header')").expand())

            # But internally, it's safe because we use:
            with page.without_exclusions():
                region = exclusion_callable(page)
            ```

        Yields:
            The page object with exclusions temporarily disabled.
        """
        old_value = self._computing_exclusions
        self._computing_exclusions = True
        try:
            yield self
        finally:
            self._computing_exclusions = old_value

    def add_region(
        self, region: "Region", name: Optional[str] = None, *, source: Optional[str] = None
    ) -> "Page":
        """
        Add a region to the page.

        Args:
            region: Region object to add
            name: Optional name for the region
            source: Optional provenance label; if provided it will be recorded on the region.

        Returns:
            Self for method chaining
        """
        # Check if it's actually a Region object
        if not isinstance(region, Region):
            raise TypeError("region must be a Region object")

        # Respect an explicitly provided source, otherwise keep any existing label.
        if source is not None:
            region.source = source
        elif getattr(region, "source", None) is None:
            region.source = "named"

        if name:
            region.name = name
            # Add to named regions dictionary (overwriting if name already exists)
            self._regions["named"][name] = region
        else:
            # Add to detected regions list (unnamed but registered)
            self._regions["detected"].append(region)

        # Add to element manager for selector queries
        self._element_mgr.add_region(region)

        return self

    def add_regions(
        self,
        regions: List["Region"],
        prefix: Optional[str] = None,
        *,
        source: Optional[str] = None,
    ) -> "Page":
        """
        Add multiple regions to the page.

        Args:
            regions: List of Region objects to add
            prefix: Optional prefix for automatic naming (regions will be named prefix_1, prefix_2, etc.)
            source: Optional provenance label applied to each region.

        Returns:
            Self for method chaining
        """
        if prefix:
            # Add with automatic sequential naming
            for i, region in enumerate(regions):
                self.add_region(region, name=f"{prefix}_{i+1}", source=source)
        else:
            # Add without names
            for region in regions:
                self.add_region(region, source=source)

        return self

    # Element manager facade helpers
    def ensure_elements_loaded(self) -> None:
        """Force the underlying element manager to load elements."""
        self._element_mgr.load_elements()

    def invalidate_element_cache(self) -> None:
        """Invalidate the cached elements so they are reloaded on next access."""
        self._element_mgr.invalidate_cache()

    def has_element_cache(self) -> bool:
        """Return True if the element manager currently holds any elements."""
        return self._element_mgr.has_elements()

    def get_all_elements_raw(self) -> List["Element"]:
        """Return all elements without applying exclusions."""
        return list(self._element_mgr.get_all_elements())

    def get_elements_by_type(self, element_type: str) -> List[Any]:
        """Return the elements for a specific backing collection (e.g. 'words')."""
        return list(self._element_mgr.get_elements(element_type))

    def add_element(self, element: Any, element_type: str = "words") -> bool:
        """Add an element to the backing collection."""
        return bool(self._element_mgr.add_element(element, element_type))

    def _infer_element_type(self, element: Any, default: str = "words") -> str:
        """Best-effort inference of element collection name for an object."""
        element_type = getattr(element, "object_type", None)
        if element_type is None and isinstance(element, dict):
            element_type = element.get("object_type")

        if isinstance(element_type, str):
            normalized = element_type.lower()
            if normalized == "word":
                return "words"
            if normalized == "char":
                return "chars"
            if normalized == "rect":
                return "rects"
            if normalized == "line":
                return "lines"
            if normalized == "region":
                return "regions"
            if normalized.endswith("s"):
                return normalized
        return default

    def remove_element(self, element: Any, element_type: Optional[str] = None) -> bool:
        """Remove an element from the backing collection."""
        target_type = element_type or self._infer_element_type(element)
        return bool(self._element_mgr.remove_element(element, target_type))

    def remove_elements_by_source(self, element_type: str, source: str) -> int:
        """Remove all elements of a given type whose source matches."""
        return int(self._element_mgr.remove_elements_by_source(element_type, source))

    def _ocr_element_manager(self) -> ElementManager:
        return self._element_mgr

    def iter_regions(self) -> List["Region"]:
        """Return a list of regions currently registered with the page."""
        return list(self._element_mgr.regions)

    def remove_regions_by_source(self, source: str) -> int:
        """Remove all registered regions that match the requested source."""
        return int(self._element_mgr.remove_elements_by_source("regions", source))

    def remove_regions(
        self,
        *,
        source: Optional[str] = None,
        region_type: Optional[str] = None,
        predicate: Optional[Callable[["Region"], bool]] = None,
    ) -> int:
        """
        Remove regions from the page based on optional filters.

        Args:
            source: Match regions whose ``region.source`` equals this string.
            region_type: Match regions whose ``region.region_type`` equals this string.
            predicate: Additional callable that returns True when a region should be removed.

        Returns:
            The number of regions removed.
        """

        def _matches(region: "Region") -> bool:
            if source is not None and getattr(region, "source", None) != source:
                return False
            if region_type is not None and getattr(region, "region_type", None) != region_type:
                return False
            if predicate is not None and not predicate(region):
                return False
            return True

        removed = 0

        # Remove from element manager, if available
        if hasattr(self, "_element_mgr") and hasattr(self._element_mgr, "regions"):
            regions_list = getattr(self._element_mgr, "regions")
            if isinstance(regions_list, list):
                to_remove = [region for region in regions_list if _matches(region)]
                for region in to_remove:
                    regions_list.remove(region)
                removed += len(to_remove)

        # Remove from detected collection
        detected = self._regions.get("detected", [])
        if detected:
            retained = [region for region in detected if not _matches(region)]
            removed += len(detected) - len(retained)
            self._regions["detected"] = retained

        # Remove from named collection
        named = self._regions.get("named", {})
        if named:
            keys_to_delete = [key for key, region in named.items() if _matches(region)]
            for key in keys_to_delete:
                del named[key]
                removed += 1

        return removed

    def _get_exclusion_regions(self, include_callable=True, debug=False) -> List["Region"]:
        """
        Get all exclusion regions for this page.
        Now handles both region-based and element-based exclusions.
        Assumes self._exclusions contains tuples of (callable/Region/Element, label, method).

        Args:
            include_callable: Whether to evaluate callable exclusion functions
            debug: Enable verbose debug logging for exclusion evaluation

        Returns:
            List of Region objects to exclude, with labels assigned.
        """
        regions = []

        # Combine page-specific exclusions with PDF-level exclusions
        all_exclusions = list(self._exclusions)  # Start with page-specific

        # Add PDF-level exclusions if we have a parent PDF
        if hasattr(self, "_parent") and self._parent and hasattr(self._parent, "_exclusions"):
            # Get existing labels to check for duplicates
            existing_labels = set()
            for exc in all_exclusions:
                if len(exc) >= 2 and exc[1]:  # Has a label
                    existing_labels.add(exc[1])

            for pdf_exclusion in self._parent._exclusions:
                # Check if this exclusion label is already in our list (avoid duplicates)
                label = pdf_exclusion[1] if len(pdf_exclusion) >= 2 else None
                if label and label in existing_labels:
                    continue  # Skip this exclusion as it's already been applied

                # Ensure consistent format (PDF exclusions might be 2-tuples, need to be 3-tuples)
                if len(pdf_exclusion) == 2:
                    # Convert to 3-tuple format with default method
                    pdf_exclusion = (pdf_exclusion[0], pdf_exclusion[1], "region")
                all_exclusions.append(pdf_exclusion)

        if debug:
            print(
                f"\nPage {self.index}: Evaluating {len(all_exclusions)} exclusions ({len(self._exclusions)} page-specific, {len(all_exclusions) - len(self._exclusions)} from PDF)"
            )

        for i, exclusion_data in enumerate(all_exclusions):
            # Handle both old format (2-tuple) and new format (3-tuple) for backward compatibility
            if len(exclusion_data) == 2:
                # Old format: (exclusion_item, label)
                exclusion_item, label = exclusion_data
                method = "region"  # Default to region for old format
            else:
                # New format: (exclusion_item, label, method)
                exclusion_item, label, method = exclusion_data

            exclusion_label = label if label else f"exclusion {i}"

            # Process callable exclusion functions
            if callable(exclusion_item) and include_callable:
                try:
                    if debug:
                        print(f"  - Evaluating callable '{exclusion_label}'...")

                    # Use context manager to prevent infinite recursion
                    with self.without_exclusions():
                        # Call the function - Expects it to return a Region or None
                        region_result = exclusion_item(self)

                    if isinstance(region_result, Region):
                        # Assign the label to the returned region
                        region_result.label = label
                        regions.append(region_result)
                        if debug:
                            print(f"    ✓ Added region from callable '{label}': {region_result}")
                    elif isinstance(region_result, ElementCollection):
                        elements_iter: Iterable[Any] = region_result.elements
                        if debug:
                            print(
                                f"    Converting ElementCollection with {len(region_result.elements)} elements to regions..."
                            )

                        for elem in elements_iter:
                            try:
                                bbox_candidate = getattr(elem, "bbox", None)
                                if (
                                    isinstance(bbox_candidate, (tuple, list))
                                    and len(bbox_candidate) == 4
                                ):
                                    bbox_coords = cast(
                                        Tuple[float, float, float, float],
                                        tuple(
                                            float(v) for v in cast(Sequence[float], bbox_candidate)
                                        ),
                                    )
                                    region = Region(self, bbox_coords, label=label)
                                    regions.append(region)
                                    if debug:
                                        print(f"      ✓ Added region from element: {bbox_coords}")
                                else:
                                    if debug:
                                        print(
                                            f"      ✗ Skipping element without valid bbox: {elem}"
                                        )
                            except Exception as e:
                                if debug:
                                    print(f"      ✗ Failed to convert element to region: {e}")
                                continue

                        if debug and region_result.elements:
                            print(
                                f"    ✓ Converted {len(region_result.elements)} elements from callable '{label}'"
                            )
                    elif isinstance(region_result, Iterable):
                        materialised = list(cast(Iterable[Any], region_result))
                        if not materialised:
                            if debug:
                                print(f"    ✗ Empty iterable returned from callable '{label}'")
                        else:
                            if debug:
                                print(
                                    f"    Converting iterable {type(region_result)} with {len(materialised)} elements to regions..."
                                )
                            for elem in materialised:
                                try:
                                    bbox_candidate = getattr(elem, "bbox", None)
                                    if (
                                        isinstance(bbox_candidate, (tuple, list))
                                        and len(bbox_candidate) == 4
                                    ):
                                        bbox_coords = cast(
                                            Tuple[float, float, float, float],
                                            tuple(
                                                float(v)
                                                for v in cast(Sequence[float], bbox_candidate)
                                            ),
                                        )
                                        region = Region(self, bbox_coords, label=label)
                                        regions.append(region)
                                        if debug:
                                            print(
                                                f"      ✓ Added region from iterable element: {bbox_coords}"
                                            )
                                    else:
                                        if debug:
                                            print(
                                                f"      ✗ Skipping iterable element without valid bbox: {elem}"
                                            )
                                except Exception as e:
                                    if debug:
                                        print(
                                            f"      ✗ Failed to convert iterable element to region: {e}"
                                        )
                                    continue
                    elif region_result:
                        # Check if it's a single Element that can be converted to a Region
                        from natural_pdf.elements.base import Element

                        if isinstance(region_result, Element) or (
                            hasattr(region_result, "bbox") and hasattr(region_result, "expand")
                        ):
                            try:
                                # Convert Element to Region using expand()
                                expanded_region = cast(Any, region_result).expand()
                                if isinstance(expanded_region, Region):
                                    expanded_region.label = label
                                    regions.append(expanded_region)
                                    if debug:
                                        print(
                                            f"    ✓ Converted Element to Region from callable '{label}': {expanded_region}"
                                        )
                                else:
                                    if debug:
                                        print(
                                            f"    ✗ Element.expand() did not return a Region: {type(expanded_region)}"
                                        )
                            except Exception as e:
                                if debug:
                                    print(f"    ✗ Failed to convert Element to Region: {e}")
                        else:
                            logger.warning(
                                f"Callable exclusion '{exclusion_label}' returned non-Region object: {type(region_result)}. Skipping."
                            )
                            if debug:
                                print(
                                    f"    ✗ Callable returned non-Region/None: {type(region_result)}"
                                )
                    else:
                        if debug:
                            print(
                                f"    ✗ Callable '{exclusion_label}' returned None, no region added"
                            )

                except Exception as e:
                    error_msg = f"Error evaluating callable exclusion '{exclusion_label}' for page {self.index}: {e}"
                    print(error_msg)
                    import traceback

                    print(f"    Traceback: {traceback.format_exc().splitlines()[-3:]}")

            # Process direct Region objects (label was assigned in add_exclusion)
            elif isinstance(exclusion_item, Region):
                regions.append(exclusion_item)  # Label is already on the Region object
                if debug:
                    print(f"  - Added direct region '{label}': {exclusion_item}")

            # Process direct Element objects - only convert to Region if method is "region"
            elif hasattr(exclusion_item, "bbox") and hasattr(exclusion_item, "expand"):
                if method == "region":
                    try:
                        # Convert Element to Region using expand()
                        expanded_region = cast(Any, exclusion_item).expand()
                        if isinstance(expanded_region, Region):
                            expanded_region.label = label
                            regions.append(expanded_region)
                            if debug:
                                print(
                                    f"  - Converted direct Element to Region '{label}': {expanded_region}"
                                )
                        else:
                            if debug:
                                print(
                                    f"  - Element.expand() did not return a Region: {type(expanded_region)}"
                                )
                    except Exception as e:
                        if debug:
                            print(f"  - Failed to convert Element to Region: {e}")
                else:
                    # method == "element" - will be handled in _filter_elements_by_exclusions
                    if debug:
                        print(
                            f"  - Skipping element '{label}' (will be handled as element-based exclusion)"
                        )

            # Process string selectors (from PDF-level exclusions)
            elif isinstance(exclusion_item, str):
                selector_str = exclusion_item
                matching_elements = self.find_all(selector_str, apply_exclusions=False)

                if debug:
                    print(
                        f"  - Evaluating selector '{exclusion_label}': found {len(matching_elements)} elements"
                    )

                if method == "region":
                    # Convert each matching element to a region
                    for el in matching_elements:
                        try:
                            bbox_coords = (
                                float(el.x0),
                                float(el.top),
                                float(el.x1),
                                float(el.bottom),
                            )
                            region = Region(self, bbox_coords, label=label)
                            regions.append(region)
                            if debug:
                                print(f"    ✓ Added region from selector match: {bbox_coords}")
                        except Exception as e:
                            if debug:
                                print(f"    ✗ Failed to create region from element: {e}")
                # If method is "element", it will be handled in _filter_elements_by_exclusions

            # Element-based exclusions are not converted to regions here
            # They will be handled separately in _filter_elements_by_exclusions

        if debug:
            print(f"Page {self.index}: Found {len(regions)} valid exclusion regions to apply")

        return regions

    def _filter_elements_by_exclusions(
        self, elements: List["Element"], debug_exclusions: bool = False
    ) -> List["Element"]:
        """
        Filters a list of elements, removing those based on exclusion rules.
        Handles both region-based exclusions (exclude all in area) and
        element-based exclusions (exclude only specific elements).

        Args:
            elements: The list of elements to filter.
            debug_exclusions: Whether to output detailed exclusion debugging info (default: False).

        Returns:
            A new list containing only the elements not excluded.
        """
        # Skip exclusion filtering if we're currently computing exclusions
        # This prevents infinite recursion when exclusion callables use find operations
        if self._computing_exclusions:
            return elements

        # Check both page-level and PDF-level exclusions
        has_page_exclusions = bool(self._exclusions)
        has_pdf_exclusions = (
            hasattr(self, "_parent")
            and self._parent
            and hasattr(self._parent, "_exclusions")
            and bool(self._parent._exclusions)
        )

        if not has_page_exclusions and not has_pdf_exclusions:
            if debug_exclusions:
                print(
                    f"Page {self.index}: No exclusions defined, returning all {len(elements)} elements."
                )
            return elements

        # Get all exclusion regions, including evaluating callable functions
        exclusion_regions = self._get_exclusion_regions(
            include_callable=True, debug=debug_exclusions
        )

        # Collect element-based exclusions
        # Store element bboxes for comparison instead of object ids
        excluded_element_bboxes = set()  # Use set for O(1) lookup

        # Process both page-level and PDF-level exclusions
        all_exclusions = list(self._exclusions) if has_page_exclusions else []
        if has_pdf_exclusions:
            all_exclusions.extend(self._parent._exclusions)

        for exclusion_data in all_exclusions:
            # Handle both old format (2-tuple) and new format (3-tuple)
            if len(exclusion_data) == 2:
                exclusion_item, label = exclusion_data
                method = "region"
            else:
                exclusion_item, label, method = exclusion_data

            # Skip callables (already handled in _get_exclusion_regions)
            if callable(exclusion_item):
                continue

            # Skip regions (already in exclusion_regions)
            if isinstance(exclusion_item, Region):
                continue

            # Handle string selectors for element-based exclusions
            if isinstance(exclusion_item, str) and method == "element":
                selector_str = exclusion_item
                matching_elements = self.find_all(selector_str, apply_exclusions=False)
                elements_iter = getattr(matching_elements, "elements", [])
                for el in cast(Iterable[Any], elements_iter):
                    bbox_vals = extract_bbox(cast(Any, el))
                    if bbox_vals is None:
                        continue
                    excluded_element_bboxes.add(bbox_vals)
                    if debug_exclusions:
                        print(
                            f"  - Added element exclusion from selector '{selector_str}': {bbox_vals}"
                        )

            # Handle element-based exclusions
            elif method == "element":
                bbox = extract_bbox(cast(Any, exclusion_item))
                if bbox is not None:
                    excluded_element_bboxes.add(bbox)
                    if debug_exclusions:
                        print(f"  - Added element exclusion with bbox {bbox}: {exclusion_item}")
                else:
                    logger.warning(
                        f"Page {self.index}: Skipping element exclusion without bounding box: {exclusion_item}"
                    )

        if debug_exclusions:
            print(
                f"Page {self.index}: Applying {len(exclusion_regions)} region exclusions "
                f"and {len(excluded_element_bboxes)} element exclusions to {len(elements)} elements."
            )

        filtered_elements = []
        region_excluded_count = 0
        element_excluded_count = 0

        for element in elements:
            exclude = False

            # Check element-based exclusions first (faster)
            element_bbox = extract_bbox(element)
            if element_bbox in excluded_element_bboxes:
                exclude = True
                element_excluded_count += 1
                if debug_exclusions:
                    print(f"    Element {element} excluded by element-based rule")
            else:
                # Check region-based exclusions
                for region in exclusion_regions:
                    # Use the region's method to check if the element is inside
                    if region._is_element_in_region(element):
                        exclude = True
                        region_excluded_count += 1
                        if debug_exclusions:
                            print(f"    Element {element} excluded by region {region}")
                        break  # No need to check other regions for this element

            if not exclude:
                filtered_elements.append(element)

        if debug_exclusions:
            print(
                f"Page {self.index}: Excluded {region_excluded_count} by regions, "
                f"{element_excluded_count} by elements, keeping {len(filtered_elements)}."
            )

        return filtered_elements

    @contextlib.contextmanager
    def _temporary_text_settings(
        self,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
    ):
        """
        Temporarily override page-level text tolerance settings and refresh caches.
        """
        overrides: Dict[str, Any] = {}
        if text_tolerance:
            overrides.update(
                {
                    key: value
                    for key, value in dict(text_tolerance).items()
                    if key
                    in {
                        "x_tolerance",
                        "y_tolerance",
                        "x_tolerance_ratio",
                        "y_tolerance_ratio",
                        "keep_blank_chars",
                    }
                }
            )

        auto_override: Optional[bool] = None
        if isinstance(auto_text_tolerance, dict):
            overrides.update(dict(auto_text_tolerance))
            auto_override = True
        else:
            auto_override = auto_text_tolerance

        if auto_override is not None:
            overrides["auto_text_tolerance"] = bool(auto_override)

        if not overrides:
            yield False
            return

        sentinel = object()
        page_config = self._config
        previous_values: Dict[str, Any] = {key: page_config.get(key, sentinel) for key in overrides}
        cache_invalidated = False
        try:
            for key, value in overrides.items():
                if value is None:
                    if key in page_config:
                        del page_config[key]
                        cache_invalidated = True
                elif page_config.get(key) != value:
                    page_config[key] = value
                    cache_invalidated = True

            if cache_invalidated:
                self._element_mgr.invalidate_cache()
            yield cache_invalidated
        finally:
            if not cache_invalidated:
                return

            restore_invalidated = False
            for key, prior in previous_values.items():
                if prior is sentinel:
                    if key in page_config:
                        del page_config[key]
                        restore_invalidated = True
                elif page_config.get(key) != prior:
                    page_config[key] = prior
                    restore_invalidated = True

            if restore_invalidated:
                self._element_mgr.invalidate_cache()

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
    ) -> Optional[Element]:
        """
        Find first element on this page matching selector OR text content.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Optional text shortcut (equivalent to ``text:contains(...)``).
            overlap: Reserved for compatibility with region APIs; ignored for pages.
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            text_tolerance: Optional dict of tolerance overrides applied temporarily.
            auto_text_tolerance: Optional overrides controlling automatic tolerance logic.
            reading_order: Whether to sort matches in reading order when applicable (default: True).
            near_threshold: Maximum distance (in points) used by the ``:near`` pseudo-class.
            engine: Optional selector engine name registered via :mod:`natural_pdf.engine_provider`.

        Returns:
            Element object or None if not found.
        """
        effective_selector = normalize_selector_input(
            selector,
            text,
            logger=logger,
            context="Page.find",
        )

        results_collection = execute_selector_query(
            self,
            effective_selector,
            text_tolerance=text_tolerance,
            auto_text_tolerance=auto_text_tolerance,
            regex=regex,
            case=case,
            reading_order=reading_order,
            near_threshold=near_threshold,
            engine=engine,
        )

        elements: List[Element] = list(results_collection.elements) if results_collection else []
        if apply_exclusions and elements:
            elements = self._filter_elements_by_exclusions(elements)

        return cast(Optional[Element], elements[0] if elements else None)

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
        """
        Find all elements on this page matching selector OR text content.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for (equivalent to 'text:contains(...)'). Accepts a
                  single string or an iterable of strings (matches any value).
            overlap: Reserved for compatibility with region APIs; ignored for pages.
            apply_exclusions: Whether to exclude elements in exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            text_tolerance: Optional dict of tolerance overrides applied temporarily.
            auto_text_tolerance: Optional overrides controlling automatic tolerance calculation.
            reading_order: Whether to sort matches in reading order (default: True).
            near_threshold: Maximum distance (in points) used by the ``:near`` pseudo-class.
            engine: Optional selector engine name registered with the selector provider.

        Returns:
            ElementCollection with matching elements.
        """
        effective_selector = normalize_selector_input(
            selector,
            text,
            logger=logger,
            context="Page.find_all",
        )

        results_collection = execute_selector_query(
            self,
            effective_selector,
            text_tolerance=text_tolerance,
            auto_text_tolerance=auto_text_tolerance,
            regex=regex,
            case=case,
            reading_order=reading_order,
            near_threshold=near_threshold,
            engine=engine,
        )

        if apply_exclusions and results_collection:
            filtered_elements = self._filter_elements_by_exclusions(results_collection.elements)
            return ElementCollection(filtered_elements)
        return results_collection

    def _apply_selector(
        self, selector_obj: Dict[str, Any], **kwargs: Any
    ) -> "ElementCollection":  # Removed apply_exclusions arg
        """
        Apply selector to page elements.
        Exclusions are now handled by the calling methods (find, find_all) if requested.

        Args:
            selector_obj: Parsed selector dictionary (single or compound OR selector)
            **kwargs: Additional filter parameters including 'regex' and 'case'

        Returns:
            ElementCollection of matching elements (unfiltered by exclusions)
        """
        from natural_pdf.selectors.parser import _calculate_aggregates, selector_to_filter_func

        selector_kwargs = dict(kwargs)
        selector_kwargs.setdefault("selector_context", self)

        def _apply_relational_only(sel: Dict[str, Any], elements: List[Any]) -> List[Any]:
            relational = sel.get("relational_pseudos")
            if not relational:
                return elements
            return _apply_relational_post_pseudos(
                self,
                {"relational_pseudos": relational},
                elements,
                selector_kwargs,
            )

        def _apply_post_only(sel: Dict[str, Any], elements: List[Any]) -> List[Any]:
            post = sel.get("post_pseudos")
            if not post:
                return elements
            return _apply_relational_post_pseudos(
                self,
                {"post_pseudos": post},
                elements,
                selector_kwargs,
            )

        # Handle compound OR selectors
        if selector_obj.get("type") == "or":
            elements_to_search = self._element_mgr.get_all_elements()

            has_aggregates = False
            for sub_selector in selector_obj.get("selectors", []):
                for attr in sub_selector.get("attributes", []):
                    value = attr.get("value")
                    if isinstance(value, dict) and value.get("type") == "aggregate":
                        has_aggregates = True
                        break
                if has_aggregates:
                    break

            aggregates: Dict[str, Any] = {}
            if has_aggregates:
                for sub_selector in selector_obj.get("selectors", []):
                    sub_type = sub_selector.get("type", "any").lower()
                    if sub_type == "text":
                        sub_elements = self._element_mgr.words
                    elif sub_type == "rect":
                        sub_elements = self._element_mgr.rects
                    elif sub_type == "line":
                        sub_elements = self._element_mgr.lines
                    elif sub_type == "region":
                        sub_elements = self._element_mgr.regions
                    else:
                        sub_elements = elements_to_search

                    sub_aggregates = _calculate_aggregates(sub_elements, sub_selector)
                    aggregates.update(sub_aggregates)

            filter_func = selector_to_filter_func(
                selector_obj, aggregates=aggregates, **selector_kwargs
            )
            matching_elements = [element for element in elements_to_search if filter_func(element)]

            matching_elements = _apply_relational_only(selector_obj, matching_elements)

            if selector_kwargs.get("reading_order", True):
                if all(hasattr(el, "top") and hasattr(el, "x0") for el in matching_elements):
                    matching_elements.sort(key=lambda el: (el.top, el.x0))
                elif matching_elements:
                    logger.warning(
                        "Cannot sort elements in reading order: Missing required attributes (top, x0)."
                    )

            # Handle collection-level pseudo-classes (:first, :last) for OR selectors
            has_first = any(
                any(p.get("name") == "first" for p in sub_selector.get("post_pseudos", []))
                for sub_selector in selector_obj.get("selectors", [])
            )
            has_last = any(
                any(p.get("name") == "last" for p in sub_selector.get("post_pseudos", []))
                for sub_selector in selector_obj.get("selectors", [])
            )

            if has_first:
                matching_elements = matching_elements[:1] if matching_elements else []
            elif has_last:
                matching_elements = matching_elements[-1:] if matching_elements else []

            return ElementCollection(matching_elements)

        element_type = selector_obj.get("type", "any").lower()
        if element_type == "text" or element_type == "word":
            elements_to_search = self._element_mgr.words
        elif element_type == "char":
            elements_to_search = self._element_mgr.chars
        elif element_type in ("rect", "rectangle"):
            elements_to_search = self._element_mgr.rects
        elif element_type == "line":
            elements_to_search = self._element_mgr.lines
        elif element_type == "region":
            elements_to_search = self._element_mgr.regions
        elif element_type == "any":
            elements_to_search = self._element_mgr.get_all_elements()
        else:
            elements_to_search = self._element_mgr.get_all_elements()

        has_aggregates = any(
            isinstance(attr.get("value"), dict) and attr["value"].get("type") == "aggregate"
            for attr in selector_obj.get("attributes", [])
        )

        aggregates: Dict[str, Any] = {}
        if has_aggregates:
            aggregates = _calculate_aggregates(elements_to_search, selector_obj)

        filter_func = selector_to_filter_func(
            selector_obj, aggregates=aggregates, **selector_kwargs
        )
        matching_elements = [element for element in elements_to_search if filter_func(element)]

        matching_elements = _apply_relational_only(selector_obj, matching_elements)

        if selector_kwargs.get("reading_order", True):
            if all(hasattr(el, "top") and hasattr(el, "x0") for el in matching_elements):
                matching_elements.sort(key=lambda el: (el.top, el.x0))
            elif matching_elements:
                logger.warning(
                    "Cannot sort elements in reading order: Missing required attributes (top, x0)."
                )

        # Handle :closest pseudo-class for fuzzy text matching
        for pseudo in selector_obj.get("pseudo_classes", []):
            name = pseudo.get("name")
            if name != "closest" or pseudo.get("args") is None:
                continue

            search_text = str(pseudo["args"]).strip()
            threshold = 0.0
            if not search_text:
                matching_elements = []
                break

            if "@" in search_text and search_text.count("@") == 1:
                text_part, threshold_part = search_text.rsplit("@", 1)
                try:
                    threshold = float(threshold_part)
                    search_text = text_part.strip()
                except (ValueError, TypeError):
                    pass

            ignore_case = not selector_kwargs.get("case", True)
            scored_elements = []
            for el in matching_elements:
                if not getattr(el, "text", None):
                    continue
                el_text = el.text.strip()
                search_term = search_text
                if ignore_case:
                    el_text = el_text.lower()
                    search_term = search_term.lower()

                ratio = _jaro_winkler_similarity(search_term, el_text)
                contains_match = search_term in el_text
                if ratio >= threshold:
                    scored_elements.append((contains_match, ratio, el))

            scored_elements.sort(key=lambda x: (x[0], x[1]), reverse=True)
            matching_elements = [entry[2] for entry in scored_elements]
            break

        matching_elements = _apply_post_only(selector_obj, matching_elements)

        return ElementCollection(matching_elements)

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

    def region(
        self,
        left: Optional[float] = None,
        top: Optional[float] = None,
        right: Optional[float] = None,
        bottom: Optional[float] = None,
        width: Union[str, float, None] = None,
        height: Optional[float] = None,
    ) -> Any:
        """
        Create a region on this page with more intuitive named parameters,
        allowing definition by coordinates or by coordinate + dimension.

        Args:
            left: Left x-coordinate (default: 0 if width not used).
            top: Top y-coordinate (default: 0 if height not used).
            right: Right x-coordinate (default: page width if width not used).
            bottom: Bottom y-coordinate (default: page height if height not used).
            width: Width definition. Can be:
                    - Numeric: The width of the region in points. Cannot be used with both left and right.
                    - String 'full': Sets region width to full page width (overrides left/right).
                    - String 'element' or None (default): Uses provided/calculated left/right,
                        defaulting to page width if neither are specified.
            height: Numeric height of the region. Cannot be used with both top and bottom.

        Returns:
            Region object for the specified coordinates

        Raises:
            ValueError: If conflicting arguments are provided (e.g., top, bottom, and height)
                        or if width is an invalid string.

        Examples:
            >>> page.region(top=100, height=50)  # Region from y=100 to y=150, default width
            >>> page.region(left=50, width=100)   # Region from x=50 to x=150, default height
            >>> page.region(bottom=500, height=50) # Region from y=450 to y=500
            >>> page.region(right=200, width=50)  # Region from x=150 to x=200
            >>> page.region(top=100, bottom=200, width="full") # Explicit full width
        """
        # Percentage support – convert strings like "30%" to absolute values
        # based on page dimensions.  X-axis params (left, right, width) use
        # page.width; Y-axis params (top, bottom, height) use page.height.

        def _pct_to_abs(val, axis: str):
            if isinstance(val, str) and val.strip().endswith("%"):
                try:
                    pct = float(val.strip()[:-1]) / 100.0
                except ValueError:
                    return val  # leave unchanged if not a number
                return pct * (self.width if axis == "x" else self.height)
            return val

        left = _pct_to_abs(left, "x")
        right = _pct_to_abs(right, "x")
        width = _pct_to_abs(width, "x")
        top = _pct_to_abs(top, "y")
        bottom = _pct_to_abs(bottom, "y")
        height = _pct_to_abs(height, "y")

        is_width_numeric = isinstance(width, (int, float))
        is_width_string = isinstance(width, str)
        width_mode = "element"  # Default mode

        if height is not None and top is not None and bottom is not None:
            raise ValueError("Cannot specify top, bottom, and height simultaneously.")
        if is_width_numeric and left is not None and right is not None:
            raise ValueError("Cannot specify left, right, and a numeric width simultaneously.")
        if is_width_string:
            width_lower = width.lower()
            if width_lower not in ["full", "element"]:
                raise ValueError("String width argument must be 'full' or 'element'.")
            width_mode = width_lower

        final_top = top
        final_bottom = bottom
        final_left = left
        final_right = right

        # Height calculations
        if height is not None:
            if top is not None:
                final_bottom = top + height
            elif bottom is not None:
                final_top = bottom - height
            else:  # Neither top nor bottom provided, default top to 0
                final_top = 0
                final_bottom = height

        # Width calculations (numeric only)
        if is_width_numeric:
            if left is not None:
                final_right = left + width
            elif right is not None:
                final_left = right - width
            else:  # Neither left nor right provided, default left to 0
                final_left = 0
                final_right = width

        # Only default coordinates if they weren't set by dimension calculation
        if final_top is None:
            final_top = 0.0
        if final_bottom is None:
            # Check if bottom should have been set by height calc
            if height is None or top is None:
                final_bottom = float(self.height)

        if final_left is None:
            final_left = 0.0
        if final_right is None:
            # Check if right should have been set by width calc
            if not is_width_numeric or left is None:
                final_right = float(self.width)

        if final_top is None or final_bottom is None or final_left is None or final_right is None:
            raise ValueError("Unable to resolve region coordinates.")

        final_left = float(final_left)
        final_top = float(final_top)
        final_right = float(final_right)
        final_bottom = float(final_bottom)

        if width_mode == "full":
            # Override left/right if mode is full
            final_left = 0
            final_right = self.width

        # Ensure coordinates are within page bounds (clamp)
        final_left = max(0.0, final_left)
        final_top = max(0.0, final_top)
        final_right = min(float(self.width), final_right)
        final_bottom = min(float(self.height), final_bottom)

        # Ensure valid box (x0<=x1, top<=bottom)
        if final_left > final_right:
            logger.warning(f"Calculated left ({final_left}) > right ({final_right}). Swapping.")
            final_left, final_right = final_right, final_left
        if final_top > final_bottom:
            logger.warning(f"Calculated top ({final_top}) > bottom ({final_bottom}). Swapping.")
            final_top, final_bottom = final_bottom, final_top

        from natural_pdf.elements.region import Region

        region = Region(self, (final_left, final_top, final_right, final_bottom))
        return region

    def get_elements(
        self, apply_exclusions=True, debug_exclusions: bool = False
    ) -> List["Element"]:
        """
        Get all elements on this page.

        Args:
            apply_exclusions: Whether to apply exclusion regions (default: True).
            debug_exclusions: Whether to output detailed exclusion debugging info (default: False).

        Returns:
            List of all elements on the page, potentially filtered by exclusions.
        """
        # Get all elements from the element manager
        all_elements = self.get_all_elements_raw()

        # Apply exclusions if requested
        if apply_exclusions:
            return self._filter_elements_by_exclusions(
                all_elements, debug_exclusions=debug_exclusions
            )
        else:
            if debug_exclusions:
                print(
                    f"Page {self.index}: get_elements returning all {len(all_elements)} elements (exclusions not applied)."
                )
            return all_elements

    def filter_elements(
        self, elements: List["Element"], selector: str, **kwargs
    ) -> List["Element"]:
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
        filter_func = selector_to_filter_func(selector_obj, **kwargs)

        # Apply the filter to the elements
        matching_elements = [element for element in elements if filter_func(element)]

        # Sort elements in reading order if requested
        if kwargs.get("reading_order", True):
            if all(hasattr(el, "top") and hasattr(el, "x0") for el in matching_elements):
                matching_elements.sort(key=lambda el: (el.top, el.x0))
            else:
                logger.warning(
                    "Cannot sort elements in reading order: Missing required attributes (top, x0)."
                )

        return matching_elements

    def until(
        self,
        selector: str,
        include_endpoint: bool = True,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
    ) -> Any:
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
        target = self.find(
            selector,
            text=text,
            apply_exclusions=apply_exclusions,
            regex=regex,
            case=case,
            text_tolerance=text_tolerance,
            auto_text_tolerance=auto_text_tolerance,
            reading_order=reading_order,
        )
        if not target:
            # If target not found, return a default region (full page)
            from natural_pdf.elements.region import Region

            return Region(self, (0, 0, self.width, self.height))

        # Create a region from the top of the page to the target
        from natural_pdf.elements.region import Region

        # Ensure target has positional attributes before using them
        target_top = getattr(target, "top", 0)
        target_bottom = getattr(target, "bottom", self.height)

        if include_endpoint:
            # Include the target element
            region = Region(self, (0, 0, self.width, target_bottom))
        else:
            # Up to the target element
            region = Region(self, (0, 0, self.width, target_top))

        region.end_element = cast(Element, target)
        return region

    def crop(self, bbox: Optional[Bounds] = None, **kwargs: Any) -> Any:
        """
        Crop the page to the specified bounding box.

        This is a direct wrapper around pdfplumber's crop method.

        Args:
            bbox: Bounding box (x0, top, x1, bottom) or None
            **kwargs: Additional parameters (top, bottom, left, right)

        Returns:
            Cropped page object (pdfplumber.Page)
        """
        # Returns the pdfplumber page object, not a natural-pdf Page
        resolved_bbox: Optional[Tuple[float, float, float, float]]
        if bbox is None:
            resolved_bbox = None
        else:
            bbox_values = extract_bbox(bbox)
            if bbox_values is None:
                raise TypeError("crop expects a 4-tuple bbox or SupportsBBox-compatible object")
            resolved_bbox = bbox_values

        crop_arg: Any = resolved_bbox if resolved_bbox is not None else None
        return self._page.crop(crop_arg, **kwargs)

    def extract_text(
        self,
        preserve_whitespace: bool = True,
        preserve_line_breaks: bool = True,
        use_exclusions: bool = True,
        debug_exclusions: bool = False,
        content_filter=None,
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
        bidi: bool = True,
    ) -> str:
        """
        Extract text from this page, respecting exclusions and using pdfplumber's
        layout engine (chars_to_textmap) if layout arguments are provided or default.

        Args:
            preserve_line_breaks: When False, collapse newlines into spaces for a flattened string.
            use_exclusions: Whether to apply exclusion regions (default: True).
                            Note: Filtering logic is now always applied if exclusions exist.
            debug_exclusions: Whether to output detailed exclusion debugging info (default: False).
            content_filter: Optional content filter to exclude specific text patterns. Can be:
                - A regex pattern string (characters matching the pattern are EXCLUDED)
                - A callable that takes text and returns True to KEEP the character
                - A list of regex patterns (characters matching ANY pattern are EXCLUDED)
            layout: Whether to enable layout-aware spacing (default: True).
            x_density: Horizontal character density override.
            y_density: Vertical line density override.
            x_tolerance: Horizontal clustering tolerance.
            y_tolerance: Vertical clustering tolerance.
            line_dir: Line reading direction override.
            char_dir: Character reading direction override.
            strip_final: When True, strip trailing whitespace from the combined text.
            strip_empty: When True, drop entirely blank lines from the output.
            bidi: Whether to apply bidi reordering when RTL text is detected (default: True).

        Returns:
            Extracted text as string, potentially with layout-based spacing.
        """
        logger.debug(
            "Page %s: extract_text called with layout=%s, x_density=%s, y_density=%s",
            self.number,
            layout,
            x_density,
            y_density,
        )
        debug = debug_exclusions

        # 1. Get Word Elements (triggers load_elements if needed)
        word_elements = self.words
        if not word_elements:
            logger.debug(f"Page {self.number}: No word elements found.")
            return ""

        # 2. Apply element-based exclusions if enabled
        # Check both page-level and PDF-level exclusions
        has_exclusions = bool(self._exclusions) or (
            hasattr(self, "_parent")
            and self._parent
            and hasattr(self._parent, "_exclusions")
            and self._parent._exclusions
        )
        if use_exclusions and has_exclusions:
            # Filter word elements through _filter_elements_by_exclusions
            # This handles both element-based and region-based exclusions
            word_elements = self._filter_elements_by_exclusions(
                word_elements, debug_exclusions=debug
            )
            if debug:
                logger.debug(
                    f"Page {self.number}: {len(word_elements)} words remaining after exclusion filtering."
                )

        # 3. Get region-based exclusions for spatial filtering
        exclusion_regions = []
        apply_exclusions_flag = use_exclusions
        if apply_exclusions_flag and has_exclusions:
            exclusion_regions = self._get_exclusion_regions(include_callable=True, debug=debug)
            if debug:
                logger.debug(
                    f"Page {self.number}: Found {len(exclusion_regions)} region exclusions for spatial filtering."
                )
        elif debug:
            logger.debug(f"Page {self.number}: Not applying exclusions.")

        # 4. Collect All Character Dictionaries from remaining Word Elements
        all_char_dicts = []
        for word in word_elements:
            all_char_dicts.extend(getattr(word, "_char_dicts", []))

        # 5. Spatially Filter Characters (only by regions, elements already filtered above)
        filtered_chars = filter_chars_spatially(
            char_dicts=all_char_dicts,
            exclusion_regions=exclusion_regions,
            target_region=None,  # No target region for full page extraction
            debug=debug,
        )

        # 5. Generate Text Layout using Utility
        # Pass page bbox as layout context
        page_bbox = (0, 0, self.width, self.height)
        # Merge PDF-level default tolerances if caller did not override
        merged_kwargs = {
            "layout": layout,
            "x_density": x_density,
            "y_density": y_density,
            "x_tolerance": x_tolerance,
            "y_tolerance": y_tolerance,
            "line_dir": line_dir,
            "char_dir": char_dir,
        }
        merged_kwargs = {
            key: value
            for key, value in merged_kwargs.items()
            if value is not None or key == "layout"
        }
        tol_keys = ["x_tolerance", "x_tolerance_ratio", "y_tolerance"]
        for k in tol_keys:
            if k not in merged_kwargs:
                if k in self._config:
                    merged_kwargs[k] = self._config[k]
                elif hasattr(self._parent, "_config") and k in self._parent._config:
                    merged_kwargs[k] = self._parent._config[k]

        # Add content_filter to kwargs if provided
        if content_filter is not None:
            merged_kwargs["content_filter"] = content_filter

        result = generate_text_layout(
            char_dicts=filtered_chars,
            layout_context_bbox=page_bbox,
            user_kwargs=merged_kwargs,
        )

        if bidi and result:
            # Quick check for any RTL character
            import unicodedata

            def _contains_rtl(s):
                return any(unicodedata.bidirectional(ch) in ("R", "AL", "AN") for ch in s)

            if _contains_rtl(result):
                try:
                    from bidi.algorithm import get_display  # type: ignore

                    from natural_pdf.utils.bidi_mirror import mirror_brackets

                    normalized_lines = []
                    for line in result.split("\n"):
                        base_dir = (
                            "R"
                            if any(
                                unicodedata.bidirectional(ch) in ("R", "AL", "AN") for ch in line
                            )
                            else "L"
                        )
                        raw_display = get_display(line, base_dir=base_dir)
                        display_line = (
                            raw_display.decode("utf-8", errors="ignore")
                            if isinstance(raw_display, bytes)
                            else str(raw_display)
                        )
                        normalized_lines.append(mirror_brackets(display_line))
                    result = "\n".join(normalized_lines)
                except ModuleNotFoundError:
                    pass  # silently skip if python-bidi not available

        if strip_empty and result:
            result = "\n".join(line for line in result.splitlines() if line.strip())

        if strip_final and result:
            result = "\n".join(line.rstrip() for line in result.splitlines()).strip()

        if result and not preserve_line_breaks:
            normalized = result.replace("\r\n", "\n").replace("\r", "\n")
            if preserve_whitespace:
                result = re.sub(r"\s*\n\s*", " ", normalized)
            else:
                result = " ".join(normalized.split())

        logger.debug(f"Page {self.number}: extract_text finished, result length: {len(result)}.")
        return result

    def extract_table(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[["Region"], Optional[str]]] = None,
        show_progress: bool = False,
        content_filter=None,
        verticals: Optional[List[float]] = None,
        horizontals: Optional[List[float]] = None,
        structure_engine: Optional[str] = None,
    ) -> TableResult:
        """
        Extract the largest table from this page using enhanced region-based extraction.

        Args:
            method: Method to use: 'tatr', 'pdfplumber', 'text', 'stream', 'lattice', or None (auto-detect).
            table_settings: Settings for pdfplumber table extraction.
            use_ocr: Whether to use OCR for text extraction (currently only applicable with 'tatr' method).
            ocr_config: OCR configuration parameters.
            text_options: Dictionary of options for the 'text' method.
            cell_extraction_func: Optional callable function that takes a cell Region object
                                    and returns its string content. For 'text' method only.
            show_progress: If True, display a progress bar during cell text extraction for the 'text' method.
            content_filter: Optional content filter to apply during cell text extraction. Can be:
                - A regex pattern string (characters matching the pattern are EXCLUDED)
                - A callable that takes text and returns True to KEEP the character
                - A list of regex patterns (characters matching ANY pattern are EXCLUDED)
            verticals: Optional list of x-coordinates for explicit vertical table lines.
            horizontals: Optional list of y-coordinates for explicit horizontal table lines.
            structure_engine: Optional structure detection engine forwarded to the underlying
                region so provider-backed engines (e.g., TATR structure consumers) can be leveraged
                before falling back to standard extraction methods.

        Returns:
            TableResult: A sequence-like object containing table rows that also provides .to_df() for pandas conversion.
        """
        # Create a full-page region and delegate to its enhanced extract_table method
        page_region = self.create_region(0, 0, self.width, self.height)
        return page_region.extract_table(
            method=method,
            table_settings=table_settings,
            use_ocr=use_ocr,
            ocr_config=ocr_config,
            text_options=text_options,
            cell_extraction_func=cell_extraction_func,
            show_progress=show_progress,
            content_filter=content_filter,
            verticals=verticals,
            horizontals=horizontals,
            structure_engine=structure_engine,
        )

    def extract_tables(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        check_tatr: bool = True,
    ) -> List[List[List[Optional[str]]]]:
        """
        Extract all tables from this page with enhanced method support.

        Args:
            method: Method to use: 'pdfplumber', 'stream', 'lattice', or None (auto-detect).
                    'stream' uses text-based strategies, 'lattice' uses line-based strategies.
                    Note: 'tatr' and 'text' methods are not supported for extract_tables.
            table_settings: Settings for pdfplumber table extraction.
            check_tatr: If True (default), first check for TATR-detected table regions
                        and extract from those before falling back to pdfplumber methods.

        Returns:
            List of tables, where each table is a list of rows, and each row is a list of cell values.
        """
        base_settings = table_settings.copy() if table_settings else {}

        def _normalise_table(table: Sequence[Sequence[Optional[str]]]) -> List[List[str]]:
            return [[cell if isinstance(cell, str) else "" for cell in row] for row in table]

        def _as_optional_rows(table: Sequence[Sequence[str]]) -> List[List[Optional[str]]]:
            return [[cell for cell in row] for row in table]

        def _process_pdfplumber_tables(
            raw_tables: Optional[Sequence[Sequence[Sequence[Any]]]],
        ) -> List[List[List[Optional[str]]]]:
            if not raw_tables:
                return []
            processed: List[List[List[Optional[str]]]] = []
            for table in raw_tables:
                normalized_rows: List[List[str]] = []
                for row in table:
                    normalized_row: List[str] = []
                    for cell in row:
                        text_value = "" if cell is None else str(cell)
                        if text_value:
                            text_value = self._apply_rtl_processing_to_text(text_value)
                        normalized_row.append(text_value)
                    normalized_rows.append(normalized_row)
                processed.append(_as_optional_rows(normalized_rows))
            return processed

        def _apply_text_strategy_overrides(settings: Dict[str, Any]) -> None:
            if "text" not in (
                settings.get("vertical_strategy"),
                settings.get("horizontal_strategy"),
            ):
                return

            if "text_x_tolerance" not in settings and "x_tolerance" not in settings:
                x_tol = self.get_config("x_tolerance", None, scope="page")
                if x_tol is not None:
                    settings.setdefault("text_x_tolerance", x_tol)
            if "text_y_tolerance" not in settings and "y_tolerance" not in settings:
                y_tol = self.get_config("y_tolerance", None, scope="page")
                if y_tol is not None:
                    settings.setdefault("text_y_tolerance", y_tol)

            if "snap_tolerance" not in settings and "snap_x_tolerance" not in settings:
                y_tol = self.get_config("y_tolerance", 1, scope="page")
                snap = max(1, round(float(y_tol) * 0.9))
                settings.setdefault("snap_tolerance", snap)
            if "join_tolerance" not in settings and "join_x_tolerance" not in settings:
                join = settings.get("snap_tolerance", 1)
                settings.setdefault("join_tolerance", join)
                settings.setdefault("join_x_tolerance", join)
                settings.setdefault("join_y_tolerance", join)

        def _run_pdfplumber(
            overrides: Optional[Dict[str, Any]] = None, *, force: bool = False
        ) -> List[List[List[Optional[str]]]]:
            settings = base_settings.copy()
            if overrides:
                for key, value in overrides.items():
                    if force or key not in settings:
                        settings[key] = value
            _apply_text_strategy_overrides(settings)
            raw_tables = self._page.extract_tables(settings)
            return _process_pdfplumber_tables(raw_tables)

        def _auto_detect_tables() -> List[List[List[Optional[str]]]]:
            strategies = (
                ("lattice", {"vertical_strategy": "lines", "horizontal_strategy": "lines"}),
                ("stream", {"vertical_strategy": "text", "horizontal_strategy": "text"}),
            )
            for label, overrides in strategies:
                try:
                    tables = _run_pdfplumber(overrides, force=True)
                    if tables:
                        logger.debug(
                            f"Page {self.number}: '{label}' strategy found {len(tables)} tables"
                        )
                        return tables
                except Exception as exc:
                    logger.debug(f"Page {self.number}: '{label}' strategy failed: {exc}")
            return []

        def _extract_tables_via_tatr() -> List[List[List[Optional[str]]]]:
            regions = self.find_all("region[type=table][model=tatr]")
            if not regions:
                logger.debug(
                    f"Page {self.number}: No TATR table regions found, using pdfplumber methods"
                )
                return []

            extracted: List[List[List[Optional[str]]]] = []
            for table_region in regions:
                table_data = table_region.extract_table(method="tatr")
                if table_data:
                    extracted.append(_as_optional_rows(_normalise_table(table_data)))

            if extracted:
                logger.debug(
                    f"Page {self.number}: Successfully extracted {len(extracted)} tables from TATR regions"
                )
            else:
                logger.debug(
                    f"Page {self.number}: TATR regions found but no tables extracted, falling back to pdfplumber"
                )
            return extracted

        if check_tatr:
            try:
                tatr_tables = _extract_tables_via_tatr()
                if tatr_tables:
                    return tatr_tables
            except Exception as exc:
                logger.debug(
                    f"Page {self.number}: Error checking TATR regions: {exc}, falling back to pdfplumber"
                )

        if method is None:
            auto_tables = _auto_detect_tables()
            if auto_tables:
                return auto_tables
            logger.debug(
                f"Page {self.number}: Auto-detection failed to find tables. Returning empty list."
            )
            return []

        settings = table_settings.copy() if table_settings else {}

        if check_tatr:
            try:
                tatr_matches = _extract_tables_via_tatr()
                if tatr_matches:
                    logger.debug(
                        f"Page {self.number}: Returning {len(tatr_matches)} tables from TATR regions."
                    )
                    return tatr_matches
            except Exception as exc:
                logger.debug(
                    f"Page {self.number}: Error checking TATR regions: {exc}; falling back to pdfplumber."
                )

        page_region = self.to_region()
        try:
            return page_region.extract_tables(method=method, table_settings=settings)
        except Exception as exc:
            logger.debug(f"Page {self.number}: Table extraction failed: {exc}")
            return []

    def _load_elements(self):
        """Load all elements from the page via ElementManager."""
        self._element_mgr.load_elements()

    def _create_char_elements(self):
        """DEPRECATED: Use self._element_mgr.chars"""
        logger.warning("_create_char_elements is deprecated. Access via self._element_mgr.chars.")
        return self._element_mgr.chars  # Delegate

    def _process_font_information(self, char_dict):
        """DEPRECATED: Handled by ElementManager"""
        logger.warning("_process_font_information is deprecated. Handled by ElementManager.")
        # ElementManager handles this internally
        pass

    def _group_chars_into_words(self, keep_spaces=True, font_attrs=None):
        """DEPRECATED: Use self._element_mgr.words"""
        logger.warning("_group_chars_into_words is deprecated. Access via self._element_mgr.words.")
        return self._element_mgr.words  # Delegate

    def _process_line_into_words(self, line_chars, keep_spaces, font_attrs):
        """DEPRECATED: Handled by ElementManager"""
        logger.warning("_process_line_into_words is deprecated. Handled by ElementManager.")
        pass

    def _check_font_attributes_match(self, char, prev_char, font_attrs):
        """DEPRECATED: Handled by ElementManager"""
        logger.warning("_check_font_attributes_match is deprecated. Handled by ElementManager.")
        pass

    def _create_word_element(self, chars, font_attrs):
        """DEPRECATED: Handled by ElementManager"""
        logger.warning("_create_word_element is deprecated. Handled by ElementManager.")
        pass

    @property
    def chars(self) -> List[Any]:
        """Get all character elements on this page."""
        return self._element_mgr.chars

    @property
    def words(self) -> List[Any]:
        """Get all word elements on this page."""
        return self._element_mgr.words

    @property
    def rects(self) -> List[Any]:
        """Get all rectangle elements on this page."""
        return self._element_mgr.rects

    @property
    def lines(self) -> List[Any]:
        """Get all line elements on this page."""
        return self._element_mgr.lines

    def add_highlight(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        color: Optional[Union[Tuple, str]] = None,
        label: Optional[str] = None,
        use_color_cycling: bool = False,
        element: Optional[Any] = None,
        annotate: Optional[List[str]] = None,
        existing: str = "append",
    ) -> "Page":
        """
        Add a highlight to a bounding box or the entire page.
        Delegates to the central HighlightingService.

        Args:
            bbox: Bounding box (x0, top, x1, bottom). If None, highlight entire page.
            color: RGBA color tuple/string for the highlight.
            label: Optional label for the highlight.
            use_color_cycling: If True and no label/color, use next cycle color.
            element: Optional original element being highlighted (for attribute extraction).
            annotate: List of attribute names from 'element' to display.
            existing: How to handle existing highlights ('append' or 'replace').

        Returns:
            Self for method chaining.
        """
        target_bbox = bbox if bbox is not None else (0, 0, self.width, self.height)
        self._highlighter.add(
            page_index=self.index,
            bbox=target_bbox,
            color=color,
            label=label,
            use_color_cycling=use_color_cycling,
            element=element,
            annotate=annotate,
            existing=existing,
        )
        return self

    def add_highlight_polygon(
        self,
        polygon: List[Tuple[float, float]],
        color: Optional[Union[Tuple, str]] = None,
        label: Optional[str] = None,
        use_color_cycling: bool = False,
        element: Optional[Any] = None,
        annotate: Optional[List[str]] = None,
        existing: str = "append",
    ) -> "Page":
        """
        Highlight a polygon shape on the page.
        Delegates to the central HighlightingService.

        Args:
            polygon: List of (x, y) points defining the polygon.
            color: RGBA color tuple/string for the highlight.
            label: Optional label for the highlight.
            use_color_cycling: If True and no label/color, use next cycle color.
            element: Optional original element being highlighted (for attribute extraction).
            annotate: List of attribute names from 'element' to display.
            existing: How to handle existing highlights ('append' or 'replace').

        Returns:
            Self for method chaining.
        """
        self._highlighter.add_polygon(
            page_index=self.index,
            polygon=polygon,
            color=color,
            label=label,
            use_color_cycling=use_color_cycling,
            element=element,
            annotate=annotate,
            existing=existing,
        )
        return self

    def save_image(
        self,
        filename: str,
        width: Optional[int] = None,
        labels: bool = True,
        legend_position: str = "right",
        render_ocr: bool = False,
        include_highlights: bool = True,  # Allow saving without highlights
        resolution: float = 144,
        **kwargs,
    ) -> "Page":
        """
        Save the page image to a file, rendering highlights via HighlightingService.

        Args:
            filename: Path to save the image to.
            width: Optional width for the output image.
            labels: Whether to include a legend.
            legend_position: Position of the legend.
            render_ocr: Whether to render OCR text.
            include_highlights: Whether to render highlights.
            resolution: Resolution in DPI for base image rendering (default: 144 DPI, equivalent to previous scale=2.0).
            **kwargs: Additional args for pdfplumber's internal to_image.

        Returns:
            Self for method chaining.
        """
        # Use export() to save the image
        if include_highlights:
            self.export(
                path=filename,
                resolution=resolution,
                width=width,
                labels=labels,
                legend_position=legend_position,
                render_ocr=render_ocr,
                **kwargs,
            )
        else:
            # For saving without highlights, use render() and save manually
            img = self.render(resolution=resolution, **kwargs)
            if img:
                # Resize if width is specified
                if width is not None and width > 0 and img.width > 0:
                    aspect_ratio = img.height / img.width
                    height = int(width * aspect_ratio)
                    try:
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                    except Exception as e:
                        logger.warning(f"Could not resize image: {e}")

                # Save the image
                try:
                    if os.path.dirname(filename):
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                    img.save(filename)
                except Exception as e:
                    logger.error(f"Failed to save image to {filename}: {e}")

        return self

    def clear_highlights(self) -> "Page":
        """
        Clear all highlights *from this specific page* via HighlightingService.

        Returns:
            Self for method chaining
        """
        self._highlighter.clear_page(self.index)
        return self

    def analyze_text_styles(
        self, options: Optional[TextStyleOptions] = None
    ) -> "ElementCollection":
        """
        Analyze text elements by style, adding attributes directly to elements.

        This method uses TextStyleAnalyzer to process text elements (typically words)
        on the page. It adds the following attributes to each processed element:
        - style_label: A descriptive or numeric label for the style group.
        - style_key: A hashable tuple representing the style properties used for grouping.
        - style_properties: A dictionary containing the extracted style properties.

        Args:
            options: Optional TextStyleOptions to configure the analysis.
                        If None, the analyzer's default options are used.

        Returns:
            ElementCollection containing all processed text elements with added style attributes.
        """
        analyzer = TextStyleAnalyzer()
        processed_elements_collection = analyzer.analyze(self, options=options)

        metadata = getattr(self, "metadata", None)
        summary = None
        if isinstance(metadata, dict):
            summary = metadata.get("text_styles_summary")
        self._text_styles_summary = summary or {}

        return processed_elements_collection

    def _create_text_elements_from_ocr(
        self, ocr_results: List[Dict[str, Any]], image_width=None, image_height=None
    ) -> List["TextElement"]:
        """DEPRECATED: Use self._element_mgr.create_text_elements_from_ocr"""
        logger.warning(
            "_create_text_elements_from_ocr is deprecated. Use self._element_mgr version."
        )
        return self._element_mgr.create_text_elements_from_ocr(
            ocr_results, image_width, image_height
        )

    def apply_ocr(
        self,
        engine: Optional[str] = None,
        options: Optional["OCROptions"] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        detect_only: bool = False,
        apply_exclusions: bool = True,
        replace: bool = True,
    ) -> "Page":
        """Apply OCR directly to this page."""
        normalized_options = normalize_ocr_options(options)
        engine_name = resolve_ocr_engine_name(
            context=self,
            requested=engine,
            options=normalized_options,
            scope="page",
        )
        resolved_languages = resolve_ocr_languages(self, languages, scope="page")
        resolved_min_conf = resolve_ocr_min_confidence(self, min_confidence, scope="page")
        resolved_device = resolve_ocr_device(self, device, scope="page")

        if replace:
            removed = self.remove_ocr_elements()
            if removed:
                logger.info(
                    f"Page {self.number}: Removed {removed} OCR elements before new OCR run."
                )

        final_resolution = (
            resolution
            if resolution is not None
            else getattr(getattr(self, "_parent", None), "_config", {}).get("resolution", 150)
        )

        try:
            ocr_payload = run_ocr_apply(
                target=self,
                context=self,
                engine_name=engine_name,
                resolution=final_resolution,
                languages=resolved_languages,
                min_confidence=resolved_min_conf,
                device=resolved_device,
                detect_only=detect_only,
                options=normalized_options,
                render_kwargs={},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Page {self.number}: OCR failed: {exc}", exc_info=True)
            raise

        image_width, image_height = ocr_payload.image_size
        if not image_width or not image_height:
            logger.error(f"Page {self.number}: OCR payload missing image dimensions.")
            return self

        scale_x = self.width / image_width if image_width else 1.0
        scale_y = self.height / image_height if image_height else 1.0
        created_elements = self.create_text_elements_from_ocr(
            ocr_payload.results, scale_x=scale_x, scale_y=scale_y
        )

        logger.info(
            f"Page {self.number}: Added {len(created_elements)} OCR elements using '{engine_name}'."
        )
        return self

    def extract_ocr_elements(
        self,
        engine: Optional[str] = None,
        options: Optional["OCROptions"] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
    ) -> List["TextElement"]:
        """
        Extract text elements using OCR *without* mutating this page's element store.

        Args:
            engine: Name of the OCR engine.
            options: Engine-specific options object or dict.
            languages: List of engine-specific language codes.
            min_confidence: Minimum confidence threshold.
            device: Device to run OCR on.
            resolution: DPI resolution for rendering page image before OCR.

        Returns:
            List of created TextElement objects derived from OCR results for this page.
        """

        logger.info(f"Page {self.number}: Extracting OCR elements (extract only)...")

        normalized_options = normalize_ocr_options(options)
        engine_name = resolve_ocr_engine_name(
            context=self,
            requested=engine,
            options=normalized_options,
            scope="page",
        )
        resolved_languages = resolve_ocr_languages(self, languages, scope="page")
        resolved_min_conf = resolve_ocr_min_confidence(self, min_confidence, scope="page")
        resolved_device = resolve_ocr_device(self, device, scope="page")

        final_resolution = resolution if resolution is not None else 150
        logger.debug(f"  Using rendering resolution: {final_resolution} DPI")

        ocr_payload = run_ocr_extract(
            target=self,
            context=self,
            engine_name=engine_name,
            resolution=final_resolution,
            languages=resolved_languages,
            min_confidence=resolved_min_conf,
            device=resolved_device,
            detect_only=False,
            options=normalized_options,
            render_kwargs={},
        )

        results = ocr_payload.results
        image_width, image_height = ocr_payload.image_size
        if not image_width or not image_height:
            logger.error("  OCR payload missing image dimensions; aborting extraction.")
            return []

        # Convert results but DO NOT add to ElementManager
        logger.debug("  Converting OCR results to TextElements (extract only)...")
        temp_elements = []
        scale_x = self.width / image_width if image_width else 1
        scale_y = self.height / image_height if image_height else 1
        for result in results:
            if not isinstance(result, Mapping):
                raise TypeError(f"OCR result has unexpected type: {type(result).__name__}")

            bbox_raw = result.get("bbox")
            text_value = result.get("text", "")
            confidence = result.get("confidence", 0.0)
            if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                raise ValueError("OCR result missing bbox")
            x0, top, x1, bottom = [float(cast(Union[int, float, str], c)) for c in bbox_raw]
            elem_data = {
                "text": str(text_value),
                "confidence": float(confidence) if confidence is not None else 0.0,
                "x0": x0 * scale_x,
                "top": top * scale_y,
                "x1": x1 * scale_x,
                "bottom": bottom * scale_y,
                "width": (x1 - x0) * scale_x,
                "height": (bottom - top) * scale_y,
                "object_type": "text",  # Using text for temporary elements
                "source": "ocr",
                "fontname": "OCR-extract",  # Different name for clarity
                "size": 10.0,
                "page_number": self.number,
            }
            temp_elements.append(TextElement(elem_data, self))

        logger.info(f"  Created {len(temp_elements)} TextElements from OCR (extract only).")
        return temp_elements

    @property
    def size(self) -> Tuple[float, float]:
        """Get the size of the page in points."""
        return (self._page.width, self._page.height)

    @property
    def layout_analyzer(self) -> "LayoutAnalyzer":
        """Get or create the layout analyzer for this page."""
        if self._layout_analyzer is None:
            self._layout_analyzer = LayoutAnalyzer(self)
        return self._layout_analyzer

    def analyze_layout(
        self,
        engine: Optional[str] = None,
        options: Optional["LayoutOptions"] = None,
        confidence: Optional[float] = None,
        classes: Optional[List[str]] = None,
        exclude_classes: Optional[List[str]] = None,
        device: Optional[str] = None,
        existing: str = "replace",
        model_name: Optional[str] = None,
        client: Optional[Any] = None,  # Add client parameter
    ) -> "ElementCollection[Region]":
        """
        Analyze the page layout using the configured layout engine.
        Adds detected Region objects to the page's element manager.

        Returns:
            ElementCollection containing the detected Region objects.
        """
        analyzer = self.layout_analyzer

        # Clear existing detected regions if 'replace' is specified
        if existing == "replace":
            self.clear_detected_layout_regions()

        # The analyzer's analyze_layout method already adds regions to the page
        # and its element manager. We just need to retrieve them.
        analyzer.analyze_layout(
            engine=engine,
            options=options,
            confidence=confidence,
            classes=classes,
            exclude_classes=exclude_classes,
            device=device,
            existing=existing,
            model_name=model_name,
            client=client,  # Pass client down
        )

        # Retrieve the detected regions from the element manager
        # Filter regions based on source='detected' and potentially the model used if available
        detected_regions = [
            r
            for r in self._element_mgr.regions
            if r.source == "detected" and (not engine or getattr(r, "model", None) == engine)
        ]

        return ElementCollection(detected_regions)

    def clear_detected_layout_regions(self) -> "Page":
        """
        Removes all regions from this page that were added by layout analysis
        (i.e., regions where `source` attribute is 'detected').

        This clears the regions both from the page's internal `_regions['detected']` list
        and from the ElementManager's internal list of regions.

        Returns:
            Self for method chaining.
        """
        removed_count = self.remove_regions(source="detected")
        if removed_count:
            logger.info(f"Page {self.index}: Cleared {removed_count} detected layout regions.")
        else:
            logger.debug(f"Page {self.index}: No detected layout regions to clear.")
        return self

    def get_section_between(
        self,
        start_element=None,
        end_element=None,
        include_boundaries="both",
        orientation="vertical",
    ) -> Optional["Region"]:  # Return Optional
        """
        Get a section between two elements on this page.

        Args:
            start_element: Element marking the start of the section
            end_element: Element marking the end of the section
            include_boundaries: How to include boundary elements: 'start', 'end', 'both', or 'none'
            orientation: 'vertical' (default) or 'horizontal' - determines section direction

        Returns:
            Region representing the section
        """
        # Create a full-page region to operate within
        page_region = self.create_region(0, 0, self.width, self.height)

        # Delegate to the region's method
        try:
            return page_region.get_section_between(
                start_element=start_element,
                end_element=end_element,
                include_boundaries=include_boundaries,
                orientation=orientation,
            )
        except Exception as e:
            logger.error(
                f"Error getting section between elements on page {self.index}: {e}", exc_info=True
            )
            return None

    def split(self, divider, **kwargs: Any) -> "ElementCollection[Region]":
        """Divide the page into sections based on the provided divider elements."""
        base_region = self.create_region(0, 0, self.width, self.height)
        return base_region.split(divider, **kwargs)

    def get_sections(
        self,
        start_elements: Union[str, Sequence[Element], ElementCollection, None] = None,
        end_elements: Union[str, Sequence[Element], ElementCollection, None] = None,
        include_boundaries: str = "start",
        y_threshold: float = 5.0,
        bounding_box: Optional[Bounds] = None,
        orientation: str = "vertical",
        **kwargs: Any,
    ) -> "ElementCollection[Region]":
        """Delegate section extraction to the Region implementation."""

        if bounding_box is not None:
            x0, top, x1, bottom = bounding_box
            base_region = self.create_region(x0, top, x1, bottom)
        else:
            base_region = self.create_region(0, 0, self.width, self.height)

        return base_region.get_sections(
            start_elements=start_elements,
            end_elements=end_elements,
            include_boundaries=include_boundaries,
            orientation=orientation,
            y_threshold=y_threshold,
            **kwargs,
        )

    def __repr__(self) -> str:
        """String representation of the page."""
        return f"<Page number={self.number} index={self.index}>"

    def show_preview(
        self,
        temporary_highlights: List[Dict],
        resolution: float = 144,
        width: Optional[int] = None,
        labels: bool = True,
        legend_position: str = "right",
        render_ocr: bool = False,
    ) -> Optional[Image.Image]:
        """
        Generates and returns a non-stateful preview image containing only
        the provided temporary highlights.

        Args:
            temporary_highlights: List of highlight data dictionaries (as prepared by
                                    ElementCollection._prepare_highlight_data).
            resolution: Resolution in DPI for rendering (default: 144 DPI, equivalent to previous scale=2.0).
            width: Optional width for the output image.
            labels: Whether to include a legend.
            legend_position: Position of the legend.
            render_ocr: Whether to render OCR text.

        Returns:
            PIL Image object of the preview, or None if rendering fails.
        """
        try:
            # Delegate rendering to the highlighter service's preview method
            img = self._highlighter.render_preview(
                page_index=self.index,
                temporary_highlights=temporary_highlights,
                resolution=resolution,
                labels=labels,
                legend_position=legend_position,
                render_ocr=render_ocr,
            )
        except AttributeError:
            logger.error("HighlightingService does not have the required 'render_preview' method.")
            return None
        except Exception as e:
            logger.error(
                f"Error calling highlighter.render_preview for page {self.index}: {e}",
                exc_info=True,
            )
            return None

        # Return the rendered image directly
        return img

    @property
    def text_style_labels(self) -> List[str]:
        """
        Get a sorted list of unique text style labels found on the page.

        Runs text style analysis with default options if it hasn't been run yet.
        To use custom options, call `analyze_text_styles(options=...)` explicitly first.

        Returns:
            A sorted list of unique style label strings.
        """
        summary: dict[str, Any] = getattr(self, "_text_styles_summary", {})
        if not summary:
            metadata = getattr(self, "metadata", None)
            if isinstance(metadata, dict):
                meta_summary = metadata.get("text_styles_summary")
                if isinstance(meta_summary, dict):
                    summary = meta_summary
                    self._text_styles_summary = summary

        if not summary:
            logger.debug(f"Page {self.number}: Running default text style analysis to get labels.")
            self.analyze_text_styles()
            summary = getattr(self, "_text_styles_summary", {})

        if summary:
            labels = {
                style_info["label"] for style_info in summary.values() if "label" in style_info
            }
            return sorted(labels)

        logger.warning(f"Page {self.number}: Text style summary not available after analysis.")
        return []

    def viewer(
        self,
        # elements_to_render: Optional[List['Element']] = None, # No longer needed, from_page handles it
        # include_source_types: List[str] = ['word', 'line', 'rect', 'region'] # No longer needed
    ) -> Optional[Any]:
        """
        Creates and returns an interactive ipywidget for exploring elements on this page.

        Uses InteractiveViewerWidget.from_page() to create the viewer.

        Returns:
            A InteractiveViewerWidget instance ready for display in Jupyter,
            or None if ipywidgets is not installed or widget creation fails.

        Raises:
            # Optional: Could raise ImportError instead of returning None
            # ImportError: If required dependencies (ipywidgets) are missing.
            ValueError: If image rendering or data preparation fails within from_page.
        """
        # Check for availability using the imported flag and class variable
        if not _IPYWIDGETS_AVAILABLE or InteractiveViewerWidget is None:
            logger.error(
                "Interactive viewer requires 'ipywidgets'. "
                'Please install with: pip install "ipywidgets>=7.0.0,<10.0.0"'
            )
            # raise ImportError("ipywidgets not found.") # Option 1: Raise error
            return None  # Option 2: Return None gracefully

        # If we reach here, InteractiveViewerWidget should be the actual class
        try:
            # Pass self (the Page object) to the factory method
            return InteractiveViewerWidget.from_page(self)
        except Exception as e:
            # Catch potential errors during widget creation (e.g., image rendering)
            logger.error(
                f"Error creating viewer widget from page {self.number}: {e}", exc_info=True
            )
            # raise # Option 1: Re-raise error (might include ValueError from from_page)
            return None  # Option 2: Return None on creation error

    def get_id(self) -> str:
        """Returns a unique identifier for the page (required by Indexable protocol)."""
        # Ensure path is safe for use in IDs (replace problematic chars)
        safe_path = re.sub(r"[^a-zA-Z0-9_-]", "_", str(self.pdf.path))
        return f"pdf_{safe_path}_page_{self.page_number}"

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata associated with the page (required by Indexable protocol)."""
        # Add content hash here for sync
        metadata = {
            "pdf_path": str(self.pdf.path),
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "content_hash": self.get_content_hash(),  # Include the hash
        }
        return metadata

    def get_content(self) -> "Page":
        """
        Returns the primary content object (self) for indexing (required by Indexable protocol).
        SearchService implementations decide how to process this (e.g., call extract_text).
        """
        return self  # Return the Page object itself

    def get_content_hash(self) -> str:
        """Returns a SHA256 hash of the extracted text content (required by Indexable for sync)."""
        # Hash the extracted text (without exclusions for consistency)
        # Consider if exclusions should be part of the hash? For now, hash raw text.
        # Using extract_text directly might be slow if called repeatedly. Cache? TODO: Optimization
        text_content = self.extract_text(
            use_exclusions=False, preserve_whitespace=False
        )  # Normalize whitespace?
        return hashlib.sha256(text_content.encode("utf-8")).hexdigest()

    def save_searchable(self, output_path: Union[str, "Path"], dpi: int = 300):
        """
        Saves the PDF page with an OCR text layer, making content searchable.

        Requires optional dependencies. Install with: pip install "natural-pdf[ocr-save]"

        Note: OCR must have been applied to the pages beforehand
                (e.g., pdf.apply_ocr()).

        Args:
            output_path: Path to save the searchable PDF.
            dpi: Resolution for rendering and OCR overlay (default 300).
        """
        # Import moved here, assuming it's always available now
        from natural_pdf.exporters.searchable_pdf import create_searchable_pdf

        # Convert pathlib.Path to string if necessary
        output_path_str = str(output_path)

        create_searchable_pdf(self, output_path_str, dpi=dpi)
        logger.info(f"Searchable PDF saved to: {output_path_str}")

    def update_text(
        self,
        transform: Callable[[Any], Optional[str]],
        *,
        selector: str = "text",
        apply_exclusions: bool = False,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[], None]] = None,  # Added progress callback
    ) -> "Page":  # Return self for chaining
        """
        Applies corrections to text elements on this page
        using a user-provided callback function, potentially in parallel.

        Finds text elements on this page matching the *selector* argument and
        calls the ``transform`` for each, passing the element itself.
        Updates the element's text if the callback returns a new string.

        Args:
            transform: A function accepting an element and returning
                        `Optional[str]` (new text or None).
            selector: CSS-like selector string to match text elements.
            apply_exclusions: Whether exclusion regions should be honoured (default: False to
                maintain backward compatibility with the base mixin behaviour).
            max_workers: The maximum number of threads to use for parallel execution.
                            If None or 0 or 1, runs sequentially.
            progress_callback: Optional callback function to call after processing each element.

        Returns:
            Self for method chaining.
        """
        logger.info(
            f"Page {self.number}: Starting text update with callback '{transform.__name__}' (max_workers={max_workers}) and selector='{selector}'"
        )

        target_elements_collection = self.find_all(
            selector=selector, apply_exclusions=apply_exclusions
        )
        target_elements = target_elements_collection.elements  # Get the list

        if not target_elements:
            logger.info(f"Page {self.number}: No text elements found to update.")
            return self

        element_pbar = None
        try:
            element_pbar = tqdm(
                total=len(target_elements),
                desc=f"Updating text Page {self.number}",
                unit="element",
                leave=False,
            )

            processed_count = 0
            updated_count = 0
            error_count = 0

            # Define the task to be run by the worker thread or sequentially
            def _process_element_task(element):
                try:
                    # Call the user-provided callback
                    corrected_text = transform(element)

                    # Validate result type
                    if corrected_text is not None and not isinstance(corrected_text, str):
                        logger.warning(
                            f"Page {self.number}: Correction callback for element '{getattr(element, 'text', '')[:20]}...' returned non-string, non-None type: {type(corrected_text)}. Skipping update."
                        )
                        return element, None, None  # Treat as no correction

                    return element, corrected_text, None  # Return element, result, no error
                except Exception as e:
                    logger.error(
                        f"Page {self.number}: Error applying correction callback to element '{getattr(element, 'text', '')[:30]}...' ({element.bbox}): {e}",
                        exc_info=False,  # Keep log concise
                    )
                    return element, None, e  # Return element, no result, error
                finally:
                    if element_pbar:
                        element_pbar.update(1)
                    if progress_callback:
                        try:
                            progress_callback()
                        except Exception as cb_e:
                            # Log error in callback itself, but don't stop processing
                            logger.error(
                                f"Page {self.number}: Error executing progress_callback: {cb_e}",
                                exc_info=False,
                            )

            # Choose execution strategy based on max_workers
            if max_workers is not None and max_workers > 1:
                logger.info(
                    f"Page {self.number}: Running text update in parallel with {max_workers} workers."
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_element = {
                        executor.submit(_process_element_task, element): element
                        for element in target_elements
                    }

                    # Process results as they complete (progress_callback called by worker)
                    for future in concurrent.futures.as_completed(future_to_element):
                        processed_count += 1
                        try:
                            element, corrected_text, error = future.result()
                            if error:
                                error_count += 1
                                # Error already logged in worker
                            elif corrected_text is not None:
                                # Apply correction if text changed
                                current_text = getattr(element, "text", None)
                                if corrected_text != current_text:
                                    element.text = corrected_text
                                    updated_count += 1
                        except Exception as exc:
                            # Catch errors from future.result() itself
                            element = future_to_element[future]  # Find original element
                            logger.error(
                                f"Page {self.number}: Internal error retrieving correction result for element {element.bbox}: {exc}",
                                exc_info=True,
                            )
                            error_count += 1
                            # Note: progress_callback was already called in the worker's finally block

            else:
                logger.info(f"Page {self.number}: Running text update sequentially.")
                for element in target_elements:
                    # Call the task function directly (it handles progress_callback)
                    processed_count += 1
                    _element, corrected_text, error = _process_element_task(element)
                    if error:
                        error_count += 1
                    elif corrected_text is not None:
                        # Apply correction if text changed
                        current_text = getattr(_element, "text", None)
                        if corrected_text != current_text:
                            _element.text = corrected_text
                            updated_count += 1

            logger.info(
                f"Page {self.number}: Text update finished. Processed: {processed_count}/{len(target_elements)}, Updated: {updated_count}, Errors: {error_count}."
            )

            return self  # Return self for chaining
        finally:
            if element_pbar:
                element_pbar.close()

    def _get_classification_content(self, model_type: str, **kwargs) -> Union[str, Image.Image]:
        if model_type == "text":
            text_content = self.extract_text(
                layout=False, use_exclusions=False
            )  # Simple join, ignore exclusions for classification
            if not text_content or text_content.isspace():
                raise ValueError("Cannot classify page with 'text' model: No text content found.")
            return text_content
        elif model_type == "vision":
            resolution = kwargs.get("resolution", 150)
            # Use render() for clean image without highlights
            img = self.render(resolution=resolution)
            if img is None:
                raise ValueError(
                    "Cannot classify page with 'vision' model: Failed to render image."
                )
            return img
        else:
            raise ValueError(f"Unsupported model_type for classification: {model_type}")

    def _get_metadata_storage(self) -> Dict[str, Any]:
        # Ensure metadata exists
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}
        return self.metadata

    @property
    def skew_angle(self) -> Optional[float]:
        """Get the detected skew angle for this page (if calculated)."""
        return self._skew_angle

    def detect_skew_angle(
        self,
        resolution: int = 72,
        grayscale: bool = True,
        force_recalculate: bool = False,
        **deskew_kwargs,
    ) -> Optional[float]:
        """Detect the skew angle of this page using the deskew provider."""
        if self._skew_angle is not None and not force_recalculate:
            logger.debug(f"Page {self.number}: Returning cached skew angle: {self._skew_angle:.2f}")
            return self._skew_angle

        try:
            angle = run_deskew_detect(
                target=self,
                context=self,
                resolution=resolution,
                grayscale=grayscale,
                deskew_kwargs=deskew_kwargs,
            )
        except ImportError:
            raise
        except Exception as exc:
            logger.warning(f"Page {self.number}: Skew detection failed: {exc}", exc_info=True)
            self._skew_angle = None
            return None

        self._skew_angle = angle
        return angle

    def deskew(
        self,
        resolution: int = 300,
        angle: Optional[float] = None,
        detection_resolution: int = 72,
        **deskew_kwargs,
    ) -> Optional[Image.Image]:
        """
        Creates and returns a deskewed PIL image of the page.

        If `angle` is not provided, it will first try to detect the skew angle
        using `detect_skew_angle` (or use the cached angle if available).

        Args:
            resolution: DPI resolution for the output deskewed image.
            angle: The specific angle (in degrees) to rotate by. If None, detects automatically.
            detection_resolution: DPI resolution used for detection if `angle` is None.
            **deskew_kwargs: Additional keyword arguments passed to `deskew.determine_skew`
                                if automatic detection is performed.

        Returns:
            A deskewed PIL.Image.Image object, or None if rendering/rotation fails.

        Raises:
            ImportError: If the 'deskew' library is not installed.
        """
        try:
            result = run_deskew_apply(
                target=self,
                context=self,
                resolution=resolution,
                angle=angle,
                detection_resolution=detection_resolution,
                grayscale=True,
                deskew_kwargs=deskew_kwargs,
            )
        except ImportError:
            raise
        except Exception as exc:
            logger.error(
                f"Page {self.number}: Error during deskewing image generation: {exc}",
                exc_info=True,
            )
            return None

        return result.image

    # Unified analysis storage (maps to metadata["analysis"])

    @property
    def analyses(self) -> Dict[str, Any]:
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}
        return self.metadata.setdefault("analysis", {})

    @analyses.setter
    def analyses(self, value: Dict[str, Any]) -> None:
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}
        self.metadata["analysis"] = value

    def inspect(self, limit: int = 30) -> "InspectionSummary":
        """
        Inspect all elements on this page with detailed tabular view.
        Equivalent to page.find_all('*').inspect().

        Args:
            limit: Maximum elements per type to show (default: 30)

        Returns:
            InspectionSummary with element tables showing coordinates,
            properties, and other details for each element
        """
        return self.find_all("*").inspect(limit=limit)

    def remove_text_layer(self) -> "Page":
        """
        Remove all text elements from this page.

        This removes all text elements (words and characters) from the page,
        effectively clearing the text layer.

        Returns:
            Self for method chaining
        """
        logger.info(f"Page {self.number}: Removing all text elements...")

        removed_words, removed_chars = self._element_mgr.clear_text_layer()

        logger.info(
            f"Page {self.number}: Removed {removed_words} words and {removed_chars} characters"
        )
        return self

    def _apply_rtl_processing_to_text(self, text: str) -> str:
        """
        Apply RTL (Right-to-Left) text processing to a string.

        This converts visual order text (as stored in PDFs) to logical order
        for proper display of Arabic, Hebrew, and other RTL scripts.

        Args:
            text: Input text string in visual order

        Returns:
            Text string in logical order
        """
        if not text or not text.strip():
            return text

        # Quick check for RTL characters - if none found, return as-is
        import unicodedata

        def _contains_rtl(s):
            return any(unicodedata.bidirectional(ch) in ("R", "AL", "AN") for ch in s)

        if not _contains_rtl(text):
            return text

        try:
            from bidi.algorithm import get_display  # type: ignore

            from natural_pdf.utils.bidi_mirror import mirror_brackets

            # Apply BiDi algorithm to convert from visual to logical order
            # Process line by line to handle mixed content properly
            processed_lines = []
            for line in text.split("\n"):
                if line.strip():
                    # Determine base direction for this line
                    base_dir = "R" if _contains_rtl(line) else "L"
                    logical_line = get_display(line, base_dir=base_dir)
                    if isinstance(logical_line, bytes):
                        try:
                            logical_line = logical_line.decode("utf-8")
                        except UnicodeDecodeError:
                            logical_line = logical_line.decode("utf-8", "ignore")
                    # Apply bracket mirroring for correct logical order
                    processed_lines.append(mirror_brackets(logical_line))
                else:
                    processed_lines.append(line)

            return "\n".join(processed_lines)

        except (ImportError, Exception):
            # If bidi library is not available or fails, return original text
            return text

    @property
    def images(self) -> List[Any]:
        """Get all embedded raster images on this page."""
        return self._element_mgr.images

    def highlights(self, show: bool = False) -> "HighlightContext":
        """
        Create a highlight context for accumulating highlights.

        This allows for clean syntax to show multiple highlight groups:

        Example:
            with page.highlights() as h:
                h.add(page.find_all('table'), label='tables', color='blue')
                h.add(page.find_all('text:bold'), label='bold text', color='red')
                h.show()

        Or with automatic display:
            with page.highlights(show=True) as h:
                h.add(page.find_all('table'), label='tables')
                h.add(page.find_all('text:bold'), label='bold')
                # Automatically shows when exiting the context

        Args:
            show: If True, automatically show highlights when exiting context

        Returns:
            HighlightContext for accumulating highlights
        """
        from natural_pdf.core.highlighting_service import HighlightContext

        return HighlightContext(self, show_on_exit=show)
