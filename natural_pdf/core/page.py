import base64
import concurrent.futures  # Added import
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
import time  # Import time
from pathlib import Path
from typing import (  # Added overload
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import pdfplumber
from PIL import Image, ImageDraw
from tqdm.auto import tqdm  # Added tqdm import

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.selectors.parser import parse_selector
from natural_pdf.utils.locks import pdf_render_lock  # Import from utils instead
from natural_pdf.utils.visualization import render_plain_page

if TYPE_CHECKING:
    import pdfplumber

    from natural_pdf.core.highlighting_service import HighlightingService
    from natural_pdf.core.pdf import PDF
    from natural_pdf.elements.base import Element

# # New Imports
import itertools

# # Deskew Imports (Conditional)
import numpy as np
from pdfplumber.utils.geometry import get_bbox_overlap, merge_bboxes, objects_to_bbox
from pdfplumber.utils.text import TEXTMAP_KWARGS, WORD_EXTRACTOR_KWARGS, chars_to_textmap

from natural_pdf.analyzers.layout.layout_analyzer import LayoutAnalyzer
from natural_pdf.analyzers.layout.layout_manager import LayoutManager
from natural_pdf.analyzers.layout.layout_options import LayoutOptions

# --- Shape Detection Mixin --- #
from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin
from natural_pdf.analyzers.text_options import TextStyleOptions
from natural_pdf.analyzers.text_structure import TextStyleAnalyzer
from natural_pdf.classification.manager import ClassificationManager  # For type hint

# # --- Classification Imports --- #
from natural_pdf.classification.mixin import ClassificationMixin  # Import classification mixin
from natural_pdf.core.element_manager import ElementManager

# Add new import
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.describe.mixin import DescribeMixin  # Import describe mixin
from natural_pdf.elements.base import Element  # Import base element
from natural_pdf.elements.text import TextElement
from natural_pdf.extraction.mixin import ExtractionMixin  # Import extraction mixin
from natural_pdf.ocr import OCRManager, OCROptions
from natural_pdf.ocr.utils import _apply_ocr_correction_to_elements
from natural_pdf.qa import DocumentQA, get_qa_engine

# --- Text update mixin import --- #
from natural_pdf.text_mixin import TextMixin
from natural_pdf.utils.locks import pdf_render_lock  # Import the lock

# # Import new utils
from natural_pdf.utils.text_extraction import filter_chars_spatially, generate_text_layout
from natural_pdf.widgets.viewer import _IPYWIDGETS_AVAILABLE, InteractiveViewerWidget

# --- End Classification Imports --- #


try:
    from deskew import determine_skew

    DESKEW_AVAILABLE = True
except ImportError:
    DESKEW_AVAILABLE = False
    determine_skew = None
# End Deskew Imports

logger = logging.getLogger(__name__)


class Page(
    TextMixin,
    ClassificationMixin,
    ExtractionMixin,
    ShapeDetectionMixin,
    DescribeMixin,
    Visualizable,
):
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
        page: "pdfplumber.page.Page",
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
        self._text_styles = None  # Lazy-loaded text style analyzer results
        self._exclusions = []  # List to store exclusion functions/regions
        self._skew_angle: Optional[float] = None  # Stores detected skew angle

        # --- ADDED --- Metadata store for mixins
        self.metadata: Dict[str, Any] = {}
        # --- END ADDED ---

        # Region management
        self._regions = {
            "detected": [],  # Layout detection results
            "named": {},  # Named regions (name -> region)
        }

        # -------------------------------------------------------------
        # Page-scoped configuration begins as a shallow copy of the parent
        # PDF-level configuration so that auto-computed tolerances or other
        # page-specific values do not overwrite siblings.
        # -------------------------------------------------------------
        self._config = dict(getattr(self._parent, "_config", {}))

        # Initialize ElementManager, passing font_attrs
        self._element_mgr = ElementManager(self, font_attrs=font_attrs, load_text=self._load_text)
        # self._highlighter = HighlightingService(self) # REMOVED - Use property accessor
        # --- NEW --- Central registry for analysis results
        self.analyses: Dict[str, Any] = {}

        # --- Get OCR Manager Instance ---
        if (
            OCRManager
            and hasattr(parent, "_ocr_manager")
            and isinstance(parent._ocr_manager, OCRManager)
        ):
            self._ocr_manager = parent._ocr_manager
            logger.debug(f"Page {self.number}: Using OCRManager instance from parent PDF.")
        else:
            self._ocr_manager = None
            if OCRManager:
                logger.warning(
                    f"Page {self.number}: OCRManager instance not found on parent PDF object."
                )

        # --- Get Layout Manager Instance ---
        if (
            LayoutManager
            and hasattr(parent, "_layout_manager")
            and isinstance(parent._layout_manager, LayoutManager)
        ):
            self._layout_manager = parent._layout_manager
            logger.debug(f"Page {self.number}: Using LayoutManager instance from parent PDF.")
        else:
            self._layout_manager = None
            if LayoutManager:
                logger.warning(
                    f"Page {self.number}: LayoutManager instance not found on parent PDF object. Layout analysis will fail."
                )

        # Initialize the internal variable with a single underscore
        self._layout_analyzer = None

        self._load_elements()
        self._to_image_cache: Dict[tuple, Optional["Image.Image"]] = {}

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

        # Handle cropping
        if crop_bbox:
            spec.crop_bbox = crop_bbox
        elif crop == "content":
            # Calculate content bounds from all elements
            elements = self.get_elements(apply_exclusions=False)
            if elements:
                # Get bounding box of all elements
                x_coords = []
                y_coords = []
                for elem in elements:
                    if hasattr(elem, "bbox") and elem.bbox:
                        x0, y0, x1, y1 = elem.bbox
                        x_coords.extend([x0, x1])
                        y_coords.extend([y0, y1])

                if x_coords and y_coords:
                    spec.crop_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        elif crop is True:
            # Crop to full page (no-op, but included for consistency)
            spec.crop_bbox = (0, 0, self.width, self.height)

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
                    elements = group.get("elements", [])
                    group_color = group.get("color", color)
                    group_label = group.get("label")

                    for elem in elements:
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

    # --- Highlighting Service Accessor ---
    @property
    def _highlighter(self) -> "HighlightingService":
        """Provides access to the parent PDF's HighlightingService."""
        if not hasattr(self._parent, "highlighter"):
            # This should ideally not happen if PDF.__init__ works correctly
            raise AttributeError("Parent PDF object does not have a 'highlighter' attribute.")
        return self._parent.highlighter

    def clear_exclusions(self) -> "Page":
        """
        Clear all exclusions from the page.
        """
        self._exclusions = []
        return self

    def add_exclusion(
        self,
        exclusion_func_or_region: Union[
            Callable[["Page"], "Region"], "Region", List[Any], Tuple[Any, ...], Any
        ],
        label: Optional[str] = None,
        method: str = "region",
    ) -> "Page":
        """
        Add an exclusion to the page. Text from these regions will be excluded from extraction.
        Ensures non-callable items are stored as Region objects if possible.

        Args:
            exclusion_func_or_region: Either a callable function returning a Region,
                                      a Region object, a list/tuple of regions or elements,
                                      or another object with a valid .bbox attribute.
            label: Optional label for this exclusion (e.g., 'header', 'footer').
            method: Exclusion method - 'region' (exclude all elements in bounding box) or
                    'element' (exclude only the specific elements). Default: 'region'.

        Returns:
            Self for method chaining

        Raises:
            TypeError: If a non-callable, non-Region object without a valid bbox is provided.
            ValueError: If method is not 'region' or 'element'.
        """
        # Validate method parameter
        if method not in ("region", "element"):
            raise ValueError(f"Invalid exclusion method '{method}'. Must be 'region' or 'element'.")

        # ------------------------------------------------------------------
        # NEW: Handle selector strings and ElementCollection instances
        # ------------------------------------------------------------------
        # If a user supplies a selector string (e.g. "text:bold") we resolve it
        # immediately *on this page* to the matching elements and turn each into
        # a Region object which is added to the internal exclusions list.
        #
        # Likewise, if an ElementCollection is passed we iterate over its
        # elements and create Regions for each one.
        # ------------------------------------------------------------------
        # Import ElementCollection from the new module path (old path removed)
        from natural_pdf.elements.element_collection import ElementCollection

        # Selector string ---------------------------------------------------
        if isinstance(exclusion_func_or_region, str):
            selector_str = exclusion_func_or_region
            matching_elements = self.find_all(selector_str, apply_exclusions=False)

            if not matching_elements:
                logger.warning(
                    f"Page {self.index}: Selector '{selector_str}' returned no elements – no exclusions added."
                )
            else:
                if method == "element":
                    # Store the actual elements for element-based exclusion
                    for el in matching_elements:
                        self._exclusions.append((el, label, method))
                        logger.debug(
                            f"Page {self.index}: Added element exclusion from selector '{selector_str}' -> {el}"
                        )
                else:  # method == "region"
                    for el in matching_elements:
                        try:
                            bbox_coords = (
                                float(el.x0),
                                float(el.top),
                                float(el.x1),
                                float(el.bottom),
                            )
                            region = Region(self, bbox_coords, label=label)
                            # Store directly as a Region tuple so we don't recurse endlessly
                            self._exclusions.append((region, label, method))
                            logger.debug(
                                f"Page {self.index}: Added exclusion region from selector '{selector_str}' -> {bbox_coords}"
                            )
                        except Exception as e:
                            # Re-raise so calling code/test sees the failure immediately
                            logger.error(
                                f"Page {self.index}: Failed to create exclusion region from element {el}: {e}",
                                exc_info=False,
                            )
                            raise
            return self  # Completed processing for selector input

        # ElementCollection -----------------------------------------------
        if isinstance(exclusion_func_or_region, ElementCollection):
            if method == "element":
                # Store the actual elements for element-based exclusion
                for el in exclusion_func_or_region:
                    self._exclusions.append((el, label, method))
                    logger.debug(
                        f"Page {self.index}: Added element exclusion from ElementCollection -> {el}"
                    )
            else:  # method == "region"
                # Convert each element to a Region and add
                for el in exclusion_func_or_region:
                    try:
                        if not (hasattr(el, "bbox") and len(el.bbox) == 4):
                            logger.warning(
                                f"Page {self.index}: Skipping element without bbox in ElementCollection exclusion: {el}"
                            )
                            continue
                        bbox_coords = tuple(float(v) for v in el.bbox)
                        region = Region(self, bbox_coords, label=label)
                        self._exclusions.append((region, label, method))
                        logger.debug(
                            f"Page {self.index}: Added exclusion region from ElementCollection element {bbox_coords}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Page {self.index}: Failed to convert ElementCollection element to Region: {e}",
                            exc_info=False,
                        )
                        raise
            return self  # Completed processing for ElementCollection input

        # ------------------------------------------------------------------
        # Existing logic (callable, Region, bbox-bearing objects)
        # ------------------------------------------------------------------
        exclusion_data = None  # Initialize exclusion data

        if callable(exclusion_func_or_region):
            # Store callable functions along with their label and method
            exclusion_data = (exclusion_func_or_region, label, method)
            logger.debug(
                f"Page {self.index}: Added callable exclusion '{label}' with method '{method}': {exclusion_func_or_region}"
            )
        elif isinstance(exclusion_func_or_region, Region):
            # Store Region objects directly, assigning the label
            exclusion_func_or_region.label = label  # Assign label
            exclusion_data = (
                exclusion_func_or_region,
                label,
                method,
            )  # Store as tuple for consistency
            logger.debug(
                f"Page {self.index}: Added Region exclusion '{label}' with method '{method}': {exclusion_func_or_region}"
            )
        elif (
            hasattr(exclusion_func_or_region, "bbox")
            and isinstance(getattr(exclusion_func_or_region, "bbox", None), (tuple, list))
            and len(exclusion_func_or_region.bbox) == 4
        ):
            if method == "element":
                # For element method, store the element directly
                exclusion_data = (exclusion_func_or_region, label, method)
                logger.debug(
                    f"Page {self.index}: Added element exclusion '{label}': {exclusion_func_or_region}"
                )
            else:  # method == "region"
                # Convert objects with a valid bbox to a Region before storing
                try:
                    bbox_coords = tuple(float(v) for v in exclusion_func_or_region.bbox)
                    # Pass the label to the Region constructor
                    region_to_add = Region(self, bbox_coords, label=label)
                    exclusion_data = (region_to_add, label, method)  # Store as tuple
                    logger.debug(
                        f"Page {self.index}: Added exclusion '{label}' with method '{method}' converted to Region from {type(exclusion_func_or_region)}: {region_to_add}"
                    )
                except (ValueError, TypeError, Exception) as e:
                    # Raise an error if conversion fails
                    raise TypeError(
                        f"Failed to convert exclusion object {exclusion_func_or_region} with bbox {getattr(exclusion_func_or_region, 'bbox', 'N/A')} to Region: {e}"
                    ) from e
        elif isinstance(exclusion_func_or_region, (list, tuple)):
            # Handle lists/tuples of regions or elements
            if not exclusion_func_or_region:
                logger.warning(f"Page {self.index}: Empty list provided for exclusion, ignoring.")
                return self

            if method == "element":
                # Store each element directly
                for item in exclusion_func_or_region:
                    if hasattr(item, "bbox") and len(getattr(item, "bbox", [])) == 4:
                        self._exclusions.append((item, label, method))
                        logger.debug(
                            f"Page {self.index}: Added element exclusion from list -> {item}"
                        )
                    else:
                        logger.warning(
                            f"Page {self.index}: Skipping item without valid bbox in list: {item}"
                        )
            else:  # method == "region"
                # Convert each item to a Region and add
                for item in exclusion_func_or_region:
                    try:
                        if isinstance(item, Region):
                            item.label = label
                            self._exclusions.append((item, label, method))
                            logger.debug(f"Page {self.index}: Added Region from list: {item}")
                        elif hasattr(item, "bbox") and len(getattr(item, "bbox", [])) == 4:
                            bbox_coords = tuple(float(v) for v in item.bbox)
                            region = Region(self, bbox_coords, label=label)
                            self._exclusions.append((region, label, method))
                            logger.debug(
                                f"Page {self.index}: Added exclusion region from list item {bbox_coords}"
                            )
                        else:
                            logger.warning(
                                f"Page {self.index}: Skipping item without valid bbox in list: {item}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Page {self.index}: Failed to convert list item to Region: {e}"
                        )
                        continue
            return self
        else:
            # Reject invalid types
            raise TypeError(
                f"Invalid exclusion type: {type(exclusion_func_or_region)}. Must be callable, Region, list/tuple of regions/elements, or have a valid .bbox attribute."
            )

        # Append the stored data (tuple of object/callable, label, and method)
        if exclusion_data:
            self._exclusions.append(exclusion_data)

        return self

    def add_region(self, region: "Region", name: Optional[str] = None) -> "Page":
        """
        Add a region to the page.

        Args:
            region: Region object to add
            name: Optional name for the region

        Returns:
            Self for method chaining
        """
        # Check if it's actually a Region object
        if not isinstance(region, Region):
            raise TypeError("region must be a Region object")

        # Set the source and name
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

    def add_regions(self, regions: List["Region"], prefix: Optional[str] = None) -> "Page":
        """
        Add multiple regions to the page.

        Args:
            regions: List of Region objects to add
            prefix: Optional prefix for automatic naming (regions will be named prefix_1, prefix_2, etc.)

        Returns:
            Self for method chaining
        """
        if prefix:
            # Add with automatic sequential naming
            for i, region in enumerate(regions):
                self.add_region(region, name=f"{prefix}_{i+1}")
        else:
            # Add without names
            for region in regions:
                self.add_region(region)

        return self

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

        if debug:
            print(f"\nPage {self.index}: Evaluating {len(self._exclusions)} exclusions")

        for i, exclusion_data in enumerate(self._exclusions):
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

                    # Temporarily clear exclusions (consider if really needed)
                    temp_original_exclusions = self._exclusions
                    self._exclusions = []

                    # Call the function - Expects it to return a Region or None
                    region_result = exclusion_item(self)

                    # Restore exclusions
                    self._exclusions = temp_original_exclusions

                    if isinstance(region_result, Region):
                        # Assign the label to the returned region
                        region_result.label = label
                        regions.append(region_result)
                        if debug:
                            print(f"    ✓ Added region from callable '{label}': {region_result}")
                    elif hasattr(region_result, "__iter__") and hasattr(region_result, "__len__"):
                        # Handle ElementCollection or other iterables
                        from natural_pdf.elements.element_collection import ElementCollection

                        if isinstance(region_result, ElementCollection) or (
                            hasattr(region_result, "__iter__") and region_result
                        ):
                            if debug:
                                print(
                                    f"    Converting {type(region_result)} with {len(region_result)} elements to regions..."
                                )

                            # Convert each element to a region
                            for elem in region_result:
                                try:
                                    if hasattr(elem, "bbox") and len(elem.bbox) == 4:
                                        bbox_coords = tuple(float(v) for v in elem.bbox)
                                        region = Region(self, bbox_coords, label=label)
                                        regions.append(region)
                                        if debug:
                                            print(
                                                f"      ✓ Added region from element: {bbox_coords}"
                                            )
                                    else:
                                        if debug:
                                            print(
                                                f"      ✗ Skipping element without valid bbox: {elem}"
                                            )
                                except Exception as e:
                                    if debug:
                                        print(f"      ✗ Failed to convert element to region: {e}")
                                    continue

                            if debug and len(region_result) > 0:
                                print(
                                    f"    ✓ Converted {len(region_result)} elements from callable '{label}'"
                                )
                        else:
                            if debug:
                                print(f"    ✗ Empty iterable returned from callable '{label}'")
                    elif region_result:
                        logger.warning(
                            f"Callable exclusion '{exclusion_label}' returned non-Region object: {type(region_result)}. Skipping."
                        )
                        if debug:
                            print(f"    ✗ Callable returned non-Region/None: {type(region_result)}")
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
        if not self._exclusions:
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
        excluded_elements = set()  # Use set for O(1) lookup

        for exclusion_data in self._exclusions:
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

            # Handle element-based exclusions
            if method == "element" and hasattr(exclusion_item, "bbox"):
                excluded_elements.add(id(exclusion_item))
                if debug_exclusions:
                    print(f"  - Added element exclusion: {exclusion_item}")

        if debug_exclusions:
            print(
                f"Page {self.index}: Applying {len(exclusion_regions)} region exclusions "
                f"and {len(excluded_elements)} element exclusions to {len(elements)} elements."
            )

        filtered_elements = []
        region_excluded_count = 0
        element_excluded_count = 0

        for element in elements:
            exclude = False

            # Check element-based exclusions first (faster)
            if id(element) in excluded_elements:
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
        selector: Optional[str] = None,  # Now optional
        *,  # Force subsequent args to be keyword-only
        text: Optional[str] = None,  # New text parameter
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> Optional[Any]:
        """
        Find first element on this page matching selector OR text content.

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
        if selector is not None and text is not None:
            raise ValueError("Provide either 'selector' or 'text', not both.")
        if selector is None and text is None:
            raise ValueError("Provide either 'selector' or 'text'.")

        # Construct selector if 'text' is provided
        effective_selector = ""
        if text is not None:
            # Escape quotes within the text for the selector string
            escaped_text = text.replace('"', '\\"').replace("'", "\\'")
            # Default to 'text:contains(...)'
            effective_selector = f'text:contains("{escaped_text}")'
            # Note: regex/case handled by kwargs passed down
            logger.debug(
                f"Using text shortcut: find(text='{text}') -> find('{effective_selector}')"
            )
        elif selector is not None:
            effective_selector = selector
        else:
            # Should be unreachable due to checks above
            raise ValueError("Internal error: No selector or text provided.")

        selector_obj = parse_selector(effective_selector)

        # Pass regex and case flags to selector function via kwargs
        kwargs["regex"] = regex
        kwargs["case"] = case

        # First get all matching elements without applying exclusions initially within _apply_selector
        results_collection = self._apply_selector(
            selector_obj, **kwargs
        )  # _apply_selector doesn't filter

        # Filter the results based on exclusions if requested
        if apply_exclusions and self._exclusions and results_collection:
            filtered_elements = self._filter_elements_by_exclusions(results_collection.elements)
            # Return the first element from the filtered list
            return filtered_elements[0] if filtered_elements else None
        elif results_collection:
            # Return the first element from the unfiltered results
            return results_collection.first
        else:
            return None

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
        *,  # Force subsequent args to be keyword-only
        text: Optional[str] = None,  # New text parameter
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> "ElementCollection":
        """
        Find all elements on this page matching selector OR text content.

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
        from natural_pdf.elements.element_collection import (  # Import here for type hint
            ElementCollection,
        )

        if selector is not None and text is not None:
            raise ValueError("Provide either 'selector' or 'text', not both.")
        if selector is None and text is None:
            raise ValueError("Provide either 'selector' or 'text'.")

        # Construct selector if 'text' is provided
        effective_selector = ""
        if text is not None:
            # Escape quotes within the text for the selector string
            escaped_text = text.replace('"', '\\"').replace("'", "\\'")
            # Default to 'text:contains(...)'
            effective_selector = f'text:contains("{escaped_text}")'
            logger.debug(
                f"Using text shortcut: find_all(text='{text}') -> find_all('{effective_selector}')"
            )
        elif selector is not None:
            effective_selector = selector
        else:
            # Should be unreachable due to checks above
            raise ValueError("Internal error: No selector or text provided.")

        selector_obj = parse_selector(effective_selector)

        # Pass regex and case flags to selector function via kwargs
        kwargs["regex"] = regex
        kwargs["case"] = case

        # First get all matching elements without applying exclusions initially within _apply_selector
        results_collection = self._apply_selector(
            selector_obj, **kwargs
        )  # _apply_selector doesn't filter

        # Filter the results based on exclusions if requested
        if apply_exclusions and self._exclusions and results_collection:
            filtered_elements = self._filter_elements_by_exclusions(results_collection.elements)
            return ElementCollection(filtered_elements)
        else:
            # Return the unfiltered collection
            return results_collection

    def _apply_selector(
        self, selector_obj: Dict, **kwargs
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
        from natural_pdf.selectors.parser import selector_to_filter_func

        # Handle compound OR selectors
        if selector_obj.get("type") == "or":
            # For OR selectors, search all elements and let the filter function decide
            elements_to_search = self._element_mgr.get_all_elements()

            # Create filter function from compound selector
            filter_func = selector_to_filter_func(selector_obj, **kwargs)

            # Apply the filter to all elements
            matching_elements = [element for element in elements_to_search if filter_func(element)]

            # Sort elements in reading order if requested
            if kwargs.get("reading_order", True):
                if all(hasattr(el, "top") and hasattr(el, "x0") for el in matching_elements):
                    matching_elements.sort(key=lambda el: (el.top, el.x0))
                else:
                    logger.warning(
                        "Cannot sort elements in reading order: Missing required attributes (top, x0)."
                    )

            # Handle collection-level pseudo-classes (:first, :last) for OR selectors
            # Note: We only apply :first/:last if they appear in any of the sub-selectors
            has_first = False
            has_last = False
            for sub_selector in selector_obj.get("selectors", []):
                for pseudo in sub_selector.get("pseudo_classes", []):
                    if pseudo.get("name") == "first":
                        has_first = True
                    elif pseudo.get("name") == "last":
                        has_last = True

            if has_first:
                matching_elements = matching_elements[:1] if matching_elements else []
            elif has_last:
                matching_elements = matching_elements[-1:] if matching_elements else []

            # Return result collection
            return ElementCollection(matching_elements)

        # Handle single selectors (existing logic)
        # Get element type to filter
        element_type = selector_obj.get("type", "any").lower()

        # Determine which elements to search based on element type
        elements_to_search = []
        if element_type == "any":
            elements_to_search = self._element_mgr.get_all_elements()
        elif element_type == "text":
            elements_to_search = self._element_mgr.words
        elif element_type == "char":
            elements_to_search = self._element_mgr.chars
        elif element_type == "word":
            elements_to_search = self._element_mgr.words
        elif element_type == "rect" or element_type == "rectangle":
            elements_to_search = self._element_mgr.rects
        elif element_type == "line":
            elements_to_search = self._element_mgr.lines
        elif element_type == "region":
            elements_to_search = self._element_mgr.regions
        else:
            elements_to_search = self._element_mgr.get_all_elements()

        # Create filter function from selector, passing any additional parameters
        filter_func = selector_to_filter_func(selector_obj, **kwargs)

        # Apply the filter to matching elements
        matching_elements = [element for element in elements_to_search if filter_func(element)]

        # Handle spatial pseudo-classes that require relationship checking
        for pseudo in selector_obj.get("pseudo_classes", []):
            name = pseudo.get("name")
            args = pseudo.get("args", "")

            if name in ("above", "below", "near", "left-of", "right-of"):
                # Find the reference element first
                from natural_pdf.selectors.parser import parse_selector

                ref_selector = parse_selector(args) if isinstance(args, str) else args
                # Recursively call _apply_selector for reference element (exclusions handled later)
                ref_elements = self._apply_selector(ref_selector, **kwargs)

                if not ref_elements:
                    return ElementCollection([])

                ref_element = ref_elements.first
                if not ref_element:
                    continue

                # Filter elements based on spatial relationship
                if name == "above":
                    matching_elements = [
                        el
                        for el in matching_elements
                        if hasattr(el, "bottom")
                        and hasattr(ref_element, "top")
                        and el.bottom <= ref_element.top
                    ]
                elif name == "below":
                    matching_elements = [
                        el
                        for el in matching_elements
                        if hasattr(el, "top")
                        and hasattr(ref_element, "bottom")
                        and el.top >= ref_element.bottom
                    ]
                elif name == "left-of":
                    matching_elements = [
                        el
                        for el in matching_elements
                        if hasattr(el, "x1")
                        and hasattr(ref_element, "x0")
                        and el.x1 <= ref_element.x0
                    ]
                elif name == "right-of":
                    matching_elements = [
                        el
                        for el in matching_elements
                        if hasattr(el, "x0")
                        and hasattr(ref_element, "x1")
                        and el.x0 >= ref_element.x1
                    ]
                elif name == "near":

                    def distance(el1, el2):
                        if not (
                            hasattr(el1, "x0")
                            and hasattr(el1, "x1")
                            and hasattr(el1, "top")
                            and hasattr(el1, "bottom")
                            and hasattr(el2, "x0")
                            and hasattr(el2, "x1")
                            and hasattr(el2, "top")
                            and hasattr(el2, "bottom")
                        ):
                            return float("inf")  # Cannot calculate distance
                        el1_center_x = (el1.x0 + el1.x1) / 2
                        el1_center_y = (el1.top + el1.bottom) / 2
                        el2_center_x = (el2.x0 + el2.x1) / 2
                        el2_center_y = (el2.top + el2.bottom) / 2
                        return (
                            (el1_center_x - el2_center_x) ** 2 + (el1_center_y - el2_center_y) ** 2
                        ) ** 0.5

                    threshold = kwargs.get("near_threshold", 50)
                    matching_elements = [
                        el for el in matching_elements if distance(el, ref_element) <= threshold
                    ]

        # Sort elements in reading order if requested
        if kwargs.get("reading_order", True):
            if all(hasattr(el, "top") and hasattr(el, "x0") for el in matching_elements):
                matching_elements.sort(key=lambda el: (el.top, el.x0))
            else:
                logger.warning(
                    "Cannot sort elements in reading order: Missing required attributes (top, x0)."
                )

        # Handle collection-level pseudo-classes (:first, :last)
        for pseudo in selector_obj.get("pseudo_classes", []):
            name = pseudo.get("name")

            if name == "first":
                matching_elements = matching_elements[:1] if matching_elements else []
            elif name == "last":
                matching_elements = matching_elements[-1:] if matching_elements else []

        # Create result collection - exclusions are handled by the calling methods (find, find_all)
        result = ElementCollection(matching_elements)

        return result

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
        left: float = None,
        top: float = None,
        right: float = None,
        bottom: float = None,
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
        # ------------------------------------------------------------------
        # Percentage support – convert strings like "30%" to absolute values
        # based on page dimensions.  X-axis params (left, right, width) use
        # page.width; Y-axis params (top, bottom, height) use page.height.
        # ------------------------------------------------------------------

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

        # --- Type checking and basic validation ---
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

        # --- Calculate Coordinates ---
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

        # --- Apply Defaults for Unset Coordinates ---
        # Only default coordinates if they weren't set by dimension calculation
        if final_top is None:
            final_top = 0
        if final_bottom is None:
            # Check if bottom should have been set by height calc
            if height is None or top is None:
                final_bottom = self.height

        if final_left is None:
            final_left = 0
        if final_right is None:
            # Check if right should have been set by width calc
            if not is_width_numeric or left is None:
                final_right = self.width

        # --- Handle width_mode == 'full' ---
        if width_mode == "full":
            # Override left/right if mode is full
            final_left = 0
            final_right = self.width

        # --- Final Validation & Creation ---
        # Ensure coordinates are within page bounds (clamp)
        final_left = max(0, final_left)
        final_top = max(0, final_top)
        final_right = min(self.width, final_right)
        final_bottom = min(self.height, final_bottom)

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
        all_elements = self._element_mgr.get_all_elements()

        # Apply exclusions if requested
        if apply_exclusions and self._exclusions:
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

    def until(self, selector: str, include_endpoint: bool = True, **kwargs) -> Any:
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
        target = self.find(selector, **kwargs)
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

        region.end_element = target
        return region

    def crop(self, bbox=None, **kwargs) -> Any:
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
        return self._page.crop(bbox, **kwargs)

    def extract_text(
        self,
        preserve_whitespace=True,
        use_exclusions=True,
        debug_exclusions=False,
        content_filter=None,
        **kwargs,
    ) -> str:
        """
        Extract text from this page, respecting exclusions and using pdfplumber's
        layout engine (chars_to_textmap) if layout arguments are provided or default.

        Args:
            use_exclusions: Whether to apply exclusion regions (default: True).
                          Note: Filtering logic is now always applied if exclusions exist.
            debug_exclusions: Whether to output detailed exclusion debugging info (default: False).
            content_filter: Optional content filter to exclude specific text patterns. Can be:
                - A regex pattern string (characters matching the pattern are EXCLUDED)
                - A callable that takes text and returns True to KEEP the character
                - A list of regex patterns (characters matching ANY pattern are EXCLUDED)
            **kwargs: Additional layout parameters passed directly to pdfplumber's
                      `chars_to_textmap` function. Common parameters include:
                      - layout (bool): If True (default), inserts spaces/newlines.
                      - x_density (float): Pixels per character horizontally.
                      - y_density (float): Pixels per line vertically.
                      - x_tolerance (float): Tolerance for horizontal character grouping.
                      - y_tolerance (float): Tolerance for vertical character grouping.
                      - line_dir (str): 'ttb', 'btt', 'ltr', 'rtl'
                      - char_dir (str): 'ttb', 'btt', 'ltr', 'rtl'
                      See pdfplumber documentation for more.

        Returns:
            Extracted text as string, potentially with layout-based spacing.
        """
        logger.debug(f"Page {self.number}: extract_text called with kwargs: {kwargs}")
        debug = kwargs.get("debug", debug_exclusions)  # Allow 'debug' kwarg

        # 1. Get Word Elements (triggers load_elements if needed)
        word_elements = self.words
        if not word_elements:
            logger.debug(f"Page {self.number}: No word elements found.")
            return ""

        # 2. Apply element-based exclusions if enabled
        if use_exclusions and self._exclusions:
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
        apply_exclusions_flag = kwargs.get("use_exclusions", use_exclusions)
        exclusion_regions = []
        if apply_exclusions_flag and self._exclusions:
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
        merged_kwargs = dict(kwargs)
        tol_keys = ["x_tolerance", "x_tolerance_ratio", "y_tolerance"]
        for k in tol_keys:
            if k not in merged_kwargs:
                if k in self._config:
                    merged_kwargs[k] = self._config[k]
                elif k in getattr(self._parent, "_config", {}):
                    merged_kwargs[k] = self._parent._config[k]

        # Add content_filter to kwargs if provided
        if content_filter is not None:
            merged_kwargs["content_filter"] = content_filter

        result = generate_text_layout(
            char_dicts=filtered_chars,
            layout_context_bbox=page_bbox,
            user_kwargs=merged_kwargs,
        )

        # --- Optional: apply Unicode BiDi algorithm for mixed RTL/LTR correctness ---
        apply_bidi = kwargs.get("bidi", True)
        if apply_bidi and result:
            # Quick check for any RTL character
            import unicodedata

            def _contains_rtl(s):
                return any(unicodedata.bidirectional(ch) in ("R", "AL", "AN") for ch in s)

            if _contains_rtl(result):
                try:
                    from bidi.algorithm import get_display  # type: ignore

                    from natural_pdf.utils.bidi_mirror import mirror_brackets

                    result = "\n".join(
                        mirror_brackets(
                            get_display(
                                line,
                                base_dir=(
                                    "R"
                                    if any(
                                        unicodedata.bidirectional(ch) in ("R", "AL", "AN")
                                        for ch in line
                                    )
                                    else "L"
                                ),
                            )
                        )
                        for line in result.split("\n")
                    )
                except ModuleNotFoundError:
                    pass  # silently skip if python-bidi not available

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
    ) -> List[List[Optional[str]]]:
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

        Returns:
            Table data as a list of rows, where each row is a list of cell values (str or None).
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
        )

    def extract_tables(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        check_tatr: bool = True,
    ) -> List[List[List[str]]]:
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
        if table_settings is None:
            table_settings = {}

        # Check for TATR-detected table regions first if enabled
        if check_tatr:
            try:
                tatr_tables = self.find_all("region[type=table][model=tatr]")
                if tatr_tables:
                    logger.debug(
                        f"Page {self.number}: Found {len(tatr_tables)} TATR table regions, extracting from those..."
                    )
                    extracted_tables = []
                    for table_region in tatr_tables:
                        try:
                            table_data = table_region.extract_table(method="tatr")
                            if table_data:  # Only add non-empty tables
                                extracted_tables.append(table_data)
                        except Exception as e:
                            logger.warning(
                                f"Failed to extract table from TATR region {table_region.bbox}: {e}"
                            )

                    if extracted_tables:
                        logger.debug(
                            f"Page {self.number}: Successfully extracted {len(extracted_tables)} tables from TATR regions"
                        )
                        return extracted_tables
                    else:
                        logger.debug(
                            f"Page {self.number}: TATR regions found but no tables extracted, falling back to pdfplumber"
                        )
                else:
                    logger.debug(
                        f"Page {self.number}: No TATR table regions found, using pdfplumber methods"
                    )
            except Exception as e:
                logger.debug(
                    f"Page {self.number}: Error checking TATR regions: {e}, falling back to pdfplumber"
                )

        # Auto-detect method if not specified (try lattice first, then stream)
        if method is None:
            logger.debug(f"Page {self.number}: Auto-detecting tables extraction method...")

            # Try lattice first
            try:
                lattice_settings = table_settings.copy()
                lattice_settings.setdefault("vertical_strategy", "lines")
                lattice_settings.setdefault("horizontal_strategy", "lines")

                logger.debug(f"Page {self.number}: Trying 'lattice' method first for tables...")
                lattice_result = self._page.extract_tables(lattice_settings)

                # Check if lattice found meaningful tables
                if (
                    lattice_result
                    and len(lattice_result) > 0
                    and any(
                        any(
                            any(cell and cell.strip() for cell in row if cell)
                            for row in table
                            if table
                        )
                        for table in lattice_result
                    )
                ):
                    logger.debug(
                        f"Page {self.number}: 'lattice' method found {len(lattice_result)} tables"
                    )
                    return lattice_result
                else:
                    logger.debug(f"Page {self.number}: 'lattice' method found no meaningful tables")

            except Exception as e:
                logger.debug(f"Page {self.number}: 'lattice' method failed: {e}")

            # Fall back to stream
            logger.debug(f"Page {self.number}: Falling back to 'stream' method for tables...")
            stream_settings = table_settings.copy()
            stream_settings.setdefault("vertical_strategy", "text")
            stream_settings.setdefault("horizontal_strategy", "text")

            return self._page.extract_tables(stream_settings)

        effective_method = method

        # Handle method aliases
        if effective_method == "stream":
            logger.debug("Using 'stream' method alias for 'pdfplumber' with text-based strategies.")
            effective_method = "pdfplumber"
            table_settings.setdefault("vertical_strategy", "text")
            table_settings.setdefault("horizontal_strategy", "text")
        elif effective_method == "lattice":
            logger.debug(
                "Using 'lattice' method alias for 'pdfplumber' with line-based strategies."
            )
            effective_method = "pdfplumber"
            table_settings.setdefault("vertical_strategy", "lines")
            table_settings.setdefault("horizontal_strategy", "lines")

        # Use the selected method
        if effective_method == "pdfplumber":
            # ---------------------------------------------------------
            # Inject auto-computed or user-specified text tolerances so
            # pdfplumber uses the same numbers we used for word grouping
            # whenever the table algorithm relies on word positions.
            # ---------------------------------------------------------
            if "text" in (
                table_settings.get("vertical_strategy"),
                table_settings.get("horizontal_strategy"),
            ):
                print("SETTING IT UP")
                pdf_cfg = getattr(self, "_config", getattr(self._parent, "_config", {}))
                if "text_x_tolerance" not in table_settings and "x_tolerance" not in table_settings:
                    x_tol = pdf_cfg.get("x_tolerance")
                    if x_tol is not None:
                        table_settings.setdefault("text_x_tolerance", x_tol)
                if "text_y_tolerance" not in table_settings and "y_tolerance" not in table_settings:
                    y_tol = pdf_cfg.get("y_tolerance")
                    if y_tol is not None:
                        table_settings.setdefault("text_y_tolerance", y_tol)

                # pdfplumber's text strategy benefits from a tight snap tolerance.
                if (
                    "snap_tolerance" not in table_settings
                    and "snap_x_tolerance" not in table_settings
                ):
                    # Derive from y_tol if available, else default 1
                    snap = max(1, round((pdf_cfg.get("y_tolerance", 1)) * 0.9))
                    table_settings.setdefault("snap_tolerance", snap)
                if (
                    "join_tolerance" not in table_settings
                    and "join_x_tolerance" not in table_settings
                ):
                    join = table_settings.get("snap_tolerance", 1)
                    table_settings.setdefault("join_tolerance", join)
                    table_settings.setdefault("join_x_tolerance", join)
                    table_settings.setdefault("join_y_tolerance", join)

            raw_tables = self._page.extract_tables(table_settings)

            # Apply RTL text processing to all extracted tables
            if raw_tables:
                processed_tables = []
                for table in raw_tables:
                    processed_table = []
                    for row in table:
                        processed_row = []
                        for cell in row:
                            if cell is not None:
                                # Apply RTL text processing to each cell
                                rtl_processed_cell = self._apply_rtl_processing_to_text(cell)
                                processed_row.append(rtl_processed_cell)
                            else:
                                processed_row.append(cell)
                        processed_table.append(processed_row)
                    processed_tables.append(processed_table)
                return processed_tables

            return raw_tables
        else:
            raise ValueError(
                f"Unknown tables extraction method: '{method}'. Choose from 'pdfplumber', 'stream', 'lattice'."
            )

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
        # Create analyzer (optionally pass default options from PDF config here)
        # For now, it uses its own defaults if options=None
        analyzer = TextStyleAnalyzer()

        # Analyze the page. The analyzer now modifies elements directly
        # and returns the collection of processed elements.
        processed_elements_collection = analyzer.analyze(self, options=options)

        # Return the collection of elements which now have style attributes
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
        """
        Apply OCR to THIS page and add results to page elements via PDF.apply_ocr.

        Args:
            engine: Name of the OCR engine.
            options: Engine-specific options object or dict.
            languages: List of engine-specific language codes.
            min_confidence: Minimum confidence threshold.
            device: Device to run OCR on.
            resolution: DPI resolution for rendering page image before OCR.
            apply_exclusions: If True (default), render page image for OCR
                              with excluded areas masked (whited out).
            detect_only: If True, only detect text bounding boxes, don't perform OCR.
            replace: If True (default), remove any existing OCR elements before
                    adding new ones. If False, add new OCR elements to existing ones.

        Returns:
            Self for method chaining.
        """
        if not hasattr(self._parent, "apply_ocr"):
            logger.error(f"Page {self.number}: Parent PDF missing 'apply_ocr'. Cannot apply OCR.")
            return self  # Return self for chaining

        # Remove existing OCR elements if replace is True
        if replace and hasattr(self, "_element_mgr"):
            logger.info(
                f"Page {self.number}: Removing existing OCR elements before applying new OCR."
            )
            self._element_mgr.remove_ocr_elements()

        logger.info(f"Page {self.number}: Delegating apply_ocr to PDF.apply_ocr.")
        # Delegate to parent PDF, targeting only this page's index
        # Pass all relevant parameters through, including apply_exclusions
        self._parent.apply_ocr(
            pages=[self.index],
            engine=engine,
            options=options,
            languages=languages,
            min_confidence=min_confidence,
            device=device,
            resolution=resolution,
            detect_only=detect_only,
            apply_exclusions=apply_exclusions,
            replace=replace,  # Pass the replace parameter to PDF.apply_ocr
        )

        # Return self for chaining
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
        Extract text elements using OCR *without* adding them to the page's elements.
        Uses the shared OCRManager instance.

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
        if not self._ocr_manager:
            logger.error(
                f"Page {self.number}: OCRManager not available. Cannot extract OCR elements."
            )
            return []

        logger.info(f"Page {self.number}: Extracting OCR elements (extract only)...")

        # Determine rendering resolution
        final_resolution = resolution if resolution is not None else 150  # Default to 150 DPI
        logger.debug(f"  Using rendering resolution: {final_resolution} DPI")

        try:
            # Get base image without highlights using the determined resolution
            # Use the global PDF rendering lock
            with pdf_render_lock:
                # Use render() for clean image without highlights
                image = self.render(resolution=final_resolution)
                if not image:
                    logger.error(
                        f"  Failed to render page {self.number} to image for OCR extraction."
                    )
                    return []
                logger.debug(f"  Rendered image size: {image.width}x{image.height}")
        except Exception as e:
            logger.error(f"  Failed to render page {self.number} to image: {e}", exc_info=True)
            return []

        # Prepare arguments for the OCR Manager call
        manager_args = {
            "images": image,
            "engine": engine,
            "languages": languages,
            "min_confidence": min_confidence,
            "device": device,
            "options": options,
        }
        manager_args = {k: v for k, v in manager_args.items() if v is not None}

        logger.debug(
            f"  Calling OCR Manager (extract only) with args: { {k:v for k,v in manager_args.items() if k != 'images'} }"
        )
        try:
            # apply_ocr now returns List[List[Dict]] or List[Dict]
            results_list = self._ocr_manager.apply_ocr(**manager_args)
            # If it returned a list of lists (batch mode), take the first list
            results = (
                results_list[0]
                if isinstance(results_list, list)
                and results_list
                and isinstance(results_list[0], list)
                else results_list
            )
            if not isinstance(results, list):
                logger.error(f"  OCR Manager returned unexpected type: {type(results)}")
                results = []
            logger.info(f"  OCR Manager returned {len(results)} results for extraction.")
        except Exception as e:
            logger.error(f"  OCR processing failed during extraction: {e}", exc_info=True)
            return []

        # Convert results but DO NOT add to ElementManager
        logger.debug(f"  Converting OCR results to TextElements (extract only)...")
        temp_elements = []
        scale_x = self.width / image.width if image.width else 1
        scale_y = self.height / image.height if image.height else 1
        for result in results:
            try:  # Added try-except around result processing
                x0, top, x1, bottom = [float(c) for c in result["bbox"]]
                elem_data = {
                    "text": result["text"],
                    "confidence": result["confidence"],
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
            except (KeyError, ValueError, TypeError) as convert_err:
                logger.warning(
                    f"  Skipping invalid OCR result during conversion: {result}. Error: {convert_err}"
                )

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
            if not self._layout_manager:
                logger.warning("LayoutManager not available, cannot create LayoutAnalyzer.")
                return None
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
        Analyze the page layout using the configured LayoutManager.
        Adds detected Region objects to the page's element manager.

        Returns:
            ElementCollection containing the detected Region objects.
        """
        analyzer = self.layout_analyzer
        if not analyzer:
            logger.error(
                "Layout analysis failed: LayoutAnalyzer not initialized (is LayoutManager available?)."
            )
            return ElementCollection([])  # Return empty collection

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
        if (
            not hasattr(self._element_mgr, "regions")
            or not hasattr(self._element_mgr, "_elements")
            or "regions" not in self._element_mgr._elements
        ):
            logger.debug(
                f"Page {self.index}: No regions found in ElementManager, nothing to clear."
            )
            self._regions["detected"] = []  # Ensure page's list is also clear
            return self

        # Filter ElementManager's list to keep only non-detected regions
        original_count = len(self._element_mgr.regions)
        self._element_mgr._elements["regions"] = [
            r for r in self._element_mgr.regions if getattr(r, "source", None) != "detected"
        ]
        new_count = len(self._element_mgr.regions)
        removed_count = original_count - new_count

        # Clear the page's specific list of detected regions
        self._regions["detected"] = []

        logger.info(f"Page {self.index}: Cleared {removed_count} detected layout regions.")
        return self

    def get_section_between(
        self, start_element=None, end_element=None, include_boundaries="both"
    ) -> Optional["Region"]:  # Return Optional
        """
        Get a section between two elements on this page.
        """
        # Create a full-page region to operate within
        page_region = self.create_region(0, 0, self.width, self.height)

        # Delegate to the region's method
        try:
            return page_region.get_section_between(
                start_element=start_element,
                end_element=end_element,
                include_boundaries=include_boundaries,
            )
        except Exception as e:
            logger.error(
                f"Error getting section between elements on page {self.index}: {e}", exc_info=True
            )
            return None

    def split(self, divider, **kwargs) -> "ElementCollection[Region]":
        """
        Divides the page into sections based on the provided divider elements.
        """
        sections = self.get_sections(start_elements=divider, **kwargs)
        top = self.region(0, 0, self.width, sections[0].top)
        sections.append(top)

        return sections

    def get_sections(
        self,
        start_elements=None,
        end_elements=None,
        include_boundaries="start",
        y_threshold=5.0,
        bounding_box=None,
    ) -> "ElementCollection[Region]":
        """
        Get sections of a page defined by start/end elements.
        Uses the page-level implementation.

        Returns:
            An ElementCollection containing the found Region objects.
        """

        # Helper function to get bounds from bounding_box parameter
        def get_bounds():
            if bounding_box:
                x0, top, x1, bottom = bounding_box
                # Clamp to page boundaries
                return max(0, x0), max(0, top), min(self.width, x1), min(self.height, bottom)
            else:
                return 0, 0, self.width, self.height

        regions = []

        # Handle cases where elements are provided as strings (selectors)
        if isinstance(start_elements, str):
            start_elements = self.find_all(start_elements).elements  # Get list of elements
        elif hasattr(start_elements, "elements"):  # Handle ElementCollection input
            start_elements = start_elements.elements

        if isinstance(end_elements, str):
            end_elements = self.find_all(end_elements).elements
        elif hasattr(end_elements, "elements"):
            end_elements = end_elements.elements

        # Ensure start_elements is a list
        if start_elements is None:
            start_elements = []
        if end_elements is None:
            end_elements = []

        valid_inclusions = ["start", "end", "both", "none"]
        if include_boundaries not in valid_inclusions:
            raise ValueError(f"include_boundaries must be one of {valid_inclusions}")

        if not start_elements:
            # Return an empty ElementCollection if no start elements
            return ElementCollection([])

        # Combine start and end elements with their type
        all_boundaries = []
        for el in start_elements:
            all_boundaries.append((el, "start"))
        for el in end_elements:
            all_boundaries.append((el, "end"))

        # Sort all boundary elements primarily by top, then x0
        try:
            all_boundaries.sort(key=lambda x: (x[0].top, x[0].x0))
        except AttributeError as e:
            logger.error(f"Error sorting boundaries: Element missing top/x0 attribute? {e}")
            return ElementCollection([])  # Cannot proceed if elements lack position

        # Process sorted boundaries to find sections
        current_start_element = None
        active_section_started = False

        for element, element_type in all_boundaries:
            if element_type == "start":
                # If we have an active section, this start implicitly ends it
                if active_section_started:
                    end_boundary_el = element  # Use this start as the end boundary
                    # Determine region boundaries
                    sec_top = (
                        current_start_element.top
                        if include_boundaries in ["start", "both"]
                        else current_start_element.bottom
                    )
                    sec_bottom = (
                        end_boundary_el.top
                        if include_boundaries not in ["end", "both"]
                        else end_boundary_el.bottom
                    )

                    if sec_top < sec_bottom:  # Ensure valid region
                        x0, _, x1, _ = get_bounds()
                        region = self.create_region(x0, sec_top, x1, sec_bottom)
                        region.start_element = current_start_element
                        region.end_element = end_boundary_el  # Mark the element that ended it
                        region.is_end_next_start = True  # Mark how it ended
                        regions.append(region)
                    active_section_started = False  # Reset for the new start

                # Set this as the potential start of the next section
                current_start_element = element
                active_section_started = True

            elif element_type == "end" and active_section_started:
                # We found an explicit end for the current section
                end_boundary_el = element
                sec_top = (
                    current_start_element.top
                    if include_boundaries in ["start", "both"]
                    else current_start_element.bottom
                )
                sec_bottom = (
                    end_boundary_el.bottom
                    if include_boundaries in ["end", "both"]
                    else end_boundary_el.top
                )

                if sec_top < sec_bottom:  # Ensure valid region
                    x0, _, x1, _ = get_bounds()
                    region = self.create_region(x0, sec_top, x1, sec_bottom)
                    region.start_element = current_start_element
                    region.end_element = end_boundary_el
                    region.is_end_next_start = False
                    regions.append(region)

                # Reset: section ended explicitly
                current_start_element = None
                active_section_started = False

        # Handle the last section if it was started but never explicitly ended
        if active_section_started:
            sec_top = (
                current_start_element.top
                if include_boundaries in ["start", "both"]
                else current_start_element.bottom
            )
            x0, _, x1, page_bottom = get_bounds()
            if sec_top < page_bottom:
                region = self.create_region(x0, sec_top, x1, page_bottom)
                region.start_element = current_start_element
                region.end_element = None  # Ended by page end
                region.is_end_next_start = False
                regions.append(region)

        return ElementCollection(regions)

    def __repr__(self) -> str:
        """String representation of the page."""
        return f"<Page number={self.number} index={self.index}>"

    def ask(
        self,
        question: Union[str, List[str], Tuple[str, ...]],
        min_confidence: float = 0.1,
        model: str = None,
        debug: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Ask a question about the page content using document QA.
        """
        try:
            from natural_pdf.qa.document_qa import get_qa_engine

            # Get or initialize QA engine with specified model
            qa_engine = get_qa_engine(model_name=model) if model else get_qa_engine()
            # Ask the question using the QA engine
            return qa_engine.ask_pdf_page(
                self, question, min_confidence=min_confidence, debug=debug, **kwargs
            )
        except ImportError:
            logger.error(
                "Question answering requires the 'natural_pdf.qa' module. Please install necessary dependencies."
            )
            return {
                "answer": None,
                "confidence": 0.0,
                "found": False,
                "page_num": self.number,
                "source_elements": [],
            }
        except Exception as e:
            logger.error(f"Error during page.ask: {e}", exc_info=True)
            return {
                "answer": None,
                "confidence": 0.0,
                "found": False,
                "page_num": self.number,
                "source_elements": [],
            }

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
            logger.error(f"HighlightingService does not have the required 'render_preview' method.")
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
        # Check if the summary attribute exists from a previous run
        if not hasattr(self, "_text_styles_summary") or not self._text_styles_summary:
            # If not, run the analysis with default options
            logger.debug(f"Page {self.number}: Running default text style analysis to get labels.")
            self.analyze_text_styles()  # Use default options

        # Extract labels from the summary dictionary
        if hasattr(self, "_text_styles_summary") and self._text_styles_summary:
            # The summary maps style_key -> {'label': ..., 'properties': ...}
            labels = {style_info["label"] for style_info in self._text_styles_summary.values()}
            return sorted(list(labels))
        else:
            # Fallback if summary wasn't created for some reason (e.g., no text elements)
            logger.warning(f"Page {self.number}: Text style summary not found after analysis.")
            return []

    def viewer(
        self,
        # elements_to_render: Optional[List['Element']] = None, # No longer needed, from_page handles it
        # include_source_types: List[str] = ['word', 'line', 'rect', 'region'] # No longer needed
    ) -> Optional["InteractiveViewerWidget"]:  # Return type hint updated
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

    # --- Indexable Protocol Methods ---
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

    # --- New Method: save_searchable ---
    def save_searchable(self, output_path: Union[str, "Path"], dpi: int = 300, **kwargs):
        """
        Saves the PDF page with an OCR text layer, making content searchable.

        Requires optional dependencies. Install with: pip install "natural-pdf[ocr-save]"

        Note: OCR must have been applied to the pages beforehand
              (e.g., pdf.apply_ocr()).

        Args:
            output_path: Path to save the searchable PDF.
            dpi: Resolution for rendering and OCR overlay (default 300).
            **kwargs: Additional keyword arguments passed to the exporter.
        """
        # Import moved here, assuming it's always available now
        from natural_pdf.exporters.searchable_pdf import create_searchable_pdf

        # Convert pathlib.Path to string if necessary
        output_path_str = str(output_path)

        create_searchable_pdf(self, output_path_str, dpi=dpi, **kwargs)
        logger.info(f"Searchable PDF saved to: {output_path_str}")

    # --- Added correct_ocr method ---
    def update_text(
        self,
        transform: Callable[[Any], Optional[str]],
        selector: str = "text",
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
            max_workers: The maximum number of threads to use for parallel execution.
                         If None or 0 or 1, runs sequentially.
            progress_callback: Optional callback function to call after processing each element.

        Returns:
            Self for method chaining.
        """
        logger.info(
            f"Page {self.number}: Starting text update with callback '{transform.__name__}' (max_workers={max_workers}) and selector='{selector}'"
        )

        target_elements_collection = self.find_all(selector=selector, apply_exclusions=False)
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
                    current_text = getattr(element, "text", None)
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
                    # --- Update internal tqdm progress bar ---
                    if element_pbar:
                        element_pbar.update(1)
                    # --- Call user's progress callback --- #
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
                # --- Parallel execution --- #
                logger.info(
                    f"Page {self.number}: Running text update in parallel with {max_workers} workers."
                )
                futures = []
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
                # --- Sequential execution --- #
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

    # --- Classification Mixin Implementation --- #
    def _get_classification_manager(self) -> "ClassificationManager":
        if not hasattr(self, "pdf") or not hasattr(self.pdf, "get_manager"):
            raise AttributeError(
                "ClassificationManager cannot be accessed: Parent PDF or get_manager method missing."
            )
        try:
            # Use the PDF's manager registry accessor
            return self.pdf.get_manager("classification")
        except (ValueError, RuntimeError, AttributeError) as e:
            # Wrap potential errors from get_manager for clarity
            raise AttributeError(f"Failed to get ClassificationManager from PDF: {e}") from e

    def _get_classification_content(
        self, model_type: str, **kwargs
    ) -> Union[str, "Image"]:  # Use "Image" for lazy import
        if model_type == "text":
            text_content = self.extract_text(
                layout=False, use_exclusions=False
            )  # Simple join, ignore exclusions for classification
            if not text_content or text_content.isspace():
                raise ValueError("Cannot classify page with 'text' model: No text content found.")
            return text_content
        elif model_type == "vision":
            # Get resolution from manager/kwargs if possible, else default
            manager = self._get_classification_manager()
            default_resolution = 150
            # Access kwargs passed to classify method if needed
            resolution = (
                kwargs.get("resolution", default_resolution)
                if "kwargs" in locals()
                else default_resolution
            )

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

    # --- Content Extraction ---

    # --- Skew Detection and Correction --- #

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
        """
        Detects the skew angle of the page image and stores it.

        Args:
            resolution: DPI resolution for rendering the page image for detection.
            grayscale: Whether to convert the image to grayscale before detection.
            force_recalculate: If True, recalculate even if an angle exists.
            **deskew_kwargs: Additional keyword arguments passed to `deskew.determine_skew`
                             (e.g., `max_angle`, `num_peaks`).

        Returns:
            The detected skew angle in degrees, or None if detection failed.

        Raises:
            ImportError: If the 'deskew' library is not installed.
        """
        if not DESKEW_AVAILABLE:
            raise ImportError(
                "Deskew library not found. Install with: pip install natural-pdf[deskew]"
            )

        if self._skew_angle is not None and not force_recalculate:
            logger.debug(f"Page {self.number}: Returning cached skew angle: {self._skew_angle:.2f}")
            return self._skew_angle

        logger.debug(f"Page {self.number}: Detecting skew angle (resolution={resolution} DPI)...")
        try:
            # Render the page at the specified detection resolution
            # Use render() for clean image without highlights
            img = self.render(resolution=resolution)
            if not img:
                logger.warning(f"Page {self.number}: Failed to render image for skew detection.")
                self._skew_angle = None
                return None

            # Convert to numpy array
            img_np = np.array(img)

            # Convert to grayscale if needed
            if grayscale:
                if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
                    gray_np = np.mean(img_np[:, :, :3], axis=2).astype(np.uint8)
                elif len(img_np.shape) == 2:
                    gray_np = img_np  # Already grayscale
                else:
                    logger.warning(
                        f"Page {self.number}: Unexpected image shape {img_np.shape} for grayscale conversion."
                    )
                    gray_np = img_np  # Try using it anyway
            else:
                gray_np = img_np  # Use original if grayscale=False

            # Determine skew angle using the deskew library
            angle = determine_skew(gray_np, **deskew_kwargs)
            self._skew_angle = angle
            logger.debug(f"Page {self.number}: Detected skew angle = {angle}")
            return angle

        except Exception as e:
            logger.warning(f"Page {self.number}: Failed during skew detection: {e}", exc_info=True)
            self._skew_angle = None
            return None

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
        if not DESKEW_AVAILABLE:
            raise ImportError(
                "Deskew library not found. Install with: pip install natural-pdf[deskew]"
            )

        # Determine the angle to use
        rotation_angle = angle
        if rotation_angle is None:
            # Detect angle (or use cached) if not explicitly provided
            rotation_angle = self.detect_skew_angle(
                resolution=detection_resolution, **deskew_kwargs
            )

        logger.debug(
            f"Page {self.number}: Preparing to deskew (output resolution={resolution} DPI). Using angle: {rotation_angle}"
        )

        try:
            # Render the original page at the desired output resolution
            # Use render() for clean image without highlights
            img = self.render(resolution=resolution)
            if not img:
                logger.error(f"Page {self.number}: Failed to render image for deskewing.")
                return None

            # Rotate if a significant angle was found/provided
            if rotation_angle is not None and abs(rotation_angle) > 0.05:
                logger.debug(f"Page {self.number}: Rotating by {rotation_angle:.2f} degrees.")
                # Determine fill color based on image mode
                fill = (255, 255, 255) if img.mode == "RGB" else 255  # White background
                # Rotate the image using PIL
                rotated_img = img.rotate(
                    rotation_angle,  # deskew provides angle, PIL rotates counter-clockwise
                    resample=Image.Resampling.BILINEAR,
                    expand=True,  # Expand image to fit rotated content
                    fillcolor=fill,
                )
                return rotated_img
            else:
                logger.debug(
                    f"Page {self.number}: No significant rotation needed (angle={rotation_angle}). Returning original render."
                )
                return img  # Return the original rendered image if no rotation needed

        except Exception as e:
            logger.error(
                f"Page {self.number}: Error during deskewing image generation: {e}", exc_info=True
            )
            return None

    # --- End Skew Detection and Correction --- #

    # ------------------------------------------------------------------
    # Unified analysis storage (maps to metadata["analysis"])
    # ------------------------------------------------------------------

    @property
    def analyses(self) -> Dict[str, Any]:
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}
        return self.metadata.setdefault("analysis", {})

    @analyses.setter
    def analyses(self, value: Dict[str, Any]):
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

        # Remove all words and chars from the element manager
        removed_words = len(self._element_mgr.words)
        removed_chars = len(self._element_mgr.chars)

        # Clear the lists
        self._element_mgr._elements["words"] = []
        self._element_mgr._elements["chars"] = []

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
                    # Apply bracket mirroring for correct logical order
                    processed_lines.append(mirror_brackets(logical_line))
                else:
                    processed_lines.append(line)

            return "\n".join(processed_lines)

        except (ImportError, Exception):
            # If bidi library is not available or fails, return original text
            return text

    @property
    def lines(self) -> List[Any]:
        """Get all line elements on this page."""
        return self._element_mgr.lines

    # ------------------------------------------------------------------
    # Image elements
    # ------------------------------------------------------------------

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
