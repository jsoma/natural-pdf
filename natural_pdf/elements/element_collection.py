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

from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin
from natural_pdf.classification.manager import ClassificationManager
from natural_pdf.classification.mixin import ClassificationMixin
from natural_pdf.collections.mixins import ApplyMixin, DirectionalCollectionMixin
from natural_pdf.core.pdf import PDF

# Add Visualizable import
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.describe.mixin import DescribeMixin, InspectMixin
from natural_pdf.elements.base import Element
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


class ElementCollection(
    Generic[T],
    ApplyMixin,
    ExportMixin,
    ClassificationMixin,
    DirectionalCollectionMixin,
    DescribeMixin,
    InspectMixin,
    Visualizable,
    MutableSequence,
):
    """Collection of PDF elements with batch operations.

    ElementCollection provides a powerful interface for working with groups of
    PDF elements (text, rectangles, lines, etc.) with batch processing capabilities.
    It implements the MutableSequence protocol for list-like behavior while adding
    specialized functionality for document analysis workflows.

    The collection integrates multiple capabilities through mixins:
    - Batch processing with .apply() method
    - Export functionality for various formats
    - AI-powered classification of element groups
    - Spatial navigation for creating related regions
    - Description and inspection capabilities
    - Element filtering and selection

    Collections support functional programming patterns and method chaining,
    making it easy to build complex document processing pipelines.

    Attributes:
        elements: List of Element objects in the collection.
        first: First element in the collection (None if empty).
        last: Last element in the collection (None if empty).

    Example:
        Basic usage:
        ```python
        pdf = npdf.PDF("document.pdf")
        page = pdf.pages[0]

        # Get collections of elements
        all_text = page.chars
        headers = page.find_all('text[size>12]:bold')

        # Collection operations
        print(f"Found {len(headers)} headers")
        header_text = headers.get_text()

        # Batch processing
        results = headers.apply(lambda el: el.fontname)
        ```

        Advanced workflows:
        ```python
        # Functional programming style
        important_text = (page.chars
                         .filter('text:contains("IMPORTANT")')
                         .apply(lambda el: el.text.upper())
                         .classify("urgency_level"))

        # Spatial navigation from collections
        content_region = headers.below(until='rect[height>2]')

        # Export functionality
        headers.save_pdf("headers_only.pdf")
        ```

    Note:
        Collections are typically created by page methods (page.chars, page.find_all())
        or by filtering existing collections. Direct instantiation is less common.
    """

    def __init__(self, elements: List[T]):
        """Initialize a collection of elements.

        Creates an ElementCollection that wraps a list of PDF elements and provides
        enhanced functionality for batch operations, filtering, and analysis.

        Args:
            elements: List of Element objects (TextElement, RectangleElement, etc.)
                to include in the collection. Can be empty for an empty collection.

        Example:
            ```python
            # Collections are usually created by page methods
            chars = page.chars  # ElementCollection[TextElement]
            rects = page.rects  # ElementCollection[RectangleElement]

            # Direct creation (advanced usage)
            selected_elements = ElementCollection([element1, element2, element3])
            ```

        Note:
            ElementCollection implements MutableSequence, so it behaves like a list
            with additional natural-pdf functionality for document processing.
        """
        self._elements = elements or []

    def _get_render_specs(
        self,
        mode: Literal["show", "render"] = "show",
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        highlights: Optional[List[Dict[str, Any]]] = None,
        crop: Union[bool, Literal["content"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        group_by: Optional[str] = None,
        bins: Optional[Union[int, List[float]]] = None,
        annotate: Optional[List[str]] = None,
        **kwargs,
    ) -> List[RenderSpec]:
        """Get render specifications for this element collection.

        Args:
            mode: Rendering mode - 'show' includes highlights, 'render' is clean
            color: Default color for highlights in show mode (or colormap name when using group_by)
            highlights: Additional highlight groups to show
            crop: Whether to crop to element bounds
            crop_bbox: Explicit crop bounds
            group_by: Attribute to group elements by for color mapping
            bins: Binning specification for quantitative data (int for equal-width bins, list for custom bins)
            annotate: List of attribute names to display on highlights
            **kwargs: Additional parameters

        Returns:
            List of RenderSpec objects, one per page with elements
        """
        if not self._elements:
            return []

        # Group elements by page
        elements_by_page = {}
        for elem in self._elements:
            if hasattr(elem, "page"):
                page = elem.page
                if page not in elements_by_page:
                    elements_by_page[page] = []
                elements_by_page[page].append(elem)

        if not elements_by_page:
            return []

        # Create RenderSpec for each page
        specs = []
        for page, page_elements in elements_by_page.items():
            spec = RenderSpec(page=page)

            # Handle cropping
            if crop_bbox:
                spec.crop_bbox = crop_bbox
            elif crop == "content" or crop is True:
                # Calculate bounds of elements on this page
                x_coords = []
                y_coords = []
                for elem in page_elements:
                    if hasattr(elem, "bbox") and elem.bbox:
                        x0, y0, x1, y1 = elem.bbox
                        x_coords.extend([x0, x1])
                        y_coords.extend([y0, y1])

                if x_coords and y_coords:
                    spec.crop_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

            # Add highlights in show mode
            if mode == "show":
                # Handle group_by parameter for quantitative/categorical grouping
                if group_by is not None:
                    # Use the improved highlighting logic from _prepare_highlight_data
                    prepared_highlights = self._prepare_highlight_data(
                        group_by=group_by, color=color, bins=bins, annotate=annotate, **kwargs
                    )

                    # Check if we have quantitative metadata to preserve
                    quantitative_metadata = None
                    for highlight_data in prepared_highlights:
                        if (
                            "quantitative_metadata" in highlight_data
                            and highlight_data["quantitative_metadata"]
                        ):
                            quantitative_metadata = highlight_data["quantitative_metadata"]
                            break

                    # Add highlights from prepared data
                    for highlight_data in prepared_highlights:
                        # Only add elements from this page
                        elem = highlight_data.get("element")
                        if elem and hasattr(elem, "page") and elem.page == page:
                            # Create the highlight dict manually to preserve quantitative metadata
                            highlight_dict = {
                                "element": elem,
                                "color": highlight_data.get("color"),
                                "label": highlight_data.get("label"),
                            }

                            # Add quantitative metadata to the first highlight
                            if quantitative_metadata and not any(
                                h.get("quantitative_metadata") for h in spec.highlights
                            ):
                                highlight_dict["quantitative_metadata"] = quantitative_metadata

                            # Add annotate if provided in the prepared data
                            if "annotate" in highlight_data:
                                highlight_dict["annotate"] = highlight_data["annotate"]
                            if "attributes_to_draw" in highlight_data:
                                highlight_dict["attributes_to_draw"] = highlight_data[
                                    "attributes_to_draw"
                                ]

                            # Extract geometry from element
                            if (
                                hasattr(elem, "polygon")
                                and hasattr(elem, "has_polygon")
                                and elem.has_polygon
                            ):
                                highlight_dict["polygon"] = elem.polygon
                            elif hasattr(elem, "bbox"):
                                highlight_dict["bbox"] = elem.bbox

                            spec.highlights.append(highlight_dict)
                else:
                    # Default behavior when no group_by is specified
                    # Determine if all elements are of the same type
                    element_types = set(type(elem).__name__ for elem in page_elements)

                    if len(element_types) == 1:
                        # All elements are the same type - use a single label
                        type_name = element_types.pop()
                        # Generate a clean label from the type name
                        base_name = (
                            type_name.replace("Element", "").replace("Region", "")
                            if type_name != "Region"
                            else "Region"
                        )
                        # Handle special cases for common types
                        if base_name == "Text":
                            shared_label = "Text Elements"
                        elif base_name == "table_cell" or (
                            hasattr(page_elements[0], "region_type")
                            and page_elements[0].region_type == "table_cell"
                        ):
                            shared_label = "Table Cells"
                        elif base_name == "table":
                            shared_label = "Tables"
                        else:
                            shared_label = f"{base_name} Elements" if base_name else "Elements"

                        # Add all elements with the same label (no color cycling)
                        for elem in page_elements:
                            # Get element highlight params with annotate
                            element_data = self._get_element_highlight_params(elem, annotate)
                            if element_data:
                                # Use add_highlight with basic params
                                spec.add_highlight(
                                    element=elem,
                                    color=color,  # Use provided color or None
                                    label=shared_label,
                                )
                                # Update last highlight with attributes if present
                                if element_data.get("attributes_to_draw") and spec.highlights:
                                    spec.highlights[-1]["attributes_to_draw"] = element_data[
                                        "attributes_to_draw"
                                    ]
                    else:
                        # Mixed types - use individual labels (existing behavior)
                        for elem in page_elements:
                            # Get element highlight params with annotate
                            element_data = self._get_element_highlight_params(elem, annotate)
                            if element_data:
                                spec.add_highlight(
                                    element=elem,
                                    color=color,
                                    label=getattr(elem, "text", None) or str(elem),
                                )
                                # Update last highlight with attributes if present
                                if element_data.get("attributes_to_draw") and spec.highlights:
                                    spec.highlights[-1]["attributes_to_draw"] = element_data[
                                        "attributes_to_draw"
                                    ]

                # Add additional highlight groups if provided
                if highlights:
                    for group in highlights:
                        group_elements = group.get("elements", [])
                        group_color = group.get("color", color)
                        group_label = group.get("label")

                        # Only add elements from this page
                        for elem in group_elements:
                            if hasattr(elem, "page") and elem.page == page:
                                spec.add_highlight(
                                    element=elem, color=group_color, label=group_label
                                )

            specs.append(spec)

        return specs

    def _get_highlighter(self):
        """Get the highlighting service for rendering.

        For ElementCollection, we get it from the first element's page.
        """
        if not self._elements:
            raise RuntimeError("Cannot get highlighter from empty ElementCollection")

        # Try to get highlighter from first element's page
        for elem in self._elements:
            if hasattr(elem, "page") and hasattr(elem.page, "_highlighter"):
                return elem.page._highlighter

        # If no elements have pages, we can't render
        raise RuntimeError(
            "Cannot find HighlightingService. ElementCollection elements don't have page access."
        )

    def __len__(self) -> int:
        """Get the number of elements in the collection."""
        return len(self._elements)

    def __getitem__(self, index: Union[int, slice]) -> Union["Element", "ElementCollection"]:
        """Get an element by index or a collection by slice."""
        if isinstance(index, slice):
            # Return a new ElementCollection for slices
            return ElementCollection(self._elements[index])
        else:
            # Return the element for integer indices
            return self._elements[index]

    def __repr__(self) -> str:
        """Return a string representation showing the element count."""
        element_type = "Mixed"
        if self._elements:
            types = set(type(el).__name__ for el in self._elements)
            if len(types) == 1:
                element_type = types.pop()
        return f"<ElementCollection[{element_type}](count={len(self)})>"

    def __add__(self, other: "ElementCollection") -> "ElementCollection":
        if not isinstance(other, ElementCollection):
            return NotImplemented
        return ElementCollection(self._elements + other._elements)

    def __setitem__(self, index, value):
        self._elements[index] = value

    def __delitem__(self, index):
        del self._elements[index]

    def insert(self, index, value):
        self._elements.insert(index, value)

    @property
    def elements(self) -> List["Element"]:
        """Get the elements in this collection."""
        return self._elements

    @property
    def first(self) -> Optional["Element"]:
        """Get the first element in the collection."""
        return self._elements[0] if self._elements else None

    @property
    def last(self) -> Optional["Element"]:
        """Get the last element in the collection."""
        return self._elements[-1] if self._elements else None

    def _are_on_multiple_pages(self) -> bool:
        """
        Check if elements in this collection span multiple pages.

        Returns:
            True if elements are on different pages, False otherwise
        """
        if not self._elements:
            return False

        # Get the page index of the first element
        if not hasattr(self._elements[0], "page"):
            return False

        first_page_idx = self._elements[0].page.index

        # Check if any element is on a different page
        return any(hasattr(e, "page") and e.page.index != first_page_idx for e in self._elements)

    def _are_on_multiple_pdfs(self) -> bool:
        """
        Check if elements in this collection span multiple PDFs.

        Returns:
            True if elements are from different PDFs, False otherwise
        """
        if not self._elements:
            return False

        # Get the PDF of the first element
        if not hasattr(self._elements[0], "page") or not hasattr(self._elements[0].page, "pdf"):
            return False

        first_pdf = self._elements[0].page.pdf

        # Check if any element is from a different PDF
        return any(
            hasattr(e, "page") and hasattr(e.page, "pdf") and e.page.pdf is not first_pdf
            for e in self._elements
        )

    def highest(self) -> Optional["Element"]:
        """
        Get element with the smallest top y-coordinate (highest on page).

        Raises:
            ValueError: If elements are on multiple pages or multiple PDFs

        Returns:
            Element with smallest top value or None if empty
        """
        if not self._elements:
            return None

        # Check if elements are on multiple pages or PDFs
        if self._are_on_multiple_pdfs():
            raise ValueError("Cannot determine highest element across multiple PDFs")
        if self._are_on_multiple_pages():
            raise ValueError("Cannot determine highest element across multiple pages")

        return min(self._elements, key=lambda e: e.top)

    def lowest(self) -> Optional["Element"]:
        """
        Get element with the largest bottom y-coordinate (lowest on page).

        Raises:
            ValueError: If elements are on multiple pages or multiple PDFs

        Returns:
            Element with largest bottom value or None if empty
        """
        if not self._elements:
            return None

        # Check if elements are on multiple pages or PDFs
        if self._are_on_multiple_pdfs():
            raise ValueError("Cannot determine lowest element across multiple PDFs")
        if self._are_on_multiple_pages():
            raise ValueError("Cannot determine lowest element across multiple pages")

        return max(self._elements, key=lambda e: e.bottom)

    def leftmost(self) -> Optional["Element"]:
        """
        Get element with the smallest x0 coordinate (leftmost on page).

        Raises:
            ValueError: If elements are on multiple pages or multiple PDFs

        Returns:
            Element with smallest x0 value or None if empty
        """
        if not self._elements:
            return None

        # Check if elements are on multiple pages or PDFs
        if self._are_on_multiple_pdfs():
            raise ValueError("Cannot determine leftmost element across multiple PDFs")
        if self._are_on_multiple_pages():
            raise ValueError("Cannot determine leftmost element across multiple pages")

        return min(self._elements, key=lambda e: e.x0)

    def rightmost(self) -> Optional["Element"]:
        """
        Get element with the largest x1 coordinate (rightmost on page).

        Raises:
            ValueError: If elements are on multiple pages or multiple PDFs

        Returns:
            Element with largest x1 value or None if empty
        """
        if not self._elements:
            return None

        # Check if elements are on multiple pages or PDFs
        if self._are_on_multiple_pdfs():
            raise ValueError("Cannot determine rightmost element across multiple PDFs")
        if self._are_on_multiple_pages():
            raise ValueError("Cannot determine rightmost element across multiple pages")

        return max(self._elements, key=lambda e: e.x1)

    def exclude_regions(self, regions: List["Region"]) -> "ElementCollection":
        """
        Remove elements that are within any of the specified regions.

        Args:
            regions: List of Region objects to exclude

        Returns:
            New ElementCollection with filtered elements
        """
        if not regions:
            return ElementCollection(self._elements)

        filtered = []
        for element in self._elements:
            exclude = False
            for region in regions:
                if region._is_element_in_region(element):
                    exclude = True
                    break
            if not exclude:
                filtered.append(element)

        return ElementCollection(filtered)

    def extract_text(
        self,
        preserve_whitespace: bool = True,
        use_exclusions: bool = True,
        strip: Optional[bool] = None,
        content_filter=None,
        **kwargs,
    ) -> str:
        """
        Extract text from all TextElements in the collection, optionally using
        pdfplumber's layout engine if layout=True is specified.

        Args:
            preserve_whitespace: Deprecated. Use layout=False for simple joining.
            use_exclusions: Deprecated. Exclusions should be applied *before* creating
                          the collection or by filtering the collection itself.
            content_filter: Optional content filter to exclude specific text patterns. Can be:
                - A regex pattern string (characters matching the pattern are EXCLUDED)
                - A callable that takes text and returns True to KEEP the character
                - A list of regex patterns (characters matching ANY pattern are EXCLUDED)
            **kwargs: Additional layout parameters passed directly to pdfplumber's
                      `chars_to_textmap` function ONLY if `layout=True` is passed.
                      See Page.extract_text docstring for common parameters.
                      If `layout=False` or omitted, performs a simple join.
            strip: Whether to strip whitespace from the extracted text.

        Returns:
            Combined text from elements, potentially with layout-based spacing.
        """
        # Filter to just TextElements that likely have _char_dicts
        text_elements = [
            el
            for el in self._elements
            if isinstance(el, TextElement) and hasattr(el, "_char_dicts")
        ]

        if not text_elements:
            return ""

        # Collect all character dictionaries
        all_char_dicts = []
        for el in text_elements:
            all_char_dicts.extend(getattr(el, "_char_dicts", []))

        if not all_char_dicts:
            # Handle case where elements exist but have no char dicts
            logger.warning(
                "ElementCollection.extract_text: No character dictionaries found in TextElements."
            )
            return " ".join(
                getattr(el, "text", "") for el in text_elements
            )  # Fallback to simple join of word text

        # Apply content filtering if provided
        if content_filter is not None:
            from natural_pdf.utils.text_extraction import _apply_content_filter

            all_char_dicts = _apply_content_filter(all_char_dicts, content_filter)

        # Check if layout is requested
        use_layout = kwargs.get("layout", False)

        if use_layout:
            logger.debug("ElementCollection.extract_text: Using layout=True path.")
            # Layout requested: Use chars_to_textmap

            # Prepare layout kwargs
            layout_kwargs = {}
            allowed_keys = set(WORD_EXTRACTOR_KWARGS) | set(TEXTMAP_KWARGS)
            for key, value in kwargs.items():
                if key in allowed_keys:
                    layout_kwargs[key] = value
            layout_kwargs["layout"] = True  # Ensure layout is True

            # Calculate overall bbox for the elements used
            collection_bbox = objects_to_bbox(all_char_dicts)
            coll_x0, coll_top, coll_x1, coll_bottom = collection_bbox
            coll_width = coll_x1 - coll_x0
            coll_height = coll_bottom - coll_top

            # Set layout parameters based on collection bounds
            # Warn if collection is sparse? TBD.
            if "layout_bbox" not in layout_kwargs:
                layout_kwargs["layout_bbox"] = collection_bbox
            if "layout_width" not in layout_kwargs:
                layout_kwargs["layout_width"] = coll_width
            if "layout_height" not in layout_kwargs:
                layout_kwargs["layout_height"] = coll_height
            # Set shifts relative to the collection's top-left
            if "x_shift" not in layout_kwargs:
                layout_kwargs["x_shift"] = coll_x0
            if "y_shift" not in layout_kwargs:
                layout_kwargs["y_shift"] = coll_top

            try:
                # Sort chars by document order (page, top, x0)
                # Need page info on char dicts for multi-page collections
                # Assuming char dicts have 'page_number' from element creation
                all_char_dicts.sort(
                    key=lambda c: (c.get("page_number", 0), c.get("top", 0), c.get("x0", 0))
                )
                textmap = chars_to_textmap(all_char_dicts, **layout_kwargs)
                result = textmap.as_string
            except Exception as e:
                logger.error(
                    f"ElementCollection: Error calling chars_to_textmap: {e}", exc_info=True
                )
                logger.warning(
                    "ElementCollection: Falling back to simple text join due to layout error."
                )
                # Fallback sorting and joining
                all_char_dicts.sort(
                    key=lambda c: (c.get("page_number", 0), c.get("top", 0), c.get("x0", 0))
                )
                result = " ".join(c.get("text", "") for c in all_char_dicts)

        else:
            # Default: Simple join without layout
            logger.debug("ElementCollection.extract_text: Using simple join (layout=False).")
            # Sort chars by document order (page, top, x0)
            all_char_dicts.sort(
                key=lambda c: (c.get("page_number", 0), c.get("top", 0), c.get("x0", 0))
            )
            # Simple join of character text
            result = "".join(c.get("text", "") for c in all_char_dicts)
            # Replace multiple spaces created by joining possibly overlapping chars? Maybe not necessary.

        # Determine final strip flag – same rule as global helper unless caller overrides
        strip_text = strip if strip is not None else (not use_layout)

        if strip_text and isinstance(result, str):
            result = "\n".join(line.rstrip() for line in result.splitlines()).strip()

        return result

    def filter(self, func: Callable[["Element"], bool]) -> "ElementCollection":
        """
        Filter elements using a function.

        Args:
            func: Function that takes an element and returns True to keep it

        Returns:
            New ElementCollection with filtered elements
        """
        return ElementCollection([e for e in self._elements if func(e)])

    def sort(self, key=None, reverse=False) -> "ElementCollection":
        """
        Sort elements by the given key function.

        Args:
            key: Function to generate a key for sorting
            reverse: Whether to sort in descending order

        Returns:
            Self for method chaining
        """
        self._elements.sort(key=key, reverse=reverse)
        return self

    def highlight(
        self,
        label: Optional[str] = None,
        color: Optional[Union[Tuple, str]] = None,
        group_by: Optional[str] = None,
        label_format: Optional[str] = None,
        distinct: bool = False,
        annotate: Optional[List[str]] = None,
        replace: bool = False,
        bins: Optional[Union[int, List[float]]] = None,
    ) -> "ElementCollection":
        """
        Adds persistent highlights for all elements in the collection to the page
        via the HighlightingService.

        By default, this APPENDS highlights to any existing ones on the page.
        To replace existing highlights, set `replace=True`.

        Uses grouping logic based on parameters (defaulting to grouping by type).

        Note: Elements must be from the same PDF for this operation to work properly,
        as each PDF has its own highlighting service.

        Args:
            label: Optional explicit label for the entire collection. If provided,
                   all elements are highlighted as a single group with this label,
                   ignoring 'group_by' and the default type-based grouping.
            color: Optional explicit color for the highlight (tuple/string), or
                   matplotlib colormap name for quantitative group_by (e.g., 'viridis', 'plasma',
                   'inferno', 'coolwarm', 'RdBu'). Applied consistently if 'label' is provided
                   or if grouping occurs.
            group_by: Optional attribute name present on the elements. If provided
                      (and 'label' is None), elements will be grouped based on the
                      value of this attribute, and each group will be highlighted
                      with a distinct label and color. Automatically detects quantitative
                      data and uses gradient colormaps when appropriate.
            label_format: Optional Python f-string to format the group label when
                          'group_by' is used. Can reference element attributes
                          (e.g., "Type: {region_type}, Conf: {confidence:.2f}").
                          If None, the attribute value itself is used as the label.
            distinct: If True, bypasses all grouping and highlights each element
                      individually with cycling colors (the previous default behavior).
                      (default: False)
            annotate: List of attribute names from the element to display directly
                      on the highlight itself (distinct from group label).
            replace: If True, existing highlights on the affected page(s)
                     are cleared before adding these highlights.
                     If False (default), highlights are appended to existing ones.
            bins: Optional binning specification for quantitative data when using group_by.
                  Can be an integer (number of equal-width bins) or a list of bin edges.
                  Only used when group_by contains quantitative data.

        Returns:
            Self for method chaining

        Raises:
            AttributeError: If 'group_by' is provided but the attribute doesn't exist
                            on some elements.
            ValueError: If 'label_format' is provided but contains invalid keys for
                        element attributes, or if elements span multiple PDFs.
        """
        # Check if elements span multiple PDFs
        if self._are_on_multiple_pdfs():
            raise ValueError("highlight() does not support elements from multiple PDFs")

        # 1. Prepare the highlight data based on parameters
        highlight_data_list = self._prepare_highlight_data(
            distinct=distinct,
            label=label,
            color=color,
            group_by=group_by,
            label_format=label_format,
            annotate=annotate,
            bins=bins,
            # 'replace' flag is handled during the add call below
        )

        # 2. Add prepared highlights to the persistent service
        if not highlight_data_list:
            return self  # Nothing to add

        # Get page and highlighter from the first element (assume uniform page)
        first_element = self._elements[0]
        if not hasattr(first_element, "page") or not hasattr(first_element.page, "_highlighter"):
            logger.warning("Cannot highlight collection: Elements lack page or highlighter access.")
            return self

        page = first_element.page
        highlighter = page._highlighter

        # Use a set to track pages affected if replacing
        pages_to_clear = set()
        # Check the 'replace' flag. If True, we replace.
        if replace:
            # Identify all unique page indices in this operation
            for data in highlight_data_list:
                pages_to_clear.add(data["page_index"])
            # Clear those pages *before* adding new highlights
            logger.debug(
                f"Highlighting with replace=True. Clearing highlights for pages: {pages_to_clear}"
            )
            for page_idx in pages_to_clear:
                highlighter.clear_page(page_idx)

        for data in highlight_data_list:
            # Call the appropriate service add method
            add_args = {
                "page_index": data["page_index"],
                "color": data["color"],  # Color determined by _prepare
                "label": data["label"],  # Label determined by _prepare
                "use_color_cycling": data.get(
                    "use_color_cycling", False
                ),  # Set by _prepare if distinct
                "element": data["element"],
                "annotate": data["annotate"],
                # Internal call to service always appends, as clearing was handled above
                "existing": "append",
            }
            if data.get("polygon"):
                add_args["polygon"] = data["polygon"]
                highlighter.add_polygon(**add_args)
            elif data.get("bbox"):
                add_args["bbox"] = data["bbox"]
                highlighter.add(**add_args)
            else:
                logger.warning(f"Skipping highlight data, no bbox or polygon found: {data}")

        return self

    def _prepare_highlight_data(
        self,
        distinct: bool = False,
        label: Optional[str] = None,
        color: Optional[Union[Tuple, str]] = None,
        group_by: Optional[str] = None,
        label_format: Optional[str] = None,
        annotate: Optional[List[str]] = None,
        bins: Optional[Union[int, List[float]]] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Determines the parameters for highlighting each element based on the strategy.

        Does not interact with the HighlightingService directly.

        Returns:
            List of dictionaries, each containing parameters for a single highlight
            (e.g., page_index, bbox/polygon, color, label, element, annotate, attributes_to_draw).
            Color and label determination happens here.
        """
        prepared_data = []
        if not self._elements:
            return prepared_data

        # Need access to the HighlightingService to determine colors correctly.
        # Use highlighting protocol to find a valid service from any element
        highlighter = None

        for element in self._elements:
            # Try direct page access first (for regular elements)
            if hasattr(element, "page") and hasattr(element.page, "_highlighter"):
                highlighter = element.page._highlighter
                break
            # Try highlighting protocol for FlowRegions and other complex elements
            elif hasattr(element, "get_highlight_specs"):
                specs = element.get_highlight_specs()
                for spec in specs:
                    if "page" in spec and hasattr(spec["page"], "_highlighter"):
                        highlighter = spec["page"]._highlighter
                        break
                if highlighter:
                    break

        if not highlighter:
            logger.warning(
                "Cannot determine highlight colors: HighlightingService not accessible from elements."
            )
            return []

        if distinct:
            logger.debug("_prepare: Distinct highlighting strategy.")
            for element in self._elements:
                # Call the service's color determination logic
                final_color = highlighter._determine_highlight_color(
                    label=None, color_input=None, use_color_cycling=True
                )
                element_data = self._get_element_highlight_params(element, annotate)
                if element_data:
                    element_data.update(
                        {"color": final_color, "label": None, "use_color_cycling": True}
                    )
                    prepared_data.append(element_data)

        elif label is not None:
            logger.debug(f"_prepare: Explicit label '{label}' strategy.")
            final_color = highlighter._determine_highlight_color(
                label=label, color_input=color, use_color_cycling=False
            )
            for element in self._elements:
                element_data = self._get_element_highlight_params(element, annotate)
                if element_data:
                    element_data.update({"color": final_color, "label": label})
                    prepared_data.append(element_data)

        elif group_by is not None:
            logger.debug("_prepare: Grouping by attribute strategy.")
            grouped_elements = self._group_elements_by_attr(group_by)

            # Collect all values for quantitative detection
            all_values = []
            for group_key, group_elements in grouped_elements.items():
                if group_elements:
                    all_values.append(group_key)

            # Import the quantitative detection function
            from natural_pdf.utils.visualization import (
                create_quantitative_color_mapping,
                detect_quantitative_data,
            )

            # Determine if we should use quantitative color mapping
            use_quantitative = detect_quantitative_data(all_values)

            if use_quantitative:
                logger.debug("  _prepare: Using quantitative color mapping.")
                # Use quantitative color mapping with specified colormap
                colormap_name = color if isinstance(color, str) else "viridis"
                value_to_color = create_quantitative_color_mapping(
                    all_values, colormap=colormap_name, bins=bins
                )

                # Store quantitative metadata for colorbar creation
                quantitative_metadata = {
                    "values": all_values,
                    "colormap": colormap_name,
                    "bins": bins,
                    "attribute": group_by,
                }

                for group_key, group_elements in grouped_elements.items():
                    if not group_elements:
                        continue
                    group_label = self._format_group_label(
                        group_key, label_format, group_elements[0], group_by
                    )

                    # Get quantitative color for this value
                    final_color = value_to_color.get(group_key)
                    if final_color is None:
                        # Fallback to traditional color assignment
                        final_color = highlighter._determine_highlight_color(
                            label=group_label, color_input=None, use_color_cycling=False
                        )

                    logger.debug(
                        f"  _prepare group '{group_label}' ({len(group_elements)} elements) -> color {final_color}"
                    )
                    for element in group_elements:
                        element_data = self._get_element_highlight_params(element, annotate)
                        if element_data:
                            element_data.update({"color": final_color, "label": group_label})
                            # Add quantitative metadata to the first element in each group
                            if not any("quantitative_metadata" in pd for pd in prepared_data):
                                element_data["quantitative_metadata"] = quantitative_metadata
                            prepared_data.append(element_data)
            else:
                logger.debug("  _prepare: Using categorical color mapping.")
                # Use traditional categorical color mapping
                for group_key, group_elements in grouped_elements.items():
                    if not group_elements:
                        continue
                    group_label = self._format_group_label(
                        group_key, label_format, group_elements[0], group_by
                    )
                    final_color = highlighter._determine_highlight_color(
                        label=group_label, color_input=None, use_color_cycling=False
                    )
                    logger.debug(
                        f"  _prepare group '{group_label}' ({len(group_elements)} elements) -> color {final_color}"
                    )
                    for element in group_elements:
                        element_data = self._get_element_highlight_params(element, annotate)
                        if element_data:
                            element_data.update({"color": final_color, "label": group_label})
                            prepared_data.append(element_data)
        else:
            logger.debug("_prepare: Default grouping strategy.")
            element_types = set(type(el).__name__ for el in self._elements)

            if len(element_types) == 1:
                type_name = element_types.pop()
                base_name = (
                    type_name.replace("Element", "").replace("Region", "")
                    if type_name != "Region"
                    else "Region"
                )
                auto_label = f"{base_name} Elements" if base_name else "Elements"
                # Determine color *before* logging or using it
                final_color = highlighter._determine_highlight_color(
                    label=auto_label, color_input=color, use_color_cycling=False
                )
                logger.debug(f"  _prepare default group '{auto_label}' -> color {final_color}")
                for element in self._elements:
                    element_data = self._get_element_highlight_params(element, annotate)
                    if element_data:
                        element_data.update({"color": final_color, "label": auto_label})
                        prepared_data.append(element_data)
            else:
                # Mixed types: Generate generic label and warn
                type_names_str = ", ".join(sorted(list(element_types)))
                auto_label = "Mixed Elements"
                logger.warning(
                    f"Highlighting collection with mixed element types ({type_names_str}) "
                    f"using generic label '{auto_label}'. Consider using 'label', 'group_by', "
                    f"or 'distinct=True' for more specific highlighting."
                )
                final_color = highlighter._determine_highlight_color(
                    label=auto_label, color_input=color, use_color_cycling=False
                )
                # Determine color *before* logging or using it (already done above for this branch)
                logger.debug(f"  _prepare default group '{auto_label}' -> color {final_color}")
                for element in self._elements:
                    element_data = self._get_element_highlight_params(element, annotate)
                    if element_data:
                        element_data.update({"color": final_color, "label": auto_label})
                        prepared_data.append(element_data)

        return prepared_data

    def _call_element_highlighter(
        self,
        element: T,
        color: Optional[Union[Tuple, str]],
        label: Optional[str],
        use_color_cycling: bool,
        annotate: Optional[List[str]],
        existing: str,
    ):
        """Low-level helper to call the appropriate HighlightingService method for an element."""
        if not hasattr(element, "page") or not hasattr(element.page, "_highlighter"):
            logger.warning(
                f"Cannot highlight element, missing 'page' attribute or page lacks highlighter access: {element}"
            )
            return

        page = element.page
        args_for_highlighter = {
            "page_index": page.index,
            "color": color,
            "label": label,
            "use_color_cycling": use_color_cycling,
            "annotate": annotate,
            "existing": existing,
            "element": element,
        }

        is_polygon = getattr(element, "has_polygon", False)
        geom_data = None
        add_method = None

        if is_polygon:
            geom_data = getattr(element, "polygon", None)
            if geom_data:
                args_for_highlighter["polygon"] = geom_data
                add_method = page._highlighter.add_polygon
        else:
            geom_data = getattr(element, "bbox", None)
            if geom_data:
                args_for_highlighter["bbox"] = geom_data
                add_method = page._highlighter.add

        if add_method and geom_data:
            try:
                add_method(**args_for_highlighter)
            except Exception as e:
                logger.error(
                    f"Error calling highlighter method for element {element} on page {page.index}: {e}",
                    exc_info=True,
                )
        elif not geom_data:
            logger.warning(f"Cannot highlight element, no bbox or polygon found: {element}")

    def _highlight_as_single_group(
        self,
        label: str,
        color: Optional[Union[Tuple, str]],
        annotate: Optional[List[str]],
        existing: str,
    ):
        """Highlights all elements with the same explicit label and color."""
        for element in self._elements:
            self._call_element_highlighter(
                element=element,
                color=color,  # Use explicit color if provided
                label=label,  # Use the explicit group label
                use_color_cycling=False,  # Use consistent color for the label
                annotate=annotate,
                existing=existing,
            )

    def _highlight_grouped_by_attribute(
        self,
        group_by: str,
        label_format: Optional[str],
        annotate: Optional[List[str]],
        existing: str,
    ):
        """Groups elements by attribute and highlights each group distinctly."""
        grouped_elements: Dict[Any, List[T]] = {}
        # Group elements by the specified attribute value
        for element in self._elements:
            try:
                group_key = getattr(element, group_by, None)
                if group_key is None:  # Handle elements missing the attribute
                    group_key = f"Missing '{group_by}'"
                # Ensure group_key is hashable (convert list/dict if necessary)
                if isinstance(group_key, (list, dict)):
                    group_key = str(group_key)

                if group_key not in grouped_elements:
                    grouped_elements[group_key] = []
                grouped_elements[group_key].append(element)
            except AttributeError:
                logger.warning(
                    f"Attribute '{group_by}' not found on element {element}. Skipping grouping."
                )
                group_key = f"Error accessing '{group_by}'"
                if group_key not in grouped_elements:
                    grouped_elements[group_key] = []
                grouped_elements[group_key].append(element)
            except TypeError:  # Handle unhashable types
                logger.warning(
                    f"Attribute value for '{group_by}' on {element} is unhashable ({type(group_key)}). Using string representation."
                )
                group_key = str(group_key)
                if group_key not in grouped_elements:
                    grouped_elements[group_key] = []
                grouped_elements[group_key].append(element)

        # Highlight each group
        for group_key, group_elements in grouped_elements.items():
            if not group_elements:
                continue

            # Determine the label for this group
            first_element = group_elements[0]  # Use first element for formatting
            group_label = None
            if label_format:
                try:
                    # Create a dict of element attributes for formatting
                    element_attrs = first_element.__dict__.copy()  # Start with element's dict
                    # Ensure the group_by key itself is present correctly
                    element_attrs[group_by] = group_key
                    group_label = label_format.format(**element_attrs)
                except KeyError as e:
                    logger.warning(
                        f"Invalid key '{e}' in label_format '{label_format}'. Using group key as label."
                    )
                    group_label = str(group_key)
                except Exception as format_e:
                    logger.warning(
                        f"Error formatting label '{label_format}': {format_e}. Using group key as label."
                    )
                    group_label = str(group_key)
            else:
                group_label = str(group_key)  # Use the attribute value as label

            logger.debug(f"  Highlighting group '{group_label}' ({len(group_elements)} elements)")

            # Highlight all elements in this group with the derived label
            for element in group_elements:
                self._call_element_highlighter(
                    element=element,
                    color=None,  # Let ColorManager choose based on label
                    label=group_label,  # Use the derived group label
                    use_color_cycling=False,  # Use consistent color for the label
                    annotate=annotate,
                    existing=existing,
                )

    def _highlight_distinctly(self, annotate: Optional[List[str]], existing: str):
        """DEPRECATED: Logic moved to _prepare_highlight_data. Kept for reference/potential reuse."""
        # This method is no longer called directly by the main highlight path.
        # The distinct logic is handled within _prepare_highlight_data.
        for element in self._elements:
            self._call_element_highlighter(
                element=element,
                color=None,  # Let ColorManager cycle
                label=None,  # No label for distinct elements
                use_color_cycling=True,  # Force cycling
                annotate=annotate,
                existing=existing,
            )

    def _render_multipage_highlights(
        self,
        specs_by_page,
        resolution,
        width,
        labels,
        legend_position,
        group_by,
        label,
        color,
        label_format,
        distinct,
        annotate,
        render_ocr,
        crop,
        stack_direction="vertical",
        stack_gap=5,
        stack_background_color=(255, 255, 255),
    ):
        """Render highlights across multiple pages and stack them."""
        from PIL import Image

        # Sort pages by index for consistent output
        sorted_pages = sorted(
            specs_by_page.keys(), key=lambda p: p.index if hasattr(p, "index") else 0
        )

        page_images = []

        for page in sorted_pages:
            element_specs = specs_by_page[page]

            # Get highlighter service from the page
            if not hasattr(page, "_highlighter"):
                logger.warning(
                    f"Page {getattr(page, 'number', '?')} has no highlighter service, skipping"
                )
                continue

            service = page._highlighter

            # Prepare highlight data for this page
            highlight_data_list = []

            for element_idx, spec in element_specs:
                # Use the element index to generate consistent colors/labels across pages
                element = spec.get(
                    "element",
                    self._elements[element_idx] if element_idx < len(self._elements) else None,
                )

                # Prepare highlight data based on grouping parameters
                if distinct:
                    # Use cycling colors for distinct mode
                    element_color = None  # Let the highlighter service pick from palette
                    use_color_cycling = True
                    element_label = (
                        f"Element_{element_idx + 1}"
                        if label is None
                        else f"{label}_{element_idx + 1}"
                    )
                elif label:
                    # Explicit label for all elements
                    element_color = color
                    use_color_cycling = color is None
                    element_label = label
                elif group_by and element:
                    # Group by attribute
                    try:
                        group_key = getattr(element, group_by, None)
                        element_label = self._format_group_label(
                            group_key, label_format, element, group_by
                        )
                        element_color = None  # Let service assign color by group
                        use_color_cycling = True
                    except:
                        element_label = f"Element_{element_idx + 1}"
                        element_color = color
                        use_color_cycling = color is None
                else:
                    # Default behavior
                    element_color = color
                    use_color_cycling = color is None
                    element_label = f"Element_{element_idx + 1}"

                # Build highlight data
                highlight_item = {
                    "page_index": spec["page_index"],
                    "bbox": spec["bbox"],
                    "polygon": spec.get("polygon"),
                    "color": element_color,
                    "label": element_label if labels else None,
                    "use_color_cycling": use_color_cycling,
                }

                # Add attributes if requested
                if annotate and element:
                    highlight_item["attributes_to_draw"] = {}
                    for attr_name in annotate:
                        try:
                            attr_value = getattr(element, attr_name, None)
                            if attr_value is not None:
                                highlight_item["attributes_to_draw"][attr_name] = attr_value
                        except:
                            pass

                highlight_data_list.append(highlight_item)

            # Calculate crop bbox if requested
            crop_bbox = None
            if crop:
                try:
                    # Get bboxes from all specs on this page
                    bboxes = [spec["bbox"] for _, spec in element_specs if spec.get("bbox")]
                    if bboxes:
                        crop_bbox = (
                            min(bbox[0] for bbox in bboxes),
                            min(bbox[1] for bbox in bboxes),
                            max(bbox[2] for bbox in bboxes),
                            max(bbox[3] for bbox in bboxes),
                        )
                except Exception as bbox_err:
                    logger.error(f"Error determining crop bbox: {bbox_err}")

            # Render this page
            try:
                img = service.render_preview(
                    page_index=page.index,
                    temporary_highlights=highlight_data_list,
                    resolution=resolution,
                    width=width,
                    labels=labels,
                    legend_position=legend_position,
                    render_ocr=render_ocr,
                    crop_bbox=crop_bbox,
                )

                if img:
                    page_images.append(img)
            except Exception as e:
                logger.error(
                    f"Error rendering page {getattr(page, 'number', '?')}: {e}", exc_info=True
                )

        if not page_images:
            logger.warning("Failed to render any pages")
            return None

        if len(page_images) == 1:
            return page_images[0]

        # Stack the images
        if stack_direction == "vertical":
            final_width = max(img.width for img in page_images)
            final_height = (
                sum(img.height for img in page_images) + (len(page_images) - 1) * stack_gap
            )

            stacked_image = Image.new("RGB", (final_width, final_height), stack_background_color)

            current_y = 0
            for img in page_images:
                # Center horizontally
                x_offset = (final_width - img.width) // 2
                stacked_image.paste(img, (x_offset, current_y))
                current_y += img.height + stack_gap
        else:  # horizontal
            final_width = sum(img.width for img in page_images) + (len(page_images) - 1) * stack_gap
            final_height = max(img.height for img in page_images)

            stacked_image = Image.new("RGB", (final_width, final_height), stack_background_color)

            current_x = 0
            for img in page_images:
                # Center vertically
                y_offset = (final_height - img.height) // 2
                stacked_image.paste(img, (current_x, y_offset))
                current_x += img.width + stack_gap

        return stacked_image

    def save(
        self,
        filename: str,
        resolution: Optional[float] = None,
        width: Optional[int] = None,
        labels: bool = True,
        legend_position: str = "right",
        render_ocr: bool = False,
    ) -> "ElementCollection":
        """
        Save the page with this collection's elements highlighted to an image file.

        Args:
            filename: Path to save the image to
            resolution: Resolution in DPI for rendering (uses global options if not specified, defaults to 144 DPI)
            width: Optional width for the output image in pixels
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            render_ocr: Whether to render OCR text with white background boxes

        Returns:
            Self for method chaining
        """
        # Apply global options as defaults, but allow explicit parameters to override
        import natural_pdf

        # Use global options if parameters are not explicitly set
        if width is None:
            width = natural_pdf.options.image.width
        if resolution is None:
            if natural_pdf.options.image.resolution is not None:
                resolution = natural_pdf.options.image.resolution
            else:
                resolution = 144  # Default resolution when none specified

        # Use export() to save the image
        self.export(
            path=filename,
            resolution=resolution,
            width=width,
            labels=labels,
            legend_position=legend_position,
            render_ocr=render_ocr,
        )
        return self

        return None

    def _group_elements_by_attr(self, group_by: str) -> Dict[Any, List[T]]:
        """Groups elements by the specified attribute."""
        grouped_elements: Dict[Any, List[T]] = {}
        for element in self._elements:
            try:
                group_key = getattr(element, group_by, None)
                if group_key is None:  # Handle elements missing the attribute
                    group_key = f"Missing '{group_by}'"
                # Ensure group_key is hashable (convert list/dict if necessary)
                if isinstance(group_key, (list, dict)):
                    group_key = str(group_key)

                if group_key not in grouped_elements:
                    grouped_elements[group_key] = []
                grouped_elements[group_key].append(element)
            except AttributeError:
                logger.warning(
                    f"Attribute '{group_by}' not found on element {element}. Skipping grouping."
                )
                group_key = f"Error accessing '{group_by}'"
                if group_key not in grouped_elements:
                    grouped_elements[group_key] = []
                grouped_elements[group_key].append(element)
            except TypeError:  # Handle unhashable types
                logger.warning(
                    f"Attribute value for '{group_by}' on {element} is unhashable ({type(group_key)}). Using string representation."
                )
                group_key = str(group_key)
                if group_key not in grouped_elements:
                    grouped_elements[group_key] = []
                grouped_elements[group_key].append(element)

        return grouped_elements

    def _format_group_label(
        self, group_key: Any, label_format: Optional[str], sample_element: T, group_by_attr: str
    ) -> str:
        """Formats the label for a group based on the key and format string."""
        if label_format:
            try:
                element_attrs = sample_element.__dict__.copy()
                element_attrs[group_by_attr] = group_key  # Ensure key is present
                return label_format.format(**element_attrs)
            except KeyError as e:
                logger.warning(
                    f"Invalid key '{e}' in label_format '{label_format}'. Using group key as label."
                )
                return str(group_key)
            except Exception as format_e:
                logger.warning(
                    f"Error formatting label '{label_format}': {format_e}. Using group key as label."
                )
                return str(group_key)
        else:
            return str(group_key)

    def _get_element_highlight_params(
        self, element: T, annotate: Optional[List[str]]
    ) -> Optional[Dict]:
        """Extracts common parameters needed for highlighting a single element."""
        # For FlowRegions and other complex elements, use highlighting protocol
        if hasattr(element, "get_highlight_specs"):
            specs = element.get_highlight_specs()
            if not specs:
                logger.warning(f"Element {element} returned no highlight specs")
                return None

            # For now, we'll use the first spec for the prepared data
            # The actual rendering will use all specs
            first_spec = specs[0]
            page = first_spec["page"]

            base_data = {
                "page_index": first_spec["page_index"],
                "element": element,
                "annotate": annotate,
                "attributes_to_draw": {},
                "bbox": first_spec.get("bbox"),
                "polygon": first_spec.get("polygon"),
                "multi_spec": len(specs) > 1,  # Flag to indicate multiple specs
                "all_specs": specs,  # Store all specs for rendering
            }

            # Extract attributes if requested
            if annotate:
                for attr_name in annotate:
                    try:
                        attr_value = getattr(element, attr_name, None)
                        if attr_value is not None:
                            base_data["attributes_to_draw"][attr_name] = attr_value
                    except AttributeError:
                        logger.warning(
                            f"Attribute '{attr_name}' not found on element {element} for annotate"
                        )

            return base_data

        # Fallback for regular elements with direct page access
        if not hasattr(element, "page"):
            logger.warning(f"Element {element} has no page attribute and no highlighting protocol")
            return None

        page = element.page

        base_data = {
            "page_index": page.index,
            "element": element,
            "annotate": annotate,
            "attributes_to_draw": {},
            "bbox": None,
            "polygon": None,
        }

        # Extract geometry
        is_polygon = getattr(element, "has_polygon", False)
        geom_data = None
        if is_polygon:
            geom_data = getattr(element, "polygon", None)
            if geom_data:
                base_data["polygon"] = geom_data
        else:
            geom_data = getattr(element, "bbox", None)
            if geom_data:
                base_data["bbox"] = geom_data

        if not geom_data:
            logger.warning(
                f"Cannot prepare highlight, no bbox or polygon found for element: {element}"
            )
            return None

        # Extract attributes if requested
        if annotate:
            for attr_name in annotate:
                try:
                    attr_value = getattr(element, attr_name, None)
                    if attr_value is not None:
                        base_data["attributes_to_draw"][attr_name] = attr_value
                except AttributeError:
                    logger.warning(
                        f"Attribute '{attr_name}' not found on element {element} for annotate"
                    )

        return base_data

    def viewer(self, title: Optional[str] = None) -> Optional["widgets.DOMWidget"]:
        """
        Creates and returns an interactive ipywidget showing ONLY the elements
        in this collection on their page background.

        Args:
            title: Optional title for the viewer window/widget.

        Returns:
            An InteractiveViewerWidget instance or None if elements lack page context.
        """
        if not self.elements:
            logger.warning("Cannot generate interactive viewer for empty collection.")
            return None

        # Assume all elements are on the same page and have .page attribute
        try:
            page = self.elements[0].page
            # Check if the page object actually has the method
            if hasattr(page, "viewer") and callable(page.viewer):
                final_title = (
                    title or f"Interactive Viewer for Collection ({len(self.elements)} elements)"
                )
                # Call the page method, passing this collection's elements
                return page.viewer(
                    elements_to_render=self.elements,
                    title=final_title,  # Pass title if Page method accepts it
                )
            else:
                logger.error("Page object is missing the 'viewer' method.")
                return None
        except AttributeError:
            logger.error(
                "Cannot generate interactive viewer: Elements in collection lack 'page' attribute."
            )
            return None
        except IndexError:
            # Should be caught by the empty check, but just in case
            logger.error(
                "Cannot generate interactive viewer: Collection unexpectedly became empty."
            )
            return None
        except Exception as e:
            logger.error(f"Error creating interactive viewer from collection: {e}", exc_info=True)
            return None

    def find(self, selector: str, **kwargs) -> "ElementCollection":
        """
        Find elements in this collection matching the selector.

        Args:
            selector: CSS-like selector string
            overlap: How to determine if elements overlap: 'full' (fully inside),
                      'partial' (any overlap), or 'center' (center point inside).
                      (default: "full")
            apply_exclusions: Whether to exclude elements in exclusion regions
        """
        return self.apply(lambda element: element.find(selector, **kwargs))

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
        Find all elements within each element of this collection matching the selector OR text,
        and return a flattened collection of all found sub-elements.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for (equivalent to 'text:contains(...)').
            overlap: How to determine if elements overlap: 'full' (fully inside),
                     'partial' (any overlap), or 'center' (center point inside).
                     (default: "full")
            apply_exclusions: Whether to apply exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            **kwargs: Additional parameters for element filtering.

        Returns:
            A new ElementCollection containing all matching sub-elements from all elements
            in this collection.
        """
        if selector is None and text is None:
            raise ValueError("Either 'selector' or 'text' must be provided to find_all.")
        if selector is not None and text is not None:
            raise ValueError("Provide either 'selector' or 'text' to find_all, not both.")

        all_found_elements: List[Element] = []
        for element in self._elements:
            if hasattr(element, "find_all") and callable(element.find_all):
                # Element.find_all returns an ElementCollection
                found_in_element: "ElementCollection" = element.find_all(
                    selector=selector,
                    text=text,
                    overlap=overlap,
                    apply_exclusions=apply_exclusions,
                    regex=regex,
                    case=case,
                    **kwargs,
                )
                if found_in_element and found_in_element.elements:
                    all_found_elements.extend(found_in_element.elements)
            # else:
            # Elements in the collection are expected to support find_all.
            # If an element type doesn't, an AttributeError will naturally occur,
            # or a more specific check/handling could be added here if needed.

        return ElementCollection(all_found_elements)

    def extract_each_text(self, **kwargs) -> List[str]:
        """
        Extract text from each element in this region.
        """
        return self.apply(
            lambda element: element.extract_text(**kwargs) if element is not None else None
        )

    def correct_ocr(
        self,
        transform: Callable[[Any], Optional[str]],
        max_workers: Optional[int] = None,
    ) -> "ElementCollection":
        """
        Applies corrections to OCR-generated text elements within this collection
        using a user-provided callback function, executed
        in parallel if `max_workers` is specified.

        Iterates through elements currently in the collection. If an element's
        'source' attribute starts with 'ocr', it calls the `transform`
        for that element, passing the element itself.

        The `transform` should contain the logic to:
        1. Determine if the element needs correction.
        2. Perform the correction (e.g., call an LLM).
        3. Return the new text (`str`) or `None`.

        If the callback returns a string, the element's `.text` is updated in place.
        Metadata updates (source, confidence, etc.) should happen within the callback.
        Elements without a source starting with 'ocr' are skipped.

        Args:
            transform: A function accepting an element and returning
                       `Optional[str]` (new text or None).
            max_workers: The maximum number of worker threads to use for parallel
                         correction on each page. If None, defaults are used.

        Returns:
            Self for method chaining.
        """
        # Delegate to the utility function
        _apply_ocr_correction_to_elements(
            elements=self._elements,
            correction_callback=transform,
            caller_info=f"ElementCollection(len={len(self._elements)})",  # Pass caller info
            max_workers=max_workers,
        )
        return self  # Return self for chaining

    def remove(self) -> int:
        """
        Remove all elements in this collection from their respective pages.

        This method removes elements from the page's _element_mgr storage.
        It's particularly useful for removing OCR elements before applying new OCR.

        Returns:
            int: Number of elements successfully removed
        """
        if not self._elements:
            return 0

        removed_count = 0

        for element in self._elements:
            # Each element should have a reference to its page
            if hasattr(element, "page") and hasattr(element.page, "_element_mgr"):
                element_mgr = element.page._element_mgr

                # Determine element type
                element_type = getattr(element, "object_type", None)
                if element_type:
                    # Convert to plural form expected by element_mgr
                    if element_type == "word":
                        element_type = "words"
                    elif element_type == "char":
                        element_type = "chars"
                    elif element_type == "rect":
                        element_type = "rects"
                    elif element_type == "line":
                        element_type = "lines"

                    # Try to remove from the element manager
                    if hasattr(element_mgr, "remove_element"):
                        success = element_mgr.remove_element(element, element_type)
                        if success:
                            removed_count += 1
                    else:
                        logger.warning("ElementManager does not have remove_element method")
            else:
                logger.warning(f"Element has no page or page has no _element_mgr: {element}")

        return removed_count

    # --- Classification Method --- #
    def classify_all(
        self,
        labels: List[str],
        model: Optional[str] = None,
        using: Optional[str] = None,
        min_confidence: float = 0.0,
        analysis_key: str = "classification",
        multi_label: bool = False,
        batch_size: int = 8,
        progress_bar: bool = True,
        **kwargs,
    ):
        """Classifies all elements in the collection in batch.

        Args:
            labels: List of category labels.
            model: Model ID (or alias 'text', 'vision').
            using: Optional processing mode ('text' or 'vision'). Inferred if None.
            min_confidence: Minimum confidence threshold.
            analysis_key: Key for storing results in element.analyses.
            multi_label: Allow multiple labels per item.
            batch_size: Size of batches passed to the inference pipeline.
            progress_bar: Display a progress bar.
            **kwargs: Additional arguments for the ClassificationManager.
        """
        if not self.elements:
            logger.info("ElementCollection is empty, skipping classification.")
            return self

        # Requires access to the PDF's manager. Assume first element has it.
        first_element = self.elements[0]
        manager_source = None
        if hasattr(first_element, "page") and hasattr(first_element.page, "pdf"):
            manager_source = first_element.page.pdf
        elif hasattr(first_element, "pdf"):  # Maybe it's a PageCollection?
            manager_source = first_element.pdf

        if not manager_source or not hasattr(manager_source, "get_manager"):
            raise RuntimeError("Cannot access ClassificationManager via elements.")

        try:
            manager = manager_source.get_manager("classification")
        except Exception as e:
            raise RuntimeError(f"Failed to get ClassificationManager: {e}") from e

        if not manager or not manager.is_available():
            raise RuntimeError("ClassificationManager is not available.")

        # Determine engine type early for content gathering
        inferred_using = manager.infer_using(model if model else manager.DEFAULT_TEXT_MODEL, using)

        # Gather content from all elements
        items_to_classify: List[Tuple[Any, Union[str, Image.Image]]] = []
        original_elements: List[Any] = []
        logger.info(
            f"Gathering content for {len(self.elements)} elements for batch classification..."
        )
        for element in self.elements:
            if not isinstance(element, ClassificationMixin):
                logger.warning(f"Skipping element (not ClassificationMixin): {element!r}")
                continue
            try:
                # Delegate content fetching to the element itself
                content = element._get_classification_content(model_type=inferred_using, **kwargs)
                items_to_classify.append(content)
                original_elements.append(element)
            except (ValueError, NotImplementedError) as e:
                logger.warning(
                    f"Skipping element {element!r}: Cannot get content for classification - {e}"
                )
            except Exception as e:
                logger.warning(
                    f"Skipping element {element!r}: Error getting classification content - {e}"
                )

        if not items_to_classify:
            logger.warning("No content could be gathered from elements for batch classification.")
            return self

        logger.info(
            f"Collected content for {len(items_to_classify)} elements. Running batch classification..."
        )

        # Call manager's batch classify
        batch_results: List[ClassificationResult] = manager.classify_batch(
            item_contents=items_to_classify,
            labels=labels,
            model_id=model,
            using=inferred_using,
            min_confidence=min_confidence,
            multi_label=multi_label,
            batch_size=batch_size,
            progress_bar=progress_bar,
            **kwargs,
        )

        # Assign results back to elements
        if len(batch_results) != len(original_elements):
            logger.error(
                f"Batch classification result count ({len(batch_results)}) mismatch "
                f"with elements processed ({len(original_elements)}). Cannot assign results."
            )
            # Decide how to handle mismatch - maybe store errors?
        else:
            logger.info(
                f"Assigning {len(batch_results)} results to elements under key '{analysis_key}'."
            )
            for element, result_obj in zip(original_elements, batch_results):
                try:
                    if not hasattr(element, "analyses") or element.analyses is None:
                        element.analyses = {}
                    element.analyses[analysis_key] = result_obj
                except Exception as e:
                    logger.warning(f"Failed to store classification result for {element!r}: {e}")

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
        Gather analysis data from all elements in the collection.

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
            logger.warning("No elements found in collection")
            return []

        all_data = []

        for i, element in enumerate(self.elements):
            # Base element information
            element_data = {
                "element_index": i,
                "element_type": getattr(element, "type", type(element).__name__),
            }

            # Add geometry if available
            for attr in ["x0", "top", "x1", "bottom", "width", "height"]:
                if hasattr(element, attr):
                    element_data[attr] = getattr(element, attr)

            # Add page information if available
            if hasattr(element, "page"):
                page = element.page
                if page:
                    element_data["page_number"] = getattr(page, "number", None)
                    element_data["pdf_path"] = (
                        getattr(page.pdf, "path", None) if hasattr(page, "pdf") else None
                    )

            # Include extracted text if requested
            if include_content and hasattr(element, "extract_text"):
                try:
                    element_data["content"] = element.extract_text(preserve_whitespace=True)
                except Exception as e:
                    logger.error(f"Error extracting text from element {i}: {e}")
                    element_data["content"] = ""

            # Save image if requested
            if include_images and hasattr(element, "to_image"):
                try:
                    # Create identifier for the element
                    pdf_name = "unknown"
                    page_num = "unknown"

                    if hasattr(element, "page") and element.page:
                        page_num = element.page.number
                        if hasattr(element.page, "pdf") and element.page.pdf:
                            pdf_name = Path(element.page.pdf.path).stem

                    # Create image filename
                    element_type = element_data.get("element_type", "element").lower()
                    image_filename = f"{pdf_name}_page{page_num}_{element_type}_{i}.{image_format}"
                    image_path = image_dir / image_filename

                    # Save image
                    element.show(path=str(image_path), resolution=image_resolution)

                    # Add relative path to data
                    element_data["image_path"] = str(Path(image_path).relative_to(image_dir.parent))
                except Exception as e:
                    logger.error(f"Error saving image for element {i}: {e}")
                    element_data["image_path"] = None

            # Add analyses data
            if hasattr(element, "analyses"):
                for key in analysis_keys:
                    if key not in element.analyses:
                        # Skip this key if it doesn't exist - elements might have different analyses
                        logger.warning(f"Analysis key '{key}' not found in element {i}")
                        continue

                    # Get the analysis result
                    analysis_result = element.analyses[key]

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

                    # Add analysis data to element data with the key as prefix
                    for k, v in analysis_data.items():
                        element_data[f"{key}.{k}"] = v

            all_data.append(element_data)

        return all_data

    def to_text_elements(
        self,
        text_content_func: Optional[Callable[["Region"], Optional[str]]] = None,
        source_label: str = "derived_from_region",
        object_type: str = "word",
        default_font_size: float = 10.0,
        default_font_name: str = "RegionContent",
        confidence: Optional[float] = None,
        add_to_page: bool = False,  # Default is False
    ) -> "ElementCollection[TextElement]":
        """
        Converts each Region in this collection to a TextElement.

        Args:
            text_content_func: A callable that takes a Region and returns its text
                               (or None). If None, all created TextElements will
                               have text=None.
            source_label: The 'source' attribute for the new TextElements.
            object_type: The 'object_type' for the TextElement's data dict.
            default_font_size: Placeholder font size.
            default_font_name: Placeholder font name.
            confidence: Confidence score.
            add_to_page: If True (default is False), also adds the created
                         TextElements to their respective page's element manager.

        Returns:
            A new ElementCollection containing the created TextElement objects.
        """
        from natural_pdf.elements.region import (  # Local import for type checking if needed or to resolve circularity
            Region,
        )
        from natural_pdf.elements.text import (  # Ensure TextElement is imported for type hint if not in TYPE_CHECKING
            TextElement,
        )

        new_text_elements: List["TextElement"] = []
        if not self.elements:  # Accesses self._elements via property
            return ElementCollection([])

        page_context_for_adding: Optional["Page"] = None
        if add_to_page:
            # Try to determine a consistent page context if adding elements
            first_valid_region_with_page = next(
                (
                    el
                    for el in self.elements
                    if isinstance(el, Region) and hasattr(el, "page") and el.page is not None
                ),
                None,
            )
            if first_valid_region_with_page:
                page_context_for_adding = first_valid_region_with_page.page
            else:
                logger.warning(
                    "Cannot add TextElements to page: No valid Region with a page attribute found in collection, or first region's page is None."
                )
                add_to_page = False  # Disable adding if no valid page context can be determined

        for element in self.elements:  # Accesses self._elements via property/iterator
            if isinstance(element, Region):
                text_el = element.to_text_element(
                    text_content=text_content_func,
                    source_label=source_label,
                    object_type=object_type,
                    default_font_size=default_font_size,
                    default_font_name=default_font_name,
                    confidence=confidence,
                )
                new_text_elements.append(text_el)

                if add_to_page:
                    if not hasattr(text_el, "page") or text_el.page is None:
                        logger.warning(
                            f"TextElement created from region {element.bbox} has no page attribute. Cannot add to page."
                        )
                        continue

                    if page_context_for_adding and text_el.page == page_context_for_adding:
                        if (
                            hasattr(page_context_for_adding, "_element_mgr")
                            and page_context_for_adding._element_mgr is not None
                        ):
                            add_as_type = (
                                "words"
                                if object_type == "word"
                                else "chars" if object_type == "char" else object_type
                            )
                            page_context_for_adding._element_mgr.add_element(
                                text_el, element_type=add_as_type
                            )
                        else:
                            page_num_str = (
                                str(page_context_for_adding.page_number)
                                if hasattr(page_context_for_adding, "page_number")
                                else "N/A"
                            )
                            logger.error(
                                f"Page context for region {element.bbox} (Page {page_num_str}) is missing '_element_mgr'. Cannot add TextElement."
                            )
                    elif page_context_for_adding and text_el.page != page_context_for_adding:
                        current_page_num_str = (
                            str(text_el.page.page_number)
                            if hasattr(text_el.page, "page_number")
                            else "Unknown"
                        )
                        context_page_num_str = (
                            str(page_context_for_adding.page_number)
                            if hasattr(page_context_for_adding, "page_number")
                            else "N/A"
                        )
                        logger.warning(
                            f"TextElement for region {element.bbox} from page {current_page_num_str} "
                            f"not added as it's different from collection's inferred page context {context_page_num_str}."
                        )
                    elif not page_context_for_adding:
                        logger.warning(
                            f"TextElement for region {element.bbox} created, but no page context was determined for adding."
                        )
            else:
                logger.warning(f"Skipping element {type(element)}, not a Region.")

        if add_to_page and page_context_for_adding:
            page_num_str = (
                str(page_context_for_adding.page_number)
                if hasattr(page_context_for_adding, "page_number")
                else "N/A"
            )
            logger.info(
                f"Created and added {len(new_text_elements)} TextElements to page {page_num_str}."
            )
        elif add_to_page and not page_context_for_adding:
            logger.info(
                f"Created {len(new_text_elements)} TextElements, but could not add to page as page context was not determined or was inconsistent."
            )
        else:  # add_to_page is False
            logger.info(f"Created {len(new_text_elements)} TextElements (not added to page).")

        return ElementCollection(new_text_elements)

    def trim(
        self,
        padding: int = 1,
        threshold: float = 0.95,
        resolution: Optional[float] = None,
        show_progress: bool = True,
    ) -> "ElementCollection":
        """
        Trim visual whitespace from each region in the collection.

        Applies the trim() method to each element in the collection,
        returning a new collection with the trimmed regions.

        Args:
            padding: Number of pixels to keep as padding after trimming (default: 1)
            threshold: Threshold for considering a row/column as whitespace (0.0-1.0, default: 0.95)
            resolution: Resolution for image rendering in DPI (default: uses global options, fallback to 144 DPI)
            show_progress: Whether to show a progress bar for the trimming operation

        Returns:
            New ElementCollection with trimmed regions
        """
        # Apply global options as defaults
        import natural_pdf

        if resolution is None:
            if natural_pdf.options.image.resolution is not None:
                resolution = natural_pdf.options.image.resolution
            else:
                resolution = 144  # Default resolution when none specified

        return self.apply(
            lambda element: element.trim(
                padding=padding, threshold=threshold, resolution=resolution
            ),
            show_progress=show_progress,
        )

    def clip(
        self,
        obj: Optional[Any] = None,
        left: Optional[float] = None,
        top: Optional[float] = None,
        right: Optional[float] = None,
        bottom: Optional[float] = None,
    ) -> "ElementCollection":
        """
        Clip each element in the collection to the specified bounds.

        This method applies the clip operation to each individual element,
        returning a new collection with the clipped elements.

        Args:
            obj: Optional object with bbox properties (Region, Element, TextElement, etc.)
            left: Optional left boundary (x0) to clip to
            top: Optional top boundary to clip to
            right: Optional right boundary (x1) to clip to
            bottom: Optional bottom boundary to clip to

        Returns:
            New ElementCollection containing the clipped elements

        Examples:
            # Clip each element to another region's bounds
            clipped_elements = collection.clip(container_region)

            # Clip each element to specific coordinates
            clipped_elements = collection.clip(left=100, right=400)

            # Mix object bounds with specific overrides
            clipped_elements = collection.clip(obj=container, bottom=page.height/2)
        """
        # --- NEW BEHAVIOUR: support per-element clipping with sequences --- #
        from collections.abc import Sequence  # Local import to avoid top-level issues

        # Detect if *obj* is a sequence meant to map one-to-one with the elements
        clip_objs = None  # type: Optional[List[Any]]
        if isinstance(obj, ElementCollection):
            clip_objs = obj.elements
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            clip_objs = list(obj)

        if clip_objs is not None:
            if len(clip_objs) != len(self._elements):
                raise ValueError(
                    f"Number of clipping objects ({len(clip_objs)}) does not match number of "
                    f"elements in collection ({len(self._elements)})."
                )

            clipped_elements = [
                el.clip(
                    obj=clip_obj,
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                )
                for el, clip_obj in zip(self._elements, clip_objs)
            ]
            return ElementCollection(clipped_elements)

        # Fallback to original behaviour: apply same clipping parameters to all elements
        return self.apply(
            lambda element: element.clip(obj=obj, left=left, top=top, right=right, bottom=bottom)
        )

    # ------------------------------------------------------------------
    # NEW METHOD: apply_ocr for collections (supports custom function)
    # ------------------------------------------------------------------
    def apply_ocr(
        self,
        function: Optional[Callable[["Region"], Optional[str]]] = None,
        *,
        show_progress: bool = True,
        **kwargs,
    ) -> "ElementCollection":
        """Apply OCR to every element in the collection.

        This is a convenience wrapper that simply iterates over the collection
        and calls ``el.apply_ocr(...)`` on each item.

        Two modes are supported depending on the arguments provided:

        1. **Built-in OCR engines** – pass parameters like ``engine='easyocr'``
           or ``languages=['en']`` and each element delegates to the global
           OCRManager.
        2. **Custom function** – pass a *callable* via the ``function`` keyword
           (alias ``ocr_function`` also recognised).  The callable will receive
           the element/region and must return the recognised text (or ``None``).
           Internally this is forwarded through the element's own
           :py:meth:`apply_ocr` implementation, so the behaviour mirrors the
           single-element API.

        Parameters
        ----------
        function : callable, optional
            Custom OCR function to use instead of the built-in engines.
        show_progress : bool, default True
            Display a tqdm progress bar while processing.
        **kwargs
            Additional parameters forwarded to each element's ``apply_ocr``.

        Returns
        -------
        ElementCollection
            *Self* for fluent chaining.
        """
        # Alias for backward-compatibility
        if function is None and "ocr_function" in kwargs:
            function = kwargs.pop("ocr_function")

        def _process(el):
            if hasattr(el, "apply_ocr"):
                if function is not None:
                    return el.apply_ocr(function=function, **kwargs)
                else:
                    return el.apply_ocr(**kwargs)
            else:
                logger.warning(
                    f"Element of type {type(el).__name__} does not support apply_ocr. Skipping."
                )
                return el

        # Use collection's apply helper for optional progress bar
        self.apply(_process, show_progress=show_progress)
        return self

    # ------------------------------------------------------------------
