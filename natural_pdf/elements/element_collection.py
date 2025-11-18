import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    MutableSequence,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

from pdfplumber.utils.geometry import get_bbox_overlap, objects_to_bbox

# New Imports
from pdfplumber.utils.text import TEXTMAP_KWARGS, WORD_EXTRACTOR_KWARGS, chars_to_textmap
from PIL import Image

from natural_pdf.collections.mixins import ApplyMixin, DirectionalCollectionMixin

# Add Visualizable import
from natural_pdf.core.highlighter_utils import resolve_highlighter
from natural_pdf.core.interfaces import SupportsBBox, SupportsElement, SupportsGeometry
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.elements.base import Element
from natural_pdf.elements.mixins.classification_batch_mixin import ClassificationBatchMixin
from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement
from natural_pdf.export.mixin import ExportMixin
from natural_pdf.ocr.utils import _apply_ocr_correction_to_elements
from natural_pdf.selectors.host_mixin import delegate_signature
from natural_pdf.utils.color_utils import format_color_value

# Potentially lazy imports for optional dependencies needed in save_pdf
try:
    import pikepdf  # type: ignore[import]
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
from natural_pdf.services.base import resolve_service
from natural_pdf.services.delegates import attach_capability
from natural_pdf.services.methods import describe_methods as _describe_methods
from natural_pdf.services.methods import navigation_methods as _navigation_methods

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ipywidgets import DOMWidget

    from natural_pdf.classification.results import ClassificationResult
    from natural_pdf.core.page import Page
    from natural_pdf.elements.region import Region
    from natural_pdf.elements.text import TextElement  # Ensure TextElement is imported
else:  # pragma: no cover - fallback when ipywidgets is unavailable at runtime
    DOMWidget = Any  # type: ignore[assignment]

T = TypeVar("T", bound=SupportsGeometry)
P = TypeVar("P", bound="Page")


class ElementCollection(
    Generic[T],
    ApplyMixin,
    ExportMixin,
    ClassificationBatchMixin,
    DirectionalCollectionMixin,
    Visualizable,
    MutableSequence[T],
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
        self._elements: List[T] = list(elements)

    @delegate_signature(_navigation_methods.below)
    def below(self, *args, **kwargs) -> "ElementCollection":
        return self.apply(lambda element: element.below(*args, **kwargs))

    @delegate_signature(_navigation_methods.above)
    def above(self, *args, **kwargs) -> "ElementCollection":
        return self.apply(lambda element: element.above(*args, **kwargs))

    @delegate_signature(_navigation_methods.left)
    def left(self, *args, **kwargs) -> "ElementCollection":
        return self.apply(lambda element: element.left(*args, **kwargs))

    @delegate_signature(_navigation_methods.right)
    def right(self, *args, **kwargs) -> "ElementCollection":
        return self.apply(lambda element: element.right(*args, **kwargs))

    def _supports_list(self) -> List[SupportsElement]:
        """Ensure elements satisfy SupportsElement protocol."""
        for element in self._elements:
            if not isinstance(element, SupportsElement):
                raise TypeError(
                    f"Element {element!r} does not satisfy SupportsElement protocol; "
                    "ensure collections are homogeneous before calling this operation."
                )
        return cast(List[SupportsElement], self._elements)

    def _get_render_specs(
        self,
        mode: Literal["show", "render"] = "show",
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        highlights: Optional[List[Dict[str, Any]]] = None,
        crop: Union[bool, int, str, "Region", Literal["wide"]] = False,
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
            crop: Cropping mode (False, True, int for padding, 'wide', or Region)
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

        # Check for FlowRegions which need special handling
        from natural_pdf.flows.region import FlowRegion

        flow_regions = []
        regular_elements = []

        for elem in self._elements:
            if isinstance(elem, FlowRegion):
                flow_regions.append(elem)
            else:
                regular_elements.append(elem)

        # Start with specs from FlowRegions (they handle their own multi-page rendering)
        all_specs = []
        specs_by_page = {}  # Track specs by page for merging

        for flow_region in flow_regions:
            # FlowRegions have their own _get_render_specs method
            flow_specs = flow_region._get_render_specs(
                mode=mode,
                color=color,
                highlights=highlights,
                crop=crop,
                crop_bbox=crop_bbox,
                **kwargs,
            )
            for spec in flow_specs:
                # Check if we already have a spec for this page
                if spec.page in specs_by_page:
                    # Merge highlights into existing spec
                    existing_spec = specs_by_page[spec.page]
                    # Add all highlights from this spec to the existing one
                    existing_spec.highlights.extend(spec.highlights)
                    # Merge crop bbox if needed
                    if spec.crop_bbox and not existing_spec.crop_bbox:
                        existing_spec.crop_bbox = spec.crop_bbox
                    elif spec.crop_bbox and existing_spec.crop_bbox:
                        # Expand crop bbox to include both
                        x0 = min(spec.crop_bbox[0], existing_spec.crop_bbox[0])
                        y0 = min(spec.crop_bbox[1], existing_spec.crop_bbox[1])
                        x1 = max(spec.crop_bbox[2], existing_spec.crop_bbox[2])
                        y1 = max(spec.crop_bbox[3], existing_spec.crop_bbox[3])
                        existing_spec.crop_bbox = (x0, y0, x1, y1)
                else:
                    # First spec for this page
                    all_specs.append(spec)
                    specs_by_page[spec.page] = spec

        # Group regular elements by page
        elements_by_page = {}
        for elem in regular_elements:
            if hasattr(elem, "page"):
                page = elem.page
                if page not in elements_by_page:
                    elements_by_page[page] = []
                elements_by_page[page].append(elem)

        if not elements_by_page and not flow_regions:
            return []

        # Create or update RenderSpec for each page with regular elements
        for page, page_elements in elements_by_page.items():
            # Check if we already have a spec for this page from FlowRegions
            existing_spec = None
            for spec in all_specs:
                if spec.page == page:
                    existing_spec = spec
                    break

            if existing_spec:
                # We'll add to the existing spec
                spec = existing_spec
            else:
                # Create new spec for this page
                spec = RenderSpec(page=page)
                all_specs.append(spec)

            # Handle cropping
            if crop_bbox:
                spec.crop_bbox = crop_bbox
            elif crop:
                # Calculate bounds of elements on this page
                x_coords = []
                y_coords = []
                for elem in page_elements:
                    if hasattr(elem, "bbox") and elem.bbox:
                        x0, y0, x1, y1 = elem.bbox
                        x_coords.extend([x0, x1])
                        y_coords.extend([y0, y1])

                if x_coords and y_coords:
                    content_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

                    if crop is True:
                        # Tight crop to content bounds
                        spec.crop_bbox = content_bbox
                    elif isinstance(crop, (int, float)):
                        # Add padding around content
                        padding = float(crop)
                        x0, y0, x1, y1 = content_bbox
                        spec.crop_bbox = (
                            max(0, x0 - padding),
                            max(0, y0 - padding),
                            min(page.width, x1 + padding),
                            min(page.height, y1 + padding),
                        )
                    elif crop == "wide":
                        # Full page width, cropped vertically to content
                        spec.crop_bbox = (0, content_bbox[1], page.width, content_bbox[3])
                    elif hasattr(crop, "bbox"):
                        # Crop to another region's bounds
                        bbox_source = cast(SupportsBBox, crop)
                        spec.crop_bbox = bbox_source.bbox

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

        return all_specs

    def _get_highlighter(self):
        """Get the highlighting service for rendering.

        For ElementCollection, we get it from the first element's page.
        """
        if not self._elements:
            raise RuntimeError("Cannot get highlighter from empty ElementCollection")

        return resolve_highlighter(self._elements)

    def __len__(self) -> int:
        """Get the number of elements in the collection."""
        return len(self._elements)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> "ElementCollection[T]": ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T, "ElementCollection[T]"]:
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

    def __add__(self, other: Union["ElementCollection[T]", T]) -> "ElementCollection[T]":
        if isinstance(other, ElementCollection):
            merged_elements = list(self._elements) + list(other)
            return ElementCollection(merged_elements)
        return ElementCollection(list(self._elements) + [other])

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...

    def __setitem__(self, index: Union[int, slice], value: Union[T, Iterable[T]]) -> None:
        self._elements[index] = value  # type: ignore[assignment]

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self._elements[index]

    def insert(self, index: int, value: T) -> None:
        self._elements.insert(index, value)

    @property
    def elements(self) -> List[T]:
        """Get the elements in this collection."""
        return self._elements

    @property
    def first(self) -> Optional[T]:
        """Get the first element in the collection."""
        return self._elements[0] if self._elements else None

    @property
    def last(self) -> Optional[T]:
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

        typed_elements = self._supports_list()
        first_page_idx = typed_elements[0].page.index

        # Check if any element is on a different page
        return any(e.page.index != first_page_idx for e in typed_elements)

    def _are_on_multiple_pdfs(self) -> bool:
        """
        Check if elements in this collection span multiple PDFs.

        Returns:
            True if elements are from different PDFs, False otherwise
        """
        if not self._elements:
            return False

        typed_elements = self._supports_list()
        first_pdf = typed_elements[0].page.pdf

        # Check if any element is from a different PDF
        return any(e.page.pdf is not first_pdf for e in typed_elements)

    def highest(self) -> Optional[T]:
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

        typed_elements = self._supports_list()
        return cast(T, min(typed_elements, key=lambda e: e.top))

    def lowest(self) -> Optional[T]:
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

        typed_elements = self._supports_list()
        return cast(T, max(typed_elements, key=lambda e: e.bottom))

    def leftmost(self) -> Optional[T]:
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

        typed_elements = self._supports_list()
        return cast(T, min(typed_elements, key=lambda e: e.x0))

    def rightmost(self) -> Optional[T]:
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

        typed_elements = self._supports_list()
        return cast(T, max(typed_elements, key=lambda e: e.x1))

    def exclude_regions(self, regions: List["Region"]) -> "ElementCollection[T]":
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
        separator: str = " ",
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
            separator: String to join text from elements. Default is a single space.
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
        # Check if we have any elements at all
        if not self._elements:
            return ""

        typed_elements = self._supports_list()

        # Check if all elements are TextElements with character data
        text_elements_with_chars = [
            el
            for el in typed_elements
            if isinstance(el, TextElement) and hasattr(el, "_char_dicts") and el._char_dicts
        ]

        # If we have a mixed collection (Regions, TextElements without chars, etc),
        # use a simpler approach: call extract_text on each element
        if len(text_elements_with_chars) < len(typed_elements):
            # Mixed collection - extract text from each element
            element_texts = []

            # Sort elements by position first
            sorted_elements = sorted(
                typed_elements,
                key=lambda el: (
                    el.page.index,
                    el.top,
                    el.x0,
                ),
            )

            for el in sorted_elements:
                if hasattr(el, "extract_text"):
                    # Call extract_text on the element (works for TextElement, Region, etc)
                    text = el.extract_text(**kwargs)
                    if text:
                        element_texts.append(text)
                elif hasattr(el, "text"):
                    # Fallback to text property if available
                    text = getattr(el, "text", "")
                    if text:
                        element_texts.append(text)

            return separator.join(element_texts)

        # All elements are TextElements with char data - use the original approach
        text_elements = text_elements_with_chars

        # Collect all character dictionaries
        all_char_dicts = []
        for el in text_elements:
            all_char_dicts.extend(getattr(el, "_char_dicts", []))

        if not all_char_dicts:
            # Handle case where elements exist but have no char dicts
            logger.debug(
                "ElementCollection.extract_text: No character dictionaries found in TextElements."
            )
            # Sort elements by position before joining
            sorted_text_elements = sorted(
                text_elements,
                key=lambda el: (el.page.index, el.top, el.x0),
            )
            return separator.join(
                getattr(el, "text", "") for el in sorted_text_elements
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
                # chars_to_textmap constructs a layout-aware text map from the supplied
                # character dictionaries, letting pdfplumber cluster characters into
                # rows/columns using the provided tolerances.
                # Sort chars by document order (page, top, x0)
                # Need page info on char dicts for multi-page collections
                # Assuming char dicts have 'page_number' from element creation
                all_char_dicts.sort(
                    key=lambda c: (c.get("page_number", 0), c.get("top", 0), c.get("x0", 0))
                )
                textmap = chars_to_textmap(all_char_dicts, **layout_kwargs)
                result = textmap.as_string
            except Exception as e:
                raise RuntimeError(
                    "ElementCollection.extract_text failed to build layout textmap"
                ) from e

        else:
            # Default: Simple join without layout
            logger.debug("ElementCollection.extract_text: Using simple join (layout=False).")
            result = separator.join(el.extract_text() for el in text_elements)

            # # Sort chars by document order (page, top, x0)
            # all_char_dicts.sort(
            #     key=lambda c: (c.get("page_number", 0), c.get("top", 0), c.get("x0", 0))
            # )
            # # Simple join of character text
            # result = "".join(c.get("text", "") for c in all_char_dicts)

        # Determine final strip flag – same rule as global helper unless caller overrides
        strip_text = strip if strip is not None else (not use_layout)

        if strip_text and isinstance(result, str):
            result = "\n".join(line.rstrip() for line in result.splitlines()).strip()

        return result

    def merge(self) -> "Region":
        """
        Merge all elements into a single region encompassing their bounding box.

        Unlike dissolve() which only connects touching elements, merge() creates
        a single region that spans from the minimum to maximum coordinates of all
        elements, regardless of whether they touch.

        Returns:
            A single Region object encompassing all elements

        Raises:
            ValueError: If the collection is empty or elements have no valid bounding boxes

        Example:
            ```python
            # Find scattered form fields and merge into one region
            fields = pdf.find_all('text:contains(Name|Date|Phone)')
            merged_region = fields.merge()

            # Extract all text from the merged area
            text = merged_region.extract_text()
            ```
        """
        if not self._elements:
            raise ValueError("Cannot merge an empty ElementCollection")

        # Collect all bounding boxes
        bboxes = []
        page = None

        for elem in self._elements:
            if hasattr(elem, "bbox") and elem.bbox:
                bboxes.append(elem.bbox)
                # Get the page from the first element that has one
                if page is None and hasattr(elem, "page"):
                    page = elem.page

        if not bboxes:
            raise ValueError("No elements with valid bounding boxes to merge")

        if page is None:
            raise ValueError("Cannot determine page for merged region")

        # Find min/max coordinates
        x_coords = []
        y_coords = []

        for bbox in bboxes:
            x0, y0, x1, y1 = bbox
            x_coords.extend([x0, x1])
            y_coords.extend([y0, y1])

        # Create encompassing bounding box
        merged_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

        # Create and return the merged region
        from natural_pdf.elements.region import Region

        return Region(page, merged_bbox)

    def sort(
        self, key: Optional[Callable[[T], Any]] = None, reverse: bool = False
    ) -> "ElementCollection[T]":
        """
        Sort elements by the given key function.

        Args:
            key: Function to generate a key for sorting
            reverse: Whether to sort in descending order

        Returns:
            Self for method chaining
        """
        self._elements.sort(key=key, reverse=reverse)  # type: ignore[arg-type]
        return self

    def exclude(self):
        """
        Excludes all elements in the collection from their respective pages.

        Since a collection can span multiple pages, this method iterates through
        all elements and calls exclude() on each one individually.

        Each element type is handled appropriately:
        - Region elements exclude everything within their bounds
        - Text/other elements exclude only the specific element, not the area

        Returns:
            Self for method chaining
        """
        for element in self._elements:
            exclude_method = getattr(element, "exclude", None)
            if callable(exclude_method):
                exclude_method()
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
    ) -> Optional[Image.Image]:
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
            return None  # Nothing to add

        # Get page and highlighter from the first element (assume uniform page)
        first_element = self._elements[0]
        if not hasattr(first_element, "page") or not hasattr(first_element.page, "_highlighter"):
            raise RuntimeError(
                "Cannot highlight collection: elements lack page/highlighter access."
            )

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
                raise ValueError(f"Highlight data missing geometry (bbox or polygon): {data}")

        return None

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
            else:
                get_specs = getattr(element, "get_highlight_specs", None)
                if not callable(get_specs):
                    continue
                specs = get_specs()
                if not isinstance(specs, Sequence):
                    continue
                for spec in specs:
                    if "page" in spec and hasattr(spec["page"], "_highlighter"):
                        highlighter = spec["page"]._highlighter
                        break
                if highlighter:
                    break

        if not highlighter:
            raise RuntimeError(
                "Cannot determine highlight colors: HighlightingService not accessible from elements."
            )

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
                # Mixed types: require explicit guidance
                type_names_str = ", ".join(sorted(list(element_types)))
                raise ValueError(
                    f"Cannot auto-highlight mixed element types ({type_names_str}). "
                    "Provide an explicit label/group_by or distinct=True."
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
            raise RuntimeError(
                f"Cannot highlight element, missing 'page' attribute or page lacks highlighter: {element}"
            )

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
            raise ValueError(f"Cannot highlight element, no bbox or polygon found: {element}")

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
            group_key: Any = None
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
            except AttributeError as exc:
                raise AttributeError(
                    f"Attribute '{group_by}' not found on element {element}"
                ) from exc
            except TypeError as exc:  # Handle unhashable types
                raise TypeError(
                    f"Attribute value for '{group_by}' on {element} is unhashable ({type(group_key)})"
                ) from exc

        # Highlight each group
        for group_key, group_elements in grouped_elements.items():
            if not group_elements:
                continue

            # Determine the label for this group
            first_element = group_elements[0]  # Use first element for formatting
            group_label = None
            if label_format:
                try:
                    element_attrs = first_element.__dict__.copy()
                    element_attrs[group_by] = group_key
                    group_label = label_format.format(**element_attrs)
                except KeyError as e:
                    raise KeyError(f"Invalid key '{e}' in label_format '{label_format}'") from e
                except Exception as format_e:
                    raise ValueError(
                        f"Error formatting label '{label_format}': {format_e}"
                    ) from format_e
            else:
                group_label = str(group_key)

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
                raise RuntimeError(
                    f"Page {getattr(page, 'number', '?')} has no highlighter service"
                )

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
                    except Exception:
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
                        except Exception:
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
            raise RuntimeError("Failed to render any pages")

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

        image_options = cast(Any, natural_pdf.options.image)

        # Use global options if parameters are not explicitly set
        if width is None:
            width = getattr(image_options, "width", None)
        if width is None:
            width = 1024  # Fallback width

        if resolution is None:
            image_resolution = getattr(image_options, "resolution", None)
            if image_resolution is not None:
                resolution = image_resolution
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
            group_key: Any
            try:
                group_key = getattr(element, group_by, None)
            except AttributeError:
                group_key = f"Missing '{group_by}'"

            if group_key is None:
                group_key = f"Missing '{group_by}'"

            if isinstance(group_key, (list, dict)):
                group_key = str(group_key)

            try:
                grouped_elements.setdefault(group_key, []).append(element)
            except TypeError:
                # Unhashable key, fall back to string repr
                fallback_key = str(group_key)
                grouped_elements.setdefault(fallback_key, []).append(element)

        return grouped_elements

    def _format_group_label(
        self, group_key: Any, label_format: Optional[str], sample_element: T, group_by_attr: str
    ) -> str:
        """Formats the label for a group based on the key and format string."""
        # Format the group_key if it's a color attribute
        formatted_key = format_color_value(group_key, attr_name=group_by_attr)

        if label_format:
            try:
                element_attrs = sample_element.__dict__.copy()
                # Use the formatted key in the attributes
                element_attrs[group_by_attr] = formatted_key  # Ensure key is present
                return label_format.format(**element_attrs)
            except KeyError as e:
                raise KeyError(f"Invalid key '{e}' in label_format '{label_format}'") from e
            except Exception as format_e:
                raise ValueError(
                    f"Error formatting label '{label_format}': {format_e}"
                ) from format_e
        else:
            return formatted_key

    def _get_element_highlight_params(
        self, element: T, annotate: Optional[List[str]]
    ) -> Optional[Dict]:
        """Extracts common parameters needed for highlighting a single element."""
        # For FlowRegions and other complex elements, use highlighting protocol
        get_specs = getattr(element, "get_highlight_specs", None)
        base_data: Dict[str, Any]
        if callable(get_specs):
            specs = get_specs()
            if not isinstance(specs, Sequence):
                raise TypeError(f"Highlight protocol on {element} returned non-sequence specs")
            if not specs:
                raise ValueError(f"Element {element} returned no highlight specs")

            # For now, we'll use the first spec for the prepared data
            # The actual rendering will use all specs
            first_spec = specs[0]
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
        else:
            page = getattr(element, "page", None)
            if page is None:
                raise AttributeError(
                    f"Element {element} has no page attribute and no highlighting protocol"
                )

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
                raise ValueError(f"Cannot prepare highlight, no bbox or polygon for {element}")

        # Extract attributes if requested
        if annotate:
            for attr_name in annotate:
                attr_value = getattr(element, attr_name, None)
                if attr_value is not None:
                    base_data["attributes_to_draw"][attr_name] = attr_value

        return base_data

    def viewer(self, title: Optional[str] = None) -> Optional["DOMWidget"]:
        """
        Creates and returns an interactive ipywidget showing ONLY the elements
        in this collection on their page background.

        Args:
            title: Optional title for the viewer window/widget.

        Returns:
            An InteractiveViewerWidget instance or None if elements lack page context.
        """
        if not self.elements:
            raise ValueError("Cannot generate interactive viewer for empty collection.")

        # Assume all elements are on the same page and have .page attribute
        try:
            page = self.elements[0].page
            # Check if the page object actually has the method
            if hasattr(page, "viewer") and callable(page.viewer):
                final_title = (
                    title or f"Interactive Viewer for Collection ({len(self.elements)} elements)"
                )
                page_viewer = cast(Any, page.viewer)
                return page_viewer(
                    elements_to_render=self.elements,
                    title=final_title,
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
    ) -> "ElementCollection":
        """
        Find the first matching element below each item in the collection.

        Args:
            selector: CSS-like selector string understood by descendant find methods.
            text: Text shortcut equivalent to ``selector='text:contains(...)'``. Accepts a single
                string or an iterable of strings.
            overlap: Optional overlap handling forwarded to region/page ``find`` helpers. When the
                downstream implementation does not accept the argument the call will retry without it.
            apply_exclusions: Whether exclusion regions should be honoured (default: True).
            regex: Whether to interpret text filters as regular expressions (default: False).
            case: Whether text comparisons should be case-sensitive (default: True).
            text_tolerance: Optional mapping of pdfplumber-style tolerance overrides applied while
                resolving matches.
            auto_text_tolerance: Optional overrides for automatic tolerance behaviour.
            reading_order: Whether matches are resolved in natural reading order (default: True).
            near_threshold: Maximum distance (in points) used by the ``:near`` pseudo-class.
            engine: Optional selector engine name registered with the selector provider.

        Returns:
            An ElementCollection built from the first match (if any) discovered beneath each element.
        """

        if selector is not None and text is not None:
            raise ValueError("Provide either 'selector' or 'text', not both.")
        if selector is None and text is None:
            raise ValueError("Provide either 'selector' or 'text'.")

        from natural_pdf.elements.element_collection import ElementCollection

        matches: List[Element] = []
        for element in self._elements:

            call_kwargs: Dict[str, Any] = {
                "selector": selector,
                "text": text,
                "apply_exclusions": apply_exclusions,
                "regex": regex,
                "case": case,
                "text_tolerance": text_tolerance,
                "auto_text_tolerance": auto_text_tolerance,
                "reading_order": reading_order,
                "near_threshold": near_threshold,
                "engine": engine,
            }

            if overlap is not None:
                call_kwargs["overlap"] = overlap

            # Remove keys with None values that downstream signatures may not accept
            filtered_kwargs = {
                key: value
                for key, value in call_kwargs.items()
                if value is not None
                or key in {"apply_exclusions", "regex", "case", "reading_order"}
            }

            try:
                found = element.find(**filtered_kwargs)  # type: ignore[misc]
            except TypeError:
                if "overlap" in filtered_kwargs:
                    filtered_kwargs.pop("overlap")
                    found = element.find(**filtered_kwargs)  # type: ignore[misc]
                else:
                    raise

            if found:
                if isinstance(found, ElementCollection):
                    matches.extend(found.elements)
                else:
                    matches.append(found)

        typed_matches = cast(List[SupportsElement], matches)
        return ElementCollection(typed_matches)

    @overload
    def find_all(
        self,
        *,
        text: Union[str, Sequence[str]],
        overlap: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
        engine: Optional[str] = None,
    ) -> "ElementCollection": ...

    @overload
    def find_all(
        self,
        selector: str,
        *,
        overlap: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
        engine: Optional[str] = None,
    ) -> "ElementCollection": ...

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
        Find all matching elements for every item in the collection and flatten the results.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for (equivalent to 'text:contains(...)'). Accepts a
                  single string or an iterable of strings (matches any value).
            overlap: How to determine if elements overlap: 'full' (fully inside),
                     'partial' (any overlap), or 'center' (center point inside).
                     Defaults to "full" when omitted.
            apply_exclusions: Whether to apply exclusion regions (default: True).
            regex: Whether to use regex for text search (`selector` or `text`) (default: False).
            case: Whether to do case-sensitive text search (`selector` or `text`) (default: True).
            text_tolerance: Optional mapping of tolerance overrides applied during selection.
            auto_text_tolerance: Optional overrides for automatic tolerance behaviour.
            reading_order: Whether to order results according to natural reading order (default: True).
            near_threshold: Maximum distance (in points) used by the ``:near`` pseudo-class.
            engine: Optional selector engine name registered with the selector provider.

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

            call_kwargs: Dict[str, Any] = {
                "selector": selector,
                "text": text,
                "apply_exclusions": apply_exclusions,
                "regex": regex,
                "case": case,
                "text_tolerance": text_tolerance,
                "auto_text_tolerance": auto_text_tolerance,
                "reading_order": reading_order,
                "near_threshold": near_threshold,
                "engine": engine,
            }

            if overlap is not None:
                call_kwargs["overlap"] = overlap

            # Remove optional entries that are None to satisfy varying signatures
            filtered_kwargs = {
                key: value
                for key, value in call_kwargs.items()
                if value is not None
                or key in {"apply_exclusions", "regex", "case", "reading_order"}
            }

            try:
                found_in_element: "ElementCollection" = element.find_all(**filtered_kwargs)  # type: ignore[misc]
            except TypeError:
                if "overlap" in filtered_kwargs:
                    filtered_kwargs.pop("overlap")
                    found_in_element = element.find_all(**filtered_kwargs)  # type: ignore[misc]
                else:
                    raise

            if found_in_element and found_in_element.elements:
                all_found_elements.extend(found_in_element.elements)
            # else:
            # Elements in the collection are expected to support find_all.
            # If an element type doesn't, an AttributeError will naturally occur,
            # or a more specific check/handling could be added here if needed.

        typed_elements = cast(List[SupportsElement], all_found_elements)
        return ElementCollection(typed_elements)

    def extract_each_text(
        self,
        order: Optional[Union[str, Callable[[T], Any]]] = None,
        *,
        newlines: bool = True,
        **kwargs,
    ) -> List[str]:
        """Return a list with the extracted text for every element.

        Parameters
        ----------
        order
            Controls the ordering of elements **before** extraction:

            * ``None`` (default) – keep the collection's current order.
            * ``callable`` – a function that will be used as ``key`` for :pyfunc:`sorted`.
            * ``"ltr"`` – left-to-right ordering (x0, then y-top).
            * ``"rtl"`` – right-to-left ordering (−x0, then y-top).
            * ``"natural"`` – natural reading order (y-top, then x0).

        Remaining keyword arguments are forwarded to each element's
        :py:meth:`extract_text` method.
        """

        def _page_index(elem: SupportsElement) -> int:
            page_obj = getattr(elem, "page", None)
            return getattr(page_obj, "index", 0) if page_obj is not None else 0

        # -- Determine ordering --------------------------------------------------
        elements: List[T] = list(self._elements)  # make a shallow copy we can sort

        if order is not None and len(elements) > 1:
            try:
                if callable(order):
                    elements.sort(key=order)
                elif isinstance(order, str):
                    preset = order.lower()
                    if preset in {"ltr", "left-to-right"}:
                        elements.sort(
                            key=lambda el: (
                                _page_index(cast(SupportsElement, el)),
                                getattr(el, "x0", 0),
                                getattr(el, "top", 0),
                            )
                        )
                    elif preset in {"rtl", "right-to-left"}:
                        elements.sort(
                            key=lambda el: (
                                _page_index(cast(SupportsElement, el)),
                                -getattr(el, "x0", 0),
                                getattr(el, "top", 0),
                            )
                        )
                    elif preset in {"natural", "tdlr", "top-down"}:
                        elements.sort(
                            key=lambda el: (
                                _page_index(cast(SupportsElement, el)),
                                getattr(el, "top", 0),
                                getattr(el, "x0", 0),
                            )
                        )
                    else:
                        # Unknown preset – silently ignore to keep original order
                        pass
            except Exception:
                # If anything goes wrong, fall back to original order
                pass

        # -- Extract ----------------------------------------------------------------
        return [
            el.extract_text(newlines=newlines, **kwargs) if el is not None else None  # type: ignore[arg-type]
            for el in elements
        ]

    def correct_ocr(
        self,
        transform: Callable[[Any], Optional[str]],
    ) -> "ElementCollection[T]":
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

        Returns:
            Self for method chaining.
        """
        # Delegate to the utility function
        _apply_ocr_correction_to_elements(
            elements=cast(Iterable[Element], self._elements),
            correction_callback=transform,
            caller_info=f"ElementCollection(len={len(self._elements)})",
        )
        return self  # Return self for chaining

    def remove_from_pages(self) -> int:
        """
        Remove all elements in this collection from their respective pages.

        This method removes elements from their respective pages via the public element APIs.
        It's particularly useful for removing OCR elements before applying new OCR.

        Returns:
            int: Number of elements successfully removed
        """
        if not self._elements:
            return 0

        removed_count = 0

        for element in self._elements:
            page_obj = getattr(element, "page", None)
            if page_obj is None or not hasattr(page_obj, "remove_element"):
                raise AttributeError(f"Element {element} has no page or page lacks remove_element")

            element_type = getattr(element, "object_type", None)
            if isinstance(element_type, str):
                normalized = element_type.lower()
                if normalized == "word":
                    element_type = "words"
                elif normalized == "char":
                    element_type = "chars"
                elif normalized == "rect":
                    element_type = "rects"
                elif normalized == "line":
                    element_type = "lines"
                elif normalized == "region":
                    element_type = "regions"

            success = bool(page_obj.remove_element(element, element_type=element_type))
            if success:
                removed_count += 1
            else:
                logger.debug("Failed to remove element from page collection: %s", element)

        return removed_count

    # Classification batch handled by ClassificationBatchMixin

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
            raise ValueError("No elements found in collection")

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
                    extractor = cast(Any, element)
                    element_data["content"] = extractor.extract_text(preserve_whitespace=True)
                except Exception as e:
                    logger.error(f"Error extracting text from element {i}: {e}")
                    element_data["content"] = ""

            # Save image if requested
            if include_images and image_dir and hasattr(element, "to_image"):
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
                    show_method = getattr(element, "show", None)
                    if callable(show_method):
                        show_method(path=str(image_path), resolution=image_resolution)
                    else:
                        raise AttributeError(
                            f"Element {i} lacks a show() method; cannot export image"
                        )

                    # Add relative path to data
                    element_data["image_path"] = str(Path(image_path).relative_to(image_dir.parent))
                except Exception as e:
                    logger.error(f"Error saving image for element {i}: {e}")
                    element_data["image_path"] = None

            analyses = getattr(element, "analyses", None)
            if isinstance(analyses, dict):
                for key in analysis_keys:
                    if key not in analyses:
                        raise KeyError(f"Analysis key '{key}' not found in element {i}")

                    analysis_result = analyses[key]

                    if hasattr(analysis_result, "to_dict"):
                        analysis_data = analysis_result.to_dict()
                    else:
                        try:
                            analysis_data = dict(analysis_result)
                        except (TypeError, ValueError):
                            analysis_data = {"raw_result": str(analysis_result)}

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

        new_text_elements: List[TextElement] = []
        if not self.elements:  # Accesses self._elements via property
            empty_collection = ElementCollection(cast(List[SupportsElement], new_text_elements))
            return cast(ElementCollection[TextElement], empty_collection)

        page_context_for_adding: Optional["Page"] = None
        if add_to_page:
            first_valid_region_with_page = next(
                (
                    el
                    for el in self.elements
                    if isinstance(el, Region) and hasattr(el, "page") and el.page is not None
                ),
                None,
            )
            if not first_valid_region_with_page:
                raise ValueError(
                    "Cannot add TextElements to page: no Region in this collection has a page."
                )
            page_context_for_adding = first_valid_region_with_page.page

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
                    if page_context_for_adding is None:
                        raise RuntimeError("Page context not initialized for adding TextElements")
                    if not hasattr(text_el, "page") or text_el.page is None:
                        raise ValueError(
                            f"TextElement created from region {element.bbox} has no page attribute."
                        )

                    if text_el.page == page_context_for_adding:
                        add_as_type = (
                            "words"
                            if object_type == "word"
                            else "chars" if object_type == "char" else object_type
                        )
                        if hasattr(page_context_for_adding, "add_element"):
                            page_context_for_adding.add_element(text_el, element_type=add_as_type)
                        else:
                            page_num_str = (
                                str(page_context_for_adding.page_number)
                                if hasattr(page_context_for_adding, "page_number")
                                else "N/A"
                            )
                            raise AttributeError(
                                f"Page context (Page {page_num_str}) does not expose add_element"
                            )
                    else:
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
                        raise ValueError(
                            f"TextElement for region {element.bbox} belongs to page {current_page_num_str}, "
                            f"but collection context is page {context_page_num_str}"
                        )
            else:
                raise TypeError(
                    f"ElementCollection.to_text_elements only supports Region elements, got {type(element)}"
                )

        if add_to_page and page_context_for_adding:
            page_num_str = (
                str(page_context_for_adding.page_number)
                if hasattr(page_context_for_adding, "page_number")
                else "N/A"
            )
            logger.info(
                f"Created and added {len(new_text_elements)} TextElements to page {page_num_str}."
            )
        else:  # add_to_page is False
            logger.info(f"Created {len(new_text_elements)} TextElements (not added to page).")

        typed_collection = ElementCollection(cast(List[SupportsElement], new_text_elements))
        return cast(ElementCollection[TextElement], typed_collection)

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

        image_options = cast(Any, natural_pdf.options.image)
        if resolution is None:
            option_resolution = getattr(image_options, "resolution", None)
            resolution = option_resolution if option_resolution is not None else 144

        def _trim_element(element: SupportsElement) -> SupportsElement:
            trim_method = getattr(element, "trim", None)
            if callable(trim_method):
                trimmed_element = trim_method(
                    padding=padding, threshold=threshold, resolution=resolution
                )
                return cast(SupportsElement, trimmed_element)
            return element

        trimmed = self.apply(_trim_element, show_progress=show_progress)
        if isinstance(trimmed, ElementCollection):
            return trimmed
        return ElementCollection(cast(List[SupportsElement], trimmed))

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

            clipped_elements: List[SupportsElement] = []
            for el, clip_obj in zip(self._elements, clip_objs):
                clip_method = getattr(el, "clip", None)
                if callable(clip_method):
                    clipped = clip_method(
                        obj=clip_obj,
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                    )
                    clipped_elements.append(cast(SupportsElement, clipped))
            return ElementCollection(clipped_elements)

        # Fallback to original behaviour: apply same clipping parameters to all elements
        def _clip_single(element: SupportsElement) -> SupportsElement:
            clip_method = getattr(element, "clip", None)
            if callable(clip_method):
                clipped_element = clip_method(
                    obj=obj, left=left, top=top, right=right, bottom=bottom
                )
                return cast(SupportsElement, clipped_element)
            return element

        clipped = self.apply(_clip_single)
        if isinstance(clipped, ElementCollection):
            return clipped
        return ElementCollection(cast(List[SupportsElement], clipped))

    def merge_connected(
        self,
        proximity_threshold: float = 5.0,
        merge_across_pages: bool = False,
        merge_non_regions: bool = False,
        text_separator: str = " ",
        preserve_order: bool = True,
    ) -> "ElementCollection":
        """
        Merge connected/adjacent regions in the collection into larger regions.

        This method identifies regions that are adjacent or overlapping (within a proximity
        threshold) and merges them into single regions. This is particularly useful for
        handling text that gets split due to font variations, accented characters, or
        other PDF rendering quirks.

        The method uses a graph-based approach (union-find) to identify connected components
        of regions and merges each component into a single region.

        Args:
            proximity_threshold: Maximum distance in points between regions to consider
                them connected. Default is 5.0 points. Use 0 for only overlapping regions.
            merge_across_pages: If True, allow merging regions from different pages.
                Default is False (only merge within same page).
            merge_non_regions: If True, attempt to merge non-Region elements by converting
                them to regions first. Default is False (skip non-Region elements).
            text_separator: String to use when joining text from merged regions.
                Default is a single space.
            preserve_order: If True, order merged text by reading order (top-to-bottom,
                left-to-right). Default is True.

        Returns:
            New ElementCollection containing the merged regions. Non-Region elements
            (if merge_non_regions=False) and elements that couldn't be merged are
            included unchanged.

        Example:
            ```python
            # Find all text regions with potential splits
            text_regions = page.find_all('region[type=text]')

            # Merge adjacent regions (useful for accented characters)
            merged = text_regions.merge_connected(proximity_threshold=2.0)

            # Extract clean text from merged regions
            for region in merged:
                print(region.extract_text())
            ```

        Note:
            - Regions are considered connected if their bounding boxes are within
              proximity_threshold distance of each other
            - The merged region's bbox encompasses all constituent regions
            - Text content is combined in reading order
            - Original metadata is preserved from the first region in each group
        """
        if not self._elements:
            return ElementCollection([])

        from natural_pdf.elements.region import Region

        # Separate Region and non-Region elements
        regions = []
        region_indices = []
        non_regions = []
        non_region_indices = []

        for i, elem in enumerate(self._elements):
            if isinstance(elem, Region):
                regions.append(elem)
                region_indices.append(i)
            else:
                non_regions.append(elem)
                non_region_indices.append(i)

        if not regions:
            # No regions to merge
            return ElementCollection(self._elements)

        # Group regions by page if not merging across pages
        page_groups = {}
        if not merge_across_pages:
            for region in regions:
                page = getattr(region, "page", None)
                if page is not None:
                    page_id = id(page)  # Use object id as unique identifier
                    if page_id not in page_groups:
                        page_groups[page_id] = []
                    page_groups[page_id].append(region)
                else:
                    # Region without page - treat as separate group
                    page_groups[id(region)] = [region]
        else:
            # All regions in one group
            page_groups = {0: regions}

        # Process each page group and collect merged regions
        all_merged_regions = []

        for page_id, page_regions in page_groups.items():
            if len(page_regions) == 1:
                # Only one region on this page, nothing to merge
                all_merged_regions.extend(page_regions)
                continue

            # Build adjacency graph using union-find
            parent = list(range(len(page_regions)))

            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            # Check all pairs of regions for connectivity
            for i in range(len(page_regions)):
                for j in range(i + 1, len(page_regions)):
                    if self._are_regions_connected(
                        page_regions[i], page_regions[j], proximity_threshold
                    ):
                        union(i, j)

            # Group regions by their connected component
            components = {}
            for i, region in enumerate(page_regions):
                root = find(i)
                if root not in components:
                    components[root] = []
                components[root].append(region)

            # Merge each component
            for component_regions in components.values():
                if len(component_regions) == 1:
                    # Single region, no merge needed
                    all_merged_regions.append(component_regions[0])
                else:
                    # Merge multiple regions
                    merged = self._merge_region_group(
                        component_regions, text_separator, preserve_order
                    )
                    all_merged_regions.append(merged)

        # Combine merged regions with non-regions (if any)
        # Reconstruct in original order as much as possible
        result_elements = []

        if not non_regions:
            # All elements were regions
            result_elements = all_merged_regions
        else:
            # Need to interleave merged regions and non-regions
            # This is a simplified approach - just append non-regions at the end
            # A more sophisticated approach would maintain relative ordering
            result_elements = all_merged_regions + non_regions

        return ElementCollection(result_elements)

    def _are_regions_connected(
        self, region1: "Region", region2: "Region", threshold: float
    ) -> bool:
        """Check if two regions are connected (adjacent or overlapping)."""
        bbox1 = region1.bbox
        bbox2 = region2.bbox
        vertical_gap: Optional[float] = None  # Reserved for future vertical-gap-aware merge logic

        # Check for overlap first
        overlap = get_bbox_overlap(bbox1, bbox2)
        if overlap is not None:
            return True

        # If no overlap and threshold is 0, regions are not connected
        if threshold == 0:
            return False

        # Check proximity - calculate minimum distance between bboxes
        # bbox format: (x0, top, x1, bottom)
        x0_1, top_1, x1_1, bottom_1 = bbox1
        x0_2, top_2, x1_2, bottom_2 = bbox2

        # Calculate horizontal distance
        if x1_1 < x0_2:
            h_dist = x0_2 - x1_1
        elif x1_2 < x0_1:
            h_dist = x0_1 - x1_2
        else:
            h_dist = 0  # Horizontally overlapping

        # Calculate vertical distance
        if bottom_1 < top_2:
            v_dist = top_2 - bottom_1
        elif bottom_2 < top_1:
            v_dist = top_1 - bottom_2
        else:
            v_dist = 0  # Vertically overlapping

        # ------------------------------------------------------------------
        # Decide connection logic based on vertical_gap parameter
        # ------------------------------------------------------------------
        if vertical_gap is not None:
            # Consider elements connected when they vertically stack within
            # the allowed gap **and** have some horizontal overlap
            horizontal_overlap = not (h_dist > 0)
            return horizontal_overlap and v_dist <= vertical_gap

        # Fallback to legacy Chebyshev distance using ``threshold``
        distance = max(h_dist, v_dist)
        return distance <= threshold

    def _merge_region_group(
        self, regions: List["Region"], text_separator: str, preserve_order: bool
    ) -> "Region":
        """Merge a group of connected regions into a single region."""
        if not regions:
            raise ValueError("Cannot merge empty region group")

        if len(regions) == 1:
            return regions[0]

        # Calculate merged bbox
        bboxes = [r.bbox for r in regions]
        x0s = [b[0] for b in bboxes]
        tops = [b[1] for b in bboxes]
        x1s = [b[2] for b in bboxes]
        bottoms = [b[3] for b in bboxes]

        merged_bbox = (min(x0s), min(tops), max(x1s), max(bottoms))

        # Use the page from the first region
        page = regions[0].page

        # Sort regions for text ordering if requested
        if preserve_order:
            # Sort by reading order: top-to-bottom, left-to-right
            sorted_regions = sorted(regions, key=lambda r: (r.top, r.x0))
        else:
            sorted_regions = regions

        # Merge text content
        text_parts = []
        for region in sorted_regions:
            try:
                text = region.extract_text()
                if text:
                    text_parts.append(text)
            except Exception:
                # Region might not have text extraction capability
                pass

        merged_text = text_separator.join(text_parts) if text_parts else None

        # Create merged region
        from natural_pdf.elements.region import Region

        merged_region = Region(
            page=page, bbox=merged_bbox, label=f"Merged ({len(regions)} regions)"
        )

        # Copy metadata from first region and add merge info
        if hasattr(regions[0], "metadata") and regions[0].metadata:
            merged_region.metadata = regions[0].metadata.copy()

        merged_region.metadata["merge_info"] = {
            "source_count": len(regions),
            "merged_text": merged_text,
            "source_bboxes": bboxes,
        }

        # If regions have region_type, preserve it if consistent
        region_types = set()
        for r in regions:
            if hasattr(r, "region_type") and r.region_type:
                region_types.add(r.region_type)

        if len(region_types) == 1:
            merged_region.region_type = region_types.pop()

        return merged_region

    def dissolve(
        self,
        padding: float = 2.0,
        *,
        vertical_gap: Optional[float] = None,
        vertical: Optional[bool] = False,
        geometry: Literal["rect", "polygon"] = "rect",
        group_by: Optional[List[str]] = None,
    ) -> "ElementCollection":
        """
        Merge connected elements based on proximity and grouping attributes.

        This method groups elements by specified attributes (if any), then finds
        connected components within each group based on a proximity threshold.
        Connected elements are merged by creating new Region objects with merged
        bounding boxes.

        Args:
            padding: Maximum chebyshev distance (in any direction) between
                elements to consider them connected **when ``vertical_gap`` is
                not provided**. Default 2.0 pt.

            vertical_gap: If given, switches to *stack-aware* dissolve:
                two elements are connected when their horizontal projections
                overlap (any amount) **and** the vertical distance between them
                is ≤ ``vertical_gap``.  This lets you combine multi-line labels
                that share the same column but have blank space between lines.

            vertical: If given, automatically sets vertical_gap to maximum to
                allow for easy vertical stacking.

            geometry: Type of geometry to use for merged regions. Currently only
                "rect" (bounding box) is supported. "polygon" will raise
                NotImplementedError.
            group_by: List of attribute names to group elements by before merging.
                Elements are grouped by exact attribute values (floats are rounded
                to 2 decimal places). If None, all elements are considered in the
                same group. Common attributes include 'size' (for TextElements),
                'font_family', 'fontname', etc.

        Returns:
            New ElementCollection containing the dissolved regions. All elements
            with bbox attributes are processed and converted to Region objects.

        Example:
            ```python
            # Dissolve elements that are close together
            dissolved = elements.dissolve(padding=5.0)

            # Group by font size before dissolving
            dissolved = elements.dissolve(padding=2.0, group_by=['size'])

            # Group by multiple attributes
            dissolved = elements.dissolve(
                padding=3.0,
                group_by=['size', 'font_family']
            )
            ```

        Note:
            - All elements with bbox attributes are processed
            - Float attribute values are rounded to 2 decimal places for grouping
            - The method uses Chebyshev distance (max of dx, dy) for proximity
            - Merged regions inherit the page from the first element in each group
            - Output is always Region objects, regardless of input element types
        """
        if geometry == "polygon":
            raise NotImplementedError("Polygon geometry is not yet supported for dissolve()")

        if geometry not in ["rect", "polygon"]:
            raise ValueError(f"Invalid geometry type: {geometry}. Must be 'rect' or 'polygon'")

        if vertical:
            vertical_gap = float("inf")

        from natural_pdf.elements.region import Region

        # Filter to elements with bbox (all elements that can be dissolved)
        elements_with_bbox: List[SupportsElement] = [
            cast(SupportsElement, elem)
            for elem in self._elements
            if hasattr(elem, "bbox") and elem.bbox
        ]

        if not elements_with_bbox:
            logger.debug("No elements with bbox found in collection for dissolve()")
            return ElementCollection([])

        # Group elements by specified attributes
        if group_by:
            grouped_elements = self._group_elements_by_attributes(elements_with_bbox, group_by)
        else:
            # All elements in one group
            grouped_elements = {(): elements_with_bbox}

        # Process each group and collect dissolved regions
        all_dissolved_regions = []

        for group_key, group_elements in grouped_elements.items():
            if not group_elements:
                continue

            logger.debug(f"Processing group {group_key} with {len(group_elements)} elements")

            # Find connected components within this group
            components = self._find_connected_components_elements(
                group_elements, padding, vertical_gap
            )

            # Merge each component
            for component_elements in components:
                if len(component_elements) == 1:
                    # Single element, convert to Region
                    elem = component_elements[0]
                    elem_type_label = getattr(elem, "type", type(elem).__name__)
                    region = Region(
                        page=elem.page,
                        bbox=elem.bbox,
                        label=f"Dissolved (1 {elem_type_label})",
                    )
                    # Copy relevant attributes from source element
                    self._copy_element_attributes_to_region(elem, region, group_by)
                    all_dissolved_regions.append(region)
                else:
                    # Merge multiple elements
                    merged = self._merge_elements_for_dissolve(component_elements, group_by)
                    all_dissolved_regions.append(merged)

        logger.debug(
            f"Dissolved {len(elements_with_bbox)} elements into {len(all_dissolved_regions)} regions"
        )

        return ElementCollection(cast(List[SupportsElement], all_dissolved_regions))

    def _group_elements_by_attributes(
        self, elements: Sequence[SupportsElement], group_by: Optional[List[str]]
    ) -> Dict[Tuple, List[SupportsElement]]:
        """Group elements by specified attributes."""
        if not group_by:
            return {(): list(elements)}

        groups: Dict[Tuple, List[SupportsElement]] = {}

        for element in elements:
            # Build group key from attribute values
            key_values = []
            for attr in group_by:
                value = None

                if hasattr(element, attr):
                    value = getattr(element, attr)
                else:
                    obj_dict = getattr(element, "_obj", None)
                    metadata = getattr(element, "metadata", None)
                    if isinstance(obj_dict, dict) and attr in obj_dict:
                        value = obj_dict[attr]
                    elif isinstance(metadata, dict) and attr in metadata:
                        value = metadata[attr]

                if isinstance(value, float):
                    value = round(value, 2)

                key_values.append(value)

            key = tuple(key_values)

            if key not in groups:
                groups[key] = []
            groups[key].append(element)

        return groups

    def _find_connected_components_elements(
        self,
        elements: List[SupportsElement],
        padding: float,
        vertical_gap: Optional[float] = None,
    ) -> List[List[SupportsElement]]:
        """Find connected components among elements using union-find."""
        if not elements:
            return []

        if len(elements) == 1:
            return [elements]

        # Build adjacency using union-find
        parent = list(range(len(elements)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Check all pairs of elements for connectivity
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                if self._are_elements_connected(elements[i], elements[j], padding, vertical_gap):
                    union(i, j)

        # Group elements by their connected component
        components = {}
        for i, element in enumerate(elements):
            root = find(i)
            if root not in components:
                components[root] = []
            components[root].append(element)

        return list(components.values())

    def _merge_elements_for_dissolve(
        self, elements: List[SupportsElement], group_by: Optional[List[str]] = None
    ) -> "Region":
        """Merge a group of elements for dissolve operation."""
        if not elements:
            raise ValueError("Cannot merge empty element group")

        if len(elements) == 1:
            elem = elements[0]
            from natural_pdf.elements.region import Region

            elem_type_label = getattr(elem, "type", type(elem).__name__)
            region = Region(
                page=elem.page, bbox=elem.bbox, label=f"Dissolved (1 {elem_type_label})"
            )
            self._copy_element_attributes_to_region(elem, region, group_by)
            return region

        # Calculate merged bbox
        bboxes = [e.bbox for e in elements]
        x0s = [b[0] for b in bboxes]
        tops = [b[1] for b in bboxes]
        x1s = [b[2] for b in bboxes]
        bottoms = [b[3] for b in bboxes]

        merged_bbox = (min(x0s), min(tops), max(x1s), max(bottoms))

        # Use the page from the first element
        page = elements[0].page

        # Count element types for label
        type_counts: Dict[str, int] = {}
        for elem in elements:
            elem_type = getattr(elem, "type", type(elem).__name__)
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1

        # Create label showing element types
        label_parts = []
        for elem_type, count in sorted(type_counts.items()):
            # Pluralize element type if count > 1
            type_label = elem_type + ("s" if count > 1 else "")
            label_parts.append(f"{count} {type_label}")
        label = f"Dissolved ({', '.join(label_parts)})"

        # Create merged region
        from natural_pdf.elements.region import Region

        merged_region = Region(page=page, bbox=merged_bbox, label=label)

        # Copy attributes from first element if they're consistent
        self._copy_element_attributes_to_region(elements[0], merged_region, group_by)

        # Check if all elements have the same region_type
        region_types = set()
        for elem in elements:
            elem_region_type = getattr(elem, "region_type", None)
            if elem_region_type:
                region_types.add(elem_region_type)

        # Handle region_type based on consistency
        if len(region_types) == 1:
            # All elements have the same region_type, preserve it
            merged_region.region_type = region_types.pop()
        elif len(region_types) > 1:
            # Multiple different region types, clear it
            merged_region.region_type = None

        # Add dissolve metadata
        merged_region.metadata["dissolve_info"] = {
            "source_count": len(elements),
            "source_bboxes": bboxes,
            "source_types": type_counts,
        }

        return merged_region

    def _are_elements_connected(
        self,
        elem1: SupportsElement,
        elem2: SupportsElement,
        threshold: float,
        vertical_gap: float | None,
    ) -> bool:
        """Check if two elements are connected (adjacent or overlapping)."""
        # Check if elements are on the same page
        # Handle edge cases where elements might not have a page attribute
        page1 = getattr(elem1, "page", None)
        page2 = getattr(elem2, "page", None)

        # If either element doesn't have a page, we can't compare pages
        # In this case, only consider them connected if both lack pages
        if page1 is None or page2 is None:
            if page1 is not page2:  # One has page, one doesn't
                return False
            # Both None - continue with proximity check
        elif page1 != page2:  # Both have pages but different
            return False

        bbox1 = elem1.bbox
        bbox2 = elem2.bbox

        # Check for overlap first
        overlap = get_bbox_overlap(bbox1, bbox2)
        if overlap is not None:
            return True

        # If no overlap and threshold is 0, elements are not connected
        if threshold == 0:
            return False

        # Check proximity - calculate minimum distance between bboxes
        # bbox format: (x0, top, x1, bottom)
        x0_1, top_1, x1_1, bottom_1 = bbox1
        x0_2, top_2, x1_2, bottom_2 = bbox2

        # Calculate horizontal distance
        if x1_1 < x0_2:
            h_dist = x0_2 - x1_1
        elif x1_2 < x0_1:
            h_dist = x0_1 - x1_2
        else:
            h_dist = 0  # Horizontally overlapping

        # Calculate vertical distance
        if bottom_1 < top_2:
            v_dist = top_2 - bottom_1
        elif bottom_2 < top_1:
            v_dist = top_1 - bottom_2
        else:
            v_dist = 0  # Vertically overlapping

        # Use Chebyshev distance (max of horizontal and vertical)
        # This creates a square proximity zone
        distance = max(h_dist, v_dist)

        if vertical_gap is not None:
            # 1. vertical distance ≤ vertical_gap
            # 2. horizontal ranges overlap OR touch
            h_overlap = (min(x1_1, x1_2) - max(x0_1, x0_2)) >= 0
            return h_overlap and v_dist <= vertical_gap

        return distance <= threshold

    def _copy_element_attributes_to_region(
        self,
        element: SupportsElement,
        region: "Region",
        group_by: Optional[List[str]] = None,
    ) -> None:
        """Copy relevant attributes from source element to region."""
        # Common text attributes to check
        text_attrs = [
            "size",
            "font_family",
            "fontname",
            "font_size",
            "font_name",
            "bold",
            "italic",
            "color",
            "text_color",
            "region_type",
        ]

        # If group_by is specified, prioritize those attributes
        attrs_to_check = (group_by or []) + text_attrs

        for attr in attrs_to_check:
            value = None

            # Try different ways to get the attribute
            if hasattr(element, attr):
                value = getattr(element, attr)
            else:
                obj_dict = getattr(element, "_obj", None)
                metadata = getattr(element, "metadata", None)
                if isinstance(obj_dict, dict) and attr in obj_dict:
                    value = obj_dict[attr]
                elif isinstance(metadata, dict) and attr in metadata:
                    value = metadata[attr]

            # Set the attribute on the region if we found a value
            if value is not None:
                # Map common attribute names
                if attr == "size" and not hasattr(region, "font_size"):
                    setattr(region, "font_size", value)
                elif attr == "fontname" and not hasattr(region, "font_name"):
                    setattr(region, "font_name", value)
                else:
                    setattr(region, attr, value)

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
           or ``languages=['en']`` and each element delegates to the shared
           engine provider.
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
            if not hasattr(el, "apply_ocr"):
                raise TypeError(f"Element of type {type(el).__name__} does not support apply_ocr")
            if function is not None:
                return el.apply_ocr(function=function, **kwargs)
            return el.apply_ocr(**kwargs)

        # Use collection's apply helper for optional progress bar
        self.apply(_process, show_progress=show_progress)
        return self

    def detect_checkboxes(
        self, *args, show_progress: bool = False, **kwargs
    ) -> "ElementCollection":
        """
        Detect checkboxes on all applicable elements in the collection.

        This method iterates through elements and calls detect_checkboxes on those
        that support it (Pages and Regions).

        Args:
            *args: Positional arguments to pass to detect_checkboxes.
            show_progress: Whether to show a progress bar during processing.
            **kwargs: Keyword arguments to pass to detect_checkboxes.

        Returns:
            A new ElementCollection containing all detected checkbox regions.
        """
        all_checkboxes = []

        def _process(el):
            if hasattr(el, "detect_checkboxes"):
                # Element supports checkbox detection
                result = el.detect_checkboxes(*args, **kwargs)
                if hasattr(result, "elements"):
                    # Result is a collection
                    all_checkboxes.extend(result.elements)
                elif isinstance(result, list):
                    # Result is a list
                    all_checkboxes.extend(result)
                elif result:
                    # Single result
                    all_checkboxes.append(result)
            return el

        # Use collection's apply helper for optional progress bar
        self.apply(_process, show_progress=show_progress, desc="Detecting checkboxes")

        return ElementCollection(all_checkboxes)

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public alias: combine
    # ------------------------------------------------------------------
    def combine(
        self,
        padding: float = 2.0,
        *,
        vertical_gap: Optional[float] = None,
        vertical: Optional[bool] = False,
        geometry: Literal["rect", "polygon"] = "rect",
        group_by: Optional[List[str]] = None,
    ) -> "ElementCollection":
        """Alias for :py:meth:`dissolve` – retained for discoverability.

        Many users find the verb *combine* more intuitive than *dissolve* when
        merging nearby or stacked elements into unified Regions.  The parameters
        are identical; see :py:meth:`dissolve` for full documentation.
        """

        return self.dissolve(
            padding=padding,
            vertical_gap=vertical_gap,
            vertical=vertical,
            geometry=geometry,
            group_by=group_by,
        )

    # ------------------------------------------------------------------
    # Describe/inspect helpers
    # ------------------------------------------------------------------
    @delegate_signature(_describe_methods.describe)
    def describe(self, **kwargs) -> Any:
        return _describe_methods.describe(self, **kwargs)

    @delegate_signature(_describe_methods.inspect)
    def inspect(self, limit: int = 30, **kwargs) -> Any:
        return _describe_methods.inspect(self, limit=limit, **kwargs)


attach_capability(ElementCollection, "describe")
