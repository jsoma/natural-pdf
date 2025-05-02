import logging
from collections.abc import MutableSequence
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
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from pdfplumber.utils.geometry import objects_to_bbox
from PIL import Image, ImageDraw, ImageFont

# New Imports
from pdfplumber.utils.text import TEXTMAP_KWARGS, WORD_EXTRACTOR_KWARGS, chars_to_textmap
from tqdm.auto import tqdm

from natural_pdf.classification.manager import ClassificationManager
from natural_pdf.classification.mixin import ClassificationMixin
from natural_pdf.collections.mixins import ApplyMixin, DirectionalCollectionMixin
from natural_pdf.core.pdf import PDF
from natural_pdf.elements.base import Element
from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement
from natural_pdf.export.mixin import ExportMixin
from natural_pdf.ocr import OCROptions
from natural_pdf.ocr.utils import _apply_ocr_correction_to_elements
from natural_pdf.selectors.parser import parse_selector, selector_to_filter_func

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.region import Region

T = TypeVar("T")
P = TypeVar("P", bound="Page")


class ElementCollection(
    Generic[T], ApplyMixin, ExportMixin, DirectionalCollectionMixin, MutableSequence
):
    """
    Collection of PDF elements with batch operations.
    """

    def __init__(self, elements: List[T]):
        """
        Initialize a collection of elements.

        Args:
            elements: List of Element objects
        """
        self._elements = elements or []

    def __len__(self) -> int:
        """Get the number of elements in the collection."""
        return len(self._elements)

    def __getitem__(self, index: int) -> "Element":
        """Get an element by index."""
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

    def extract_text(self, preserve_whitespace=True, use_exclusions=True, **kwargs) -> str:
        """
        Extract text from all TextElements in the collection, optionally using
        pdfplumber's layout engine if layout=True is specified.

        Args:
            preserve_whitespace: Deprecated. Use layout=False for simple joining.
            use_exclusions: Deprecated. Exclusions should be applied *before* creating
                          the collection or by filtering the collection itself.
            **kwargs: Additional layout parameters passed directly to pdfplumber's
                      `chars_to_textmap` function ONLY if `layout=True` is passed.
                      See Page.extract_text docstring for common parameters.
                      If `layout=False` or omitted, performs a simple join.

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
        include_attrs: Optional[List[str]] = None,
        replace: bool = False,
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
            color: Optional explicit color for the highlight (tuple/string). Applied
                   consistently if 'label' is provided or if grouping occurs.
            group_by: Optional attribute name present on the elements. If provided
                      (and 'label' is None), elements will be grouped based on the
                      value of this attribute, and each group will be highlighted
                      with a distinct label and color.
            label_format: Optional Python f-string to format the group label when
                          'group_by' is used. Can reference element attributes
                          (e.g., "Type: {region_type}, Conf: {confidence:.2f}").
                          If None, the attribute value itself is used as the label.
            distinct: If True, bypasses all grouping and highlights each element
                      individually with cycling colors (the previous default behavior).
                      (default: False)
            include_attrs: List of attribute names from the element to display directly
                           on the highlight itself (distinct from group label).
            replace: If True, existing highlights on the affected page(s)
                     are cleared before adding these highlights.
                     If False (default), highlights are appended to existing ones.

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
            include_attrs=include_attrs,
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
                "include_attrs": data["include_attrs"],
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
        include_attrs: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Determines the parameters for highlighting each element based on the strategy.

        Does not interact with the HighlightingService directly.

        Returns:
            List of dictionaries, each containing parameters for a single highlight
            (e.g., page_index, bbox/polygon, color, label, element, include_attrs, attributes_to_draw).
            Color and label determination happens here.
        """
        prepared_data = []
        if not self._elements:
            return prepared_data

        # Need access to the HighlightingService to determine colors correctly.
        highlighter = None
        first_element = self._elements[0]
        if hasattr(first_element, "page") and hasattr(first_element.page, "_highlighter"):
            highlighter = first_element.page._highlighter
        else:
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
                element_data = self._get_element_highlight_params(element, include_attrs)
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
                element_data = self._get_element_highlight_params(element, include_attrs)
                if element_data:
                    element_data.update({"color": final_color, "label": label})
                    prepared_data.append(element_data)

        elif group_by is not None:
            logger.debug("_prepare: Grouping by attribute strategy.")
            grouped_elements = self._group_elements_by_attr(group_by)
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
                    element_data = self._get_element_highlight_params(element, include_attrs)
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
                    element_data = self._get_element_highlight_params(element, include_attrs)
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
                    element_data = self._get_element_highlight_params(element, include_attrs)
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
        include_attrs: Optional[List[str]],
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
            "include_attrs": include_attrs,
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
        include_attrs: Optional[List[str]],
        existing: str,
    ):
        """Highlights all elements with the same explicit label and color."""
        for element in self._elements:
            self._call_element_highlighter(
                element=element,
                color=color,  # Use explicit color if provided
                label=label,  # Use the explicit group label
                use_color_cycling=False,  # Use consistent color for the label
                include_attrs=include_attrs,
                existing=existing,
            )

    def _highlight_grouped_by_attribute(
        self,
        group_by: str,
        label_format: Optional[str],
        include_attrs: Optional[List[str]],
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
                    include_attrs=include_attrs,
                    existing=existing,
                )

    def _highlight_distinctly(self, include_attrs: Optional[List[str]], existing: str):
        """DEPRECATED: Logic moved to _prepare_highlight_data. Kept for reference/potential reuse."""
        # This method is no longer called directly by the main highlight path.
        # The distinct logic is handled within _prepare_highlight_data.
        for element in self._elements:
            self._call_element_highlighter(
                element=element,
                color=None,  # Let ColorManager cycle
                label=None,  # No label for distinct elements
                use_color_cycling=True,  # Force cycling
                include_attrs=include_attrs,
                existing=existing,
            )

    def show(
        self,
        # --- Visualization Parameters ---
        group_by: Optional[str] = None,
        label: Optional[str] = None,
        color: Optional[Union[Tuple, str]] = None,
        label_format: Optional[str] = None,
        distinct: bool = False,
        include_attrs: Optional[List[str]] = None,
        # --- Rendering Parameters ---
        scale: float = 2.0,
        labels: bool = True,  # Use 'labels' consistent with service
        legend_position: str = "right",
        render_ocr: bool = False,
    ) -> Optional["Image.Image"]:
        """
        Generates a temporary preview image highlighting elements in this collection
        on their page, ignoring any persistent highlights.

        Currently only supports collections where all elements are on the same page
        of the same PDF.

        Allows grouping and coloring elements based on attributes, similar to the
        persistent `highlight()` method, but only for this temporary view.

        Args:
            group_by: Attribute name to group elements by for distinct colors/labels.
            label: Explicit label for all elements (overrides group_by).
            color: Explicit color for all elements (if label used) or base color.
            label_format: F-string to format group labels if group_by is used.
            distinct: Highlight each element distinctly (overrides group_by/label).
            include_attrs: Attributes to display on individual highlights.
            scale: Scale factor for rendering image.
            labels: Whether to include a legend for the temporary highlights.
            legend_position: Position of the legend ('right', 'left', 'top', 'bottom').
            render_ocr: Whether to render OCR text.

        Returns:
            PIL Image object of the temporary preview, or None if rendering fails or
            elements span multiple pages/PDFs.

        Raises:
            ValueError: If the collection is empty or elements are on different pages/PDFs.
        """
        if not self._elements:
            raise ValueError("Cannot show an empty collection.")

        # Check if elements are on multiple PDFs
        if self._are_on_multiple_pdfs():
            raise ValueError(
                "show() currently only supports collections where all elements are from the same PDF."
            )

        # Check if elements are on multiple pages
        if self._are_on_multiple_pages():
            raise ValueError(
                "show() currently only supports collections where all elements are on the same page."
            )

        # Get the page and highlighting service from the first element
        first_element = self._elements[0]
        if not hasattr(first_element, "page") or not first_element.page:
            logger.warning("Cannot show collection: First element has no associated page.")
            return None
        page = first_element.page
        if not hasattr(page, "pdf") or not page.pdf:
            logger.warning("Cannot show collection: Page has no associated PDF object.")
            return None

        service = page._highlighter
        if not service:
            logger.warning("Cannot show collection: PDF object has no highlighting service.")
            return None

        # 1. Prepare temporary highlight data based on grouping parameters
        # This returns a list of dicts, suitable for render_preview
        highlight_data_list = self._prepare_highlight_data(
            distinct=distinct,
            label=label,
            color=color,
            group_by=group_by,
            label_format=label_format,
            include_attrs=include_attrs,
        )

        if not highlight_data_list:
            logger.warning("No highlight data generated for show(). Rendering clean page.")
            # Render the page without any temporary highlights
            highlight_data_list = []

        # 2. Call render_preview on the HighlightingService
        try:
            return service.render_preview(
                page_index=page.index,
                temporary_highlights=highlight_data_list,
                scale=scale,
                labels=labels,  # Use 'labels'
                legend_position=legend_position,
                render_ocr=render_ocr,
            )
        except Exception as e:
            logger.error(f"Error calling highlighting_service.render_preview: {e}", exc_info=True)
            return None

    def save(
        self,
        filename: str,
        scale: float = 2.0,
        width: Optional[int] = None,
        labels: bool = True,
        legend_position: str = "right",
        render_ocr: bool = False,
    ) -> "ElementCollection":
        """
        Save the page with this collection's elements highlighted to an image file.

        Args:
            filename: Path to save the image to
            scale: Scale factor for rendering
            width: Optional width for the output image in pixels
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            render_ocr: Whether to render OCR text with white background boxes

        Returns:
            Self for method chaining
        """
        # Use to_image to generate and save the image
        self.to_image(
            path=filename,
            scale=scale,
            width=width,
            labels=labels,
            legend_position=legend_position,
            render_ocr=render_ocr,
        )
        return self

    def to_image(
        self,
        path: Optional[str] = None,
        scale: float = 2.0,
        width: Optional[int] = None,
        labels: bool = True,
        legend_position: str = "right",
        render_ocr: bool = False,
    ) -> Optional["Image.Image"]:
        """
        Generate an image of the page with this collection's elements highlighted,
        optionally saving it to a file.

        Args:
            path: Optional path to save the image to
            scale: Scale factor for rendering
            width: Optional width for the output image in pixels (height calculated to maintain aspect ratio)
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            render_ocr: Whether to render OCR text with white background boxes

        Returns:
            PIL Image of the page with elements highlighted, or None if no valid page
        """
        # Get the page from the first element (if available)
        if self._elements and hasattr(self._elements[0], "page"):
            page = self._elements[0].page
            # Generate the image using to_image
            return page.to_image(
                path=path,
                scale=scale,
                width=width,
                labels=labels,
                legend_position=legend_position,
                render_ocr=render_ocr,
            )
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
        self, element: T, include_attrs: Optional[List[str]]
    ) -> Optional[Dict]:
        """Extracts common parameters needed for highlighting a single element."""
        if not hasattr(element, "page"):
            return None
        page = element.page

        base_data = {
            "page_index": page.index,
            "element": element,
            "include_attrs": include_attrs,
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
        if include_attrs:
            for attr_name in include_attrs:
                try:
                    attr_value = getattr(element, attr_name, None)
                    if attr_value is not None:
                        base_data["attributes_to_draw"][attr_name] = attr_value
                except AttributeError:
                    logger.warning(
                        f"Attribute '{attr_name}' not found on element {element} for include_attrs"
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
            apply_exclusions: Whether to exclude elements in exclusion regions
        """
        return self.apply(lambda element: element.find(selector, **kwargs))

    def extract_each_text(self, **kwargs) -> List[str]:
        """
        Extract text from each element in this region.
        """
        return self.apply(
            lambda element: element.extract_text(**kwargs) if element is not None else None
        )

    def correct_ocr(
        self,
        correction_callback: Callable[[Any], Optional[str]],
        max_workers: Optional[int] = None,
    ) -> "ElementCollection":
        """
        Applies corrections to OCR-generated text elements within this collection
        using a user-provided callback function, executed
        in parallel if `max_workers` is specified.

        Iterates through elements currently in the collection. If an element's
        'source' attribute starts with 'ocr', it calls the `correction_callback`
        for that element, passing the element itself.

        The `correction_callback` should contain the logic to:
        1. Determine if the element needs correction.
        2. Perform the correction (e.g., call an LLM).
        3. Return the new text (`str`) or `None`.

        If the callback returns a string, the element's `.text` is updated in place.
        Metadata updates (source, confidence, etc.) should happen within the callback.
        Elements without a source starting with 'ocr' are skipped.

        Args:
            correction_callback: A function accepting an element and returning
                                 `Optional[str]` (new text or None).
            max_workers: The maximum number of worker threads to use for parallel
                         correction on each page. If None, defaults are used.

        Returns:
            Self for method chaining.
        """
        # Delegate to the utility function
        _apply_ocr_correction_to_elements(
            elements=self._elements,
            correction_callback=correction_callback,
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
        max_workers: Optional[int] = None,
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
            max_workers: (Not currently used for classification batching which is
                         handled by the underlying pipeline).
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
                    element.to_image(
                        path=str(image_path), resolution=image_resolution, include_highlights=True
                    )

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


class PageCollection(Generic[P], ApplyMixin):
    """
    A collection of PDF pages with cross-page operations.

    This class provides methods for working with multiple pages, such as finding
    elements across pages, extracting text from page ranges, and more.
    """

    def __init__(self, pages: List[P]):
        """
        Initialize a page collection.

        Args:
            pages: List of Page objects
        """
        self.pages = pages

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

    def extract_text(self, keep_blank_chars=True, apply_exclusions=True, **kwargs) -> str:
        """
        Extract text from all pages in the collection.

        Args:
            keep_blank_chars: Whether to keep blank characters (default: True)
            apply_exclusions: Whether to apply exclusion regions (default: True)
            **kwargs: Additional extraction parameters

        Returns:
            Combined text from all pages
        """
        texts = []
        for page in self.pages:
            text = page.extract_text(
                keep_blank_chars=keep_blank_chars, apply_exclusions=apply_exclusions, **kwargs
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
        page_indices = [p.index for p in self.pages]

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
        Find all elements matching the selector OR text across all pages in the collection.

        Provide EITHER `selector` OR `text`, but not both.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for (equivalent to 'text:contains(...)').
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
                apply_exclusions=apply_exclusions,
                regex=regex,
                case=case,
                **kwargs,
            )
            if elements:
                all_elements.extend(elements.elements)

        return ElementCollection(all_elements)

    def correct_ocr(
        self,
        correction_callback: Callable[[Any], Optional[str]],
        max_workers: Optional[int] = None,
    ) -> "PageCollection[P]":
        """
        Applies corrections to OCR-generated text elements across all pages
        in this collection using a user-provided callback function, executed
        in parallel if `max_workers` is specified.

        This method delegates to the parent PDF's `correct_ocr` method,
        targeting all pages within this collection.

        Args:
            correction_callback: A function that accepts a single argument (an element
                                 object) and returns `Optional[str]` (new text or None).
            max_workers: The maximum number of worker threads to use for parallel
                         correction on each page. If None, defaults are used.

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If the collection is empty, pages lack a parent PDF reference,
                          or the parent PDF lacks the `correct_ocr` method.
        """
        if not self.pages:
            logger.warning("Cannot correct OCR for an empty PageCollection.")
            # Return self even if empty to maintain chaining consistency
            return self

        # Assume all pages share the same parent PDF object
        parent_pdf = self.pages[0]._parent
        if (
            not parent_pdf
            or not hasattr(parent_pdf, "correct_ocr")
            or not callable(parent_pdf.correct_ocr)
        ):
            raise RuntimeError(
                "Parent PDF reference not found or parent PDF lacks the required 'correct_ocr' method."
            )

        page_indices = [p.index for p in self.pages]
        logger.info(
            f"PageCollection: Delegating correct_ocr to parent PDF for page indices: {page_indices} with max_workers={max_workers}."
        )

        # Delegate the call to the parent PDF object for the relevant pages
        # Pass the max_workers parameter down
        parent_pdf.correct_ocr(
            correction_callback=correction_callback,
            pages=page_indices,
            max_workers=max_workers,  # Pass it here
        )

        return self

    def get_sections(
        self,
        start_elements=None,
        end_elements=None,
        new_section_on_page_break=False,
        boundary_inclusion="both",
    ) -> List["Region"]:
        """
        Extract sections from a page collection based on start/end elements.

        Args:
            start_elements: Elements or selector string that mark the start of sections
            end_elements: Elements or selector string that mark the end of sections
            new_section_on_page_break: Whether to start a new section at page boundaries (default: False)
            boundary_inclusion: How to include boundary elements: 'start', 'end', 'both', or 'none' (default: 'both')

        Returns:
            List of Region objects representing the extracted sections
        """
        # Find start and end elements across all pages
        if isinstance(start_elements, str):
            start_elements = self.find_all(start_elements).elements

        if isinstance(end_elements, str):
            end_elements = self.find_all(end_elements).elements

        # If no start elements, return empty list
        if not start_elements:
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
                start_elements.append(top_region)

        # Get all elements from all pages and sort them in document order
        all_elements = []
        for page in self.pages:
            elements = page.get_elements()
            all_elements.extend(elements)

        # Sort by page index, then vertical position, then horizontal position
        all_elements.sort(key=lambda e: (e.page.index, e.top, e.x0))

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
        section_boundaries.sort(
            key=lambda x: (
                x["page_idx"],
                x["index"] if x["index"] != -1 else (0 if x["type"] == "start" else float("inf")),
            )
        )

        # Generate sections
        sections = []
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

                # If both elements are on the same page, use the page's get_section_between
                if start_element.page == end_element.page:
                    section = start_element.page.get_section_between(
                        start_element, end_element, boundary_inclusion
                    )
                    sections.append(section)
                else:
                    # Create a multi-page section
                    from natural_pdf.elements.region import Region

                    # Get the start and end pages
                    start_page = start_element.page
                    end_page = end_element.page

                    # Create a combined region
                    combined_region = Region(
                        start_page, (0, start_element.top, start_page.width, start_page.height)
                    )
                    combined_region._spans_pages = True
                    combined_region._page_range = (start_page.index, end_page.index)
                    combined_region.start_element = start_element
                    combined_region.end_element = end_element

                    # Get all elements that fall within this multi-page region
                    combined_elements = []

                    # Get elements from the first page
                    first_page_elements = [
                        e
                        for e in all_elements
                        if e.page == start_page and e.top >= start_element.top
                    ]
                    combined_elements.extend(first_page_elements)

                    # Get elements from middle pages (if any)
                    for page_idx in range(start_page.index + 1, end_page.index):
                        middle_page_elements = [e for e in all_elements if e.page.index == page_idx]
                        combined_elements.extend(middle_page_elements)

                    # Get elements from the last page
                    last_page_elements = [
                        e
                        for e in all_elements
                        if e.page == end_page and e.bottom <= end_element.bottom
                    ]
                    combined_elements.extend(last_page_elements)

                    # Store the elements in the combined region
                    combined_region._multi_page_elements = combined_elements

                    sections.append(combined_region)

                current_start = None

            # If it's another start boundary and we have a current start (for splitting by starts only)
            elif boundary["type"] == "start" and current_start is not None and not end_elements:
                # Create a section from current_start to just before this boundary
                start_element = current_start["element"]

                # Find the last element before this boundary on the same page
                if start_element.page == boundary["element"].page:
                    # Find elements on this page
                    page_elements = [e for e in all_elements if e.page == start_element.page]
                    # Sort by position
                    page_elements.sort(key=lambda e: (e.top, e.x0))

                    # Find the last element before the boundary
                    end_idx = (
                        page_elements.index(boundary["element"]) - 1
                        if boundary["element"] in page_elements
                        else -1
                    )
                    end_element = page_elements[end_idx] if end_idx >= 0 else None

                    # Create the section
                    section = start_element.page.get_section_between(
                        start_element, end_element, boundary_inclusion
                    )
                    sections.append(section)
                else:
                    # Cross-page section - create from current_start to the end of its page
                    from natural_pdf.elements.region import Region

                    start_page = start_element.page

                    region = Region(
                        start_page, (0, start_element.top, start_page.width, start_page.height)
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
                last_page_elements.sort(key=lambda e: (e.top, e.x0))
                end_element = last_page_elements[-1] if last_page_elements else None

                # Create a multi-page section
                from natural_pdf.elements.region import Region

                if start_page == last_page:
                    # Simple case - both on same page
                    section = start_page.get_section_between(
                        start_element, end_element, boundary_inclusion
                    )
                    sections.append(section)
                else:
                    # Create a multi-page section
                    combined_region = Region(
                        start_page, (0, start_element.top, start_page.width, start_page.height)
                    )
                    combined_region._spans_pages = True
                    combined_region._page_range = (start_page.index, last_page.index)
                    combined_region.start_element = start_element
                    combined_region.end_element = end_element

                    # Get all elements that fall within this multi-page region
                    combined_elements = []

                    # Get elements from the first page
                    first_page_elements = [
                        e
                        for e in all_elements
                        if e.page == start_page and e.top >= start_element.top
                    ]
                    combined_elements.extend(first_page_elements)

                    # Get elements from middle pages (if any)
                    for page_idx in range(start_page.index + 1, last_page.index):
                        middle_page_elements = [e for e in all_elements if e.page.index == page_idx]
                        combined_elements.extend(middle_page_elements)

                    # Get elements from the last page
                    last_page_elements = [
                        e
                        for e in all_elements
                        if e.page == last_page
                        and (end_element is None or e.bottom <= end_element.bottom)
                    ]
                    combined_elements.extend(last_page_elements)

                    # Store the elements in the combined region
                    combined_region._multi_page_elements = combined_elements

                    sections.append(combined_region)
            else:
                # With start_elements only, create a section to the end of the current page
                from natural_pdf.elements.region import Region

                region = Region(
                    start_page, (0, start_element.top, start_page.width, start_page.height)
                )
                region.start_element = start_element
                sections.append(region)

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
        page_indices = [p.index for p in self.pages]
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

    def to_image(
        self,
        page_width: int = 300,
        cols: Optional[int] = 4,
        rows: Optional[int] = None,
        max_pages: Optional[int] = None,
        spacing: int = 10,
        add_labels: bool = True,
        show_category: bool = False,  # Add new flag
    ) -> Optional["Image.Image"]:
        """
        Generate a grid of page images for this collection.

        Args:
            page_width: Width in pixels for rendering individual pages
            cols: Number of columns in grid (default: 4)
            rows: Number of rows in grid (calculated automatically if None)
            max_pages: Maximum number of pages to include (default: all)
            spacing: Spacing between page thumbnails in pixels
            add_labels: Whether to add page number labels
            show_category: Whether to add category and confidence labels (if available)

        Returns:
            PIL Image of the page grid or None if no pages
        """
        if not self.pages:
            logger.warning("Cannot generate image for empty PageCollection")
            return None

        # Limit pages if max_pages is specified
        pages_to_render = self.pages[:max_pages] if max_pages else self.pages

        # Load font once outside the loop
        font = ImageFont.load_default(16) if add_labels else None

        # Render individual page images
        page_images = []
        for page in pages_to_render:
            img = page.to_image(width=page_width)

            # Add page number label
            if add_labels and font:  # Check if font was loaded
                draw = ImageDraw.Draw(img)
                pdf_name = Path(page.pdf.path).stem if hasattr(page, "pdf") and page.pdf else ""
                label_text = f"p{page.number} - {pdf_name}"

                # Add category if requested and available
                if show_category:
                    category = getattr(page, "category", None)
                    confidence = getattr(page, "category_confidence", None)
                    if category is not None and confidence is not None:
                        category_str = f"{category} {confidence:.3f}"
                        label_text += f"\n{category_str}"

                # Calculate bounding box for multi-line text
                # Use (5, 5) as top-left anchor for textbbox calculation for padding
                # Use multiline_textbbox for accurate bounds with newlines
                bbox = draw.multiline_textbbox((5, 5), label_text, font=font)
                # Add padding to the calculated bbox for the white background
                bg_rect = (bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2)

                # Draw white background rectangle
                draw.rectangle(bg_rect, fill=(255, 255, 255))

                # Draw the potentially multi-line text using multiline_text
                draw.multiline_text((5, 5), label_text, fill=(0, 0, 0), font=font)

            page_images.append(img)

        # Calculate grid dimensions if not provided
        if not rows and not cols:
            # Default to a square-ish grid
            cols = min(4, int(len(page_images) ** 0.5) + 1)
            rows = (len(page_images) + cols - 1) // cols
        elif rows and not cols:
            cols = (len(page_images) + rows - 1) // rows
        elif cols and not rows:
            rows = (len(page_images) + cols - 1) // cols

        # Get maximum dimensions for consistent grid cells
        max_width = max(img.width for img in page_images)
        max_height = max(img.height for img in page_images)

        # Create grid image
        grid_width = cols * max_width + (cols + 1) * spacing
        grid_height = rows * max_height + (rows + 1) * spacing
        grid_img = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

        # Place images in grid
        for i, img in enumerate(page_images):
            if i >= rows * cols:
                break

            row = i // cols
            col = i % cols

            x = col * max_width + (col + 1) * spacing
            y = row * max_height + (row + 1) * spacing

            grid_img.paste(img, (x, y))

        return grid_img
