"""Element Manager for natural-pdf.

This module handles the loading, creation, and management of PDF elements like
characters, words, rectangles, lines, and images extracted from a page. The
ElementManager class serves as the central coordinator for element lifecycle
management and provides enhanced word extraction capabilities.

The module includes:
- Element creation and caching for performance
- Custom word extraction that respects font boundaries
- OCR coordinate transformation and integration
- Text decoration detection (underline, strikethrough, highlights)
- Performance optimizations for bulk text processing
"""

import logging
import statistics
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Tuple

from natural_pdf.core.decoration_detector import DecorationDetector
from natural_pdf.core.element_loader import ElementLoader
from natural_pdf.core.element_store import ElementStore
from natural_pdf.core.ocr_converter import OCRConverter
from natural_pdf.core.word_engine import WordEngine, WordEngineOptions
from natural_pdf.elements.image import ImageElement
from natural_pdf.elements.line import LineElement
from natural_pdf.elements.rect import RectangleElement
from natural_pdf.elements.text import TextElement

logger = logging.getLogger(__name__)

CharDirection = Literal["ltr", "rtl", "ttb", "btt"]

# ------------------------------------------------------------------
#  Default decoration-detection parameters (magic numbers centralised)
# ------------------------------------------------------------------

STRIKE_DEFAULTS = {
    "thickness_tol": 1.5,  # pt ; max height of line/rect to be considered strike
    "horiz_tol": 1.0,  # pt ; vertical tolerance for horizontality
    "coverage_ratio": 0.7,  # proportion of glyph width to be overlapped
    "band_top_frac": 0.35,  # fraction of glyph height above top baseline band
    "band_bottom_frac": 0.65,  # fraction below top (same used internally)
}

UNDERLINE_DEFAULTS = {
    "thickness_tol": 1.5,
    "horiz_tol": 1.0,
    "coverage_ratio": 0.8,
    "band_frac": 0.25,  # height fraction above baseline
    "below_pad": 0.7,  # pt ; pad below baseline
}

HIGHLIGHT_DEFAULTS = {
    "height_min_ratio": 0.6,  # rect height relative to char height lower bound
    "height_max_ratio": 2.0,  # upper bound
    "coverage_ratio": 0.6,  # horizontal overlap with glyph
    "color_saturation_min": 0.4,  # HSV S >
    "color_value_min": 0.4,  # HSV V >
}


@contextmanager
def disable_text_sync():
    """Temporarily disable text synchronization for performance.

    This context manager is used when bulk-updating text content where character-level
    synchronization is not needed, such as during bidi processing or large-scale
    text transformations. It prevents exponential recursion issues with Arabic/RTL
    text processing by bypassing the normal text property setter.

    Yields:
        None: The context where text synchronization is disabled.

    Example:
        ```python
        with disable_text_sync():
            for element in text_elements:
                element.text = process_arabic_text(element.text)
        # Text sync automatically restored after the block
        ```

    Note:
        This optimization is critical for performance when processing documents
        with complex text layouts or right-to-left scripts that would otherwise
        trigger expensive character synchronization operations.
    """
    # Save original property so we can restore it afterwards
    original_property = TextElement.text
    original_getter = original_property.fget
    original_setter = original_property.fset

    if original_getter is None or original_setter is None:
        # If the property is oddly configured, skip the optimisation entirely.
        yield
        return

    # Create a fast setter that skips sync
    def fast_setter(self: TextElement, value: str) -> None:
        self._obj["text"] = value
        if hasattr(self, "_layout_text_cache"):
            self._layout_text_cache = value
        if hasattr(self, "_text_manually_set"):
            self._text_manually_set = True

    # Apply fast setter
    fast_property = property(original_getter, fast_setter)
    setattr(TextElement, "text", fast_property)

    try:
        yield
    finally:
        # Restore original setter
        setattr(TextElement, "text", original_property)


class ElementManager:
    """
    Manages the loading, creation, and retrieval of elements from a PDF page.

    This class centralizes the element management functionality previously
    contained in the Page class, providing better separation of concerns.
    """

    def __init__(self, page, font_attrs=None, load_text: bool = True):
        """
        Initialize the ElementManager.

        Args:
            page: The parent Page object
            font_attrs: Font attributes to consider when grouping characters into words.
                       Default: ['fontname', 'size', 'bold', 'italic']
                       None: Only consider spatial relationships
                       List: Custom attributes to consider
            load_text: Whether to load text elements from the PDF (default: True).
        """
        self._page = page
        self._store = ElementStore()
        self._load_text = load_text
        # Default to splitting by fontname, size, bold, italic if not specified
        # Renamed internal variable for clarity
        self._word_split_attributes = (
            ["fontname", "size", "bold", "italic"] if font_attrs is None else font_attrs
        )
        self._word_engine = WordEngine(
            self._word_split_attributes,
            load_text=self._load_text,
        )
        self._element_loader = ElementLoader(page_number=page.number)
        self._decorations = DecorationDetector(page)
        self._ocr_converter = OCRConverter(page)

    def load_elements(self):
        """
        Load all elements from the page (lazy loading).
        Uses WordEngine for word grouping.
        """
        if self._store.is_populated():
            return
        with self._store.transaction():
            if self._store.is_populated():
                return
            self._populate_store()

    def _populate_store(self) -> None:
        logger.debug(f"Page {self._page.number}: Loading elements...")

        # 1. Prepare character dictionaries only if loading text
        if self._load_text:
            native_chars = getattr(self._page._page, "chars", []) or []
            prepared_char_dicts = self._element_loader.prepare_native_chars(native_chars)
        else:
            prepared_char_dicts = []
            logger.debug(f"Page {self._page.number}: Skipping text loading (load_text=False)")

        if self._load_text and prepared_char_dicts:
            self._decorations.annotate_chars(prepared_char_dicts)

        word_options = self._build_word_engine_options(prepared_char_dicts)
        generated_words = self._word_engine.generate_words(
            prepared_char_dicts,
            options=word_options,
            create_word_element=self._create_word_element,
            propagate_decorations=self._decorations.propagate_to_words,
            disable_text_sync=disable_text_sync,
        )
        logger.debug(
            "Page %s: Generated %d words using WordEngine.",
            self._page.number,
            len(generated_words),
        )

        # 4. Load other elements (rects, lines)
        rect_elements = [RectangleElement(r, self._page) for r in self._page._page.rects]
        line_elements = [LineElement(l, self._page) for l in self._page._page.lines]
        image_elements = [ImageElement(i, self._page) for i in self._page._page.images]
        logger.debug(
            f"Page {self._page.number}: Loaded {len(rect_elements)} rects, {len(line_elements)} lines, {len(image_elements)} images."
        )

        elements_data = {
            "chars": [TextElement(c_dict, self._page) for c_dict in prepared_char_dicts],
            "words": generated_words,
            "rects": rect_elements,
            "lines": line_elements,
            "images": image_elements,
        }

        if hasattr(self._page, "_regions") and (
            "detected" in self._page._regions
            or "named" in self._page._regions
            or "checkbox" in self._page._regions
        ):
            regions = []
            if "detected" in self._page._regions:
                regions.extend(self._page._regions["detected"])
            if "named" in self._page._regions:
                regions.extend(self._page._regions["named"].values())
            if "checkbox" in self._page._regions:
                regions.extend(self._page._regions["checkbox"])
            elements_data["regions"] = regions
            logger.debug(f"Page {self._page.number}: Added {len(regions)} regions.")
        else:
            elements_data["regions"] = []

        logger.debug(f"Page {self._page.number}: Element loading complete.")
        self._store.replace(elements_data)

    @property
    def element_loader(self) -> ElementLoader:
        """Expose the ElementLoader instance for hosts that need native char enrichment."""
        return self._element_loader

    def _build_word_engine_options(
        self, prepared_char_dicts: List[Dict[str, Any]]
    ) -> WordEngineOptions:
        page_config = self._page._config
        pdf_config = getattr(self._page, "_parent")._config

        def _resolve_bool(key: str, default: bool) -> bool:
            value = page_config.get(key)
            if value is None:
                value = pdf_config.get(key, default)
            return bool(value) if value is not None else default

        def _resolve_numeric(key: str) -> Optional[float]:
            value = page_config.get(key)
            if value is None:
                value = pdf_config.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            return None

        auto_text_tolerance = page_config.get("auto_text_tolerance")
        if auto_text_tolerance is None:
            auto_text_tolerance = pdf_config.get("auto_text_tolerance", True)
        auto_text_tolerance = bool(auto_text_tolerance)

        xt = page_config.get("x_tolerance")
        if xt is None:
            xt = pdf_config.get("x_tolerance")
        if isinstance(xt, (int, float)):
            xt = float(xt)
        else:
            xt = None

        yt = page_config.get("y_tolerance")
        if yt is None:
            yt = pdf_config.get("y_tolerance")
        if isinstance(yt, (int, float)):
            yt = float(yt)
        else:
            yt = None

        if auto_text_tolerance and prepared_char_dicts:
            sizes = [
                c.get("size")
                for c in prepared_char_dicts
                if isinstance(c.get("size"), (int, float))
            ]
            if sizes:
                median_size = statistics.median(sizes)
                if xt is None:
                    xt = 0.25 * median_size
                    page_config["x_tolerance"] = xt
                if yt is None:
                    yt = 0.6 * median_size
                    page_config["y_tolerance"] = yt

        if xt is None:
            xt = 3.0
        if yt is None:
            yt = 3.0

        options = WordEngineOptions(
            page_number=self._page.number,
            x_tolerance=xt,
            y_tolerance=yt,
            x_tolerance_ratio=_resolve_numeric("x_tolerance_ratio"),
            y_tolerance_ratio=_resolve_numeric("y_tolerance_ratio"),
            keep_blank_chars=_resolve_bool("keep_blank_chars", True),
            use_text_flow=bool(pdf_config.get("use_text_flow", False)),
        )
        return options

    def _create_word_element(self, word_dict: Dict[str, Any]) -> TextElement:
        """
        Create a TextElement (type 'word') from a word dictionary generated
        by NaturalWordExtractor/pdfplumber.

        Args:
            word_dict: Dictionary representing the word, including geometry,
                       text, and attributes copied from the first char
                       (e.g., fontname, size, bold, italic).

        Returns:
            TextElement representing the word.
        """
        # word_dict already contains calculated geometry (x0, top, x1, bottom, etc.)
        # and text content. We just need to ensure our required fields exist
        # and potentially set the source.

        # Start with a copy of the word_dict
        element_data = word_dict.copy()

        # Ensure required TextElement fields are present or add defaults
        element_data.setdefault("object_type", "word")  # Set type to 'word'
        element_data.setdefault("page_number", self._page.number)
        # Determine source based on attributes present (e.g., if 'confidence' exists, it's likely OCR)
        # This assumes the word_dict carries over some hint from its chars.
        # A simpler approach: assume 'native' unless fontname is 'OCR'.
        element_data.setdefault(
            "source", "ocr" if element_data.get("fontname") == "OCR" else "native"
        )
        element_data.setdefault(
            "confidence", 1.0 if element_data["source"] == "native" else 0.0
        )  # Default confidence

        # Bold/italic should already be in word_dict if they were split attributes,
        # copied from the first (representative) char by pdfplumber's merge_chars.
        # Ensure they exist for TextElement initialization.
        element_data.setdefault("bold", False)
        element_data.setdefault("italic", False)

        # Ensure fontname and size exist
        element_data.setdefault("fontname", "Unknown")
        element_data.setdefault("size", 0)

        # Store the constituent char dicts (passed alongside word_dict from extractor)
        # We need to modify the caller (load_elements) to pass this.
        # For now, assume it might be passed in word_dict for placeholder.
        element_data["_char_dicts"] = word_dict.get("_char_dicts", [])  # Store char list

        return TextElement(element_data, self._page)

    def create_text_elements_from_ocr(self, ocr_results, scale_x=None, scale_y=None):
        """
        Convert OCR results to TextElement objects AND adds them to the manager's
        'words' and 'chars' lists.

        This method should be called AFTER initial elements (native) might have
        been loaded, as it appends to the existing lists.

        Args:
            ocr_results: List of OCR results dictionaries with 'text', 'bbox', 'confidence'.
                         Confidence can be None for detection-only results.
            scale_x: Factor to convert image x-coordinates to PDF coordinates.
            scale_y: Factor to convert image y-coordinates to PDF coordinates.

        Returns:
            List of created TextElement word objects that were added.
        """
        self.load_elements()
        store = self._element_store()

        scale_x = float(scale_x) if scale_x is not None else 1.0
        scale_y = float(scale_y) if scale_y is not None else 1.0

        logger.debug(
            f"Page {self._page.number}: Adding {len(ocr_results)} OCR results as elements. Scale: x={scale_x:.2f}, y={scale_y:.2f}"
        )

        word_elements, char_elements = self._ocr_converter.convert(
            ocr_results, scale_x=scale_x, scale_y=scale_y
        )

        if word_elements:
            words = list(store.get("words", []))
            words.extend(word_elements)
            self._store.set("words", words)
        if char_elements:
            chars = list(store.get("chars", []))
            chars.extend(char_elements)
            self._store.set("chars", chars)

        logger.info(
            f"Page {self._page.number}: Appended {len(word_elements)} OCR TextElements (words) and corresponding char entries."
        )
        return list(word_elements)

    def _element_store(self) -> Dict[str, List[Any]]:
        """Return the cached element mapping, ensuring it is populated."""
        self.load_elements()
        return self._store.data_view()

    def add_element(self, element, element_type="words"):
        """
        Add an element to the managed elements.

        Args:
            element: The element to add
            element_type: The type of element ('words', 'chars', etc.)

        Returns:
            True if added successfully, False otherwise
        """
        # Load elements if not already loaded
        # Add to the appropriate list
        store = self._element_store()

        if element_type in store:
            # Avoid adding duplicates
            if element not in store[element_type]:
                store[element_type].append(element)
                self._store.mark_dirty([element_type])
                return True
            else:
                # logger.debug(f"Element already exists in {element_type}: {element}")
                return False  # Indicate it wasn't newly added

        return False

    def add_region(self, region, name=None):
        """
        Add a region to the managed elements.

        Args:
            region: The region to add
            name: Optional name for the region

        Returns:
            True if added successfully, False otherwise
        """
        store = self._element_store()

        # Make sure regions is in _elements
        # Add to elements for selector queries
        regions = list(store.get("regions", []))
        if region not in regions:
            regions.append(region)
            self._store.set("regions", regions)
            return True

        return False

    def get_elements(self, element_type=None):
        """
        Get all elements of the specified type, or all elements if type is None.

        Args:
            element_type: Optional element type ('words', 'chars', 'rects', 'lines', 'regions' etc.)

        Returns:
            List of elements
        """
        # Load elements if not already loaded
        try:
            store = self._element_store()
        except RuntimeError:
            return []

        if element_type:
            return list(store.get(element_type, []))

        # Combine all element types
        all_elements: List[Any] = []
        for elements in store.values():
            all_elements.extend(elements)

        return all_elements

    def get_all_elements(self):
        """
        Get all elements from all types.

        Returns:
            List of all elements
        """
        try:
            store = self._element_store()
        except RuntimeError:
            return []

        all_elements: List[Any] = []
        for elements in store.values():
            all_elements.extend(elements)
        return all_elements

    @property
    def chars(self):
        """Get all character elements."""
        store = self._element_store()
        return list(store.get("chars", []))

    def invalidate_cache(self):
        """Invalidate the cached elements, forcing a reload on next access."""
        self._store.clear()
        logger.debug(f"Page {self._page.number}: ElementManager cache invalidated")

    @property
    def words(self):
        """Get all word elements."""
        store = self._element_store()
        return list(store.get("words", []))

    @property
    def rects(self):
        """Get all rectangle elements."""
        store = self._element_store()
        return list(store.get("rects", []))

    @property
    def lines(self):
        """Get all line elements."""
        store = self._element_store()
        return list(store.get("lines", []))

    @property
    def regions(self):
        """Get all region elements."""
        store = self._element_store()
        return list(store.get("regions", []))

    @property
    def images(self):
        """Get all image elements."""
        store = self._element_store()
        return list(store.get("images", []))

    def remove_ocr_elements(self):
        """
        Remove all elements with source="ocr" from the elements dictionary.
        This should be called before adding new OCR elements if replacement is desired.

        Returns:
            int: Number of OCR elements removed
        """
        store = self._element_store()

        removed_count = 0

        # Filter out OCR elements from words
        if "words" in store:
            original_words = store["words"]
            filtered_words = [
                word for word in original_words if getattr(word, "source", None) != "ocr"
            ]
            word_diff = len(original_words) - len(filtered_words)
            if word_diff:
                self._store.set("words", filtered_words)
                removed_count += word_diff

        # Filter out OCR elements from chars
        if "chars" in store:
            original_chars = store["chars"]
            filtered_chars = [
                char
                for char in original_chars
                if (isinstance(char, dict) and char.get("source") != "ocr")
                or (not isinstance(char, dict) and getattr(char, "source", None) != "ocr")
            ]
            diff = len(original_chars) - len(filtered_chars)
            if diff:
                self._store.set("chars", filtered_chars)
                removed_count += diff

        logger.info(f"Page {self._page.number}: Removed {removed_count} OCR elements.")
        return removed_count

    def remove_element(self, element, element_type="words"):
        """
        Remove a specific element from the managed elements.

        Args:
            element: The element to remove
            element_type: The type of element ('words', 'chars', etc.)

        Returns:
            bool: True if removed successfully, False otherwise
        """
        store = self._element_store()

        # Check if the collection exists
        if element_type not in store:
            raise KeyError(f"Element collection '{element_type}' does not exist")

        # Try to remove the element
        try:
            if element in store[element_type]:
                store[element_type].remove(element)
                logger.debug(f"Removed element from {element_type}: {element}")
                self._store.mark_dirty([element_type])
                return True
            else:
                logger.debug(f"Element not found in {element_type}: {element}")
                return False
        except Exception as e:
            logger.error(f"Error removing element from {element_type}: {e}", exc_info=True)
            return False

    def remove_elements_by_source(self, element_type: str, source: str) -> int:
        """Remove all elements of ``element_type`` whose ``source`` attribute matches ``source``."""
        store = self._element_store()

        if element_type not in store:
            return 0

        elements = store[element_type]
        filtered = [element for element in elements if getattr(element, "source", None) != source]
        removed = len(elements) - len(filtered)
        if removed:
            self._store.set(element_type, filtered)
            logger.info(
                "Page %s: Removed %d '%s' element(s) with source '%s'.",
                getattr(self._page, "number", "?"),
                removed,
                element_type,
                source,
            )
        return removed

    def clear_text_layer(self) -> tuple[int, int]:
        """Remove all word and character elements tracked by this manager."""
        store = self._element_store()

        removed_words = len(store.get("words", []))
        removed_chars = len(store.get("chars", []))

        if "words" in store:
            self._store.set("words", [])

        if "chars" in store:
            self._store.set("chars", [])

        return removed_words, removed_chars

    def has_elements(self) -> bool:
        """
        Check if any significant elements (words, rects, lines, regions)
        have been loaded or added.

        Returns:
            True if any elements exist, False otherwise.
        """
        try:
            store = self._element_store()
        except RuntimeError:
            return False

        for key in ["words", "rects", "lines", "regions"]:
            if store.get(key):
                return True

        return False
