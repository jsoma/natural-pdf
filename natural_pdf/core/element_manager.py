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
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

from natural_pdf.core.decoration_detector import DecorationDetector
from natural_pdf.core.element_loader import ElementLoader
from natural_pdf.core.element_store import ElementStore
from natural_pdf.core.ocr_converter import OCRConverter
from natural_pdf.core.word_engine import WordEngine, WordEngineOptions
from natural_pdf.elements.image import ImageElement
from natural_pdf.elements.line import LineElement
from natural_pdf.elements.rect import RectangleElement
from natural_pdf.elements.text import TextElement, disable_text_sync

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
        self._raw_char_dicts: Optional[List[Dict[str, Any]]] = None
        self._preserved_elements: Dict[str, List[Any]] = {}
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

    def _mark_content_mutated(self) -> None:
        bump = getattr(self._page, "_bump_text_state_version", None)
        if callable(bump):
            bump()

    @staticmethod
    def _char_origin_key(char: Dict[str, Any]) -> Tuple[Any, ...]:
        """Return a stable key for a native glyph across element reloads."""

        assigned = char.get("_natural_pdf_origin_key")
        if isinstance(assigned, tuple):
            return assigned

        def _coordinate(name: str) -> Any:
            value = char.get(name)
            return round(float(value), 6) if isinstance(value, (int, float)) else value

        return (
            *(_coordinate(name) for name in ("x0", "top", "x1", "bottom")),
            char.get("fontname"),
            char.get("size"),
            char.get("source"),
            0,
        )

    @classmethod
    def _assign_native_origin_keys(cls, chars: List[Dict[str, Any]]) -> None:
        """Assign deterministic occurrence ordinals to coincident native glyphs."""

        occurrences: Dict[Tuple[Any, ...], int] = {}
        for char in chars:
            base = cls._char_origin_key(char)[:-1]
            ordinal = occurrences.get(base, 0)
            occurrences[base] = ordinal + 1
            char["_natural_pdf_origin_key"] = (*base, ordinal)

    @staticmethod
    def _element_source(element: Any) -> Optional[str]:
        obj = getattr(element, "_obj", None)
        if isinstance(obj, dict):
            source = obj.get("source")
            if source is not None:
                return str(source)
            # TextElement.source deliberately defaults missing native sources
            # to ``"pdf"``.  Keep source-based removal consistent with that
            # public contract for manually constructed/native-like elements.
            fallback = getattr(element, "source", None)
            return str(fallback) if fallback is not None else None
        source = getattr(element, "source", None)
        return str(source) if source is not None else None

    def _capture_preserved_elements(self) -> Dict[str, List[Any]]:
        """Capture overlays and explicit edits that must survive a native reload."""

        if not self._store.is_populated():
            return {}

        store = self._store.data_view()
        words = list(store.get("words", []))
        preserved_words = [
            word
            for word in words
            if self._element_source(word) not in {None, "native", "pdf"}
            or bool(getattr(word, "_text_user_edited", False))
        ]
        chars = list(store.get("chars", []))
        preserved_char_dict_ids = {
            id(char_dict)
            for word in preserved_words
            for char_dict in getattr(word, "_char_dicts", ())
            if isinstance(char_dict, dict)
        }

        preserved_chars = [
            char
            for char in chars
            if id(getattr(char, "_obj", None)) in preserved_char_dict_ids
            or self._element_source(char) not in {None, "native", "pdf"}
            or bool(getattr(char, "_text_user_edited", False))
        ]

        preserved: Dict[str, List[Any]] = {
            "words": preserved_words,
            "chars": preserved_chars,
            # Regions are synthetic registry entries, never raw pdfplumber objects.
            "regions": list(store.get("regions", [])),
        }
        for kind in ("rects", "lines", "images"):
            preserved[kind] = [
                element
                for element in store.get(kind, [])
                if self._element_source(element) not in {None, "native", "pdf"}
            ]
        return {kind: values for kind, values in preserved.items() if values}

    def _filter_preserved_native_chars(
        self, prepared_char_dicts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove edited native glyph origins before fresh word grouping."""

        edited_origin_keys = {
            origin_key
            for word in self._preserved_elements.get("words", ())
            if self._element_source(word) in {None, "native", "pdf"}
            and bool(getattr(word, "_text_user_edited", False))
            for origin_key in getattr(word, "_native_origin_keys", ())
        }
        edited_origin_keys.update(
            self._char_origin_key(char._obj)
            for char in self._preserved_elements.get("chars", ())
            if self._element_source(char) in {None, "native", "pdf"}
            and bool(getattr(char, "_text_user_edited", False))
            and isinstance(getattr(char, "_obj", None), dict)
        )
        if not edited_origin_keys:
            return prepared_char_dicts
        return [
            char
            for char in prepared_char_dicts
            if self._char_origin_key(char) not in edited_origin_keys
        ]

    @staticmethod
    def _extend_unique(target: List[Any], additions: List[Any]) -> None:
        seen = {id(item) for item in target}
        for item in additions:
            if id(item) not in seen:
                target.append(item)
                seen.add(id(item))

    def _merge_preserved_elements(
        self,
        elements_data: Dict[str, List[Any]],
        prepared_char_dicts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge captured overlays into freshly generated native elements."""

        preserved = self._preserved_elements
        self._preserved_elements = {}
        if not preserved:
            return prepared_char_dicts

        preserved_words = list(preserved.get("words", []))
        self._extend_unique(elements_data["words"], preserved_words)
        for kind in ("rects", "lines", "images", "regions"):
            self._extend_unique(elements_data[kind], list(preserved.get(kind, [])))

        preserved_chars = list(preserved.get("chars", []))
        known_char_dict_ids = {id(getattr(char, "_obj", None)) for char in preserved_chars}
        for word in preserved_words:
            for char_dict in getattr(word, "_char_dicts", ()):
                if not isinstance(char_dict, dict) or id(char_dict) in known_char_dict_ids:
                    continue
                preserved_chars.append(TextElement(char_dict, self._page))
                known_char_dict_ids.add(id(char_dict))

        if not preserved_words and not preserved_chars:
            return prepared_char_dicts

        # A mixed native/overlay layer is materialized so every word can point
        # at stable char objects after native char counts change.
        native_chars = [TextElement(char, self._page) for char in prepared_char_dicts]
        combined_chars = native_chars
        self._extend_unique(combined_chars, preserved_chars)
        elements_data["chars"] = combined_chars
        self._reindex_words(elements_data["words"], combined_chars)
        return []

    @staticmethod
    def _reindex_words(words: List[Any], chars: List[Any]) -> None:
        dict_to_index = {
            id(char._obj): index
            for index, char in enumerate(chars)
            if isinstance(getattr(char, "_obj", None), dict)
        }
        for word in words:
            char_dicts = [
                item for item in getattr(word, "_char_dicts", ()) if isinstance(item, dict)
            ]
            indices = [dict_to_index[id(item)] for item in char_dicts if id(item) in dict_to_index]
            word._char_indices = indices
            word._char_dicts = [chars[index]._obj for index in indices]

    def load_elements(self):
        """
        Load all elements from the page (lazy loading).
        Uses WordEngine for word grouping.
        """
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
            self._assign_native_origin_keys(prepared_char_dicts)
            prepared_char_dicts = self._filter_preserved_native_chars(prepared_char_dicts)
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
            "chars": [],
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

        prepared_char_dicts = self._merge_preserved_elements(elements_data, prepared_char_dicts)
        self._raw_char_dicts = prepared_char_dicts or None

        logger.debug(f"Page {self._page.number}: Element loading complete.")
        self._store.replace(elements_data)

    def _materialize_chars(self) -> List[TextElement]:
        """Return character TextElements, creating wrappers only on demand."""
        store = self._element_store()
        chars = store.get("chars", [])
        if chars:
            return list(chars)

        raw_char_dicts = self._raw_char_dicts or []
        if not raw_char_dicts:
            return []

        char_elements = [TextElement(c_dict, self._page) for c_dict in raw_char_dicts]
        self._raw_char_dicts = None
        self._store.set("chars", char_elements)
        return list(char_elements)

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
                    xt = 0.35 * median_size
                    page_config["x_tolerance"] = xt
                if yt is None:
                    yt = 0.6 * median_size
                    page_config["y_tolerance"] = yt

        if xt is None:
            xt = 3.0
        if yt is None:
            yt = 3.0

        # Resolve space_gap_ratio: None means use default (0.12), 0 disables
        sgr = _resolve_numeric("space_gap_ratio")

        # When auto_text_tolerance is active and the user hasn't set an
        # explicit x_tolerance_ratio, use a ratio so that the tolerance
        # scales with each character's font size.  This handles documents
        # with mixed font sizes (e.g. 9pt body + 15pt titles).
        xtr = _resolve_numeric("x_tolerance_ratio")
        if xtr is None and auto_text_tolerance:
            xtr = 0.35

        options = WordEngineOptions(
            page_number=self._page.number,
            x_tolerance=xt,
            y_tolerance=yt,
            x_tolerance_ratio=xtr,
            y_tolerance_ratio=_resolve_numeric("y_tolerance_ratio"),
            keep_blank_chars=_resolve_bool("keep_blank_chars", True),
            use_text_flow=bool(pdf_config.get("use_text_flow", False)),
            space_gap_ratio=sgr,
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

        element = TextElement(element_data, self._page)
        if element.source == "native":
            element._native_origin_keys = tuple(
                self._char_origin_key(char)
                for char in element._char_dicts
                if isinstance(char, dict)
            )
        return element

    def create_text_elements_from_ocr(
        self,
        ocr_results,
        scale_x=None,
        scale_y=None,
        *,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        engine_name: str | None = None,
    ):
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

        scale_x = float(scale_x) if scale_x is not None else 1.0
        scale_y = float(scale_y) if scale_y is not None else 1.0

        logger.debug(
            f"Page {self._page.number}: Adding {len(ocr_results)} OCR results as elements. Scale: x={scale_x:.2f}, y={scale_y:.2f}"
        )

        word_elements, char_elements = self._ocr_converter.convert(
            ocr_results,
            scale_x=scale_x,
            scale_y=scale_y,
            offset_x=offset_x,
            offset_y=offset_y,
            engine_name=engine_name,
        )

        with self._store.transaction():
            store = self._element_store()
            if char_elements:
                chars = self._materialize_chars()
                first_new_index = len(chars)
                chars.extend(char_elements)
                self._store.set("chars", chars)
                char_offset = 0
                for word in word_elements:
                    if word._obj.get("text") is None or char_offset >= len(char_elements):
                        continue
                    char_index = first_new_index + char_offset
                    char_offset += 1
                    word._char_indices = [char_index]
                    word._char_dicts = [chars[char_index]._obj]
            if word_elements:
                words = list(store.get("words", []))
                words.extend(word_elements)
                self._store.set("words", words)

            if word_elements or char_elements:
                self._mark_content_mutated()

        logger.info(
            f"Page {self._page.number}: Appended {len(word_elements)} OCR TextElements (words) and corresponding char entries."
        )
        return list(word_elements)

    def _element_store(self) -> Dict[str, List[Any]]:
        """Return the manager-owned mapping, ensuring it is populated.

        Direct writes bypass semantic revision tracking.  New code must use
        ElementManager mutation methods (or explicitly call
        ``_mark_content_mutated`` after a legacy direct write).
        """
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
        normalized_type = {"word": "words", "char": "chars"}.get(element_type, element_type)
        with self._store.transaction():
            store = self._element_store()
            if normalized_type == "chars":
                self._materialize_chars()
                store = self._element_store()
            if normalized_type not in store:
                return False

            existing = store[normalized_type]
            if element in existing or (
                normalized_type == "chars"
                and any(
                    getattr(item, "_obj", None) is getattr(element, "_obj", object())
                    for item in existing
                )
            ):
                return False

            existing.append(element)
            dirty_kinds = [normalized_type]
            if normalized_type == "words" and isinstance(element, TextElement):
                chars = self._materialize_chars()
                known_dict_ids = {
                    id(char._obj): index
                    for index, char in enumerate(chars)
                    if isinstance(getattr(char, "_obj", None), dict)
                }
                indices: List[int] = []
                for char_dict in getattr(element, "_char_dicts", ()):
                    if not isinstance(char_dict, dict):
                        continue
                    char_index = known_dict_ids.get(id(char_dict))
                    if char_index is None:
                        char_index = len(chars)
                        chars.append(TextElement(char_dict, self._page))
                        known_dict_ids[id(char_dict)] = char_index
                    indices.append(char_index)
                element._char_indices = indices
                element._char_dicts = [chars[index]._obj for index in indices]
                self._store.set("chars", chars)
                dirty_kinds.append("chars")
            elif normalized_type == "chars":
                self._reindex_words(list(store.get("words", [])), list(existing))

            self._store.mark_dirty(dirty_kinds)
            self._mark_content_mutated()
            return True

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
            self._mark_content_mutated()
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

        if element_type in ("char", "chars"):
            return self._materialize_chars()

        if element_type:
            return list(store.get(element_type, []))

        return self.get_all_elements()

    def get_all_elements(self, *, include_chars: bool = True):
        """
        Get all elements from all types.

        Args:
            include_chars: Whether native character TextElements should be
                included. Internal summary/layout paths can set this to False
                when character-level elements would be ignored downstream.

        Returns:
            List of all elements
        """
        try:
            store = self._element_store()
        except RuntimeError:
            return []

        all_elements: List[Any] = []
        for element_type, elements in store.items():
            if element_type == "chars":
                if not include_chars:
                    continue
                all_elements.extend(self._materialize_chars())
            else:
                all_elements.extend(elements)
        return all_elements

    @staticmethod
    def _set_even_char_geometry(
        char: TextElement, word: TextElement, index: int, count: int
    ) -> None:
        """Fit replacement chars inside the unchanged word bounding box."""

        if count <= 0:
            return
        width = max(float(word.x1) - float(word.x0), 0.0) / count
        x0 = float(word.x0) + (width * index)
        x1 = float(word.x0) + (width * (index + 1))
        char._obj.update(
            {
                "x0": x0,
                "x1": x1,
                "top": word.top,
                "bottom": word.bottom,
                "width": width,
                "height": word.height,
                "adv": width,
                "object_type": "char",
            }
        )

    def update_text_element(self, element: TextElement, value: str) -> bool:
        """Apply a canonical text mutation and repair all shared char indices.

        Returns ``False`` only for detached elements that are not owned by this
        manager, allowing TextElement to use its legacy local fallback.
        """

        if getattr(element, "page", None) is not self._page:
            return False

        with self._store.transaction():
            store = self._element_store()
            element_type = getattr(element, "type", "")
            if element_type == "word" and element in store.get("words", []):
                self._update_word_text(element, value, store)
            elif element_type == "char":
                chars = self._materialize_chars()
                if element not in chars:
                    return False
                self._update_char_text(element, value, store, chars)
            else:
                return False
            # Publish the semantic revision before releasing the same RLock
            # that protects the canonical word/char update.
            self._mark_content_mutated()
        return True

    def _update_word_text(
        self, element: TextElement, value: str, store: Dict[str, List[Any]]
    ) -> None:
        chars = self._materialize_chars()
        original_chars = list(chars)
        words = list(store.get("words", []))
        linked_indices = [
            index
            for index in getattr(element, "_char_indices", ())
            if isinstance(index, int) and 0 <= index < len(original_chars)
        ]
        if not linked_indices:
            linked_dict_ids = {
                id(item) for item in getattr(element, "_char_dicts", ()) if isinstance(item, dict)
            }
            linked_indices = [
                index
                for index, char in enumerate(original_chars)
                if id(getattr(char, "_obj", None)) in linked_dict_ids
            ]

        linked_chars = [original_chars[index] for index in linked_indices]
        old_count = len(linked_chars)
        raw_text = "".join(char.text for char in linked_chars)
        visible_text = element.text
        leading_layout, trailing_layout = self._hidden_boundary_whitespace(raw_text, visible_text)
        if value[:1].isspace():
            leading_layout = ""
        if value[-1:].isspace():
            trailing_layout = ""
        replacement_values = [*leading_layout, *value, *trailing_layout]

        if self._element_source(element) in {None, "native", "pdf"} and not hasattr(
            element, "_native_origin_keys"
        ):
            element._native_origin_keys = tuple(
                self._char_origin_key(char._obj)
                for char in linked_chars
                if isinstance(getattr(char, "_obj", None), dict)
            )

        other_word_char_dict_ids = {
            id(char_dict)
            for word in words
            if word is not element
            for char_dict in self._word_char_dicts(word, original_chars)
        }
        # ``_char_indices`` is the documented memory-efficient representation.
        # Materialize its identities before rebuilding the shared char list so
        # index-only peer words can be reindexed after positions shift.
        for word in words:
            if not getattr(word, "_char_dicts", None):
                word._char_dicts = self._word_char_dicts(word, original_chars)
        template = (
            linked_chars[-1]._obj.copy()
            if linked_chars
            else {
                **element._obj,
                "object_type": "char",
                "source": element.source,
            }
        )
        replacement_chars: List[TextElement] = []
        for index, char_value in enumerate(replacement_values):
            if index < len(linked_chars):
                existing = linked_chars[index]
                char = (
                    TextElement(existing._obj.copy(), self._page)
                    if id(existing._obj) in other_word_char_dict_ids
                    else existing
                )
            else:
                char = TextElement(template.copy(), self._page)
            char._set_text_value(char_value)
            if len(replacement_values) != old_count:
                self._set_even_char_geometry(char, element, index, len(replacement_values))
            replacement_chars.append(char)

        linked_set = set(linked_indices)
        removable_indices = {
            index
            for index in linked_set
            if id(original_chars[index]._obj) not in other_word_char_dict_ids
        }
        insertion_at = min(linked_indices) if linked_indices else len(original_chars)
        retained_chars = [
            char for index, char in enumerate(original_chars) if index not in removable_indices
        ]
        retained_before = sum(
            1
            for index in range(insertion_at)
            if index < len(original_chars) and index not in removable_indices
        )
        rebuilt_chars = (
            retained_chars[:retained_before] + replacement_chars + retained_chars[retained_before:]
        )

        element._set_text_value(value, user_edit=True)
        element._char_dicts = [char._obj for char in replacement_chars]
        self._reindex_words(words, rebuilt_chars)
        self._store.set("chars", rebuilt_chars)
        self._store.mark_dirty(["words"])

    @staticmethod
    def _hidden_boundary_whitespace(raw_text: str, visible_text: str) -> Tuple[str, str]:
        """Return layout-only leading/trailing whitespace hidden by a word."""

        if raw_text.strip() != visible_text:
            return "", ""
        leading = raw_text[: len(raw_text) - len(raw_text.lstrip())]
        trailing = raw_text[len(raw_text.rstrip()) :]
        return leading, trailing

    def _update_char_text(
        self,
        element: TextElement,
        value: str,
        store: Dict[str, List[Any]],
        chars: List[TextElement],
    ) -> None:
        char_index = chars.index(element)
        affected_words: List[TextElement] = []
        for word in store.get("words", []):
            if char_index in getattr(word, "_char_indices", ()) or any(
                item is element._obj for item in getattr(word, "_char_dicts", ())
            ):
                affected_words.append(word)

        # Resolve inferred-space expansion before changing the raw char value;
        # afterward the old word text would no longer match the raw glyph run.
        from natural_pdf.text.operations import _expand_word_chars_for_injected_spaces

        word_states: Dict[int, Tuple[List[Dict[str, Any]], str, str]] = {}
        for word in affected_words:
            char_dicts = self._word_char_dicts(word, chars)
            raw_text = "".join(str(char_dict.get("text", "")) for char_dict in char_dicts)
            visible_text = word.text
            expanded = _expand_word_chars_for_injected_spaces(char_dicts, visible_text)
            leading, trailing = self._hidden_boundary_whitespace(raw_text, visible_text)
            word_states[id(word)] = (expanded or char_dicts, leading, trailing)

        element._set_text_value(value, user_edit=True)

        for word in affected_words:
            char_dicts, leading, trailing = word_states[id(word)]
            if not getattr(word, "_char_dicts", None):
                word._char_dicts = char_dicts
            text = "".join(
                str(char_dict.get("text", ""))
                for char_dict in char_dicts
                if isinstance(char_dict, dict)
            )
            if leading and text.startswith(leading):
                text = text[len(leading) :]
            if trailing and text.endswith(trailing):
                text = text[: -len(trailing)]
            word._set_text_value(text, user_edit=True)

        self._store.mark_dirty(["chars", "words"])

    @property
    def chars(self):
        """Get all character elements."""
        return self._materialize_chars()

    def invalidate_cache(self, *, preserve_overlays: bool = True):
        """Rebuild native elements without discarding synthetic/manual overlays."""

        with self._store.transaction():
            had_content = self._store.is_populated() or self._raw_char_dicts is not None
            if preserve_overlays:
                captured = self._capture_preserved_elements()
                for kind, values in captured.items():
                    existing = self._preserved_elements.setdefault(kind, [])
                    self._extend_unique(existing, values)
            else:
                self._preserved_elements = {}
            self._raw_char_dicts = None
            self._store.clear()
            if had_content:
                self._mark_content_mutated()
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

    @staticmethod
    def _center_is_in_bbox(element: Any, bbox: Tuple[float, float, float, float]) -> bool:
        """Return whether an element's center lies in an inclusive target box."""

        obj = element if isinstance(element, dict) else getattr(element, "_obj", None)
        if isinstance(obj, dict):
            values = tuple(obj.get(key) for key in ("x0", "top", "x1", "bottom"))
        else:
            candidate_bbox = getattr(element, "bbox", None)
            values = candidate_bbox if isinstance(candidate_bbox, (list, tuple)) else ()

        if len(values) != 4 or not all(isinstance(value, (int, float)) for value in values):
            return False

        x0, top, x1, bottom = (float(value) for value in values)
        center_x = (x0 + x1) / 2.0
        center_y = (top + bottom) / 2.0
        bx0, btop, bx1, bbottom = bbox
        return bx0 <= center_x <= bx1 and btop <= center_y <= bbottom

    @staticmethod
    def _word_char_dicts(word: Any, chars: List[Any]) -> List[Dict[str, Any]]:
        char_dicts = [
            char_dict
            for char_dict in getattr(word, "_char_dicts", ())
            if isinstance(char_dict, dict)
        ]
        if char_dicts:
            return char_dicts
        return [
            chars[index]._obj
            for index in getattr(word, "_char_indices", ())
            if isinstance(index, int)
            and 0 <= index < len(chars)
            and isinstance(getattr(chars[index], "_obj", None), dict)
        ]

    def _text_after_char_removal(
        self,
        word: TextElement,
        old_char_dicts: List[Dict[str, Any]],
        removed_dict_ids: set[int],
    ) -> str:
        """Rebuild visible word text after constituent chars are removed."""

        from natural_pdf.text.operations import _expand_word_chars_for_injected_spaces

        raw_text = "".join(str(char.get("text", "")) for char in old_char_dicts)
        visible_text = word.text
        leading, trailing = self._hidden_boundary_whitespace(raw_text, visible_text)
        expanded = _expand_word_chars_for_injected_spaces(old_char_dicts, visible_text)
        sequence = expanded if expanded is not None else old_char_dicts
        had_inferred_spaces = expanded is not None and len(expanded) != len(old_char_dicts)
        text = "".join(
            str(char.get("text", "")) for char in sequence if id(char) not in removed_dict_ids
        )
        if leading and text.startswith(leading):
            text = text[len(leading) :]
        if trailing and text.endswith(trailing):
            text = text[: -len(trailing)]
        return text.strip() if had_inferred_spaces else text

    def _remove_text_members(
        self,
        store: Dict[str, List[Any]],
        *,
        words_to_remove: Iterable[Any] = (),
        chars_to_remove: Iterable[Any] = (),
    ) -> Tuple[int, int]:
        """Remove text objects and repair both layers by backing-dict identity."""

        words = list(store.get("words", []))
        chars = self._materialize_chars()
        words_to_remove = tuple(words_to_remove)
        chars_to_remove = tuple(chars_to_remove)
        selected_word_ids = {id(word) for word in words_to_remove}
        selected_char_ids = {id(char) for char in chars_to_remove}
        selected_char_dict_ids = {
            id(char._obj)
            for char in chars_to_remove
            if isinstance(getattr(char, "_obj", None), dict)
        }

        retained_words = [word for word in words if id(word) not in selected_word_ids]
        retained_word_dict_ids = {
            id(char_dict)
            for word in retained_words
            for char_dict in self._word_char_dicts(word, chars)
        }
        removed_word_dict_ids = {
            id(char_dict)
            for word in words
            if id(word) in selected_word_ids
            for char_dict in self._word_char_dicts(word, chars)
        }
        removed_dict_ids = selected_char_dict_ids | (removed_word_dict_ids - retained_word_dict_ids)

        rebuilt_chars = [
            char
            for char in chars
            if id(char) not in selected_char_ids
            and id(getattr(char, "_obj", None)) not in removed_dict_ids
        ]
        available_dict_ids = {
            id(char._obj) for char in rebuilt_chars if isinstance(getattr(char, "_obj", None), dict)
        }

        coherent_words: List[Any] = []
        for word in retained_words:
            old_char_dicts = self._word_char_dicts(word, chars)
            kept_char_dicts = [
                char_dict
                for char_dict in old_char_dicts
                if id(char_dict) in available_dict_ids and id(char_dict) not in removed_dict_ids
            ]
            if old_char_dicts and not kept_char_dicts:
                continue
            # Preserve backing identities even when this word started in the
            # index-only representation; _reindex_words operates by identity.
            word._char_dicts = kept_char_dicts
            if len(kept_char_dicts) != len(old_char_dicts):
                kept_dict_ids = {id(item) for item in kept_char_dicts}
                removed_from_word = {
                    id(char_dict)
                    for char_dict in old_char_dicts
                    if id(char_dict) not in kept_dict_ids
                }
                word._set_text_value(
                    self._text_after_char_removal(word, old_char_dicts, removed_from_word),
                    user_edit=True,
                )
            coherent_words.append(word)

        self._reindex_words(coherent_words, rebuilt_chars)
        if len(coherent_words) != len(words):
            self._store.set("words", coherent_words)
        elif selected_char_ids or selected_char_dict_ids:
            self._store.mark_dirty(["words"])
        if len(rebuilt_chars) != len(chars):
            self._store.set("chars", rebuilt_chars)

        return len(words) - len(coherent_words), len(chars) - len(rebuilt_chars)

    def remove_text_elements_in_bbox(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
        *,
        sources: Optional[Iterable[str]] = None,
        predicate: Optional[Callable[[Any], bool]] = None,
    ) -> Tuple[int, int]:
        """Remove matching text elements within ``bbox``.

        Geometry is decided by word center so a Region OCR operation cannot
        erase unrelated words elsewhere on the same page. Character entries
        linked to removed words are removed with those words, keeping the two
        layers consistent. ``bbox=None`` targets the entire page. ``predicate``
        can further restrict removal without teaching this manager about every
        feature-specific text subtype.
        """

        normalized_sources = set(sources) if sources is not None else None

        def source_matches(element: Any) -> bool:
            source_ok = (
                normalized_sources is None or self._element_source(element) in normalized_sources
            )
            return source_ok and (predicate is None or bool(predicate(element)))

        def geometry_matches(element: Any) -> bool:
            return bbox is None or self._center_is_in_bbox(element, bbox)

        with self._store.transaction():
            store = self._element_store()
            words = list(store.get("words", []))
            chars = self._materialize_chars()

            removed_words = [
                word for word in words if source_matches(word) and geometry_matches(word)
            ]
            all_linked_char_dict_ids = {
                id(char_dict) for word in words for char_dict in self._word_char_dicts(word, chars)
            }
            standalone_chars = [
                char
                for char in chars
                if id(getattr(char, "_obj", None)) not in all_linked_char_dict_ids
                and source_matches(char)
                and geometry_matches(char)
            ]
            _, removed_char_count = self._remove_text_members(
                store,
                words_to_remove=removed_words,
                chars_to_remove=standalone_chars,
            )
            if removed_words or removed_char_count:
                self._mark_content_mutated()
        return len(removed_words), removed_char_count

    def remove_ocr_elements(self, bbox: Optional[Tuple[float, float, float, float]] = None) -> int:
        """Remove OCR text within an optional geometry scope."""

        removed_words, removed_chars = self.remove_text_elements_in_bbox(bbox, sources={"ocr"})
        removed_count = removed_words + removed_chars
        logger.info("Page %s: Removed %d OCR elements.", self._page.number, removed_count)
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
        normalized_type = {"word": "words", "char": "chars"}.get(element_type, element_type)
        with self._store.transaction():
            store = self._element_store()
            if normalized_type == "chars":
                self._materialize_chars()
                store = self._element_store()
            if normalized_type not in store:
                raise KeyError(f"Element collection '{normalized_type}' does not exist")
            if element not in store[normalized_type]:
                logger.debug("Element not found in %s: %s", normalized_type, element)
                return False

            if normalized_type == "words":
                self._remove_text_members(store, words_to_remove=(element,))
            elif normalized_type == "chars":
                self._remove_text_members(store, chars_to_remove=(element,))
            else:
                remaining = [item for item in store[normalized_type] if item is not element]
                self._store.set(normalized_type, remaining)
            self._mark_content_mutated()

        logger.debug("Removed element from %s: %s", normalized_type, element)
        return True

    def remove_elements_by_source(self, element_type: str, source: str) -> int:
        """Remove all elements of ``element_type`` whose ``source`` attribute matches ``source``."""
        normalized_type = {"word": "words", "char": "chars"}.get(element_type, element_type)
        with self._store.transaction():
            store = self._element_store()
            if normalized_type == "chars":
                self._materialize_chars()
                store = self._element_store()
            if normalized_type not in store:
                return 0

            selected = [
                element
                for element in store[normalized_type]
                if self._element_source(element) == source
            ]
            removed = len(selected)
            if not removed:
                return 0
            if normalized_type == "words":
                self._remove_text_members(store, words_to_remove=selected)
            elif normalized_type == "chars":
                self._remove_text_members(store, chars_to_remove=selected)
            else:
                selected_ids = {id(element) for element in selected}
                self._store.set(
                    normalized_type,
                    [
                        element
                        for element in store[normalized_type]
                        if id(element) not in selected_ids
                    ],
                )
            self._mark_content_mutated()

        logger.info(
            "Page %s: Removed %d '%s' element(s) with source '%s'.",
            getattr(self._page, "number", "?"),
            removed,
            normalized_type,
            source,
        )
        return removed

    def clear_text_layer(
        self, bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> tuple[int, int]:
        """Remove all text within an optional geometry scope."""

        return self.remove_text_elements_in_bbox(bbox)

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
