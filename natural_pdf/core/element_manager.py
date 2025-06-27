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
import re
from contextlib import contextmanager
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple, Union

from pdfplumber.utils.text import WordExtractor

from natural_pdf.elements.image import ImageElement
from natural_pdf.elements.line import LineElement
from natural_pdf.elements.rect import RectangleElement
from natural_pdf.elements.text import TextElement

logger = logging.getLogger(__name__)

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
    # Save original setter
    original_setter = TextElement.text.fset

    # Create a fast setter that skips sync
    def fast_setter(self, value):
        self._obj["text"] = value
        # Skip character synchronization for performance

    # Apply fast setter
    TextElement.text = property(TextElement.text.fget, fast_setter)

    try:
        yield
    finally:
        # Restore original setter
        TextElement.text = property(TextElement.text.fget, original_setter)


class NaturalWordExtractor(WordExtractor):
    """Custom WordExtractor that splits words based on specified character attributes.

    This class extends pdfplumber's WordExtractor to provide more intelligent word
    segmentation that respects font boundaries and other character attributes.
    It prevents words from spanning across different fonts, sizes, or styles,
    which is essential for maintaining semantic meaning in document analysis.

    The extractor considers multiple character attributes when determining word
    boundaries, ensuring that visually distinct text elements (like bold headers
    mixed with regular text) are properly separated into distinct words.

    Attributes:
        font_attrs: List of character attributes to consider for word boundaries.
            Common attributes include 'fontname', 'size', 'flags', etc.

    Example:
        ```python
        # Create extractor that splits on font and size changes
        extractor = NaturalWordExtractor(['fontname', 'size'])

        # Extract words with font-aware boundaries
        words = extractor.extract_words(page_chars)

        # Each word will have consistent font properties
        for word in words:
            print(f"'{word['text']}' in {word['fontname']} size {word['size']}")
        ```
    in addition to pdfplumber's default spatial logic.
    """

    def __init__(self, word_split_attributes: List[str], extra_attrs: List[str], *args, **kwargs):
        """
        Initialize the extractor.

        Args:
            word_split_attributes: List of character attributes (keys in char dict)
                                   that should trigger a word split if they differ
                                   between adjacent characters.
            extra_attrs: List of character attributes (keys in char dict)
                         to copy from the first char of a word into the
                         resulting word dictionary.
            *args: Positional arguments passed to WordExtractor parent.
            **kwargs: Keyword arguments passed to WordExtractor parent.
        """
        self.word_split_attributes = word_split_attributes or []
        # Remove our custom arg before passing to parent
        # (Though WordExtractor likely ignores unknown kwargs)
        # Ensure it's removed if it exists in kwargs
        if "word_split_attributes" in kwargs:
            del kwargs["word_split_attributes"]
        # Pass extra_attrs to the parent constructor
        kwargs["extra_attrs"] = extra_attrs
        super().__init__(*args, **kwargs)

    def char_begins_new_word(
        self,
        prev_char: Dict[str, Any],
        curr_char: Dict[str, Any],
        direction: str,
        x_tolerance: float,
        y_tolerance: float,
    ) -> bool:
        """
        Determine if curr_char begins a new word, considering spatial and
        attribute differences.
        """
        # 1. Check pdfplumber's spatial logic first
        spatial_split = super().char_begins_new_word(
            prev_char, curr_char, direction, x_tolerance, y_tolerance
        )
        if spatial_split:
            return True

        # 2. Check for differences in specified attributes
        if self.word_split_attributes:
            for attr in self.word_split_attributes:
                # Use .get() for safety, although _prepare_char_dicts should ensure presence
                if prev_char.get(attr) != curr_char.get(attr):
                    logger.debug(
                        f"Splitting word due to attribute mismatch on '{attr}': {prev_char.get(attr)} != {curr_char.get(attr)}"
                    )
                    return True  # Attribute mismatch forces a new word

        # If both spatial and attribute checks pass, it's the same word
        return False


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
        self._elements = None  # Lazy-loaded
        self._load_text = load_text
        # Default to splitting by fontname, size, bold, italic if not specified
        # Renamed internal variable for clarity
        self._word_split_attributes = (
            ["fontname", "size", "bold", "italic"] if font_attrs is None else font_attrs
        )

    def load_elements(self):
        """
        Load all elements from the page (lazy loading).
        Uses NaturalWordExtractor for word grouping.
        """
        if self._elements is not None:
            return

        logger.debug(f"Page {self._page.number}: Loading elements...")

        # 1. Prepare character dictionaries only if loading text
        if self._load_text:
            prepared_char_dicts = self._prepare_char_dicts()
            logger.debug(
                f"Page {self._page.number}: Prepared {len(prepared_char_dicts)} character dictionaries."
            )
        else:
            prepared_char_dicts = []
            logger.debug(f"Page {self._page.number}: Skipping text loading (load_text=False)")

        # -------------------------------------------------------------
        # Detect strikethrough (horizontal strike-out lines) on raw
        # characters BEFORE we run any word-grouping.  This way the
        # NaturalWordExtractor can use the presence/absence of a
        # "strike" attribute to decide whether two neighbouring chars
        # belong to the same word.
        # -------------------------------------------------------------

        if self._load_text and prepared_char_dicts:
            try:
                self._mark_strikethrough_chars(prepared_char_dicts)
            except (
                Exception
            ) as strike_err:  # pragma: no cover – strike detection must never crash loading
                logger.warning(
                    f"Page {self._page.number}: Strikethrough detection failed – {strike_err}",
                    exc_info=True,
                )

        # -------------------------------------------------------------
        # Detect underlines on raw characters (must come after strike so
        # both attributes are present before word grouping).
        # -------------------------------------------------------------

        if self._load_text and prepared_char_dicts:
            try:
                self._mark_underline_chars(prepared_char_dicts)
            except Exception as u_err:  # pragma: no cover
                logger.warning(
                    f"Page {self._page.number}: Underline detection failed – {u_err}",
                    exc_info=True,
                )

        # Detect highlights
        if self._load_text and prepared_char_dicts:
            try:
                self._mark_highlight_chars(prepared_char_dicts)
            except Exception as h_err:
                logger.warning(
                    f"Page {self._page.number}: Highlight detection failed – {h_err}",
                    exc_info=True,
                )

        # Create a mapping from character dict to index for efficient lookup
        if self._load_text:
            char_to_index = {}
            for idx, char_dict in enumerate(prepared_char_dicts):
                key = (
                    char_dict.get("x0", 0),
                    char_dict.get("top", 0),
                    char_dict.get("text", ""),
                )
                char_to_index[key] = idx
        else:
            char_to_index = {}

        # 2. Instantiate the custom word extractor
        # Prefer page-level config over PDF-level for tolerance lookup
        word_elements: List[TextElement] = []

        # Get config objects (needed for auto_text_tolerance check)
        page_config = getattr(self._page, "_config", {})
        pdf_config = getattr(self._page._parent, "_config", {})

        # Initialize tolerance variables
        xt = None
        yt = None
        use_flow = pdf_config.get("use_text_flow", False)

        if self._load_text and prepared_char_dicts:
            # Start with any explicitly supplied tolerances (may be None)
            xt = page_config.get("x_tolerance", pdf_config.get("x_tolerance"))
            yt = page_config.get("y_tolerance", pdf_config.get("y_tolerance"))

        # ------------------------------------------------------------------
        # Auto-adaptive tolerance: scale based on median character size when
        # requested and explicit values are absent.
        # ------------------------------------------------------------------
        if self._load_text and pdf_config.get("auto_text_tolerance", True):
            import statistics

            sizes = [c.get("size", 0) for c in prepared_char_dicts if c.get("size")]
            median_size = None
            if sizes:
                median_size = statistics.median(sizes)
                if xt is None:
                    xt = 0.25 * median_size  # ~kerning width
                    # Record back to page config for downstream users
                    page_config["x_tolerance"] = xt
                if yt is None:
                    yt = 0.6 * median_size  # ~line spacing fraction
                    page_config["y_tolerance"] = yt

            # Warn users when the page's font size is extremely small –
            # this is often the root cause of merged-row/column issues.
            if median_size and median_size < 6:  # 6 pt is unusually small
                logger.warning(
                    f"Page {self._page.number}: Median font size is only {median_size:.1f} pt; "
                    f"auto-set x_tolerance={xt:.2f}, y_tolerance={yt:.2f}. "
                    "If the output looks wrong you can override these values via "
                    "PDF(..., text_tolerance={'x_tolerance': X, 'y_tolerance': Y}, "
                    "auto_text_tolerance=False)."
                )

        # Fallback to pdfplumber defaults if still None
        if xt is None:
            xt = 3
        if yt is None:
            yt = 3

        # List of attributes to preserve on word objects
        attributes_to_preserve = list(
            set(
                self._word_split_attributes
                + [
                    "non_stroking_color",
                    "strike",
                    "underline",
                    "highlight",
                    "highlight_color",
                ]
            )
        )

        # ------------------------------------------------------------------
        # NEW: Detect direction (LTR vs RTL) per visual line and feed
        #       pdfplumber's WordExtractor with the correct settings.
        # -------------------------------------------------------------
        import unicodedata

        def _is_rtl_char(ch: str) -> bool:
            """Return True if the character has an RTL bidi class."""
            if not ch:
                return False
            # If string has more than one character take first (works for most PDFs)
            first = ch[0]
            try:
                return unicodedata.bidirectional(first) in ("R", "AL", "AN")
            except Exception:
                return False

        # Helper: group characters into visual lines using y-tolerance
        sorted_chars_for_line_grouping = sorted(
            prepared_char_dicts,
            key=lambda c: (round(c.get("top", 0) / max(yt, 1)) * yt, c.get("x0", 0)),
        )

        lines: List[List[Dict[str, Any]]] = []
        current_line_key = None
        for char_dict in sorted_chars_for_line_grouping:
            top_val = char_dict.get("top", 0)
            line_key = round(top_val / max(yt, 1))  # bucket index
            if current_line_key is None or line_key != current_line_key:
                # start new line bucket
                lines.append([])
                current_line_key = line_key
            lines[-1].append(char_dict)

        # Process each line separately with direction detection
        for line_chars in lines:
            if not line_chars:
                continue
            # Determine RTL ratio
            rtl_count = sum(1 for ch in line_chars if _is_rtl_char(ch.get("text", "")))
            ltr_count = len(line_chars) - rtl_count
            # Consider RTL if it has strictly more RTL than LTR strong characters
            is_rtl_line = rtl_count > ltr_count

            # Build a WordExtractor tailored for this line's direction
            if is_rtl_line:
                line_dir = "ttb"  # horizontal lines stacked top→bottom
                # Feed characters in right→left x-order; extractor can then treat
                # them as left-to-right so that resulting text stays logical.
                char_dir = "ltr"
            else:
                line_dir = "ttb"
                char_dir = "ltr"

            extractor = NaturalWordExtractor(
                word_split_attributes=self._word_split_attributes
                + ["strike", "underline", "highlight"],
                extra_attrs=attributes_to_preserve,
                x_tolerance=xt,
                y_tolerance=yt,
                keep_blank_chars=True,
                use_text_flow=use_flow,
                line_dir=line_dir,
                char_dir=char_dir,
            )

            # Prepare character sequence for the extractor:
            # Always feed characters in spatial order (x0 ascending)
            # PDF stores glyphs in visual order, so this gives us the visual sequence
            line_chars_for_extractor = sorted(line_chars, key=lambda c: c.get("x0", 0))

            try:
                word_tuples = extractor.iter_extract_tuples(line_chars_for_extractor)
            except Exception as e:  # pragma: no cover
                logger.error(
                    f"Word extraction failed on line (rtl={is_rtl_line}) of page {self._page.number}: {e}",
                    exc_info=True,
                )
                word_tuples = []

            for word_dict, char_list in word_tuples:
                # Memory optimisation for char indices
                char_indices = []
                for char_dict in char_list:
                    key = (
                        char_dict.get("x0", 0),
                        char_dict.get("top", 0),
                        char_dict.get("text", ""),
                    )
                    # char_to_index dict built earlier in load_elements
                    if key in char_to_index:
                        char_indices.append(char_to_index[key])
                word_dict["_char_indices"] = char_indices
                word_dict["_char_dicts"] = char_list  # keep for back-compat
                # Create and append TextElement
                word_element = self._create_word_element(word_dict)
                word_elements.append(word_element)

                # Decide if this individual word contains RTL characters; safer than relying
                # on the whole-line heuristic.
                rtl_in_word = any(_is_rtl_char(ch.get("text", "")) for ch in char_list)
                if rtl_in_word:
                    # Convert from visual order (from PDF) to logical order using bidi
                    try:
                        from bidi.algorithm import get_display  # type: ignore

                        from natural_pdf.utils.bidi_mirror import mirror_brackets

                        with disable_text_sync():
                            # word_element.text is currently in visual order (from PDF)
                            # Convert to logical order using bidi with auto direction detection
                            logical_text = get_display(word_element.text, base_dir="L")
                            # Apply bracket mirroring for logical order
                            word_element.text = mirror_brackets(logical_text)
                    except Exception:
                        pass

        # ------------------------------------------------------------------
        #  Propagate per-char strikethrough info up to word level.
        # ------------------------------------------------------------------

        if prepared_char_dicts:
            for w in word_elements:
                strike_chars = 0
                total_chars = 0
                if getattr(w, "_char_indices", None):
                    for idx in w._char_indices:
                        if 0 <= idx < len(prepared_char_dicts):
                            total_chars += 1
                            if prepared_char_dicts[idx].get("strike"):
                                strike_chars += 1
                elif getattr(w, "_char_dicts", None):
                    for ch in w._char_dicts:
                        total_chars += 1
                        if ch.get("strike"):
                            strike_chars += 1

                if total_chars:
                    w._obj["strike"] = (strike_chars / total_chars) >= 0.6
                else:
                    w._obj["strike"] = False

                # underline propagation
                ul_chars = 0
                if getattr(w, "_char_indices", None):
                    for idx in w._char_indices:
                        if 0 <= idx < len(prepared_char_dicts):
                            if prepared_char_dicts[idx].get("underline"):
                                ul_chars += 1
                elif getattr(w, "_char_dicts", None):
                    ul_chars = sum(1 for ch in w._char_dicts if ch.get("underline"))

                if total_chars:
                    w._obj["underline"] = (ul_chars / total_chars) >= 0.6
                else:
                    w._obj["underline"] = False

                # highlight propagation
                hl_chars = 0
                if getattr(w, "_char_indices", None):
                    for idx in w._char_indices:
                        if 0 <= idx < len(prepared_char_dicts):
                            if prepared_char_dicts[idx].get("highlight"):
                                hl_chars += 1
                elif getattr(w, "_char_dicts", None):
                    hl_chars = sum(1 for ch in w._char_dicts if ch.get("highlight"))

                if total_chars:
                    w._obj["highlight"] = (hl_chars / total_chars) >= 0.6
                else:
                    w._obj["highlight"] = False

                # Determine dominant highlight color among chars
                if w._obj.get("highlight"):
                    color_counts = {}
                    source_iter = (
                        (prepared_char_dicts[idx] for idx in w._char_indices)
                        if getattr(w, "_char_indices", None)
                        else w._char_dicts if getattr(w, "_char_dicts", None) else []
                    )
                    for chd in source_iter:
                        if chd.get("highlight") and chd.get("highlight_color") is not None:
                            col = chd["highlight_color"]
                            color_counts[col] = color_counts.get(col, 0) + 1

                    if color_counts:
                        dominant_color = max(color_counts.items(), key=lambda t: t[1])[0]
                        try:
                            w._obj["highlight_color"] = (
                                tuple(dominant_color)
                                if isinstance(dominant_color, (list, tuple))
                                else dominant_color
                            )
                        except Exception:
                            w._obj["highlight_color"] = dominant_color

        # generated_words defaults to empty list if text loading is disabled
        generated_words = word_elements if self._load_text else []
        logger.debug(
            f"Page {self._page.number}: Generated {len(generated_words)} words using NaturalWordExtractor."
        )

        # 4. Load other elements (rects, lines)
        rect_elements = [RectangleElement(r, self._page) for r in self._page._page.rects]
        line_elements = [LineElement(l, self._page) for l in self._page._page.lines]
        image_elements = [ImageElement(i, self._page) for i in self._page._page.images]
        logger.debug(
            f"Page {self._page.number}: Loaded {len(rect_elements)} rects, {len(line_elements)} lines, {len(image_elements)} images."
        )

        # 5. Create the final elements dictionary
        self._elements = {
            # Store original char elements if needed (e.g., for visualization/debugging)
            # We re-create them here from the prepared dicts
            "chars": [TextElement(c_dict, self._page) for c_dict in prepared_char_dicts],
            "words": generated_words,
            "rects": rect_elements,
            "lines": line_elements,
            "images": image_elements,
        }

        # Add regions if they exist
        if hasattr(self._page, "_regions") and (
            "detected" in self._page._regions or "named" in self._page._regions
        ):
            regions = []
            if "detected" in self._page._regions:
                regions.extend(self._page._regions["detected"])
            if "named" in self._page._regions:
                regions.extend(self._page._regions["named"].values())
            self._elements["regions"] = regions
            logger.debug(f"Page {self._page.number}: Added {len(regions)} regions.")
        else:
            self._elements["regions"] = []  # Ensure key exists

        logger.debug(f"Page {self._page.number}: Element loading complete.")

        # If per-word BiDi was skipped, generated_words already stay in logical order.

    def _prepare_char_dicts(self) -> List[Dict[str, Any]]:
        """
        Prepares a list of character dictionaries from native PDF characters,
        augmenting them with necessary attributes like bold/italic flags.
        This method focuses ONLY on native characters. OCR results are
        handled separately by create_text_elements_from_ocr.

        Returns:
            List of augmented native character dictionaries.
        """
        prepared_dicts = []
        processed_native_ids = set()  # To track processed native chars

        # 1. Process Native PDF Characters
        native_chars = self._page._page.chars or []
        logger.debug(f"Page {self._page.number}: Preparing {len(native_chars)} native char dicts.")
        for i, char_dict in enumerate(native_chars):
            # Create a temporary TextElement for analysis ONLY
            # We need to ensure the char_dict has necessary keys first
            if not all(k in char_dict for k in ["x0", "top", "x1", "bottom", "text"]):
                logger.warning(f"Skipping native char dict due to missing keys: {char_dict}")
                continue

            temp_element = TextElement(char_dict, self._page)

            # Augment the original dictionary
            augmented_dict = char_dict.copy()  # Work on a copy
            augmented_dict["bold"] = temp_element.bold
            augmented_dict["italic"] = temp_element.italic
            augmented_dict["source"] = "native"
            # Copy color if it exists
            if "non_stroking_color" in char_dict:
                augmented_dict["non_stroking_color"] = char_dict["non_stroking_color"]
            # Ensure basic required keys are present
            augmented_dict.setdefault("upright", True)
            augmented_dict.setdefault("fontname", "Unknown")
            augmented_dict.setdefault("size", 0)
            augmented_dict.setdefault("highlight_color", None)
            # Ensure decoration keys exist for safe grouping
            augmented_dict.setdefault("strike", False)
            augmented_dict.setdefault("underline", False)
            augmented_dict.setdefault("highlight", False)

            prepared_dicts.append(augmented_dict)
            # Use a unique identifier if available (e.g., tuple of key properties)
            # Simple approach: use index for now, assuming list order is stable here
            processed_native_ids.add(i)

        # 2. Remove OCR Processing from this method
        # OCR results will be added later via create_text_elements_from_ocr

        logger.debug(
            f"Page {self._page.number}: Total prepared native char dicts: {len(prepared_dicts)}"
        )
        return prepared_dicts

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
        added_word_elements = []
        if self._elements is None:
            # Trigger loading of native elements if not already done
            logger.debug(
                f"Page {self._page.number}: create_text_elements_from_ocr triggering initial load_elements."
            )
            self.load_elements()

        # Ensure scales are valid numbers
        scale_x = float(scale_x) if scale_x is not None else 1.0
        scale_y = float(scale_y) if scale_y is not None else 1.0

        logger.debug(
            f"Page {self._page.number}: Adding {len(ocr_results)} OCR results as elements. Scale: x={scale_x:.2f}, y={scale_y:.2f}"
        )

        # Ensure the target lists exist in the _elements dict
        if self._elements is None:
            logger.error(
                f"Page {self._page.number}: _elements dictionary is None after load_elements call in create_text_elements_from_ocr. Cannot add OCR elements."
            )
            return []  # Cannot proceed

        if "words" not in self._elements:
            self._elements["words"] = []
        if "chars" not in self._elements:
            self._elements["chars"] = []

        for result in ocr_results:
            try:
                x0_img, top_img, x1_img, bottom_img = map(float, result["bbox"])
                height_img = bottom_img - top_img
                pdf_x0 = x0_img * scale_x
                pdf_top = top_img * scale_y
                pdf_x1 = x1_img * scale_x
                pdf_bottom = bottom_img * scale_y
                pdf_height = (bottom_img - top_img) * scale_y

                # Handle potential None confidence
                raw_confidence = result.get("confidence")
                confidence_value = (
                    float(raw_confidence) if raw_confidence is not None else None
                )  # Keep None if it was None
                ocr_text = result.get("text")  # Get text, will be None if detect_only

                # Create the TextElement for the word
                word_element_data = {
                    "text": ocr_text,
                    "x0": pdf_x0,
                    "top": pdf_top,
                    "x1": pdf_x1,
                    "bottom": pdf_bottom,
                    "width": (x1_img - x0_img) * scale_x,
                    "height": pdf_height,
                    "object_type": "word",  # Treat OCR results as whole words
                    "source": "ocr",
                    "confidence": confidence_value,  # Use the handled confidence
                    "fontname": "OCR",  # Use consistent OCR fontname
                    "size": (
                        round(pdf_height) if pdf_height > 0 else 10.0
                    ),  # Use calculated PDF height for size
                    "page_number": self._page.number,
                    "bold": False,
                    "italic": False,
                    "upright": True,
                    "doctop": pdf_top + self._page._page.initial_doctop,
                    "strike": False,
                    "underline": False,
                    "highlight": False,
                    "highlight_color": None,
                }

                # Create the representative char dict for this OCR word
                ocr_char_dict = word_element_data.copy()
                ocr_char_dict["object_type"] = "char"
                ocr_char_dict.setdefault("adv", ocr_char_dict.get("width", 0))
                # Ensure decoration keys
                ocr_char_dict.setdefault("strike", False)
                ocr_char_dict.setdefault("underline", False)
                ocr_char_dict.setdefault("highlight", False)
                ocr_char_dict.setdefault("highlight_color", None)

                # Add the char dict list to the word data before creating TextElement
                word_element_data["_char_dicts"] = [ocr_char_dict]  # Store itself as its only char

                word_elem = TextElement(word_element_data, self._page)
                added_word_elements.append(word_elem)

                # Append the word element to the manager's list
                self._elements["words"].append(word_elem)

                # Only add a representative char dict if text actually exists
                if ocr_text is not None:
                    # This char dict represents the entire OCR word as a single 'char'.
                    char_dict_data = ocr_char_dict  # Use the one we already created
                    char_dict_data["object_type"] = "char"  # Mark as char type
                    char_dict_data.setdefault("adv", char_dict_data.get("width", 0))

                    # Create a TextElement for the char representation
                    # Ensure _char_dicts is handled correctly by TextElement constructor
                    # For an OCR word represented as a char, its _char_dicts can be a list containing its own data
                    char_element_specific_data = char_dict_data.copy()
                    char_element_specific_data["_char_dicts"] = [char_dict_data.copy()]

                    ocr_char_as_element = TextElement(char_element_specific_data, self._page)
                    self._elements["chars"].append(
                        ocr_char_as_element
                    )  # Append TextElement instance

            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Failed to process OCR result: {result}. Error: {e}", exc_info=True)
                continue

        logger.info(
            f"Page {self._page.number}: Appended {len(added_word_elements)} TextElements (words) and corresponding char dicts from OCR results."
        )
        return added_word_elements

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
        self.load_elements()

        # Add to the appropriate list
        if element_type in self._elements:
            # Avoid adding duplicates
            if element not in self._elements[element_type]:
                self._elements[element_type].append(element)
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
        # Load elements if not already loaded
        self.load_elements()

        # Make sure regions is in _elements
        if "regions" not in self._elements:
            self._elements["regions"] = []

        # Add to elements for selector queries
        if region not in self._elements["regions"]:
            self._elements["regions"].append(region)
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
        self.load_elements()

        if element_type:
            return self._elements.get(element_type, [])

        # Combine all element types
        all_elements = []
        for elements in self._elements.values():
            all_elements.extend(elements)

        return all_elements

    def get_all_elements(self):
        """
        Get all elements from all types.

        Returns:
            List of all elements
        """
        # Load elements if not already loaded
        self.load_elements()

        # Combine all element types
        all_elements = []
        if self._elements:  # Ensure _elements is not None
            for elements in self._elements.values():
                if isinstance(elements, list):  # Ensure we only extend lists
                    all_elements.extend(elements)
        return all_elements

    @property
    def chars(self):
        """Get all character elements."""
        self.load_elements()
        return self._elements.get("chars", [])

    @property
    def words(self):
        """Get all word elements."""
        self.load_elements()
        return self._elements.get("words", [])

    @property
    def rects(self):
        """Get all rectangle elements."""
        self.load_elements()
        return self._elements.get("rects", [])

    @property
    def lines(self):
        """Get all line elements."""
        self.load_elements()
        return self._elements.get("lines", [])

    @property
    def regions(self):
        """Get all region elements."""
        self.load_elements()
        return self._elements.get("regions", [])

    @property
    def images(self):
        """Get all image elements."""
        self.load_elements()
        return self._elements.get("images", [])

    def remove_ocr_elements(self):
        """
        Remove all elements with source="ocr" from the elements dictionary.
        This should be called before adding new OCR elements if replacement is desired.

        Returns:
            int: Number of OCR elements removed
        """
        # Load elements if not already loaded
        self.load_elements()

        removed_count = 0

        # Filter out OCR elements from words
        if "words" in self._elements:
            original_len = len(self._elements["words"])
            self._elements["words"] = [
                word for word in self._elements["words"] if getattr(word, "source", None) != "ocr"
            ]
            removed_count += original_len - len(self._elements["words"])

        # Filter out OCR elements from chars
        if "chars" in self._elements:
            original_len = len(self._elements["chars"])
            self._elements["chars"] = [
                char
                for char in self._elements["chars"]
                if (isinstance(char, dict) and char.get("source") != "ocr")
                or (not isinstance(char, dict) and getattr(char, "source", None) != "ocr")
            ]
            removed_count += original_len - len(self._elements["chars"])

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
        # Load elements if not already loaded
        self.load_elements()

        # Check if the collection exists
        if element_type not in self._elements:
            logger.warning(f"Cannot remove element: collection '{element_type}' does not exist")
            return False

        # Try to remove the element
        try:
            if element in self._elements[element_type]:
                self._elements[element_type].remove(element)
                logger.debug(f"Removed element from {element_type}: {element}")
                return True
            else:
                logger.debug(f"Element not found in {element_type}: {element}")
                return False
        except Exception as e:
            logger.error(f"Error removing element from {element_type}: {e}", exc_info=True)
            return False

    def has_elements(self) -> bool:
        """
        Check if any significant elements (words, rects, lines, regions)
        have been loaded or added.

        Returns:
            True if any elements exist, False otherwise.
        """
        self.load_elements()

        for key in ["words", "rects", "lines", "regions"]:
            if self._elements.get(key):
                return True

        return False

    # ------------------------------------------------------------------
    #  Strikethrough detection (horizontal strike-out lines)
    # ------------------------------------------------------------------

    def _mark_strikethrough_chars(
        self,
        char_dicts: List[Dict[str, Any]],
        *,
        thickness_tol: float = 1.5,
        horiz_tol: float = 1.0,
        coverage_ratio: float = 0.7,
        band_top: float = 0.35,
        band_bottom: float = 0.65,
    ) -> None:
        """Annotate character dictionaries with a boolean ``strike`` flag.

        Args
        ----
        char_dicts : list
            The list that _prepare_char_dicts() returned – *modified in place*.
        thickness_tol : float
            Maximum height (in PDF pts) for a path to be considered a strike.
        horiz_tol : float
            Vertical tolerance when deciding if a pdfplumber ``line`` object
            is horizontal (|y0-y1| ≤ horiz_tol).
        coverage_ratio : float
            Minimum proportion of the glyph's width that must be overlapped
            by a candidate line.
        band_top, band_bottom : float
            Fractions of the glyph's height that define the central band in
            which a line must fall to count as a strikethrough.  Defaults to
            35–65 %.
        """

        # -------------------------------------------------------------
        # Collect candidate horizontal primitives (lines + skinny rects)
        # -------------------------------------------------------------
        raw_lines = list(getattr(self._page._page, "lines", []))
        raw_rects = list(getattr(self._page._page, "rects", []))

        candidates: List[Tuple[float, float, float, float]] = []  # (x0, y0, x1, y1)

        # pdfplumber line objects – treat those whose angle ≈ 0°
        for ln in raw_lines:
            y0 = min(ln.get("y0", 0), ln.get("y1", 0))
            y1 = max(ln.get("y0", 0), ln.get("y1", 0))
            if abs(y1 - y0) <= horiz_tol:  # horizontal
                candidates.append((ln.get("x0", 0), y0, ln.get("x1", 0), y1))

        # Thin rectangles that act as drawn lines
        pg_height = self._page.height
        for rc in raw_rects:
            rb0 = rc.get("y0", 0)
            rb1 = rc.get("y1", 0)
            y0_raw = min(rb0, rb1)
            y1_raw = max(rb0, rb1)
            if (y1_raw - y0_raw) <= thickness_tol:
                # Convert from PDF (origin bottom-left) to top-based coords used by chars
                y0 = pg_height - y1_raw  # upper edge distance from top
                y1 = pg_height - y0_raw  # lower edge distance from top
                candidates.append((rc.get("x0", 0), y0, rc.get("x1", 0), y1))

        if not candidates:
            return  # nothing to mark

        # -------------------------------------------------------------
        # Walk through characters and flag those crossed by a candidate
        # -------------------------------------------------------------
        for ch in char_dicts:
            ch.setdefault("strike", False)  # default value
            try:
                x0, top, x1, bottom = ch["x0"], ch["top"], ch["x1"], ch["bottom"]
            except KeyError:
                continue  # skip malformed char dict

            width = x1 - x0
            height = bottom - top
            if width <= 0 or height <= 0:
                continue

            mid_y0 = top + band_top * height
            mid_y1 = top + band_bottom * height

            # Check each candidate line for overlap
            for lx0, ly0, lx1, ly1 in candidates:
                if (ly0 >= (mid_y0 - 1.0)) and (ly1 <= (mid_y1 + 1.0)):  # lies inside central band
                    overlap = min(x1, lx1) - max(x0, lx0)
                    if overlap > 0 and (overlap / width) >= coverage_ratio:
                        ch["strike"] = True
                        break  # no need to check further lines

        # Done – char_dicts mutated in place

    # ------------------------------------------------------------------
    #  Underline detection
    # ------------------------------------------------------------------

    def _mark_underline_chars(
        self,
        char_dicts: List[Dict[str, Any]],
        *,
        thickness_tol: float = None,
        horiz_tol: float = None,
        coverage_ratio: float = None,
        band_frac: float = None,
        below_pad: float = None,
    ) -> None:
        """Annotate character dicts with ``underline`` flag."""

        # Allow user overrides via PDF._config["underline_detection"]
        pdf_cfg = getattr(self._page._parent, "_config", {}).get("underline_detection", {})

        thickness_tol = (
            thickness_tol
            if thickness_tol is not None
            else pdf_cfg.get("thickness_tol", UNDERLINE_DEFAULTS["thickness_tol"])
        )
        horiz_tol = (
            horiz_tol
            if horiz_tol is not None
            else pdf_cfg.get("horiz_tol", UNDERLINE_DEFAULTS["horiz_tol"])
        )
        coverage_ratio = (
            coverage_ratio
            if coverage_ratio is not None
            else pdf_cfg.get("coverage_ratio", UNDERLINE_DEFAULTS["coverage_ratio"])
        )
        band_frac = (
            band_frac
            if band_frac is not None
            else pdf_cfg.get("band_frac", UNDERLINE_DEFAULTS["band_frac"])
        )
        below_pad = (
            below_pad
            if below_pad is not None
            else pdf_cfg.get("below_pad", UNDERLINE_DEFAULTS["below_pad"])
        )

        raw_lines = list(getattr(self._page._page, "lines", []))
        raw_rects = list(getattr(self._page._page, "rects", []))

        candidates: List[Tuple[float, float, float, float]] = []

        for ln in raw_lines:
            y0 = min(ln.get("y0", 0), ln.get("y1", 0))
            y1 = max(ln.get("y0", 0), ln.get("y1", 0))
            if abs(y1 - y0) <= horiz_tol and (
                (ln.get("x1", 0) - ln.get("x0", 0)) < self._page.width * 0.95
            ):  # ignore full-width rules
                candidates.append((ln.get("x0", 0), y0, ln.get("x1", 0), y1))

        pg_height = self._page.height
        for rc in raw_rects:
            rb0 = rc.get("y0", 0)
            rb1 = rc.get("y1", 0)
            y0_raw = min(rb0, rb1)
            y1_raw = max(rb0, rb1)
            if (y1_raw - y0_raw) <= thickness_tol and (
                (rc.get("x1", 0) - rc.get("x0", 0)) < self._page.width * 0.95
            ):
                y0 = pg_height - y1_raw
                y1 = pg_height - y0_raw
                candidates.append((rc.get("x0", 0), y0, rc.get("x1", 0), y1))

        if not candidates:
            for ch in char_dicts:
                ch.setdefault("underline", False)
            return

        # group candidates by y within tolerance 0.5 to detect repeating table borders
        y_groups: Dict[int, int] = {}
        for _, y0, _, y1 in candidates:
            key = int((y0 + y1) / 2)
            y_groups[key] = y_groups.get(key, 0) + 1

        table_y = {k for k, v in y_groups.items() if v >= 3}

        # filter out candidates on those y values
        filtered_candidates = [c for c in candidates if int((c[1] + c[3]) / 2) not in table_y]

        # annotate chars
        for ch in char_dicts:
            ch.setdefault("underline", False)
            try:
                x0, top, x1, bottom = ch["x0"], ch["top"], ch["x1"], ch["bottom"]
            except KeyError:
                continue

            width = x1 - x0
            height = bottom - top
            if width <= 0 or height <= 0:
                continue

            band_top = bottom - band_frac * height
            band_bottom = bottom + below_pad  # allow some distance below baseline

            for lx0, ly0, lx1, ly1 in filtered_candidates:
                if (ly0 >= band_top - 1) and (ly1 <= band_bottom + 1):
                    overlap = min(x1, lx1) - max(x0, lx0)
                    if overlap > 0 and (overlap / width) >= coverage_ratio:
                        ch["underline"] = True
                        break

    # ------------------------------------------------------------------
    #  Highlight detection
    # ------------------------------------------------------------------

    def _mark_highlight_chars(self, char_dicts: List[Dict[str, Any]]) -> None:
        """Detect PDF marker-style highlights and set ``highlight`` on char dicts."""

        cfg = getattr(self._page._parent, "_config", {}).get("highlight_detection", {})

        height_min_ratio = cfg.get("height_min_ratio", HIGHLIGHT_DEFAULTS["height_min_ratio"])
        height_max_ratio = cfg.get("height_max_ratio", HIGHLIGHT_DEFAULTS["height_max_ratio"])
        coverage_ratio = cfg.get("coverage_ratio", HIGHLIGHT_DEFAULTS["coverage_ratio"])

        raw_rects = list(getattr(self._page._page, "rects", []))
        pg_height = self._page.height

        # Build list of candidate highlight rectangles (convert to top-based coords)
        highlight_rects = []
        for rc in raw_rects:
            if rc.get("stroke", False):
                continue  # border stroke, not fill-only
            if not rc.get("fill", False):
                continue

            fill_col = rc.get("non_stroking_color")
            # We keep colour as metadata but no longer filter on it
            if fill_col is None:
                continue

            y0_rect = min(rc.get("y0", 0), rc.get("y1", 0))
            y1_rect = max(rc.get("y0", 0), rc.get("y1", 0))
            rheight = y1_rect - y0_rect
            highlight_rects.append(
                (rc.get("x0", 0), y0_rect, rc.get("x1", 0), y1_rect, rheight, fill_col)
            )

        if not highlight_rects:
            for ch in char_dicts:
                ch.setdefault("highlight", False)
            return

        for ch in char_dicts:
            ch.setdefault("highlight", False)
            try:
                x0_raw, y0_raw, x1_raw, y1_raw = ch["x0"], ch["y0"], ch["x1"], ch["y1"]
            except KeyError:
                continue

            width = x1_raw - x0_raw
            height = y1_raw - y0_raw
            if width <= 0 or height <= 0:
                continue

            for rx0, ry0, rx1, ry1, rheight, rcolor in highlight_rects:
                # height ratio check relative to char
                ratio = rheight / height if height else 0
                if ratio < height_min_ratio or ratio > height_max_ratio:
                    continue

                # vertical containment in raw coords
                if not (y0_raw + 1 >= ry0 and y1_raw - 1 <= ry1):
                    continue

                overlap = min(x1_raw, rx1) - max(x0_raw, rx0)
                if overlap > 0 and (overlap / width) >= coverage_ratio:
                    ch["highlight"] = True
                    try:
                        ch["highlight_color"] = (
                            tuple(rcolor) if isinstance(rcolor, (list, tuple)) else rcolor
                        )
                    except Exception:
                        ch["highlight_color"] = rcolor
                    break
