"""
Text element classes for natural-pdf.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pdfplumber.utils.text import chars_to_textmap

from natural_pdf.elements.base import Element
from natural_pdf.text.facades import ScalarTextMixin
from natural_pdf.text.font_style import detect_bold_style, detect_italic_style, resolve_fontname

if TYPE_CHECKING:
    from natural_pdf.core.page import Page


_TEXT_SYNC_SUPPRESSION_DEPTH: ContextVar[int] = ContextVar(
    "natural_pdf_text_sync_suppression_depth", default=0
)


@contextmanager
def disable_text_sync():
    """Suppress char synchronization in the current task/thread only.

    This is reserved for initial word construction, where the backing character
    layer is already authoritative.  A ContextVar avoids the process-wide class
    monkeypatch previously used by ElementManager.
    """

    token = _TEXT_SYNC_SUPPRESSION_DEPTH.set(_TEXT_SYNC_SUPPRESSION_DEPTH.get() + 1)
    try:
        yield
    finally:
        _TEXT_SYNC_SUPPRESSION_DEPTH.reset(token)


class TextElement(ScalarTextMixin, Element):
    """
    Represents a text element in a PDF.

    This class is a wrapper around pdfplumber's character objects,
    providing additional functionality for text extraction and analysis.
    """

    def __init__(self, obj: Dict[str, Any], page: "Page"):
        """
        Initialize a text element.

        Args:
            obj: The underlying pdfplumber object. For OCR text elements,
                 should include 'text', 'bbox', 'source', and 'confidence'
            page: The parent Page object
        """
        # Add object_type if not present
        if "object_type" not in obj:
            obj["object_type"] = "text"

        super().__init__(obj, page)

        # Memory optimization: Store character indices instead of full dictionaries
        # This reduces memory usage by ~50% by avoiding character data duplication
        self._char_indices = obj.pop("_char_indices", [])

        # Backward compatibility: Keep _char_dicts for existing code
        # But prefer _char_indices when available to save memory
        self._char_dicts = obj.pop("_char_dicts", [])
        self._layout_text_cache: Optional[str] = None
        self._text_manually_set: bool = False
        # Provenance is deliberately separate from the layout/cache flag above.
        # Construction-time BiDi and inferred-space rewrites are not user edits
        # and therefore must not pin native words across a regrouping reload.
        self._text_user_edited: bool = False

    @property
    def chars(self):
        """Get constituent character elements efficiently.

        Uses character indices when available to avoid memory duplication,
        falls back to _char_dicts for backward compatibility.
        """
        if self._char_indices:
            # Memory-efficient approach: access characters by index
            char_elements = self.page.get_elements_by_type("chars")
            return [char_elements[i] for i in self._char_indices if i < len(char_elements)]

        # Backward compatibility: convert _char_dicts to TextElement objects
        if self._char_dicts:
            return [TextElement(char_dict, self.page) for char_dict in self._char_dicts]

        return []

    @property
    def text(self) -> str:
        """Get the text content."""
        if self._layout_text_cache is not None:
            return self._layout_text_cache

        stored = self._obj.get("text", "")
        if self._text_manually_set or not stored:
            self._layout_text_cache = stored
            return stored

        char_dicts = self._resolve_char_dicts_for_textmap()
        if char_dicts:
            try:
                textmap = chars_to_textmap(char_dicts, layout=False)
                computed = textmap.as_string
                if computed:
                    self._obj["text"] = computed
                    self._layout_text_cache = computed
                    return computed
            except Exception:  # pragma: no cover
                pass

        self._layout_text_cache = stored
        return stored

    @text.setter
    def text(self, value: str):
        """Mutate text and keep the page's canonical word/char views coherent."""
        if not isinstance(value, str):
            raise TypeError("TextElement.text must be assigned a string")

        if value == self.text:
            return

        if _TEXT_SYNC_SUPPRESSION_DEPTH.get():
            self._set_text_value(value)
            return

        manager = getattr(self.page, "_element_mgr", None)
        synchronizer = getattr(manager, "update_text_element", None)
        if callable(synchronizer) and synchronizer(self, value):
            return

        # Detached/legacy TextElements still maintain their own character data.
        self._detach_local_text_state()
        self._set_text_value(value, user_edit=True)
        self._sync_local_char_dicts(value)

        # A real manager returning False has authoritatively classified this
        # element as detached, so its local edit is not a page mutation.
        if callable(synchronizer):
            return

        from natural_pdf.services._text_state import bump_text_state

        bump_text_state(self.page, elements=(self,))

    def _set_text_value(self, value: str, *, user_edit: bool = False) -> None:
        """Set only this element's cached/stored value without synchronization."""

        self._obj["text"] = value
        self._layout_text_cache = value
        self._text_manually_set = True
        if user_edit:
            self._text_user_edited = True

    def _detach_local_text_state(self) -> None:
        """Copy backing dictionaries before a detached local mutation."""

        resolved = self._resolve_char_dicts_for_textmap()
        self._obj = self._obj.copy()
        self._char_dicts = [char_dict.copy() for char_dict in resolved]
        self._char_indices = []

    def _sync_local_char_dicts(self, value: str) -> None:
        """Synchronize detached legacy char dictionaries without silent failure."""

        original = [item for item in self._char_dicts if isinstance(item, dict)]
        if not original and not value:
            self._char_dicts = []
            self._char_indices = []
            return

        template = original[-1].copy() if original else self._obj.copy()
        template["object_type"] = "char"
        updated: List[Dict[str, Any]] = []
        for index, char in enumerate(value):
            char_dict = original[index].copy() if index < len(original) else template.copy()
            char_dict["text"] = char
            updated.append(char_dict)

        self._char_dicts = updated
        if self._char_indices:
            self._char_indices = self._char_indices[: len(updated)]

    @property
    def source(self) -> str:
        """Get the source of this text element (pdf or ocr)."""
        return self._obj.get("source", "pdf")

    @property
    def ocr_engine(self) -> Optional[str]:
        """Get the OCR engine that produced this element, if any."""
        return self._obj.get("ocr_engine")

    @property
    def confidence(self) -> float:
        """Get the confidence score for OCR text elements."""
        return self._obj.get("confidence", 1.0)

    @property
    def fontname(self) -> str:
        """Get the font name."""
        return resolve_fontname(self._obj)

    @property
    def font_family(self) -> str:
        """
        Get a cleaner font family name by stripping PDF-specific prefixes.

        PDF font names often include prefixes like 'ABCDEF+' followed by the font name
        or unique identifiers. This method attempts to extract a more readable font name.
        """
        font = self.fontname

        # Remove common PDF font prefixes (e.g., 'ABCDEF+')
        if "+" in font:
            font = font.split("+", 1)[1]

        # Try to extract common font family names
        common_fonts = [
            "Arial",
            "Helvetica",
            "Times",
            "Courier",
            "Calibri",
            "Cambria",
            "Georgia",
            "Verdana",
            "Tahoma",
            "Trebuchet",
        ]

        for common in common_fonts:
            if common.lower() in font.lower():
                return common

        return font

    def _resolve_char_dicts_for_textmap(self) -> List[Dict[str, Any]]:
        """Return raw char dictionaries suitable for chars_to_textmap."""
        if self._char_dicts:
            dicts: List[Dict[str, Any]] = []
            for ch in self._char_dicts:
                if isinstance(ch, dict):
                    dicts.append(ch)
                elif hasattr(ch, "_obj"):
                    dicts.append(ch._obj)
            if dicts:
                return dicts

        if self._char_indices:
            char_elements = self.page.get_elements_by_type("chars")
            dicts = []
            for idx in self._char_indices:
                if idx < len(char_elements):
                    char_el = char_elements[idx]
                    if hasattr(char_el, "_obj"):
                        dicts.append(char_el._obj)
                    elif isinstance(char_el, dict):
                        dicts.append(char_el)
            return dicts

        return []

    @property
    def font_variant(self) -> str:
        """
        Get the font variant identifier (prefix before the '+' in PDF font names).

        PDF embeds font subsets with unique identifiers like 'AAAAAB+FontName'.
        Different variants of the same base font will have different prefixes.
        This can be used to differentiate text that looks different despite
        having the same font name and size.

        Returns:
            The font variant prefix, or empty string if no variant is present
        """
        font = self.fontname

        # Extract the prefix before '+' if it exists
        if "+" in font:
            return font.split("+", 1)[0]

        return ""

    @property
    def size(self) -> float:
        """Get the font size."""
        return self._obj.get("size", 0)

    @property
    def color(self) -> tuple:
        """Get the text color (RGB tuple)."""
        from natural_pdf.utils.color_utils import normalize_pdf_color

        return normalize_pdf_color(self._obj.get("non_stroking_color"))

    def _extract_scalar_text(self) -> str:
        """Return the element's current logical-order text without transforms."""

        return self.text or ""

    def contains(self, substring: str, case_sensitive: bool = True) -> bool:
        """
        Check if this text element contains a substring.

        Args:
            substring: The substring to check for
            case_sensitive: Whether the check is case-sensitive

        Returns:
            True if the text contains the substring
        """
        if case_sensitive:
            return substring in self.text
        else:
            return substring.lower() in self.text.lower()

    def matches(self, pattern: str) -> bool:
        """
        Check if this text element matches a regular expression pattern.

        Args:
            pattern: Regular expression pattern

        Returns:
            True if the text matches the pattern
        """
        import re

        return bool(re.search(pattern, self.text))

    @property
    def bold(self) -> bool:
        """
        Check if the text is bold based on multiple indicators in the PDF.

        PDFs encode boldness in several ways:
        1. Font name containing 'bold' or 'black'
        2. Font descriptor flags (bit 2 indicates bold)
        3. StemV value (thickness of vertical stems)
        4. Font weight values (700+ is typically bold)
        5. Text rendering mode 2 (fill and stroke)
        """
        # Check font name (original method)
        return detect_bold_style(self._obj)

    @property
    def italic(self) -> bool:
        """
        Check if the text is italic based on multiple indicators in the PDF.

        PDFs encode italic (oblique) text in several ways:
        1. Font name containing 'italic' or 'oblique'
        2. Font descriptor flags (bit 6 indicates italic)
        3. Text with non-zero slant angle
        """
        # Check font name (original method)
        return detect_italic_style(self._obj)

    @property
    def strike(self) -> bool:  # alias: struck
        """True if this element (word/char) is marked as strikethrough."""
        # Two possible storage places: raw object dict (comes from extractor
        # via extra_attrs) or metadata (if later pipeline stages mutate).
        return bool(self._obj.get("strike") or self.metadata.get("decoration", {}).get("strike"))

    # Back-compat alias
    @property
    def struck(self) -> bool:  # noqa: D401
        return self.strike

    # -----------------------------
    #  Underline decoration
    # -----------------------------

    @property
    def underline(self) -> bool:
        """True if element is underlined."""
        return bool(
            self._obj.get("underline") or self.metadata.get("decoration", {}).get("underline")
        )

    # -----------------------------
    #  Highlight decoration
    # -----------------------------

    @property
    def is_highlighted(self) -> bool:
        """True if element (char/word) is marked as highlighted in the original PDF."""
        return bool(
            self._obj.get("highlight")
            or self._obj.get("is_highlighted")
            or self.metadata.get("decoration", {}).get("highlight")
        )

    @property
    def highlight_color(self):
        """Return RGB(A) tuple of highlight colour if stored."""
        # Check _obj first, being careful with falsy values like 0.0
        if "highlight_color" in self._obj:
            return self._obj["highlight_color"]
        # Fall back to metadata
        return self.metadata.get("decoration", {}).get("highlight_color")

    def __repr__(self) -> str:
        """String representation of the text element."""
        if self.text:
            preview = self.text[:10] + "..." if len(self.text) > 10 else self.text
        else:
            preview = "..."
        font_style = []
        if self.bold:
            font_style.append("bold")
        if self.italic:
            font_style.append("italic")
        if self.strike:
            font_style.append("strike")
        if self.underline:
            font_style.append("underline")
        if self.is_highlighted:
            font_style.append("highlight")
        style_str = f", style={font_style}" if font_style else ""

        # Use font_family for display but include raw fontname and variant
        font_display = self.font_family
        variant = self.font_variant
        variant_str = f", variant='{variant}'" if variant else ""

        if font_display != self.fontname and "+" in self.fontname:
            base_font = self.fontname.split("+", 1)[1]
            font_display = f"{font_display} ({base_font})"

        color_info = ""
        if self.is_highlighted and self.highlight_color is not None:
            color_info = f", highlight_color={self.highlight_color}"

        return f"<TextElement text='{preview}' font='{font_display}'{variant_str} size={self.size}{style_str}{color_info} bbox={self.bbox}>"

    def font_info(self) -> dict:
        """
        Get detailed font information for this text element.

        Returns a dictionary with all available font-related properties,
        useful for debugging font detection issues.
        """
        info = {
            "text": self.text,
            "fontname": self.fontname,
            "font_family": self.font_family,
            "font_variant": self.font_variant,
            "size": self.size,
            "bold": self.bold,
            "italic": self.italic,
            "color": self.color,
        }

        # Include raw font properties from the PDF
        font_props = [
            "flags",
            "stemv",
            "StemV",
            "weight",
            "FontWeight",
            "render_mode",
            "stroke_width",
            "lineWidth",
        ]

        for prop in font_props:
            if prop in self._obj:
                info[f"raw_{prop}"] = self._obj[prop]

        return info

    @property
    def visual_text(self) -> str:
        """Return the text converted to *visual* order using the Unicode BiDi algorithm.

        This helper is intentionally side-effect–free: it does **not** mutate
        ``self.text`` or the underlying character dictionaries.  It should be
        used by UI / rendering code that needs human-readable RTL/LTR mixing.
        """
        logical = self.text
        if not logical:
            return logical

        # Quick check – bail out if no RTL chars to save import/CPU.
        import unicodedata

        if not any(unicodedata.bidirectional(ch) in ("R", "AL", "AN") for ch in logical):
            return logical

        try:
            from bidi.algorithm import get_display  # type: ignore

            from natural_pdf.utils.bidi_mirror import mirror_brackets

            # Convert from logical order to visual order
            visual = get_display(logical, base_dir="R")
            return mirror_brackets(str(visual))
        except Exception:
            # If python-bidi is missing or errors, fall back to logical order
            return logical
