from __future__ import annotations

import itertools
import logging
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from pdfplumber.utils.text import WordExtractor

from natural_pdf.elements.text import TextElement
from natural_pdf.utils.bidi_mirror import mirror_brackets

logger = logging.getLogger(__name__)

CharDirection = str


@dataclass(frozen=True)
class WordEngineOptions:
    page_number: int
    x_tolerance: float
    y_tolerance: float
    keep_blank_chars: bool
    x_tolerance_ratio: Optional[float]
    y_tolerance_ratio: Optional[float]
    use_text_flow: bool


class NaturalWordExtractor(WordExtractor):
    """Custom WordExtractor that splits words based on specified character attributes."""

    def __init__(self, word_split_attributes: List[str], extra_attrs: List[str], *args, **kwargs):
        self.word_split_attributes = word_split_attributes or []
        if "word_split_attributes" in kwargs:
            del kwargs["word_split_attributes"]
        kwargs["extra_attrs"] = extra_attrs
        super().__init__(*args, **kwargs)

    def char_begins_new_word(
        self,
        prev_char: Dict[str, Any],
        curr_char: Dict[str, Any],
        direction: CharDirection,
        x_tolerance: float,
        y_tolerance: float,
    ) -> bool:
        spatial_split = super().char_begins_new_word(
            prev_char, curr_char, direction, x_tolerance, y_tolerance
        )
        if spatial_split:
            return True

        if self.word_split_attributes:
            for attr in self.word_split_attributes:
                if prev_char.get(attr) != curr_char.get(attr):
                    logger.debug(
                        "Splitting word due to attribute mismatch on '%s': %s != %s",
                        attr,
                        prev_char.get(attr),
                        curr_char.get(attr),
                    )
                    return True

        return False


class WordEngine:
    """Encapsulates word extraction and BiDi/decoration propagation."""

    def __init__(
        self,
        word_split_attributes: Sequence[str],
        *,
        load_text: bool = True,
    ) -> None:
        self._split_attrs = list(word_split_attributes)
        self._load_text = load_text

    def generate_words(
        self,
        prepared_char_dicts: List[Dict[str, Any]],
        *,
        options: WordEngineOptions,
        create_word_element: Callable[[Dict[str, Any]], TextElement],
        propagate_decorations: Callable[[List[TextElement], List[Dict[str, Any]]], None],
        disable_text_sync: Callable[..., Any],
    ) -> List[TextElement]:
        if not self._load_text or not prepared_char_dicts:
            return []

        char_to_index = {
            (
                char_dict.get("x0", 0),
                char_dict.get("top", 0),
                char_dict.get("text", ""),
            ): idx
            for idx, char_dict in enumerate(prepared_char_dicts)
        }

        xt = options.x_tolerance
        yt = options.y_tolerance
        x_tolerance_ratio = options.x_tolerance_ratio
        y_tolerance_ratio = options.y_tolerance_ratio
        keep_blank_chars = options.keep_blank_chars
        use_flow = options.use_text_flow

        attributes_to_preserve = list(
            set(
                self._split_attrs
                + [
                    "non_stroking_color",
                    "strike",
                    "underline",
                    "highlight",
                    "highlight_color",
                ]
            )
        )

        sorted_chars = sorted(
            prepared_char_dicts,
            key=lambda c: (round(c.get("top", 0) / max(yt, 1)) * yt, c.get("x0", 0)),
        )

        lines: List[List[Dict[str, Any]]] = []
        current_line_key = None
        for char_dict in sorted_chars:
            top_val = char_dict.get("top", 0)
            line_key = round(top_val / max(yt, 1))
            if current_line_key is None or line_key != current_line_key:
                lines.append([])
                current_line_key = line_key
            lines[-1].append(char_dict)

        word_elements: List[TextElement] = []

        for line_chars in lines:
            if not line_chars:
                continue

            rtl_count = sum(
                1
                for ch in line_chars
                if unicodedata.bidirectional((ch.get("text") or " ")[0]) in ("R", "AL", "AN")
            )
            ltr_count = len(line_chars) - rtl_count
            is_rtl_line = rtl_count > ltr_count

            line_dir = "ttb"
            char_dir = "ltr"

            extractor = NaturalWordExtractor(
                word_split_attributes=self._split_attrs + ["strike", "underline", "highlight"],
                extra_attrs=attributes_to_preserve,
                x_tolerance=xt,
                y_tolerance=yt,
                x_tolerance_ratio=x_tolerance_ratio,
                y_tolerance_ratio=y_tolerance_ratio,
                keep_blank_chars=keep_blank_chars,
                use_text_flow=use_flow,
                line_dir=line_dir,
                char_dir=char_dir,
            )

            line_chars_for_extractor = sorted(line_chars, key=lambda c: c.get("x0", 0))

            try:
                word_tuples = extractor.iter_extract_tuples(line_chars_for_extractor)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error(
                    "Word extraction failed on page %s (rtl=%s): %s",
                    options.page_number,
                    is_rtl_line,
                    exc,
                    exc_info=True,
                )
                word_tuples = []

            for word_dict, char_list in word_tuples:
                char_indices: List[int] = []
                for char_dict in char_list:
                    key = (
                        char_dict.get("x0", 0),
                        char_dict.get("top", 0),
                        char_dict.get("text", ""),
                    )
                    if key in char_to_index:
                        char_indices.append(char_to_index[key])
                word_dict["_char_indices"] = char_indices
                word_dict["_char_dicts"] = char_list

                word_element = create_word_element(word_dict)
                word_elements.append(word_element)

                rtl_in_word = any(
                    unicodedata.bidirectional((ch.get("text") or " ")[0]) in ("R", "AL", "AN")
                    for ch in char_list
                )
                if rtl_in_word:
                    try:
                        from bidi.algorithm import get_display  # type: ignore

                        with disable_text_sync():
                            element_text_obj = word_element.text
                            if isinstance(element_text_obj, bytes):
                                element_text = element_text_obj.decode("utf-8", errors="ignore")
                            else:
                                element_text = str(element_text_obj)
                            logical_text = get_display(element_text, base_dir="L")
                            word_element.text = mirror_brackets(str(logical_text))
                    except Exception:
                        continue

        propagate_decorations(word_elements, prepared_char_dicts)
        return word_elements
