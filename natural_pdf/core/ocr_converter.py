from __future__ import annotations

from typing import List, Sequence, Tuple

from natural_pdf.elements.text import TextElement


class OCRConverter:
    """Converts OCR service output into TextElement word/char entries."""

    def __init__(self, page) -> None:
        self._page = page

    def convert(
        self,
        ocr_results: Sequence[dict],
        *,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> Tuple[List[TextElement], List[TextElement]]:
        words: List[TextElement] = []
        chars: List[TextElement] = []

        scale_x = float(scale_x)
        scale_y = float(scale_y)
        offset_x = float(offset_x)
        offset_y = float(offset_y)

        for result in ocr_results:
            try:
                x0_img, top_img, x1_img, bottom_img = map(float, result["bbox"])
            except Exception:
                continue

            pdf_x0 = offset_x + (x0_img * scale_x)
            pdf_top = offset_y + (top_img * scale_y)
            pdf_x1 = offset_x + (x1_img * scale_x)
            pdf_bottom = offset_y + (bottom_img * scale_y)
            pdf_height = (bottom_img - top_img) * scale_y

            raw_confidence = result.get("confidence")
            confidence_value = float(raw_confidence) if raw_confidence is not None else None
            ocr_text = result.get("text")

            word_element_data = {
                "text": ocr_text,
                "x0": pdf_x0,
                "top": pdf_top,
                "x1": pdf_x1,
                "bottom": pdf_bottom,
                "width": (x1_img - x0_img) * scale_x,
                "height": pdf_height,
                "object_type": "word",
                "source": "ocr",
                "confidence": confidence_value,
                "fontname": "OCR",
                "size": round(pdf_height) if pdf_height > 0 else 10.0,
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

            ocr_char_dict = word_element_data.copy()
            ocr_char_dict["object_type"] = "char"
            ocr_char_dict.setdefault("adv", ocr_char_dict.get("width", 0))
            ocr_char_dict.setdefault("highlight", False)
            ocr_char_dict.setdefault("highlight_color", None)

            word_element_data["_char_dicts"] = [ocr_char_dict.copy()]

            word_elem = TextElement(word_element_data, self._page)
            words.append(word_elem)

            if ocr_text is not None:
                char_dict = ocr_char_dict.copy()
                char_dict["object_type"] = "char"
                char_dict.setdefault("adv", char_dict.get("width", 0))
                char_element_specific_data = char_dict.copy()
                char_element_specific_data["_char_dicts"] = [char_dict.copy()]
                chars.append(TextElement(char_element_specific_data, self._page))

        return words, chars
