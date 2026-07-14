from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, List, Tuple

from natural_pdf.elements.text import TextElement
from natural_pdf.exceptions import OCRError


def validate_ocr_image_size(image_size: Any) -> tuple[float, float]:
    """Return a finite positive OCR image size or raise :class:`OCRError`."""

    if isinstance(image_size, (str, bytes, bytearray, Mapping)):
        raise OCRError("Invalid OCR payload: image_size must contain width and height.")
    try:
        values = list(image_size)
    except TypeError as exc:
        raise OCRError("Invalid OCR payload: image_size must contain width and height.") from exc
    if len(values) != 2 or any(isinstance(value, bool) for value in values):
        raise OCRError("Invalid OCR payload: image_size must contain width and height.")
    try:
        width, height = (float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OCRError("Invalid OCR payload: image_size values must be numeric.") from exc
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise OCRError("Invalid OCR payload: image_size values must be finite and positive.")
    return width, height


def validate_classic_ocr_results(
    ocr_results: Any,
    *,
    detection_only: bool | None = None,
) -> List[Mapping[str, Any]]:
    """Validate every classic OCR result before any result is converted.

    Numeric strings are accepted for bbox and confidence values for compatibility
    with public third-party engines, but all numeric values must be finite.  The
    returned list also safely materializes one-shot iterables before conversion.
    """

    if isinstance(ocr_results, (str, bytes, bytearray, Mapping)):
        raise OCRError("Invalid classic OCR payload: results must be an iterable of mappings.")
    try:
        results = list(ocr_results)
    except TypeError as exc:
        raise OCRError(
            "Invalid classic OCR payload: results must be an iterable of mappings."
        ) from exc

    normalized_results: List[Mapping[str, Any]] = []
    for index, result in enumerate(results):
        prefix = f"Invalid classic OCR payload at result {index}"
        if not isinstance(result, Mapping):
            raise OCRError(f"{prefix}: expected a mapping, got {type(result).__name__}.")

        detection_flag = result.get("_ocr_detection_only", False)
        if not isinstance(detection_flag, bool):
            raise OCRError(f"{prefix}: '_ocr_detection_only' must be a boolean when provided.")
        if detection_only is False and detection_flag:
            raise OCRError(
                f"{prefix}: a recognition payload cannot contain detection-only entries."
            )
        is_detection = detection_only is True or detection_flag

        bbox = result.get("bbox")
        if isinstance(bbox, (str, bytes, bytearray, Mapping)):
            raise OCRError(f"{prefix}: 'bbox' must contain exactly four coordinates.")
        try:
            bbox_values = list(bbox)
        except TypeError as exc:
            raise OCRError(f"{prefix}: 'bbox' must contain exactly four coordinates.") from exc
        if len(bbox_values) != 4:
            raise OCRError(f"{prefix}: 'bbox' must contain exactly four coordinates.")
        try:
            x0, top, x1, bottom = (float(value) for value in bbox_values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise OCRError(f"{prefix}: 'bbox' coordinates must be numeric.") from exc
        if any(isinstance(value, bool) for value in bbox_values) or not all(
            math.isfinite(value) for value in (x0, top, x1, bottom)
        ):
            raise OCRError(f"{prefix}: 'bbox' coordinates must be finite numbers.")
        if x1 <= x0 or bottom <= top:
            raise OCRError(f"{prefix}: 'bbox' must be ordered with x1 > x0 and bottom > top.")

        if is_detection:
            text = result.get("text")
            if text is not None and not isinstance(text, str):
                raise OCRError(f"{prefix}: detection 'text' must be a string or None.")
        elif (
            "text" not in result
            or not isinstance(result["text"], str)
            or not result["text"].strip()
        ):
            raise OCRError(f"{prefix}: recognition 'text' must be a non-empty string.")

        confidence = result.get("confidence")
        confidence_value = None
        if confidence is not None:
            if isinstance(confidence, bool):
                raise OCRError(f"{prefix}: 'confidence' must be a finite number or None.")
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError, OverflowError) as exc:
                raise OCRError(f"{prefix}: 'confidence' must be a finite number or None.") from exc
            if not math.isfinite(confidence_value):
                raise OCRError(f"{prefix}: 'confidence' must be a finite number or None.")

        normalized_result = dict(result)
        normalized_result["bbox"] = (x0, top, x1, bottom)
        if confidence is not None:
            normalized_result["confidence"] = confidence_value
        normalized_results.append(normalized_result)

    return normalized_results


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
        engine_name: str | None = None,
    ) -> Tuple[List[TextElement], List[TextElement]]:
        # Validate the complete payload before constructing even one element.
        # This keeps mixed valid/invalid payloads atomic.
        validated_results = validate_classic_ocr_results(ocr_results)

        words: List[TextElement] = []
        chars: List[TextElement] = []

        scale_x = float(scale_x)
        scale_y = float(scale_y)
        offset_x = float(offset_x)
        offset_y = float(offset_y)

        for result in validated_results:
            x0_img, top_img, x1_img, bottom_img = map(float, result["bbox"])

            pdf_x0 = offset_x + (x0_img * scale_x)
            pdf_top = offset_y + (top_img * scale_y)
            pdf_x1 = offset_x + (x1_img * scale_x)
            pdf_bottom = offset_y + (bottom_img * scale_y)
            pdf_height = (bottom_img - top_img) * scale_y

            raw_confidence = result.get("confidence")
            confidence_value = float(raw_confidence) if raw_confidence is not None else None
            ocr_text = result.get("text")
            detection_only = bool(result.get("_ocr_detection_only", False))

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
                "ocr_engine": engine_name,
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
            if detection_only:
                word_element_data["ocr_detection_only"] = True

            # Pass through extra metadata from OCR engines (e.g. source_category)
            source_category = result.get("source_category")
            if source_category:
                word_element_data["source_category"] = source_category

            ocr_char_dict = word_element_data.copy()
            ocr_char_dict["object_type"] = "char"
            ocr_char_dict.setdefault("adv", ocr_char_dict.get("width", 0))
            ocr_char_dict.setdefault("highlight", False)
            ocr_char_dict.setdefault("highlight_color", None)

            word_element_data["_char_dicts"] = [] if detection_only else [ocr_char_dict.copy()]

            word_elem = TextElement(word_element_data, self._page)
            words.append(word_elem)

            if ocr_text is not None and not detection_only:
                char_dict = ocr_char_dict.copy()
                char_dict["object_type"] = "char"
                char_dict.setdefault("adv", char_dict.get("width", 0))
                char_element_specific_data = char_dict.copy()
                char_element_specific_data["_char_dicts"] = [char_dict.copy()]
                chars.append(TextElement(char_element_specific_data, self._page))

        return words, chars
