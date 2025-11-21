"""Character loading/enrichment helpers used by ElementManager."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from natural_pdf.text.font_style import detect_bold_style, detect_italic_style

logger = logging.getLogger(__name__)


class ElementLoader:
    """Prepare native pdfplumber character dictionaries for downstream services."""

    _REQUIRED_KEYS = ("x0", "top", "x1", "bottom", "text")

    def __init__(self, *, page_number: int) -> None:
        self._page_number = page_number

    def prepare_native_chars(self, native_chars: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return augmented copies of the provided native character dictionaries."""
        prepared: List[Dict[str, Any]] = []

        for idx, char_dict in enumerate(native_chars or []):
            missing_keys = [key for key in self._REQUIRED_KEYS if key not in char_dict]
            if missing_keys:
                raise KeyError(
                    f"Page {self._page_number}: native char dict missing keys {missing_keys}: {char_dict}"
                )

            augmented = char_dict.copy()
            augmented["bold"] = detect_bold_style(augmented)
            augmented["italic"] = detect_italic_style(augmented)
            augmented["source"] = augmented.get("source") or "native"

            # Preserve non-stroking color if available
            if "non_stroking_color" in char_dict:
                augmented["non_stroking_color"] = char_dict["non_stroking_color"]

            # Ensure required defaults exist for downstream grouping
            augmented.setdefault("upright", True)
            augmented.setdefault("fontname", "Unknown")
            augmented.setdefault("size", 0)
            augmented.setdefault("highlight_color", None)
            augmented.setdefault("strike", False)
            augmented.setdefault("underline", False)
            augmented.setdefault("highlight", False)

            prepared.append(augmented)

        logger.debug(
            "Page %s: Prepared %d native char dicts.",
            self._page_number,
            len(prepared),
        )
        return prepared


__all__ = ["ElementLoader"]
