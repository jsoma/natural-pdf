from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from natural_pdf.elements.text import TextElement

logger = logging.getLogger(__name__)


class OCRMixin:
    """Shared OCR helpers for Page-like objects backed by an ElementManager."""

    def _ocr_element_manager(self):
        raise NotImplementedError

    def remove_ocr_elements(self) -> int:
        return int(self._ocr_element_manager().remove_ocr_elements())

    def clear_text_layer(self) -> Tuple[int, int]:
        return self._ocr_element_manager().clear_text_layer()

    def create_text_elements_from_ocr(
        self,
        ocr_results: Any,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
    ) -> List[Any]:
        return self._ocr_element_manager().create_text_elements_from_ocr(
            ocr_results, scale_x=scale_x, scale_y=scale_y
        )

    def apply_ocr(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def extract_ocr_elements(self, *args: Any, **kwargs: Any) -> List[TextElement]:
        raise NotImplementedError
