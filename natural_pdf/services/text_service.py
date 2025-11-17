from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Protocol, Sequence

from natural_pdf.services.base import resolve_service
from natural_pdf.services.registry import register_delegate

logger = logging.getLogger(__name__)


class SupportsFindAll(Protocol):
    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Sequence[str] | str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[dict] = None,
        auto_text_tolerance: Optional[dict] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
    ) -> Any: ...


class TextService:
    """Shared text update helpers formerly provided by TextMixin."""

    def __init__(self, _context) -> None:  # context reserved for future use
        self._context = _context

    @register_delegate("text", "correct_ocr")
    def correct_ocr(
        self,
        host: SupportsFindAll,
        transform: Callable[[Any], Optional[str]],
        *,
        apply_exclusions: bool = False,
    ):
        return self.update_text(
            host,
            transform=transform,
            selector="text[source=ocr]",
            apply_exclusions=apply_exclusions,
        )

    @register_delegate("text", "update_text")
    def update_text(
        self,
        host: SupportsFindAll,
        transform: Callable[[Any], Optional[str]],
        *,
        selector: str = "text",
        apply_exclusions: bool = False,
    ):
        if not callable(transform):
            raise TypeError("transform must be callable")

        finder = getattr(host, "find_all", None)
        if finder is None:
            raise NotImplementedError(
                f"{host.__class__.__name__} must implement `update_text` explicitly "
                "(no `find_all` method found)."
            )

        try:
            elements_collection = finder(selector=selector, apply_exclusions=apply_exclusions)
        except Exception as exc:  # pragma: no cover – defensive
            raise RuntimeError(
                f"Failed to gather elements with selector '{selector}': {exc}"
            ) from exc

        elements_iter = getattr(elements_collection, "elements", elements_collection)
        updated = 0

        for element in elements_iter:
            if not hasattr(element, "text"):
                continue

            new_text = transform(element)
            if new_text is not None and isinstance(new_text, str) and new_text != element.text:
                element.text = new_text
                updated += 1

        logger.info(
            "%s.update_text – processed %d element(s); updated %d.",
            host.__class__.__name__,
            len(elements_iter),
            updated,
        )

        return host
