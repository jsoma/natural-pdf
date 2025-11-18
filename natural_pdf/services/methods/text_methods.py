"""Text update helpers that proxy to the text service."""

from __future__ import annotations

from typing import Any, Callable, Optional

from natural_pdf.services.base import resolve_service


def update_text(
    self,
    transform: Callable[[Any], Optional[str]],
    *,
    selector: str = "text",
    apply_exclusions: bool = False,
    **ignored: Any,
):
    """Apply a text transformation via the shared text service."""

    service = resolve_service(self, "text")
    return service.update_text(
        self,
        transform=transform,
        selector=selector,
        apply_exclusions=apply_exclusions,
    )


def correct_ocr(
    self,
    transform: Callable[[Any], Optional[str]],
    *,
    apply_exclusions: bool = False,
    **ignored: Any,
):
    """Convenience wrapper for updating only OCR-derived text elements."""

    service = resolve_service(self, "text")
    return service.correct_ocr(
        self,
        transform=transform,
        apply_exclusions=apply_exclusions,
    )


__all__ = ["update_text", "correct_ocr"]
