"""Shared OCR wrapper functions that delegate to the OCR service."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from natural_pdf.services.base import resolve_service


def apply_ocr(
    self,
    *,
    replace: bool = True,
    engine: Optional[str] = None,
    options: Optional[Any] = None,
    languages: Optional[List[str]] = None,
    min_confidence: Optional[float] = None,
    device: Optional[str] = None,
    resolution: Optional[int] = None,
    detect_only: bool = False,
    apply_exclusions: bool = True,
    **kwargs: Any,
):
    """Apply OCR through the configured service and return the host for chaining."""

    service = resolve_service(self, "ocr")
    service.apply_ocr(
        self,
        engine=engine,
        options=options,
        languages=languages,
        min_confidence=min_confidence,
        device=device,
        resolution=resolution,
        detect_only=detect_only,
        apply_exclusions=apply_exclusions,
        replace=replace,
        **kwargs,
    )
    return self


def extract_ocr_elements(
    self,
    *,
    engine: Optional[str] = None,
    options: Optional[Any] = None,
    languages: Optional[List[str]] = None,
    min_confidence: Optional[float] = None,
    device: Optional[str] = None,
    resolution: Optional[int] = None,
) -> List[Any]:
    """Run OCR without mutating the host and return the new text elements."""

    service = resolve_service(self, "ocr")
    return service.extract_ocr_elements(
        self,
        engine=engine,
        options=options,
        languages=languages,
        min_confidence=min_confidence,
        device=device,
        resolution=resolution,
    )


def apply_custom_ocr(
    self,
    *,
    ocr_function,
    source_label: str = "custom-ocr",
    replace: bool = True,
    confidence: Optional[float] = None,
    add_to_page: bool = True,
):
    """Route custom OCR callables through the OCR service."""

    service = resolve_service(self, "ocr")
    service.apply_custom_ocr(
        self,
        ocr_function=ocr_function,
        source_label=source_label,
        replace=replace,
        confidence=confidence,
        add_to_page=add_to_page,
    )
    return self


def remove_ocr_elements(self) -> int:
    """Remove OCR-derived elements via the OCR service."""

    service = resolve_service(self, "ocr")
    return service.remove_ocr_elements(self)


def clear_text_layer(self) -> Tuple[int, int]:
    """Clear OCR-created text layers via the OCR service."""

    service = resolve_service(self, "ocr")
    return service.clear_text_layer(self)


def create_text_elements_from_ocr(
    self,
    ocr_results: Any,
    *,
    scale_x: Optional[float] = None,
    scale_y: Optional[float] = None,
):
    """Convert OCR results into text elements via the OCR service."""

    service = resolve_service(self, "ocr")
    return service.create_text_elements_from_ocr(
        self,
        ocr_results,
        scale_x=scale_x,
        scale_y=scale_y,
    )


__all__ = [
    "apply_ocr",
    "extract_ocr_elements",
    "apply_custom_ocr",
    "remove_ocr_elements",
    "clear_text_layer",
    "create_text_elements_from_ocr",
]
