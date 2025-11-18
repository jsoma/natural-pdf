"""Layout helper functions shared by host wrappers."""

from __future__ import annotations

from typing import Any, List, Optional

from natural_pdf.analyzers.layout.layout_options import LayoutOptions
from natural_pdf.services.base import resolve_service


def analyze_layout(
    self,
    *,
    engine: Optional[str] = None,
    options: Optional[LayoutOptions] = None,
    confidence: Optional[float] = None,
    classes: Optional[List[str]] = None,
    exclude_classes: Optional[List[str]] = None,
    device: Optional[str] = None,
    existing: str = "replace",
    model_name: Optional[str] = None,
    client: Optional[Any] = None,
) -> Any:
    """Delegate layout analysis to the configured layout service."""

    service = resolve_service(self, "layout")
    return service.analyze_layout(
        self,
        engine=engine,
        options=options,
        confidence=confidence,
        classes=classes,
        exclude_classes=exclude_classes,
        device=device,
        existing=existing,
        model_name=model_name,
        client=client,
    )


__all__ = ["analyze_layout"]
