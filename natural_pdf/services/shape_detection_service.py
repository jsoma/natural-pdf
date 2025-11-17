from __future__ import annotations

from typing import Any

from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin
from natural_pdf.services.registry import register_delegate


class _ShapeDetectionProxy(ShapeDetectionMixin):
    """Proxy that exposes mixin helpers while delegating attribute access to the host."""

    def __init__(self, host: Any):
        object.__setattr__(self, "_host", host)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._host, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._host, name, value)


class ShapeDetectionService:
    """Service wrapper around the legacy ShapeDetectionMixin helpers."""

    def __init__(self, context):
        self._context = context

    @register_delegate("shapes", "detect_lines")
    def detect_lines(self, host: Any, **kwargs) -> Any:
        pdfs = getattr(host, "pdfs", None)
        if pdfs is not None:
            for pdf in pdfs:
                pages = getattr(pdf, "pages", None)
                if pages is None:
                    continue
                for page in pages:
                    detector = getattr(page, "detect_lines", None)
                    if callable(detector):
                        detector(**kwargs)
            return host

        pages = getattr(host, "pages", None)
        if pages is not None and not hasattr(host, "_page"):
            for page in pages:
                detector = getattr(page, "detect_lines", None)
                if callable(detector):
                    detector(**kwargs)
            return host

        proxy = _ShapeDetectionProxy(host)
        ShapeDetectionMixin.detect_lines(proxy, **kwargs)
        return host
