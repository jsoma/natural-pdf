from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Type

from natural_pdf.services.registry import register_delegate

if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin

_SHAPE_PROXY_CLASS: Optional[Type[Any]] = None


def _load_shape_detection_mixin() -> Type["ShapeDetectionMixin"]:
    from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin

    return ShapeDetectionMixin


def _shape_proxy_factory(host: Any):
    global _SHAPE_PROXY_CLASS
    if _SHAPE_PROXY_CLASS is None:
        mixin_cls = _load_shape_detection_mixin()

        class _Proxy(mixin_cls):  # type: ignore[misc]
            def __init__(self, wrapped):
                object.__setattr__(self, "_host", wrapped)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._host, name)

            def __setattr__(self, name: str, value: Any) -> None:
                setattr(self._host, name, value)

            @property
            def page(self):
                host = object.__getattribute__(self, "_host")
                return getattr(host, "page", host)

        _SHAPE_PROXY_CLASS = _Proxy
    return _SHAPE_PROXY_CLASS(host)


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

        proxy = _shape_proxy_factory(host)
        proxy.detect_lines(**kwargs)
        return host
