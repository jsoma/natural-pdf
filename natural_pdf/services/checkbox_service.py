from __future__ import annotations

from typing import Any, Sequence, cast

from natural_pdf.analyzers.checkbox.mixin import CheckboxDetectionMixin
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.services.registry import register_delegate


class _CheckboxProxy(CheckboxDetectionMixin):
    """Proxy that exposes mixin helpers while delegating attribute access to the host."""

    def __init__(self, host: Any):
        object.__setattr__(self, "_host", host)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._host, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._host, name, value)


class CheckboxDetectionService:
    """Service wrapper around the legacy CheckboxDetectionMixin logic."""

    def __init__(self, context):
        self._context = context

    @register_delegate("checkbox", "detect_checkboxes")
    def detect_checkboxes(self, host: Any, **kwargs) -> ElementCollection:
        pdfs = getattr(host, "pdfs", None)
        if pdfs is not None:
            combined = []
            for pdf in cast(Sequence[Any], pdfs):
                detector = getattr(pdf, "detect_checkboxes", None)
                if callable(detector):
                    result = detector(**kwargs)
                    if result:
                        combined.extend(getattr(result, "elements", result))
            return ElementCollection(combined)

        pages = getattr(host, "pages", None)
        if pages is not None and not hasattr(host, "_page"):
            per_page_kwargs = dict(kwargs)
            show_progress = per_page_kwargs.pop("show_progress", True)
            iterator = pages
            if show_progress:
                try:
                    from tqdm.auto import tqdm

                    iterator = tqdm(pages, desc="Detecting checkboxes")
                except Exception:  # pragma: no cover - optional dependency
                    pass

            combined = []
            for page in iterator:
                detector = getattr(page, "detect_checkboxes", None)
                if callable(detector):
                    result = detector(**per_page_kwargs)
                    if result:
                        combined.extend(getattr(result, "elements", result))
            return ElementCollection(combined)

        proxy = _CheckboxProxy(host)
        result = CheckboxDetectionMixin.detect_checkboxes(proxy, **kwargs)
        return result
