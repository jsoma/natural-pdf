from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Sequence, cast

from natural_pdf.analyzers.checkbox.mixin import CheckboxDetectionMixin
from natural_pdf.services.registry import register_delegate

if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf.elements.element_collection import ElementCollection


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
    def detect_checkboxes(self, host: Any, **kwargs) -> "ElementCollection":
        from natural_pdf.elements.element_collection import ElementCollection

        pdfs = getattr(host, "pdfs", None)
        if pdfs is not None:
            combined: list[Any] = []
            for pdf in cast(Sequence[Any], pdfs):
                detector = getattr(pdf, "detect_checkboxes", None)
                if callable(detector):
                    result = detector(**kwargs)
                    self._extend_matches(combined, result)
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
                    self._extend_matches(combined, result)
            return ElementCollection(combined)

        proxy = _CheckboxProxy(host)
        result = CheckboxDetectionMixin.detect_checkboxes(proxy, **kwargs)
        return result

    @staticmethod
    def _extend_matches(destination: list[Any], result: Any) -> None:
        from natural_pdf.elements.element_collection import ElementCollection

        if result is None:
            return
        if isinstance(result, ElementCollection):
            destination.extend(result.elements)
            return
        elements_attr = getattr(result, "elements", None)
        if isinstance(elements_attr, list):
            destination.extend(elements_attr)
            return
        if isinstance(result, Iterable):
            destination.extend(result)
            return
        destination.append(result)
