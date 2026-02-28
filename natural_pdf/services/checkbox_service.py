"""Checkbox detection service — direct dispatch matching LayoutService pattern."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from natural_pdf.services.registry import register_delegate

if TYPE_CHECKING:
    from natural_pdf.elements.element_collection import ElementCollection

logger = logging.getLogger(__name__)


class CheckboxDetectionService:
    """Service for checkbox detection across Page/Region/PageCollection/PDF."""

    def __init__(self, context):
        self._context = context

    def _normalize_engine_arg(self, args, kwargs):
        """Allow the first positional arg to act as the engine name."""
        if not args:
            return kwargs

        if len(args) > 1:
            raise TypeError(
                "detect_checkboxes() accepts at most one positional argument (the engine name)"
            )

        if "engine" in kwargs and kwargs.get("engine") is not None:
            raise TypeError("detect_checkboxes() got multiple values for 'engine'")

        engine_arg = args[0]
        if engine_arg is None:
            return kwargs

        if not isinstance(engine_arg, str):
            raise TypeError(
                "The positional argument to detect_checkboxes() must be a string engine name; "
                f"received {type(engine_arg).__name__!r}"
            )

        merged = dict(kwargs)
        merged["engine"] = engine_arg
        return merged

    @register_delegate("checkbox", "detect_checkboxes")
    def detect_checkboxes(self, host: Any, *args, **kwargs) -> "ElementCollection":
        kwargs = self._normalize_engine_arg(args, kwargs)

        from natural_pdf.core.page import Page
        from natural_pdf.core.page_collection import PageCollection
        from natural_pdf.core.pdf import PDF
        from natural_pdf.elements.region import Region

        if isinstance(host, Page):
            return self._detect_page(host, **kwargs)
        if isinstance(host, Region):
            return self._detect_region(host, **kwargs)
        if isinstance(host, PageCollection):
            return self._detect_page_collection(host, **kwargs)
        if isinstance(host, PDF):
            return self._detect_pdf(host, **kwargs)

        # Fallback: check for pages attribute (duck typing)
        pages = getattr(host, "pages", None)
        if pages is not None:
            return self._detect_page_collection_like(host, **kwargs)

        raise TypeError(f"Host type {type(host)!r} does not support checkbox detection.")

    def _detect_page(self, page, **kwargs) -> "ElementCollection":
        from natural_pdf.analyzers.checkbox.checkbox_analyzer import CheckboxAnalyzer
        from natural_pdf.elements.element_collection import ElementCollection

        analyzer = CheckboxAnalyzer(page)
        regions = analyzer.detect_checkboxes(**kwargs)
        return ElementCollection(regions)

    def _detect_region(self, region, **kwargs) -> "ElementCollection":
        # For regions, detect on the full page then filter to region bounds
        from natural_pdf.elements.element_collection import ElementCollection

        page = region._page if hasattr(region, "_page") else region.page
        page_results = self._detect_page(page, **kwargs)

        # Filter to regions whose center falls within the target region
        rx0, ry0, rx1, ry1 = region.bbox
        filtered = []
        for r in page_results:
            cx = (r.bbox[0] + r.bbox[2]) / 2
            cy = (r.bbox[1] + r.bbox[3]) / 2
            if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
                filtered.append(r)

        return ElementCollection(filtered)

    def _detect_page_collection(self, collection, **kwargs) -> "ElementCollection":
        from natural_pdf.elements.element_collection import ElementCollection

        show_progress = kwargs.pop("show_progress", True)
        pages_iter = collection.pages
        if show_progress:
            try:
                from tqdm.auto import tqdm

                pages_iter = tqdm(collection.pages, desc="Detecting checkboxes")
            except Exception:
                pass

        all_regions = []
        for page in pages_iter:
            page_regions = self._detect_page(page, **kwargs)
            if page_regions:
                all_regions.extend(page_regions.elements)

        return ElementCollection(all_regions)

    def _detect_pdf(self, pdf, **kwargs) -> "ElementCollection":
        return self._detect_page_collection(pdf.pages, **kwargs)

    def _detect_page_collection_like(self, host, **kwargs) -> "ElementCollection":
        """Handle duck-typed hosts with .pages attribute."""
        from natural_pdf.elements.element_collection import ElementCollection

        all_regions = []
        for page in host.pages:
            detector = getattr(page, "detect_checkboxes", None)
            if callable(detector):
                result = detector(**kwargs)
                if result is not None:
                    if hasattr(result, "elements"):
                        all_regions.extend(result.elements)
                    elif hasattr(result, "__iter__"):
                        all_regions.extend(result)

        return ElementCollection(all_regions)
