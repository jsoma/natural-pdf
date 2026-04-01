"""Form cell detection service — dispatches across Page/Region/PageCollection/PDF."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from natural_pdf.services.registry import register_delegate

if TYPE_CHECKING:
    from natural_pdf.elements.element_collection import ElementCollection

logger = logging.getLogger(__name__)


class FormCellService:
    """Service for form cell detection across Page/Region/PageCollection/PDF."""

    def __init__(self, context):
        self._context = context

    @register_delegate("form_cell", "detect_form_cells")
    def detect_form_cells(self, host: Any, *args, **kwargs) -> "ElementCollection":
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

        pages = getattr(host, "pages", None)
        if pages is not None:
            return self._detect_page_collection(host, **kwargs)

        raise TypeError(f"Host type {type(host)!r} does not support form cell detection.")

    def _detect_page(self, page, **kwargs) -> "ElementCollection":
        from natural_pdf.analyzers.form_cells.form_cell_analyzer import FormCellAnalyzer
        from natural_pdf.elements.element_collection import ElementCollection

        analyzer = FormCellAnalyzer(page)
        regions = analyzer.detect_form_cells(**kwargs)
        return ElementCollection(regions)

    def _detect_region(self, region, **kwargs) -> "ElementCollection":
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
        pages_iter = collection.pages if hasattr(collection, "pages") else collection
        if show_progress:
            from tqdm.auto import tqdm

            total = len(pages_iter) if hasattr(pages_iter, "__len__") else None
            pages_iter = tqdm(pages_iter, total=total, desc="Detecting form cells")

        all_regions = []
        for page in pages_iter:
            page_regions = self._detect_page(page, **kwargs)
            if page_regions:
                all_regions.extend(page_regions.elements)

        return ElementCollection(all_regions)

    def _detect_pdf(self, pdf, **kwargs) -> "ElementCollection":
        return self._detect_page_collection(pdf.pages, **kwargs)
