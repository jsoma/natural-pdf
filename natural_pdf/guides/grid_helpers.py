"""Grid-specific helper utilities for Guides."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Set, Tuple

from natural_pdf.core.interfaces import HasPages, HasSinglePage

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
else:  # pragma: no cover
    Page = Any  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


def collect_constituent_pages(regions: Sequence[Any]) -> Set["Page"]:
    """Collect the unique pages referenced by the supplied regions."""
    pages: Set["Page"] = set()
    for region in regions:
        candidates: Tuple["Page", ...] = ()
        if isinstance(region, HasPages):
            candidates = tuple(region.pages)
        elif isinstance(region, HasSinglePage):
            candidate = getattr(region, "page", None)
            candidates = (candidate,) if candidate is not None else ()
        else:
            page_candidate = getattr(region, "page", None)
            if page_candidate is not None:
                candidates = (page_candidate,)

        for page in candidates:
            if page is not None and hasattr(page, "add_element"):
                pages.add(page)

    return pages


def register_regions_with_pages(
    pages: Iterable["Page"],
    table_region: Any,
    row_regions: Sequence[Any],
    column_regions: Sequence[Any],
    cell_regions: Sequence[Any],
    *,
    log: Optional[logging.Logger] = None,
) -> None:
    """Register logical table regions with the provided pages."""
    logger_to_use = log or logger
    page_list = tuple(pages)

    for page in page_list:
        page.add_element(table_region, element_type="regions")
        logger_to_use.debug(
            "Registered multi-page table with page %s", getattr(page, "page_number", "?")
        )

    for page in page_list:
        for row in row_regions:
            page.add_element(row, element_type="regions")
        for column in column_regions:
            page.add_element(column, element_type="regions")
        for cell in cell_regions:
            page.add_element(cell, element_type="regions")
