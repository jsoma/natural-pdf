from __future__ import annotations

from typing import List, Sequence

from natural_pdf.analyzers.guides.grid_helpers import (
    collect_constituent_pages,
    register_regions_with_pages,
)
from natural_pdf.core.interfaces import HasPages, HasSinglePage


class DummyPage:
    def __init__(self, page_number: int):
        self.page_number = page_number
        self.calls: List[tuple[object, str]] = []

    def add_element(self, element, element_type: str = "regions"):
        self.calls.append((element, element_type))


class MultiPageRegion(HasPages):
    def __init__(self, pages: Sequence[DummyPage]):
        self._pages = tuple(pages)

    @property
    def pages(self) -> Sequence[DummyPage]:
        return self._pages


class SinglePageRegion(HasSinglePage):
    def __init__(self, page: DummyPage):
        self._page = page

    @property
    def page(self) -> DummyPage:
        return self._page


class AttrPageRegion:
    def __init__(self, page: DummyPage):
        self.page = page


def test_collect_constituent_pages_handles_protocols_and_attrs():
    pages = [DummyPage(1), DummyPage(2), DummyPage(3)]
    regions = [
        MultiPageRegion(pages[:2]),
        SinglePageRegion(pages[2]),
        AttrPageRegion(pages[0]),
    ]

    result = collect_constituent_pages(regions)

    assert result == {pages[0], pages[1], pages[2]}


def test_register_regions_with_pages_adds_all_regions():
    pages = [DummyPage(1), DummyPage(2)]
    table = object()
    rows = [object(), object()]
    cols = [object()]
    cells = [object(), object(), object()]

    register_regions_with_pages(pages, table, rows, cols, cells)

    expected_calls = 1 + len(rows) + len(cols) + len(cells)
    for page in pages:
        assert len(page.calls) == expected_calls
        assert all(element_type == "regions" for _, element_type in page.calls)
        assert page.calls[0][0] is table
