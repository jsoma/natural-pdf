from __future__ import annotations

import warnings

import pytest

from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides
from natural_pdf.services.table_service import TableService
from natural_pdf.tables.engines.text import TextTablesEngine
from natural_pdf.tables.utils.cells import extract_cell_value


class FakeWord:
    def __init__(self, text, bbox):
        self.text = text
        self.bbox = bbox
        self.x0, self.top, self.x1, self.bottom = bbox

    def extract_text(self):
        return self.text


class FakeCell:
    def __init__(self, row, col, bbox, *, value="fallback", source="guides_temp"):
        self.metadata = {"row_index": row, "col_index": col}
        self.x0, self.top, self.x1, self.bottom = bbox
        self.bbox = bbox
        self.source = source
        self.region_type = "table-cell"
        self.normalized_type = "table-cell"
        self.value = value
        self.extract_calls = 0

    def extract_text(self, *args, **kwargs):
        self.extract_calls += 1
        return self.value

    def apply_ocr(self, **kwargs):
        return None

    def _apply_content_filter_to_text(self, text, content_filter):
        if callable(content_filter):
            return "".join(char for char in text if content_filter(char))
        if content_filter == "strip_drop":
            return text.replace("DROP", "").strip()
        return text


class FakePage:
    def __init__(self, cells):
        self.cells = cells

    def find_all(self, selector, **kwargs):
        assert selector == "region[type=table_cell]"
        return self.cells


class FakeTableRegion:
    def __init__(self, cells, words=()):
        self.source = "guides_temp"
        self.region_type = "table"
        self.normalized_type = "table"
        self.bbox = (0, 0, 20, 20)
        self.page = FakePage(cells)
        self.words = list(words)
        self.find_all_calls = []

    def intersects(self, cell):
        return True

    def find_all(self, selector, **kwargs):
        self.find_all_calls.append((selector, kwargs))
        assert selector == "text"
        return self.words


def _four_cells():
    return [
        FakeCell(0, 0, (0, 0, 10, 10)),
        FakeCell(0, 1, (10, 0, 20, 10)),
        FakeCell(1, 0, (0, 10, 10, 20)),
        FakeCell(1, 1, (10, 10, 20, 20)),
    ]


def test_guide_cell_shortcut_passes_cell_extraction_func_and_callback_wins():
    cells = _four_cells()
    host = FakeTableRegion(cells, words=[FakeWord("ignored", (1, 1, 2, 2))])

    result = TableService(None).extract_table(
        host,
        cell_extraction_func=lambda cell: f"r{cell.metadata['row_index']}c{cell.metadata['col_index']}",
        cell_extract="words",
        cell_overlap="partial",
    )

    assert list(result) == [["r0c0", "r0c1"], ["r1c0", "r1c1"]]
    assert host.find_all_calls == []


def test_large_cell_callback_warns_once_for_guide_cell_shortcut():
    cells = [FakeCell(i, 0, (0, i, 10, i + 1)) for i in range(501)]
    host = FakeTableRegion(cells)

    with pytest.warns(UserWarning, match="cell_extraction_func runs once per table cell"):
        TableService(None).extract_table(host, cell_extraction_func=lambda cell: "x")


def test_cell_callback_does_not_warn_at_threshold():
    cells = [FakeCell(i, 0, (0, i, 10, i + 1)) for i in range(500)]
    host = FakeTableRegion(cells)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TableService(None).extract_table(host, cell_extraction_func=lambda cell: "x")

    assert not any(
        "cell_extraction_func runs once per table cell" in str(w.message) for w in caught
    )


def test_cell_callback_errors_are_propagated():
    cell = FakeCell(0, 0, (0, 0, 10, 10))

    def callback(_cell):
        raise ValueError("callback failed")

    with pytest.raises(ValueError, match="callback failed"):
        extract_cell_value(cell, cell_extraction_func=callback)


def test_cell_callback_rejects_invalid_return_type():
    cell = FakeCell(0, 0, (0, 0, 10, 10))

    with pytest.raises(TypeError, match="must return str or None"):
        extract_cell_value(cell, cell_extraction_func=lambda _cell: 42)


def test_text_engine_adds_cell_context_to_extraction_errors():
    class Region:
        page = None

        def analyze_text_table_structure(self, **_kwargs):
            return {"cells": [{"top": 0, "left": 0, "width": 10, "height": 10}]}

    class Page:
        def region(self, **_kwargs):
            raise OSError("region unavailable")

    Region.page = Page()
    with pytest.raises(RuntimeError, match=r"row 0, column 0") as exc_info:
        TextTablesEngine().extract_tables(
            context=None,
            region=Region(),
            cell_extraction_func=lambda _cell: "x",
        )
    assert isinstance(exc_info.value.__cause__, OSError)


@pytest.mark.parametrize(
    ("overlap", "expected"),
    [
        ("center", [["A SPAN", "B"], ["C", "D"]]),
        ("full", [["A", "B"], ["C", "D"]]),
        ("partial", [["A SPAN", "B SPAN"], ["C", "D"]]),
    ],
)
def test_guide_cell_words_fast_path_bins_by_overlap(overlap, expected):
    cells = _four_cells()
    words = [
        FakeWord("A", (1, 1, 2, 2)),
        FakeWord("B", (11, 1, 12, 2)),
        FakeWord("C", (1, 11, 2, 12)),
        FakeWord("D", (11, 11, 12, 12)),
        FakeWord("SPAN", (9.2, 1, 10.2, 2)),
    ]
    host = FakeTableRegion(cells, words=words)

    result = TableService(None).extract_table(
        host,
        cell_extract="words",
        cell_overlap=overlap,
    )

    assert list(result) == expected
    assert all(cell.extract_calls == 0 for cell in cells)
    assert host.find_all_calls == [("text", {"overlap": "partial", "apply_exclusions": True})]


def test_guide_cell_words_fast_path_applies_newlines_and_content_filter():
    cells = [FakeCell(0, 0, (0, 0, 20, 10))]
    words = [FakeWord("KEEP\nDROP", (1, 1, 5, 2))]
    host = FakeTableRegion(cells, words=words)

    result = TableService(None).extract_table(
        host,
        cell_extract="words",
        cell_newlines=False,
        content_filter="strip_drop",
    )

    assert list(result) == [["KEEP"]]


def test_guides_extract_table_words_uses_tiny_text_fixture():
    pdf = PDF("pdfs/tiny-text-tables.pdf")
    try:
        page = pdf.pages[0]
        guides = Guides(
            verticals=[70, 90, 112],
            horizontals=[55, 57.3],
            context=page,
        )

        result = guides.extract_table(
            cell_padding=0,
            cell_extract="words",
            cell_overlap="partial",
            cell_newlines=False,
            header=None,
        )

        rows = list(result)
        assert rows[0][0] == "OFFICER"
        assert "DATE OF REPORT" in rows[0][1]
    finally:
        pdf.close()
