from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from natural_pdf.flows.collections import FlowRegionCollection
from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion
from natural_pdf.tables import TableResult


class _DummyRegion:
    """Minimal Region stand-in for FlowRegion tests."""

    def __init__(self, rows: List[List[str]], page_number: int):
        self.page = SimpleNamespace(number=page_number)
        self._rows = rows
        self.extract_table_calls: List[Dict[str, Any]] = []
        self.extract_tables_calls: List[Dict[str, Any]] = []

    def extract_table(self, **kwargs: Any) -> TableResult:
        self.extract_table_calls.append(kwargs)
        return TableResult(list(self._rows))

    def extract_tables(self, **kwargs: Any) -> List[TableResult]:
        self.extract_tables_calls.append(kwargs)
        return [TableResult(list(self._rows))]


def _flow_region_with(rows_per_region: List[List[List[Optional[str]]]]) -> FlowRegion:
    regions = [
        _DummyRegion(rows=rows, page_number=index + 1) for index, rows in enumerate(rows_per_region)
    ]
    flow = SimpleNamespace()
    return FlowRegion(flow=flow, constituent_regions=regions)


def _flow_with_segments(rows_per_segment: List[List[List[Optional[str]]]]) -> Flow:
    segments = [
        _DummyRegion(rows=rows, page_number=index + 1)
        for index, rows in enumerate(rows_per_segment)
    ]
    flow = Flow.__new__(Flow)
    flow.segments = segments  # type: ignore[attr-defined]
    flow._analysis_region_cache = None
    return flow


def _flow_region_collection(
    rows_per_flow_region: List[List[List[List[Optional[str]]]]],
) -> FlowRegionCollection:
    flow_regions = [_flow_region_with(rows) for rows in rows_per_flow_region]
    return FlowRegionCollection(flow_regions)


def test_flow_region_extract_table_passes_extended_arguments():
    """FlowRegion should forward the new Region table options to every segment."""

    rows = [
        [["A", "B"], ["row1", "row2"]],
        [["C", "D"], ["row3", "row4"]],
    ]
    flow_region = _flow_region_with(rows)

    table_settings = {"vertical_strategy": "text"}
    text_options = {"coordinate_grouping_tolerance": 2}

    result = flow_region.extract_table(
        method="stream",
        table_settings=table_settings,
        text_options=text_options,
        content_filter=r"\d",
        apply_exclusions=False,
        verticals=[10.0],
        horizontals=[20.0],
    )

    assert isinstance(result, TableResult)
    # Combined rows should include every region in flow order
    assert list(result) == rows[0] + rows[1]

    for region in flow_region.constituent_regions:
        kwargs = region.extract_table_calls[-1]
        assert kwargs["content_filter"] == r"\d"
        assert kwargs["apply_exclusions"] is False
        assert kwargs["verticals"] == [10.0]
        assert kwargs["horizontals"] == [20.0]
        # table_settings/text_options should be copied before dispatch
        assert kwargs["table_settings"] == table_settings
        assert kwargs["table_settings"] is not table_settings
        assert kwargs["text_options"] == text_options
        assert kwargs["text_options"] is not text_options


def test_flow_region_extract_tables_copies_table_settings_per_segment():
    rows = [
        [["H1", "H2"], ["r1", "r2"]],
        [["H1", "H2"], ["r3", "r4"]],
    ]
    flow_region = _flow_region_with(rows)
    table_settings = {"horizontal_strategy": "lines"}

    tables = flow_region.extract_tables(method="pdfplumber", table_settings=table_settings)

    # Each region contributed a table, so aggregated list should have two entries
    assert len(tables) == 2
    for region in flow_region.constituent_regions:
        kwargs = region.extract_tables_calls[-1]
        assert kwargs["table_settings"] == table_settings
        assert kwargs["table_settings"] is not table_settings


def test_flow_extract_tables_aggregates_all_segments():
    rows = [
        [["H1", "H2"], ["r1", "r2"]],
        [["H1", "H2"], ["r3", "r4"]],
    ]
    flow = _flow_with_segments(rows)
    table_settings = {"vertical_strategy": "text"}

    tables = flow.extract_tables(method="stream", table_settings=table_settings)

    assert len(tables) == 2
    assert isinstance(tables[0], TableResult)
    assert isinstance(tables[1], TableResult)
    assert list(tables[0]) == rows[0]
    assert list(tables[1]) == rows[1]

    for segment in flow.segments:
        kwargs = segment.extract_tables_calls[-1]
        assert kwargs["table_settings"] == table_settings
        assert kwargs["table_settings"] is not table_settings


def test_flow_region_collection_extract_table_returns_per_region_results():
    flows = [
        [
            [["A1", "A2"], ["A3", "A4"]],
        ],
        [
            [["B1", "B2"], ["B3", "B4"]],
        ],
    ]
    collection = _flow_region_collection(flows)

    results = collection.extract_table(method="stream")
    assert len(results) == len(flows)
    for idx, table_result in enumerate(results):
        assert isinstance(table_result, TableResult)
        assert list(table_result) == flows[idx][0]


def test_flow_region_collection_extract_tables_flattens_results():
    flows = [
        [
            [["A1", "A2"], ["A3", "A4"]],
        ],
        [
            [["B1", "B2"], ["B3", "B4"]],
        ],
    ]
    collection = _flow_region_collection(flows)

    tables = collection.extract_tables(method="lattice")
    assert len(tables) == len(flows)
    assert isinstance(tables[0], TableResult)
    assert isinstance(tables[1], TableResult)
    assert list(tables[0]) == flows[0][0]
    assert list(tables[1]) == flows[1][0]


def test_flow_region_extract_table_recovers_row_dropped_at_segment_seam():
    """A row at the top of a post-seam segment must not be dropped.

    In pdfs/multicolumn.pdf, "Table one" starts at the bottom of column 1 and
    continues at the top of column 2. The renderer never redrew the table
    border after the column break, so the ruling line above row 25 does not
    exist in the second segment and pdfplumber's lattice detection starts at
    the next ruling. The flow path must recover that seam row.
    """
    from pathlib import Path

    import pytest

    from natural_pdf import PDF

    source = Path("pdfs/multicolumn.pdf")
    if not source.exists():
        pytest.skip("Test requires pdfs/multicolumn.pdf fixture")

    pdf = PDF(str(source))
    try:
        page = pdf.pages[0]
        left = page.region(left=0, right=page.width / 3)
        mid = page.region(left=page.width / 3, right=2 * page.width / 3)
        right = page.region(left=2 * page.width / 3, right=page.width)
        flow = Flow(segments=[left, mid, right], arrangement="vertical")

        table_one = flow.find('text:contains("Table one")').below(
            until='text:contains("Table two")', include_endpoint=False
        )

        # The seam row is present in the text layer...
        assert "25 8115" in table_one.extract_text()

        rows = list(table_one.extract_table())
        first_col = [row[0] for row in rows]

        # ...and must be present in the extracted table too.
        assert ["25", "8115"] == rows[25]
        assert first_col == ["index"] + [str(i) for i in range(1, 40)]

        # Total row count matches the per-segment sum: 25 rows from the first
        # segment (header + rows 1-24) and 15 from the second (rows 25-39,
        # including the recovered seam row).
        assert len(rows) == 40

        # Single-region extraction semantics are unchanged: the recovery only
        # applies on the flow path.
        seg_first, seg_second = table_one.constituent_regions
        assert len(list(seg_first.extract_table())) == 25
        assert len(list(seg_second.extract_table())) == 14
    finally:
        pdf.close()
