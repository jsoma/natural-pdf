from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

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

    def extract_tables(self, **kwargs: Any) -> List[List[List[str]]]:
        self.extract_tables_calls.append(kwargs)
        return [list(self._rows)]


def _flow_region_with(rows_per_region: List[List[List[str]]]) -> FlowRegion:
    regions = [
        _DummyRegion(rows=rows, page_number=index + 1) for index, rows in enumerate(rows_per_region)
    ]
    flow = SimpleNamespace()
    return FlowRegion(flow=flow, constituent_regions=regions)


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
