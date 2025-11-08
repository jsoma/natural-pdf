from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

from natural_pdf.analyzers.guides.flow_adapter import FlowGuideAdapter, RegionGrid


class DummyGuides:
    def __init__(self, flow_region, verticals, horizontals):
        self._flow_region = flow_region
        self.is_flow_region = True
        self._verticals = verticals
        self._horizontals = horizontals

    def _flow_context(self):
        return self._flow_region

    def _flow_constituent_regions(self):
        return self._flow_region.constituent_regions

    @property
    def vertical(self):
        return self._verticals

    @property
    def horizontal(self):
        return self._horizontals


class _HashableNamespace(SimpleNamespace):
    def __hash__(self):
        return id(self)


def _stub_region(bbox, page_number=1, row_index=None, col_index=None, region_type="table_row"):
    region = _HashableNamespace(
        bbox=bbox,
        page=SimpleNamespace(page_number=page_number),
        metadata={},
        region_type=region_type,
        source="guides",
    )
    region.x0, region.top, region.x1, region.bottom = bbox
    if row_index is not None:
        region.metadata["row_index"] = row_index
    if col_index is not None:
        region.metadata["col_index"] = col_index
    return region


def _region_grid(region, rows, columns, cells):
    return RegionGrid(region=region, table=None, rows=rows, columns=columns, cells=cells)


def test_flow_adapter_horizontal_merges_columns():
    region_a = _stub_region((0, 0, 50, 100))
    region_b = _stub_region((50, 0, 100, 100))
    flow = SimpleNamespace()
    flow_region = SimpleNamespace(flow=flow, constituent_regions=[region_a, region_b])
    guides = DummyGuides(flow_region, verticals=[0, 100], horizontals=[0, 100])
    adapter = FlowGuideAdapter(guides)

    # Build fake grids that simulate a single column split across two regions
    col_a = _stub_region((40, 0, 50, 100), col_index=1, region_type="table_column")
    col_b = _stub_region((50, 0, 60, 100), col_index=0, region_type="table_column")
    cell_a = _stub_region((40, 0, 50, 50), row_index=0, col_index=1, region_type="table_cell")
    cell_b = _stub_region((50, 0, 60, 50), row_index=0, col_index=0, region_type="table_cell")
    row_a = _stub_region((0, 0, 50, 50), row_index=0)
    row_b = _stub_region((50, 0, 100, 50), row_index=0)

    region_grids: List[RegionGrid] = [
        _region_grid(region_a, rows=[row_a], columns=[col_a], cells=[cell_a]),
        _region_grid(region_b, rows=[row_b], columns=[col_b], cells=[cell_b]),
    ]

    rows, cols, cells = adapter.stitch_region_results(region_grids, "horizontal", source="guides")

    assert rows, "Rows should be stitched across horizontal regions"
    assert any(col.metadata.get("is_multi_page") for col in cols)
    assert any(cell.metadata.get("is_multi_page") for cell in cells)


def test_flow_adapter_unknown_orientation_returns_fragments():
    region = _stub_region((0, 0, 50, 100))
    flow = SimpleNamespace()
    flow_region = SimpleNamespace(flow=flow, constituent_regions=[region])
    guides = DummyGuides(flow_region, verticals=[0, 50], horizontals=[0, 50])
    adapter = FlowGuideAdapter(guides)

    row = _stub_region((0, 0, 50, 25), row_index=0)
    column = _stub_region((0, 0, 25, 100), col_index=0, region_type="table_column")
    cell = _stub_region((0, 0, 25, 25), row_index=0, col_index=0, region_type="table_cell")

    region_grids = [_region_grid(region, rows=[row], columns=[column], cells=[cell])]
    rows, cols, cells = adapter.stitch_region_results(region_grids, "unknown", source="guides")

    assert rows == [row]
    assert cols == [column]
    assert cells == [cell]
