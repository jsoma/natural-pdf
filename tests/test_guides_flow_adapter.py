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


class _BuilderGuides:
    calls = []

    def __init__(self, verticals=None, horizontals=None, context=None):
        self.vertical = verticals or []
        self.horizontal = horizontals or []
        self.context = context
        self._flow_guides = {}
        self.is_flow_region = False

    def _build_grid_single_page(
        self,
        *,
        target,
        source,
        cell_padding,
        include_outer_boundaries,
    ):
        self.__class__.calls.append(include_outer_boundaries)
        return {
            "counts": {"table": 1, "rows": 0, "columns": 0, "cells": 0},
            "regions": {"table": object(), "rows": [], "columns": [], "cells": []},
        }


class _AdapterBuildGuides(_BuilderGuides):
    def __init__(self, flow_region=None, verticals=None, horizontals=None, context=None):
        active_context = flow_region if flow_region is not None else context
        super().__init__(verticals=verticals, horizontals=horizontals, context=active_context)
        self._flow_region = active_context
        self.is_flow_region = True

    def _flow_context(self):
        return self._flow_region

    def _flow_constituent_regions(self):
        return self._flow_region.constituent_regions


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


def test_flow_adapter_forwards_include_outer_boundaries():
    region = _stub_region((0, 0, 50, 100))
    flow_region = SimpleNamespace(flow=SimpleNamespace(), constituent_regions=[region])

    guides = _AdapterBuildGuides(flow_region, verticals=[0, 50], horizontals=[0, 100])
    _BuilderGuides.calls = []

    adapter = FlowGuideAdapter(guides)
    adapter.build_region_grids(
        source="guides",
        cell_padding=0.5,
        include_outer_boundaries=True,
    )

    assert _BuilderGuides.calls == [True]


def test_flow_adapter_raises_on_mismatched_stitch_counts():
    region_a = _stub_region((0, 0, 50, 100))
    region_b = _stub_region((0, 100, 50, 200))
    flow_region = SimpleNamespace(flow=SimpleNamespace(), constituent_regions=[region_a, region_b])
    guides = DummyGuides(flow_region, verticals=[0, 50], horizontals=[0, 100, 200])
    adapter = FlowGuideAdapter(guides)

    row_a0 = _stub_region((0, 0, 50, 50), row_index=0)
    row_a1 = _stub_region((0, 50, 50, 100), row_index=1)
    row_b0 = _stub_region((0, 100, 50, 150), row_index=0)

    region_grids = [
        _region_grid(region_a, rows=[row_a0, row_a1], columns=[], cells=[]),
        _region_grid(region_b, rows=[row_b0], columns=[], cells=[]),
    ]

    with pytest.raises(ValueError, match="Inconsistent multi-page row counts"):
        adapter.stitch_region_results(region_grids, "horizontal", source="guides")
