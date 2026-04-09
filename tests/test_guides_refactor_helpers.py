from __future__ import annotations

from unittest.mock import Mock

from natural_pdf.analyzers.guides import Guides
from natural_pdf.analyzers.guides._grid_builder import (
    build_single_page_grid,
    remove_temporary_grid_regions,
)
from natural_pdf.analyzers.guides._targets import resolve_page_for_materialization
from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion


def test_resolve_page_for_materialization_accepts_page_region_and_flowregion(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region = page.region(0, 0, 50, 50)
    flow = Flow(segments=[page], arrangement="vertical")
    flow_region = FlowRegion(flow=flow, constituent_regions=[region], source_flow_element=None)

    assert resolve_page_for_materialization(page) is page
    assert resolve_page_for_materialization(region) is page
    assert resolve_page_for_materialization(flow_region) is page


def test_remove_temporary_grid_regions_filters_only_canonical_types():
    page = Mock()
    canonical = Mock()
    canonical.source = "guides_temp"
    canonical.region_type = "table-cell"
    canonical.bbox = (0, 0, 10, 10)

    legacy = Mock()
    legacy.source = "guides_temp"
    legacy.region_type = "table_cell"
    legacy.bbox = (0, 0, 10, 10)

    other_source = Mock()
    other_source.source = "other"
    other_source.region_type = "table-cell"
    other_source.bbox = (0, 0, 10, 10)

    page.iter_regions.return_value = [canonical, legacy, other_source]

    removed = remove_temporary_grid_regions([page], source="guides_temp")

    assert removed == [canonical]
    page.remove_element.assert_called_once_with(canonical, element_type="regions")


def test_single_page_grid_builder_matches_guides_build_grid(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    guides = Guides(verticals=[0, 50, 100], horizontals=[0, 40, 80], context=page)

    helper_result = build_single_page_grid(
        target_obj=page,
        verticals=guides.vertical,
        horizontals=guides.horizontal,
        source="helper_grid",
        cell_padding=0.5,
        include_outer_boundaries=False,
    )
    public_result = guides.build_grid(source="public_grid")

    assert helper_result.counts.as_dict() == public_result["counts"]
    assert helper_result.regions.table is not None
    assert len(helper_result.regions.rows) == len(public_result["regions"]["rows"])
    assert len(helper_result.regions.columns) == len(public_result["regions"]["columns"])
    assert len(helper_result.regions.cells) == len(public_result["regions"]["cells"])

    remove_temporary_grid_regions([page], source="helper_grid")
    remove_temporary_grid_regions([page], source="public_grid")


def test_guides_build_grid_keeps_legacy_dict_shape(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    guides = Guides(verticals=[0, 50], horizontals=[0, 40], context=page)

    result = guides.build_grid(source="legacy_shape")

    assert set(result.keys()) == {"counts", "regions"}
    assert set(result["counts"].keys()) == {"table", "rows", "columns", "cells"}
    assert set(result["regions"].keys()) == {"table", "rows", "columns", "cells"}

    remove_temporary_grid_regions([page], source="legacy_shape")
