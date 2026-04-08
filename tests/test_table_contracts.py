from __future__ import annotations

from unittest.mock import Mock, patch

from natural_pdf.analyzers.guides import Guides
from natural_pdf.tables.result import TableResult


def test_guides_extract_table_accepts_flowregion_page_property():
    mock_page = Mock()
    mock_page.iter_regions.return_value = []
    mock_page.remove_element.return_value = True
    mock_page.add_element.return_value = True

    flow_region = Mock()
    flow_region.x0 = 0
    flow_region.top = 0
    flow_region.page = mock_page

    guides = Guides(verticals=[100, 200], horizontals=[100, 200], context=flow_region)

    mock_table_region = Mock()
    mock_table_region.extract_table.return_value = TableResult([["ok"]])
    mock_grid_result = {
        "regions": {"table": mock_table_region, "rows": [], "columns": [], "cells": []}
    }

    with patch.object(guides, "build_grid", return_value=mock_grid_result):
        result = guides.extract_table()

    assert list(result) == [["ok"]]


def test_text_table_structure_cache_keys_on_options(monkeypatch, practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region = page.to_region()
    calls = {"count": 0}

    def _fake_find_text_based_tables(**kwargs):
        calls["count"] += 1
        return {
            "horizontal_edges": [],
            "vertical_edges": [],
            "cells": [{"left": 0, "top": 0, "right": 10, "bottom": 10}],
            "intersections": {},
        }

    monkeypatch.setattr(
        "natural_pdf.elements.region.find_text_based_tables", _fake_find_text_based_tables
    )

    region.analyze_text_table_structure(join_tolerance=3)
    region.analyze_text_table_structure(join_tolerance=3)
    region.analyze_text_table_structure(join_tolerance=4)

    assert calls["count"] == 2


def test_text_table_structure_cache_invalidates_on_text_state_change(
    monkeypatch, practice_pdf_fresh
):
    page = practice_pdf_fresh.pages[0]
    region = page.to_region()
    calls = {"count": 0}

    def _fake_find_text_based_tables(**kwargs):
        calls["count"] += 1
        return {
            "horizontal_edges": [],
            "vertical_edges": [],
            "cells": [{"left": 0, "top": 0, "right": 10, "bottom": 10}],
            "intersections": {},
        }

    monkeypatch.setattr(
        "natural_pdf.elements.region.find_text_based_tables", _fake_find_text_based_tables
    )

    region.analyze_text_table_structure()
    page._bump_text_state_version()
    region.analyze_text_table_structure()

    assert calls["count"] == 2


def test_region_type_selectors_accept_underscore_alias_for_hyphenated_types(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, 10, 10)
    region.region_type = "table-cell"
    region.normalized_type = "table-cell"
    page.add_region(region, source="test")

    matches = page.find_all("region[type=table_cell]")

    assert region in matches.elements
