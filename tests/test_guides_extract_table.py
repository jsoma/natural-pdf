#!/usr/bin/env python3
"""Test the new guides.extract_table() method."""

from unittest.mock import Mock, PropertyMock, patch

import pytest

from natural_pdf.analyzers.guides import Guides


def test_guides_extract_table_basic():
    """Test that guides.extract_table() works for basic case."""

    # Create mock page
    mock_page = Mock()
    mock_page._element_mgr = Mock()
    mock_page._element_mgr.regions = []
    mock_page.width = 612
    mock_page.height = 792

    # Create guides with some vertical and horizontal lines
    guides = Guides(verticals=[100, 200, 300], horizontals=[100, 150, 200], context=mock_page)

    # Mock the build_grid method to return a result
    mock_table_region = Mock()
    mock_table_result = Mock()
    mock_table_region.extract_table.return_value = mock_table_result

    mock_grid_result = {
        "counts": {"table": 1, "rows": 2, "columns": 2, "cells": 4},
        "regions": {"table": mock_table_region, "rows": [], "columns": [], "cells": []},
    }

    with patch.object(guides, "build_grid", return_value=mock_grid_result):
        # Call extract_table
        result = guides.extract_table()

        # Verify that build_grid was called with correct parameters
        guides.build_grid.assert_called_once_with(
            target=mock_page,
            source="guides_temp",
            cell_padding=0.5,
            include_outer_boundaries=False,
            multi_page="auto",
        )

        # Verify that extract_table was called on the table region
        mock_table_region.extract_table.assert_called_once()

        # Verify result
        assert result == mock_table_result


def test_guides_extract_table_with_parameters():
    """Test that guides.extract_table() passes parameters correctly."""

    # Create mock page
    mock_page = Mock()
    mock_page._element_mgr = Mock()
    mock_page._element_mgr.regions = []

    # Create guides
    guides = Guides(verticals=[100, 200], horizontals=[100, 200], context=mock_page)

    # Mock the build_grid method
    mock_table_region = Mock()
    mock_table_result = Mock()
    mock_table_region.extract_table.return_value = mock_table_result

    mock_grid_result = {
        "regions": {"table": mock_table_region, "rows": [], "columns": [], "cells": []}
    }

    with patch.object(guides, "build_grid", return_value=mock_grid_result):
        # Call with custom parameters
        result = guides.extract_table(
            method="tatr", use_ocr=True, cell_padding=1.0, include_outer_boundaries=True
        )

        # Verify build_grid was called with custom parameters
        guides.build_grid.assert_called_once_with(
            target=mock_page,
            source="guides_temp",
            cell_padding=1.0,
            include_outer_boundaries=True,
            multi_page="auto",
        )

        # Verify extract_table was called with custom parameters
        mock_table_region.extract_table.assert_called_once_with(
            method="tatr",
            table_settings=None,
            use_ocr=True,
            ocr_config=None,
            text_options=None,
            cell_extraction_func=None,
            show_progress=False,
            content_filter=None,
            apply_exclusions=True,
        )


def test_guides_extract_table_cleanup_on_success():
    """Test that temporary regions are cleaned up after successful extraction."""

    # Create mock temporary regions that should be cleaned up
    temp_region1 = Mock()
    temp_region1.source = "guides_temp"
    temp_region1.region_type = "table"

    temp_region2 = Mock()
    temp_region2.source = "guides_temp"
    temp_region2.region_type = "table_cell"

    other_region = Mock()
    other_region.source = "other_source"
    other_region.region_type = "table"

    # Make sure regions is a proper list that can be iterated over
    regions_list = [temp_region1, temp_region2, other_region]

    # Create mock element manager - use spec_set to prevent attribute creation
    mock_element_manager = Mock(spec_set=["regions", "remove_element"])
    mock_element_manager.regions = regions_list
    mock_element_manager.remove_element = Mock()

    # Create mock page
    mock_page = Mock(spec_set=["_element_mgr", "width", "height"])
    mock_page._element_mgr = mock_element_manager
    mock_page.width = 612
    mock_page.height = 792

    # Create guides
    guides = Guides(verticals=[100, 200], horizontals=[100, 200], context=mock_page)

    # Mock successful table extraction
    mock_table_region = Mock()
    mock_table_result = Mock()
    mock_table_region.extract_table.return_value = mock_table_result

    mock_grid_result = {"regions": {"table": mock_table_region}}

    with patch.object(guides, "build_grid", return_value=mock_grid_result):
        result = guides.extract_table()

        # Verify cleanup was called for temp regions only
        expected_calls = [mock_element_manager.remove_element.call_args_list]

        # Should have called remove_element twice (for the two temp regions)
        assert mock_element_manager.remove_element.call_count == 2

        # Verify the temp regions were removed (other_region should not be removed)
        removed_regions = [
            call[0][0] for call in mock_element_manager.remove_element.call_args_list
        ]
        assert temp_region1 in removed_regions
        assert temp_region2 in removed_regions
        assert other_region not in removed_regions


def test_guides_extract_table_cleanup_on_failure():
    """Test that temporary regions are cleaned up even when extraction fails."""

    # Create mock temporary region
    temp_region = Mock()
    temp_region.source = "guides_temp"
    temp_region.region_type = "table"
    # Make sure regions is a proper list that can be iterated over
    regions_list = [temp_region]

    # Create mock element manager - use spec_set to prevent attribute creation
    mock_element_manager = Mock(spec_set=["regions", "remove_element"])
    mock_element_manager.regions = regions_list
    mock_element_manager.remove_element = Mock()

    # Create mock page
    mock_page = Mock(spec_set=["_element_mgr", "width", "height"])
    mock_page._element_mgr = mock_element_manager
    mock_page.width = 612
    mock_page.height = 792

    # Create guides
    guides = Guides(verticals=[100, 200], horizontals=[100, 200], context=mock_page)

    # Mock build_grid to return valid result but extract_table to fail
    mock_table_region = Mock()
    mock_table_region.extract_table.side_effect = ValueError("Extraction failed")

    mock_grid_result = {"regions": {"table": mock_table_region}}

    with patch.object(guides, "build_grid", return_value=mock_grid_result):
        # Should raise the extraction error
        with pytest.raises(ValueError, match="Extraction failed"):
            guides.extract_table()

        # But cleanup should still happen
        mock_element_manager.remove_element.assert_called_once_with(
            temp_region, element_type="regions"
        )


def test_guides_extract_table_no_table_region():
    """Test error when no table region is created."""

    mock_page = Mock()
    mock_page._element_mgr = Mock()
    mock_page._element_mgr.regions = []

    guides = Guides(
        verticals=[100],  # Only one vertical - can't make table
        horizontals=[100, 200],
        context=mock_page,
    )

    # Mock build_grid to return no table region
    mock_grid_result = {"regions": {"table": None}}

    with patch.object(guides, "build_grid", return_value=mock_grid_result):
        with pytest.raises(ValueError, match="No table region was created"):
            guides.extract_table()


def test_guides_extract_table_multi_page_list():
    """Test handling of multi-page case where table region is a list."""

    mock_page = Mock()
    mock_page._element_mgr = Mock()
    mock_page._element_mgr.regions = []

    guides = Guides(verticals=[100, 200], horizontals=[100, 200], context=mock_page)

    # Mock multi-page result with list of table regions
    mock_table_region1 = Mock()
    mock_table_region2 = Mock()
    mock_table_result = Mock()
    mock_table_region1.extract_table.return_value = mock_table_result

    mock_grid_result = {
        "regions": {"table": [mock_table_region1, mock_table_region2]}  # List of regions
    }

    with patch.object(guides, "build_grid", return_value=mock_grid_result):
        result = guides.extract_table()

        # Should use the first table region
        mock_table_region1.extract_table.assert_called_once()
        mock_table_region2.extract_table.assert_not_called()

        assert result == mock_table_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
