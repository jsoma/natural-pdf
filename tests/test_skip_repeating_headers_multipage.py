#!/usr/bin/env python3
"""Test merge_headers functionality for multi-page tables."""

import warnings
from unittest.mock import Mock

import pytest

from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion
from natural_pdf.tables import TableResult


def test_merge_headers_auto_detection():
    """Test auto-detection of repeated headers across segments."""

    # Mock segments (pages) with repeated headers
    mock_segment1 = Mock()
    mock_segment2 = Mock()
    mock_segment3 = Mock()

    # Simulate table data from each page/segment
    # Page 1: Header + data
    page1_data = [
        ["Name", "Age", "City"],  # Header
        ["Alice", "25", "New York"],  # Data
        ["Bob", "30", "London"],  # Data
    ]

    # Page 2: Repeated header + more data
    page2_data = [
        ["Name", "Age", "City"],  # Repeated header (should be removed)
        ["Charlie", "35", "Paris"],  # Data
        ["David", "40", "Tokyo"],  # Data
    ]

    # Page 3: Repeated header + final data
    page3_data = [
        ["Name", "Age", "City"],  # Repeated header (should be removed)
        ["Eve", "28", "Berlin"],  # Data
    ]

    # Mock the extract_table method for each segment
    mock_segment1.extract_table.return_value = TableResult(page1_data)
    mock_segment2.extract_table.return_value = TableResult(page2_data)
    mock_segment3.extract_table.return_value = TableResult(page3_data)

    # Create a mock FlowRegion
    mock_flow = Mock()
    flow_region = FlowRegion(
        flow=mock_flow,
        constituent_regions=[mock_segment1, mock_segment2, mock_segment3],
        source_flow_element=None,
    )

    # Test auto-detection (default behavior) - should emit warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = flow_region.extract_table()

        # Check that warning was issued
        assert len(w) == 1
        assert "Detected repeated headers" in str(w[0].message)
        assert issubclass(w[0].category, UserWarning)

    result_rows = list(result)

    print("Auto-detection results:")
    print(f"Total rows: {len(result_rows)}")
    for i, row in enumerate(result_rows):
        print(f"  Row {i}: {row}")

    # Should have: header + 2 data from page1 + 2 data from page2 + 1 data from page3 = 6 rows
    expected_rows = [
        ["Name", "Age", "City"],  # Original header
        ["Alice", "25", "New York"],  # Page 1 data
        ["Bob", "30", "London"],  # Page 1 data
        ["Charlie", "35", "Paris"],  # Page 2 data (header removed)
        ["David", "40", "Tokyo"],  # Page 2 data
        ["Eve", "28", "Berlin"],  # Page 3 data (header removed)
    ]

    assert result_rows == expected_rows, f"Expected {expected_rows}, got {result_rows}"
    print("✅ Auto-detection test passed!")


def test_merge_headers_explicit_control():
    """Test explicit control of header merging."""

    # Same setup as above
    mock_segment1 = Mock()
    mock_segment2 = Mock()

    page1_data = [["Name", "Age"], ["Alice", "25"]]
    page2_data = [["Name", "Age"], ["Bob", "30"]]  # Repeated header

    mock_segment1.extract_table.return_value = TableResult(page1_data)
    mock_segment2.extract_table.return_value = TableResult(page2_data)

    mock_flow = Mock()
    flow_region = FlowRegion(
        flow=mock_flow, constituent_regions=[mock_segment1, mock_segment2], source_flow_element=None
    )

    # Test with merge_headers=False (keep all rows)
    result_keep = flow_region.extract_table(merge_headers=False)
    result_keep_rows = list(result_keep)

    print("Keep repeating headers:")
    print(f"Total rows: {len(result_keep_rows)}")
    for i, row in enumerate(result_keep_rows):
        print(f"  Row {i}: {row}")

    # Should keep all rows including repeated header
    expected_keep = [
        ["Name", "Age"],  # Original header
        ["Alice", "25"],  # Page 1 data
        ["Name", "Age"],  # Repeated header (kept)
        ["Bob", "30"],  # Page 2 data
    ]

    assert result_keep_rows == expected_keep, f"Expected {expected_keep}, got {result_keep_rows}"

    # Test with merge_headers=True (remove repeated headers) - should emit warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_skip = flow_region.extract_table(merge_headers=True)

        # Check that warning was issued
        assert len(w) == 1
        assert "Removing repeated headers" in str(w[0].message)
        assert issubclass(w[0].category, UserWarning)

    result_skip_rows = list(result_skip)

    print("Merge headers:")
    print(f"Total rows: {len(result_skip_rows)}")
    for i, row in enumerate(result_skip_rows):
        print(f"  Row {i}: {row}")

    # Should remove repeated headers
    expected_skip = [
        ["Name", "Age"],  # Original header
        ["Alice", "25"],  # Page 1 data
        ["Bob", "30"],  # Page 2 data (repeated header removed)
    ]

    assert result_skip_rows == expected_skip, f"Expected {expected_skip}, got {result_skip_rows}"
    print("✅ Explicit control test passed!")


def test_no_repeated_headers():
    """Test behavior when headers don't repeat."""

    mock_segment1 = Mock()
    mock_segment2 = Mock()

    # Page 1: Header + data
    page1_data = [["Name", "Age"], ["Alice", "25"]]
    # Page 2: Just data (no repeated header)
    page2_data = [["Bob", "30"], ["Charlie", "35"]]

    mock_segment1.extract_table.return_value = TableResult(page1_data)
    mock_segment2.extract_table.return_value = TableResult(page2_data)

    mock_flow = Mock()
    flow_region = FlowRegion(
        flow=mock_flow, constituent_regions=[mock_segment1, mock_segment2], source_flow_element=None
    )

    # Auto-detection should detect no repeating headers (no warning expected)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = flow_region.extract_table()

        # Should not emit warning since no headers are being merged
        assert len(w) == 0

    result_rows = list(result)

    print("No repeated headers:")
    print(f"Total rows: {len(result_rows)}")
    for i, row in enumerate(result_rows):
        print(f"  Row {i}: {row}")

    # Should keep all rows (no headers were repeated)
    expected = [
        ["Name", "Age"],  # Original header
        ["Alice", "25"],  # Page 1 data
        ["Bob", "30"],  # Page 2 data
        ["Charlie", "35"],  # Page 2 data
    ]

    assert result_rows == expected, f"Expected {expected}, got {result_rows}"
    print("✅ No repeated headers test passed!")


def test_inconsistent_header_pattern_error():
    """Test that inconsistent header patterns raise ValueError."""

    mock_segment1 = Mock()
    mock_segment2 = Mock()
    mock_segment3 = Mock()

    # Page 1: Header + data
    page1_data = [["Name", "Age"], ["Alice", "25"]]
    # Page 2: Repeated header + data
    page2_data = [["Name", "Age"], ["Bob", "30"]]  # Has repeated header
    # Page 3: Just data (no repeated header) - INCONSISTENT!
    page3_data = [["Charlie", "35"], ["David", "40"]]  # No repeated header

    mock_segment1.extract_table.return_value = TableResult(page1_data)
    mock_segment2.extract_table.return_value = TableResult(page2_data)
    mock_segment3.extract_table.return_value = TableResult(page3_data)

    mock_flow = Mock()
    flow_region = FlowRegion(
        flow=mock_flow,
        constituent_regions=[mock_segment1, mock_segment2, mock_segment3],
        source_flow_element=None,
    )

    # Should raise ValueError due to inconsistent pattern
    with pytest.raises(ValueError, match="Inconsistent header pattern"):
        flow_region.extract_table()

    print("✅ Inconsistent header pattern error test passed!")


def test_inconsistent_header_pattern_error_reverse():
    """Test inconsistent patterns where first segment has no header but later ones do."""

    mock_segment1 = Mock()
    mock_segment2 = Mock()
    mock_segment3 = Mock()

    # Page 1: Header + data
    page1_data = [["Name", "Age"], ["Alice", "25"]]
    # Page 2: Just data (no repeated header)
    page2_data = [["Bob", "30"], ["Charlie", "35"]]  # No repeated header
    # Page 3: Repeated header + data - INCONSISTENT!
    page3_data = [["Name", "Age"], ["David", "40"]]  # Has repeated header

    mock_segment1.extract_table.return_value = TableResult(page1_data)
    mock_segment2.extract_table.return_value = TableResult(page2_data)
    mock_segment3.extract_table.return_value = TableResult(page3_data)

    mock_flow = Mock()
    flow_region = FlowRegion(
        flow=mock_flow,
        constituent_regions=[mock_segment1, mock_segment2, mock_segment3],
        source_flow_element=None,
    )

    # Should raise ValueError due to inconsistent pattern
    with pytest.raises(ValueError, match="Inconsistent header pattern"):
        flow_region.extract_table()

    print("✅ Reverse inconsistent header pattern error test passed!")


def test_warning_only_once():
    """Test that warning is only emitted once even with multiple headers."""

    mock_segment1 = Mock()
    mock_segment2 = Mock()
    mock_segment3 = Mock()

    # All segments have repeated headers
    page1_data = [["Name", "Age"], ["Alice", "25"]]
    page2_data = [["Name", "Age"], ["Bob", "30"]]
    page3_data = [["Name", "Age"], ["Charlie", "35"]]

    mock_segment1.extract_table.return_value = TableResult(page1_data)
    mock_segment2.extract_table.return_value = TableResult(page2_data)
    mock_segment3.extract_table.return_value = TableResult(page3_data)

    mock_flow = Mock()
    flow_region = FlowRegion(
        flow=mock_flow,
        constituent_regions=[mock_segment1, mock_segment2, mock_segment3],
        source_flow_element=None,
    )

    # Should only emit one warning despite multiple headers being removed
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = flow_region.extract_table()

        # Should emit exactly one warning
        assert len(w) == 1
        assert "Detected repeated headers" in str(w[0].message)

    # Verify result is correct
    result_rows = list(result)
    expected = [
        ["Name", "Age"],  # Original header
        ["Alice", "25"],  # Page 1 data
        ["Bob", "30"],  # Page 2 data (header removed)
        ["Charlie", "35"],  # Page 3 data (header removed)
    ]

    assert result_rows == expected, f"Expected {expected}, got {result_rows}"
    print("✅ Warning only once test passed!")


if __name__ == "__main__":
    test_merge_headers_auto_detection()
    test_merge_headers_explicit_control()
    test_no_repeated_headers()
    test_inconsistent_header_pattern_error()
    test_inconsistent_header_pattern_error_reverse()
    test_warning_only_once()
    print("\n🎉 All multi-page header tests passed!")
