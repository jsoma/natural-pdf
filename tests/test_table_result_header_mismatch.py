#!/usr/bin/env python3
"""Test TableResult.to_df() with header/body column count mismatch."""

import pytest

from natural_pdf.tables.result import TableResult


def test_header_body_column_mismatch():
    """Test that header/body column count mismatch falls back to no header."""
    # Case where header row has 1 column but body rows have 6 columns
    rows = [
        [
            "Recipient: Amount Expended: Approved Use Category: Expenditure Terms: Disbursement Purpose Outcomes"
        ],
        [
            "Recipient:",
            "Amount Expended:",
            "Approved Use Category:",
            "Expenditure Terms:",
            "Disbursement Purpose",
            "Outcomes",
        ],
        ["John Doe", "$1000", "Training", "Grant", "Job training program", "Improved skills"],
    ]

    table_result = TableResult(rows)

    # This should not raise an error and should fallback to no header
    df = table_result.to_df()

    # Should have all 3 rows (header mismatch detected, so all rows treated as body)
    assert df.shape == (3, 6)

    # Column names should be default integer indices since no header was used
    assert list(df.columns) == [0, 1, 2, 3, 4, 5]

    # First row should contain the original "header" that had mismatched columns
    assert (
        df.iloc[0, 0]
        == "Recipient: Amount Expended: Approved Use Category: Expenditure Terms: Disbursement Purpose Outcomes"
    )

    # Second row should be the actual header row
    assert df.iloc[1, 0] == "Recipient:"
    assert df.iloc[1, 5] == "Outcomes"


def test_normal_header_preserved():
    """Test that normal tables with matching header/body columns work correctly."""
    rows = [["Name", "Age", "City"], ["Alice", "25", "New York"], ["Bob", "30", "San Francisco"]]

    table_result = TableResult(rows)
    df = table_result.to_df()

    # Should have 2 body rows, 3 columns with proper headers
    assert df.shape == (2, 3)
    assert list(df.columns) == ["Name", "Age", "City"]

    # First data row should be Alice
    assert df.iloc[0, 0] == "Alice"
    assert df.iloc[0, 2] == "New York"


def test_multi_row_header_mismatch():
    """Test mismatch detection with multi-row headers."""
    rows = [
        ["Name Age City"],  # Merged header - 1 column
        ["First Last Years Location"],  # Another merged header - 1 column
        ["John", "Doe", "25", "NYC"],  # Data row - 4 columns
        ["Jane", "Smith", "30", "LA"],  # Data row - 4 columns
    ]

    table_result = TableResult(rows)

    # Try with multi-row header
    df = table_result.to_df(header=[0, 1])

    # Should fallback to no header due to mismatch
    assert df.shape == (4, 4)  # All rows included
    assert list(df.columns) == [0, 1, 2, 3]  # Default column names


def test_explicit_header_none_unaffected():
    """Test that explicit header=None still works the same."""
    rows = [["Merged header row"], ["Col1", "Col2", "Col3"], ["Data1", "Data2", "Data3"]]

    table_result = TableResult(rows)

    # Explicit header=None should work the same as before
    df = table_result.to_df(header=None)

    assert df.shape == (3, 3)
    assert list(df.columns) == [0, 1, 2]

    # All rows should be preserved
    assert df.iloc[0, 0] == "Merged header row"
    assert df.iloc[1, 0] == "Col1"


def test_empty_body_rows():
    """Test edge case with only header row."""
    rows = [["Header1", "Header2", "Header3"]]

    table_result = TableResult(rows)
    df = table_result.to_df()

    # Should have empty body but proper headers
    assert df.shape == (0, 3)
    assert list(df.columns) == ["Header1", "Header2", "Header3"]


def test_inconsistent_body_column_counts():
    """Test when body rows have different column counts."""
    rows = [
        ["A", "B", "C"],  # Header: 3 columns
        ["1", "2", "3"],  # Body row: 3 columns
        ["4", "5"],  # Body row: 2 columns
        ["6", "7", "8", "9"],  # Body row: 4 columns (max)
    ]

    table_result = TableResult(rows)

    # Should fallback to no header since max body columns (4) != header columns (3)
    df = table_result.to_df()

    # Should fallback to no header due to mismatch with max columns
    assert list(df.columns) == [0, 1, 2, 3]  # Default column names
    assert df.shape[0] == 4  # All 4 rows included as body

    # First row should be the original header
    assert df.iloc[0, 0] == "A"
    assert df.iloc[0, 1] == "B"
    assert df.iloc[0, 2] == "C"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
