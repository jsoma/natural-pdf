#!/usr/bin/env python3
"""Test TableResult.to_df() keep_blank parameter."""

import pandas as pd
import pytest

from natural_pdf.tables.result import TableResult


def test_keep_blank_false_default():
    """Test that empty strings are converted to NaN by default."""
    rows = [
        ["Name", "Age", "Notes"],
        ["Alice", "25", "Manager"],
        ["Bob", "", ""],  # Empty age and notes
        ["", "30", "Developer"],  # Empty name
    ]

    table_result = TableResult(rows)
    df = table_result.to_df()

    # Should convert empty strings to NaN by default
    assert (df == "").sum().sum() == 0  # No empty strings
    assert df.isna().sum().sum() == 3  # 3 NaN values
    assert pd.isna(df.iloc[1, 1])  # Bob's age is NaN
    assert pd.isna(df.iloc[2, 0])  # Third person's name is NaN


def test_keep_blank_true():
    """Test that empty strings are preserved when keep_blank=True."""
    rows = [
        ["Name", "Age", "Notes"],
        ["Alice", "25", "Manager"],
        ["Bob", "", ""],  # Empty age and notes
        ["", "30", "Developer"],  # Empty name
    ]

    table_result = TableResult(rows)
    df = table_result.to_df(keep_blank=True)

    # Empty strings should be preserved
    assert (df == "").sum().sum() == 3  # 3 empty strings
    assert df.isna().sum().sum() == 0  # No NaN values
    assert df.iloc[1, 1] == ""  # Bob's age is empty string
    assert df.iloc[2, 0] == ""  # Third person's name is empty string


def test_keep_blank_with_whitespace():
    """Test that only empty strings (not whitespace) are converted."""
    rows = [
        ["Item", "Value"],
        ["A", ""],  # Empty string
        ["B", "  "],  # Spaces
        ["C", "\t"],  # Tab
        ["D", "text"],  # Normal text
    ]

    table_result = TableResult(rows)

    # Default behavior (keep_blank=False)
    df_default = table_result.to_df()
    assert pd.isna(df_default.iloc[0, 1])  # Empty string → NaN
    assert df_default.iloc[1, 1] == "  "  # Spaces preserved
    assert df_default.iloc[2, 1] == "\t"  # Tab preserved
    assert df_default.iloc[3, 1] == "text"  # Normal text preserved
    assert df_default.isna().sum().sum() == 1  # Only 1 NaN value

    # With keep_blank=True
    df_keep = table_result.to_df(keep_blank=True)
    assert df_keep.iloc[0, 1] == ""  # Empty string preserved
    assert df_keep.iloc[1, 1] == "  "  # Spaces preserved
    assert df_keep.iloc[2, 1] == "\t"  # Tab preserved
    assert df_keep.iloc[3, 1] == "text"  # Normal text preserved
    assert df_keep.isna().sum().sum() == 0  # No NaN values


def test_keep_blank_with_no_header():
    """Test keep_blank works with header=None."""
    rows = [["", "B", ""], ["D", "", "F"]]

    table_result = TableResult(rows)

    # Default: convert to NaN
    df_default = table_result.to_df(header=None)
    assert df_default.isna().sum().sum() == 3  # 3 empty strings converted
    assert pd.isna(df_default.iloc[0, 0])  # First row, first col
    assert pd.isna(df_default.iloc[1, 1])  # Second row, second col

    # Keep blank: preserve empty strings
    df_keep = table_result.to_df(header=None, keep_blank=True)
    assert df_keep.isna().sum().sum() == 0  # No NaN values
    assert (df_keep == "").sum().sum() == 3  # 3 empty strings preserved


def test_keep_blank_numerical_analysis():
    """Test that default behavior enables numerical analysis."""
    rows = [
        ["Product", "Price", "Quantity"],
        ["Apple", "1.50", "10"],
        ["Banana", "", "20"],  # Missing price
        ["Orange", "2.00", ""],  # Missing quantity
    ]

    table_result = TableResult(rows)

    # Default behavior (keep_blank=False) - better for analysis
    df_default = table_result.to_df()
    price_mean_default = pd.to_numeric(df_default["Price"], errors="coerce").mean()

    # With keep_blank=True - preserves empty strings
    df_keep = table_result.to_df(keep_blank=True)
    price_mean_keep = pd.to_numeric(df_keep["Price"], errors="coerce").mean()

    # Both should give same result since pd.to_numeric handles both cases
    assert abs(price_mean_default - price_mean_keep) < 0.001

    # But pandas operations work more naturally with NaN (default)
    complete_rows = df_default.dropna()
    assert len(complete_rows) == 1  # Only Apple row is complete

    # With keep_blank=True, dropna() doesn't help with empty strings
    complete_rows_keep = df_keep.dropna()
    assert len(complete_rows_keep) == 3  # All rows since no NaN values


def test_keep_blank_backward_compatibility():
    """Test that the new default doesn't break functionality."""
    rows = [["A", "B", "C"], ["1", "2", "3"], ["4", "5", "6"]]

    table_result = TableResult(rows)

    # No empty strings, so behavior should be identical
    df_implicit_default = table_result.to_df()
    df_explicit_false = table_result.to_df(keep_blank=False)
    df_explicit_true = table_result.to_df(keep_blank=True)
    df_property = table_result.df  # Uses default

    # All should be identical when no empty strings present
    assert df_implicit_default.equals(df_explicit_false)
    assert df_implicit_default.equals(df_explicit_true)
    assert df_implicit_default.equals(df_property)

    # No NaN values in any of them
    assert df_implicit_default.isna().sum().sum() == 0
    assert df_explicit_false.isna().sum().sum() == 0
    assert df_explicit_true.isna().sum().sum() == 0
    assert df_property.isna().sum().sum() == 0


def test_keep_blank_with_all_empty():
    """Test behavior when all values are empty strings."""
    rows = [
        ["", "", ""],
        ["", "", ""],
    ]

    table_result = TableResult(rows)

    # Default behavior - convert to NaN
    df_default = table_result.to_df(header=None)
    assert df_default.isna().sum().sum() == 6
    assert (df_default == "").sum().sum() == 0

    # With keep_blank=True - preserve empty strings
    df_keep = table_result.to_df(header=None, keep_blank=True)
    assert (df_keep == "").sum().sum() == 6
    assert df_keep.isna().sum().sum() == 0


def test_keep_blank_mixed_with_header_mismatch():
    """Test keep_blank with header mismatch fallback."""
    rows = [
        ["Single merged header"],  # 1 column
        ["A", "B", "C"],  # 3 columns - triggers fallback
        ["", "E", ""],  # Empty values
    ]

    table_result = TableResult(rows)

    # Default: convert to NaN
    df_default = table_result.to_df()
    assert list(df_default.columns) == [0, 1, 2]  # Default column names
    assert df_default.shape == (3, 3)  # All rows preserved
    # 2 empty strings from last row + 2 None values from first row's missing columns
    expected_na_count = 4
    assert df_default.isna().sum().sum() == expected_na_count

    # With keep_blank=True
    df_keep = table_result.to_df(keep_blank=True)
    assert list(df_keep.columns) == [0, 1, 2]  # Default column names
    assert df_keep.shape == (3, 3)  # All rows preserved
    # Only 2 None values from first row's missing columns (empty strings preserved)
    assert df_keep.isna().sum().sum() == 2
    assert (df_keep == "").sum().sum() == 2  # 2 empty strings preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
