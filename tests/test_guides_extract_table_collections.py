#!/usr/bin/env python3
"""Test extract_table functionality with collections."""

from pathlib import Path

import pandas as pd
import pytest

from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides


def find_test_pdf():
    """Find the test PDF file."""
    pdf_path = Path("pdfs/24480polcompleted.pdf")
    return pdf_path if pdf_path.exists() else None


@pytest.mark.skipif(find_test_pdf() is None, reason="Test PDF not found")
def test_extract_table_from_collection_basic():
    """Test basic extract_table functionality with page collections."""
    pdf = PDF("pdfs/24480polcompleted.pdf")
    pages = pdf.pages[4:7]  # Just test first 3 pages for speed
    pdf.add_exclusion(lambda page: page.find_all("text:regex(\\d+ Records Found)"))

    columns = [
        "Number",
        "Date Occurred",
        "Time Occurred",
        "Location",
        "Call Type",
        "Description",
        "Disposition",
        "Main Officer",
    ]

    # Create guide with static vertical, dynamic horizontal
    guide = Guides(pages[0])
    guide.vertical.from_content(columns, outer="last")
    guide.horizontal.from_content(lambda p: p.find_all("text:starts-with(NF-)"))

    # Extract table
    table_result = guide.extract_table(pages, header=columns)

    # Verify it returns TableResult
    from natural_pdf.tables.result import TableResult

    assert isinstance(table_result, TableResult)

    # Convert to DataFrame
    df = table_result.to_df()

    # Check basic properties
    assert len(df.columns) == 8
    assert list(df.columns) == columns
    assert len(df) > 0  # Should have some rows

    # Check that all rows have the expected number of columns
    for _, row in df.iterrows():
        assert len(row) == 8


@pytest.mark.skipif(find_test_pdf() is None, reason="Test PDF not found")
def test_extract_table_collection_header_options():
    """Test different header options with collections."""
    pdf = PDF("pdfs/24480polcompleted.pdf")
    pages = pdf.pages[4:6]  # Just 2 pages

    guide = Guides(pages[0])
    guide.vertical.divide(8)
    guide.horizontal.from_content(lambda p: p.find_all("text:starts-with(NF-)"))

    # Test header="first" (default)
    result1 = guide.extract_table(pages, header="first")
    df1 = result1.to_df()
    assert isinstance(df1.columns[0], str)  # Should use first row as headers

    # Test header=None
    result2 = guide.extract_table(pages, header=None)
    df2 = result2.to_df(header=None)  # Need to pass header=None to to_df as well
    assert isinstance(df2.columns[0], int)  # Should use numeric indices

    # Test custom headers
    custom_headers = ["A", "B", "C", "D", "E", "F", "G", "H"]
    result3 = guide.extract_table(pages, header=custom_headers)
    df3 = result3.to_df(header=custom_headers)  # Pass custom headers to to_df
    assert list(df3.columns) == custom_headers


@pytest.mark.skipif(find_test_pdf() is None, reason="Test PDF not found")
def test_extract_table_collection_matches_manual():
    """Test that extract_table with collections matches manual approach."""
    pdf = PDF("pdfs/24480polcompleted.pdf")
    pages = pdf.pages[4:8]  # Test 4 pages
    pdf.add_exclusion(lambda page: page.find_all("text:regex(\\d+ Records Found)"))

    columns = [
        "Number",
        "Date Occurred",
        "Time Occurred",
        "Location",
        "Call Type",
        "Description",
        "Disposition",
        "Main Officer",
    ]

    # Manual approach
    base = Guides(pages[0])
    base.vertical.from_content(columns, outer="last")
    base.horizontal.from_content(pages[0].find_all("text:starts-with(NF-)"))

    manual_dfs = [base.extract_table().to_df()]

    for page in pages[1:]:
        guides = Guides(page)
        guides.vertical = base.vertical
        guides.horizontal.from_content(page.find_all("text:starts-with(NF-)"))
        manual_dfs.append(guides.extract_table().to_df(header=columns))

    df_manual = pd.concat(manual_dfs, ignore_index=True)

    # New approach
    guide = Guides(pages[0])
    guide.vertical.from_content(columns, outer="last")
    guide.horizontal.from_content(lambda p: p.find_all("text:starts-with(NF-)"))

    table_result = guide.extract_table(pages, header=columns)
    df_new = table_result.to_df()

    # Compare
    assert df_manual.shape == df_new.shape
    assert list(df_manual.columns) == list(df_new.columns)

    # Compare content (allowing for minor differences)
    for i in range(len(df_manual)):
        for j in range(len(df_manual.columns)):
            val1 = df_manual.iloc[i, j]
            val2 = df_new.iloc[i, j]
            if pd.isna(val1):
                assert pd.isna(val2)
            else:
                assert val1 == val2


def test_extract_table_empty_collection():
    """Test extract_table with empty page collection."""
    # This test doesn't need a real PDF, but we need a dummy context
    from natural_pdf.tables.result import TableResult

    # Create guide with bounds instead of page/region context
    guide = Guides(bounds=(0, 0, 100, 100))
    guide.vertical.add([25, 50, 75])
    guide.horizontal.add([25, 50, 75])

    result = guide.extract_table([])
    assert isinstance(result, TableResult)
    assert len(result) == 0

    df = result.to_df()
    assert len(df) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
