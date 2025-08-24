"""Test handling of PDFs with negative bounds in pdfplumber."""

import pytest

from natural_pdf import PDF


def test_extract_table_with_negative_bounds():
    """Test that extract_table works with PDFs that have negative bounds in pdfplumber.

    Some PDFs have page bounds that start with negative coordinates in pdfplumber,
    which can cause issues when trying to crop regions that assume (0,0) origin.
    """
    # This specific PDF has pdfplumber bounds of approximately:
    # (-14.4, -14.4, 827.28, 580.8) instead of (0, 0, 841.68, 595.2)
    pdf = PDF("https://www.pak-ks.org/desk/inc/media/EB62887E-EDF3-4CE4-B4D5-DEC69D53A2EF.pdf")
    page = pdf.pages[0]

    # This should not raise a ValueError about bounding box not being within parent
    table = page.extract_table()
    assert table is not None

    # Convert to dataframe to verify we got actual data
    df = table.to_df(header=None)
    assert len(df) > 0
    assert len(df.columns) > 0

    # Also test with extract_tables (multiple tables)
    tables = page.extract_tables()
    assert len(tables) >= 0  # Should not raise an error

    # Test with a custom region that might extend beyond pdfplumber bounds
    region = page.create_region(0, 0, page.width, page.height)
    region_table = region.extract_table()
    # Should either extract a table or return None, but not raise an error
    assert region_table is None or len(region_table) >= 0


def test_region_crop_with_out_of_bounds():
    """Test that regions handle out-of-bounds cropping gracefully."""
    pdf = PDF("https://www.pak-ks.org/desk/inc/media/EB62887E-EDF3-4CE4-B4D5-DEC69D53A2EF.pdf")
    page = pdf.pages[0]

    # Create a region that definitely extends beyond any reasonable page bounds
    huge_region = page.create_region(-100, -100, 2000, 2000)

    # Should handle gracefully without errors
    table = huge_region.extract_table()
    assert table is None or len(table) >= 0

    # Test with method='lattice' as well (uses different code path)
    table_lattice = huge_region.extract_table(method="lattice")
    assert table_lattice is None or len(table_lattice) >= 0
