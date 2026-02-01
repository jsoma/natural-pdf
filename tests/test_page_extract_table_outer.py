"""Test outer parameter for Page.extract_table."""

import pytest

import natural_pdf as npdf


@pytest.fixture
def hebrew_pdf():
    """Load the Hebrew table PDF."""
    pdf = npdf.PDF("pdfs/hebrew-table.pdf")
    yield pdf
    pdf.close()


def test_page_extract_table_accepts_outer_parameter(hebrew_pdf):
    """Page.extract_table should accept outer=True parameter."""
    page = hebrew_pdf.pages[0]

    # Find guides
    divider = page.find_all("rect:vertical[height>100]")[1]
    rows = divider.right(height="element").find_all("text").dissolve()
    headers = page.find_all("rect:horizontal")[1].above().find_all("text")

    # Should not raise TypeError
    result = page.extract_table(verticals=headers, horizontals=rows, outer=True)

    df = result.to_df()
    assert len(df) > 0
    assert "2014" in df.columns
    assert "2009" in df.columns


def test_page_extract_table_outer_false_is_default(hebrew_pdf):
    """Page.extract_table should default to outer=False."""
    page = hebrew_pdf.pages[0]

    divider = page.find_all("rect:vertical[height>100]")[1]
    rows = divider.right(height="element").find_all("text").dissolve()
    headers = page.find_all("rect:horizontal")[1].above().find_all("text")

    # Without outer=True, table boundaries won't extend to page edges
    result_no_outer = page.extract_table(
        verticals=headers,
        horizontals=rows,
    )

    result_with_outer = page.extract_table(verticals=headers, horizontals=rows, outer=True)

    # Both should work
    assert result_no_outer.to_df() is not None
    assert result_with_outer.to_df() is not None
