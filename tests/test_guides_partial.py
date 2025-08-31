"""Test partial guides functionality (only vertical or only horizontal guides)."""

import pytest

from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides


def test_guides_extract_table_with_only_verticals(practice_pdf):
    """Test that extract_table works with only vertical guides."""
    page = practice_pdf[0]

    # Get some text elements to use as column markers
    text_elements = page.find_all("text")[:4]

    if len(text_elements) >= 2:
        guides = Guides(page)
        guides.vertical.from_content(text_elements)

        # Should work with only vertical guides
        result = guides.extract_table()

        # Should return a TableResult
        assert hasattr(result, "to_df")  # TableResult has to_df method
        assert isinstance(result, list) is False  # It's a TableResult, not a raw list


def test_guides_extract_table_with_only_horizontals(practice_pdf):
    """Test that extract_table works with only horizontal guides."""
    page = practice_pdf[0]

    guides = Guides(page)
    guides.horizontal.from_lines(n=5)  # Add some horizontal guides

    # Should work with only horizontal guides
    result = guides.extract_table()

    # Should return a TableResult
    assert hasattr(result, "to_df")


def test_page_extract_table_with_verticals(practice_pdf):
    """Test that page.extract_table accepts verticals parameter."""
    page = practice_pdf[0]

    # Create some vertical positions
    verticals = [100, 200, 300, 400]

    # Should work with explicit verticals
    result = page.extract_table(verticals=verticals)

    # Should return a TableResult
    assert hasattr(result, "to_df")  # TableResult has to_df method
    # Can still use it like a list
    if len(result) > 0:
        assert isinstance(result[0], list)


def test_page_extract_table_with_horizontals(practice_pdf):
    """Test that page.extract_table accepts horizontals parameter."""
    page = practice_pdf[0]

    # Create some horizontal positions
    horizontals = [100, 200, 300]

    # Should work with explicit horizontals
    result = page.extract_table(horizontals=horizontals)

    # Should return a TableResult
    assert hasattr(result, "to_df")


def test_page_extract_table_with_both_guides(practice_pdf):
    """Test page.extract_table with both verticals and horizontals."""
    page = practice_pdf[0]

    verticals = [100, 200, 300]
    horizontals = [100, 200, 300]

    result = page.extract_table(verticals=verticals, horizontals=horizontals)

    # Should return a TableResult with expected dimensions
    assert hasattr(result, "to_df")
    if len(result) > 0:
        # Should have 2 rows (3 horizontals = 2 rows)
        assert len(result) <= 3  # Might be less if no content
        # Should have 2 columns (3 verticals = 2 columns)
        assert len(result[0]) <= 3


def test_guides_from_headers_then_extract(atlanta_pdf):
    """Test the full workflow: from_headers -> extract_table."""
    page = atlanta_pdf[0]

    # Find potential headers in the page
    headers = page.find_all("text")[:5]

    if len(headers) >= 3:
        guides = Guides(page)
        guides.vertical.from_headers(headers)

        # Extract table with only vertical guides
        result = guides.extract_table()

        # Should work and return TableResult
        assert hasattr(result, "to_df")

        # Convert to DataFrame to verify it works
        df = result.to_df()
        assert df is not None
