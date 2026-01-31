"""Test stripe boundary detection for outer table extraction."""

import pytest


@pytest.fixture
def oklahoma_pdf():
    """Load the Oklahoma license PDF."""
    from natural_pdf import PDF

    return PDF("pdfs/m27.pdf")


def test_outer_detects_stripe_boundaries_page2(oklahoma_pdf):
    """Test that outer=True detects stripe boundaries on page 2.

    Page 2 of m27.pdf has alternating row stripes that don't cover the
    first and last data rows. With outer=True, these should be captured
    by auto-detecting the stripe pattern and adding outer boundaries.
    """
    page = oklahoma_pdf[1]  # Page 2

    headers = page.region(top=55, height=30).find_all("text").expand(1).dissolve().expand(-1)
    region = page.find("text:contains('LICENSEE NAME')").below(
        until="text:regex(Page \\d)", include_endpoint=False
    )

    # Extract with outer=True - should auto-detect stripes
    result = region.extract_table(verticals=headers, outer=True)

    # Should capture all 23 license entries including first and last
    assert len(result) == 23

    # Check first and last entries are captured
    licenses = [row[0] for row in result if row[0]]
    assert "632426" in licenses  # First entry (was missing without stripe detection)
    assert "604941" in licenses  # Last entry (was missing without stripe detection)


def test_outer_still_works_page1(oklahoma_pdf):
    """Test that outer=True still works on page 1 (which already worked)."""
    page = oklahoma_pdf[0]  # Page 1

    headers = page.region(top=55, height=30).find_all("text").expand(1).dissolve().expand(-1)
    region = page.find("text:contains('LICENSEE NAME')").below(
        until="text:regex(Page \\d)", include_endpoint=False
    )

    result = region.extract_table(verticals=headers, outer=True)

    # Page 1 should still have 25 rows
    assert len(result) == 25

    # First and last entries
    assert result[0][0] == "648765"
    assert result[-1][0] == "543149"


def test_outer_works_without_stripes(practice_pdf_fresh):
    """Test that outer=True works on PDFs without stripe rectangles."""
    page = practice_pdf_fresh[0]

    # Create a region without stripes
    region = page.region(top=100, bottom=400)

    # Should not error even without stripes
    result = region.extract_table(verticals=[100, 200, 300, 400], outer=True)

    # Should return some rows (content boundaries used as horizontals)
    assert hasattr(result, "to_df")  # It's a TableResult


@pytest.fixture
def practice_pdf_fresh():
    """Load the practice PDF with fresh state."""
    from natural_pdf import PDF

    return PDF("pdfs/01-practice.pdf")
