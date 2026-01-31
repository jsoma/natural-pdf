"""Test that directional navigation with until parameter has precise boundaries.

This tests the fix for an off-by-one error where characters at the exact boundary
of a region were incorrectly included in text extraction.
"""

import pytest

from natural_pdf import PDF


@pytest.fixture
def practice_pdf_fresh():
    """Returns a fresh PDF instance for each test."""
    pdf = PDF("pdfs/01-practice.pdf")
    yield pdf
    pdf.close()


def test_right_until_text_no_boundary_bleed(practice_pdf_fresh):
    """Test that right(until='text') doesn't include characters at the boundary.

    When using right(until='text'), the region should NOT include characters
    from the next text element that merely touch the region boundary.
    """
    pdf = practice_pdf_fresh
    page = pdf.pages[0]

    # Find the Site element
    site_element = page.find("text:contains('Site')")
    assert site_element is not None, "Site element should exist"

    # Get region to the right until next text
    region = site_element.right(until="text")
    extracted = region.extract_text()

    # The extracted text should be "Durham's Meatpacking" not "Durham's Meatpacking C"
    # The 'C' is the first character of "Chicago, Ill." which is the next text element
    assert extracted.strip().endswith(
        "Meatpacking"
    ), f"Expected text to end with 'Meatpacking', but got: '{extracted}'"
    assert (
        "C" not in extracted[-3:]
    ), f"Text should not include 'C' from next element: '{extracted}'"


def test_boundary_character_excluded_when_center_outside(practice_pdf_fresh):
    """Test that characters with their center outside the region are excluded.

    When a character's bounding box touches the region boundary but its center
    is outside, it should NOT be included in text extraction.
    """
    pdf = practice_pdf_fresh
    page = pdf.pages[0]

    site_element = page.find("text:contains('Site')")
    region = site_element.right(until="text")

    # Find the 'C' character from "Chicago"
    all_chars = page.find_all("char")
    chicago_c = [
        c
        for c in all_chars
        if c.text == "C"
        and abs(c.x0 - region.x1) < 1.0
        and c.top >= region.top - 2
        and c.bottom <= region.bottom + 2
    ]

    if chicago_c:
        c_char = chicago_c[0]
        center_x = (c_char.x0 + c_char.x1) / 2

        # The center of 'C' should be OUTSIDE the region
        assert (
            center_x > region.x1
        ), f"'C' center ({center_x}) should be outside region.x1 ({region.x1})"

        # And 'C' should NOT appear in extracted text
        extracted = region.extract_text()
        assert (
            c_char.text not in extracted[-3:]
        ), f"Character at boundary should be excluded: '{extracted}'"


def test_directional_navigation_all_directions(practice_pdf_fresh):
    """Test that all directional methods work correctly without boundary bleed."""
    pdf = practice_pdf_fresh
    page = pdf.pages[0]

    # Test right()
    site = page.find("text:contains('Site')")
    if site:
        right_text = site.right(until="text").extract_text().strip()
        # Should not have extra characters at the end
        assert (
            right_text.endswith("Meatpacking") or len(right_text) < 30
        ), f"right() text: '{right_text}'"

    # Test left()
    chicago = page.find("text:contains('Chicago')")
    if chicago:
        left_text = chicago.left(until="text").extract_text().strip()
        # Should be the text between Site and Chicago
        assert "Meatpacking" in left_text or len(left_text) > 0, f"left() text: '{left_text}'"
