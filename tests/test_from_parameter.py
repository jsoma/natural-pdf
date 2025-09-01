"""Tests for the from= parameter in directional navigation methods."""

import pytest

import natural_pdf as npdf
from natural_pdf.elements.region import Region


def test_below_from_parameter():
    """Test the from= parameter for below() method with overlapping text."""
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Find a text element
    text_elem = page.find("text:contains('the')")
    assert text_elem is not None

    # Test default behavior (from='start')
    region_default = text_elem.below(height=50, until="text")

    # Test from='end' (old strict behavior)
    region_end = text_elem.below(height=50, until="text", anchor="end")

    # Test from='center'
    region_center = text_elem.below(height=50, until="text", anchor="center")

    # Test explicit edge names
    region_top = text_elem.below(height=50, until="text", anchor="top")
    region_bottom = text_elem.below(height=50, until="text", anchor="bottom")

    # Verify that from='start' and from='top' are equivalent for below()
    assert region_default.bbox == region_top.bbox

    # Verify that from='end' and from='bottom' are equivalent for below()
    assert region_end.bbox == region_bottom.bbox

    # Verify that different from values may capture different text
    # (depending on whether there's overlapping text)
    # This is hard to assert without knowing the specific content,
    # but we can at least verify the regions are created
    assert isinstance(region_default, Region)
    assert isinstance(region_end, Region)
    assert isinstance(region_center, Region)


def test_above_from_parameter():
    """Test the from= parameter for above() method."""
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Find a text element in the middle of the page
    text_elem = page.find("text:contains('the')")
    assert text_elem is not None

    # Test different from values
    region_start = text_elem.above(height=50, until="text", anchor="start")
    region_end = text_elem.above(height=50, until="text", anchor="end")
    region_center = text_elem.above(height=50, until="text", anchor="center")

    # Test explicit edge names
    region_bottom = text_elem.above(height=50, until="text", anchor="bottom")
    region_top = text_elem.above(height=50, until="text", anchor="top")

    # Verify that from='start' and from='bottom' are equivalent for above()
    assert region_start.bbox == region_bottom.bbox

    # Verify that from='end' and from='top' are equivalent for above()
    assert region_end.bbox == region_top.bbox


def test_left_right_from_parameter():
    """Test the from= parameter for left() and right() methods."""
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Find a text element
    text_elem = page.find("text:contains('the')")
    assert text_elem is not None

    # Test right() with different from values
    region_right_start = text_elem.right(width=50, until="text", anchor="start")
    region_right_end = text_elem.right(width=50, until="text", anchor="end")
    region_right_left = text_elem.right(width=50, until="text", anchor="left")
    region_right_right = text_elem.right(width=50, until="text", anchor="right")

    # Verify that from='start' and from='left' are equivalent for right()
    assert region_right_start.bbox == region_right_left.bbox

    # Verify that from='end' and from='right' are equivalent for right()
    assert region_right_end.bbox == region_right_right.bbox

    # Test left() with different from values
    region_left_start = text_elem.left(width=50, until="text", anchor="start")
    region_left_end = text_elem.left(width=50, until="text", anchor="end")
    region_left_right = text_elem.left(width=50, until="text", anchor="right")
    region_left_left = text_elem.left(width=50, until="text", anchor="left")

    # Verify that from='start' and from='right' are equivalent for left()
    assert region_left_start.bbox == region_left_right.bbox

    # Verify that from='end' and from='left' are equivalent for left()
    assert region_left_end.bbox == region_left_left.bbox


def test_from_center():
    """Test that from='center' works for all directions."""
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    text_elem = page.find("text:contains('the')")
    assert text_elem is not None

    # Test center for all directions
    region_below_center = text_elem.below(height=50, until="text", anchor="center")
    region_above_center = text_elem.above(height=50, until="text", anchor="center")
    region_left_center = text_elem.left(width=50, until="text", anchor="center")
    region_right_center = text_elem.right(width=50, until="text", anchor="center")

    # All should create valid regions
    assert isinstance(region_below_center, Region)
    assert isinstance(region_above_center, Region)
    assert isinstance(region_left_center, Region)
    assert isinstance(region_right_center, Region)


def test_overlapping_text_capture():
    """Test that from='start' can capture overlapping text while from='end' cannot."""
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # This test would be more meaningful with a PDF that has known overlapping elements
    # For now, we just verify the functionality works
    text_elem = page.find("text:contains('the')")
    assert text_elem is not None

    # Get regions with different from values
    region_from_start = text_elem.below(until="text", anchor="start")
    region_from_end = text_elem.below(until="text", anchor="end")

    # Both should be valid regions
    assert region_from_start is not None
    assert region_from_end is not None

    # The regions might be different if there's overlapping text
    # We can't assert they're different without knowing the PDF content,
    # but we can verify they're both valid
    if region_from_start.bbox != region_from_end.bbox:
        # If they're different, from_start should extend higher (smaller top value)
        # because it starts looking from the top of the element
        assert region_from_start.top <= region_from_end.top


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
