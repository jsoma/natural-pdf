"""Tests for the from= parameter in directional navigation methods."""

import pytest

from natural_pdf.elements.region import Region


def test_below_from_parameter(practice_pdf):
    """Test the from= parameter for below() method with overlapping text."""
    page = practice_pdf.pages[0]

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

    # Verify that from='start' and from='bottom' are equivalent for below()
    # (start = boundary where below region begins = source's bottom edge)
    assert region_default.bbox == region_bottom.bbox

    # Verify that from='end' and from='top' are equivalent for below()
    # (end = opposite edge, allows finding overlapping elements)
    assert region_end.bbox == region_top.bbox

    # Verify that different from values may capture different text
    # (depending on whether there's overlapping text)
    # This is hard to assert without knowing the specific content,
    # but we can at least verify the regions are created
    assert isinstance(region_default, Region)
    assert isinstance(region_end, Region)
    assert isinstance(region_center, Region)


def test_above_from_parameter(practice_pdf):
    """Test the from= parameter for above() method."""
    page = practice_pdf.pages[0]

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

    # Verify that from='start' and from='top' are equivalent for above()
    # (start = boundary where above region begins = source's top edge)
    assert region_start.bbox == region_top.bbox

    # Verify that from='end' and from='bottom' are equivalent for above()
    # (end = opposite edge, allows finding overlapping elements)
    assert region_end.bbox == region_bottom.bbox


def test_left_right_from_parameter(practice_pdf):
    """Test the from= parameter for left() and right() methods."""
    page = practice_pdf.pages[0]

    # Find a text element
    text_elem = page.find("text:contains('the')")
    assert text_elem is not None

    # Test right() with different from values
    region_right_start = text_elem.right(width=50, until="text", anchor="start")
    region_right_end = text_elem.right(width=50, until="text", anchor="end")
    region_right_left = text_elem.right(width=50, until="text", anchor="left")
    region_right_right = text_elem.right(width=50, until="text", anchor="right")

    # Verify that from='start' and from='right' are equivalent for right()
    # (start = boundary where right region begins = source's right edge)
    assert region_right_start.bbox == region_right_right.bbox

    # Verify that from='end' and from='left' are equivalent for right()
    # (end = opposite edge, allows finding overlapping elements)
    assert region_right_end.bbox == region_right_left.bbox

    # Test left() with different from values
    region_left_start = text_elem.left(width=50, until="text", anchor="start")
    region_left_end = text_elem.left(width=50, until="text", anchor="end")
    region_left_right = text_elem.left(width=50, until="text", anchor="right")
    region_left_left = text_elem.left(width=50, until="text", anchor="left")

    # Verify that from='start' and from='left' are equivalent for left()
    # (start = boundary where left region begins = source's left edge)
    assert region_left_start.bbox == region_left_left.bbox

    # Verify that from='end' and from='right' are equivalent for left()
    # (end = opposite edge, allows finding overlapping elements)
    assert region_left_end.bbox == region_left_right.bbox


def test_from_center(practice_pdf):
    """Test that from='center' works for all directions."""
    page = practice_pdf.pages[0]

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


def test_overlapping_text_capture(practice_pdf):
    """Test that anchor='end' can capture overlapping text while anchor='start' cannot."""
    page = practice_pdf.pages[0]

    # This test would be more meaningful with a PDF that has known overlapping elements
    # For now, we just verify the functionality works
    text_elem = page.find("text:contains('the')")
    assert text_elem is not None

    # Get regions with different anchor values
    region_from_start = text_elem.below(until="text", anchor="start")
    region_from_end = text_elem.below(until="text", anchor="end")

    # Both should be valid regions
    assert region_from_start is not None
    assert region_from_end is not None

    # The regions might be different if there's overlapping text
    # We can't assert they're different without knowing the PDF content,
    # but we can verify they're both valid
    if region_from_start.bbox != region_from_end.bbox:
        # If they're different, from_end can capture overlapping elements
        # (uses source's top edge as reference), while from_start only finds
        # elements strictly below (uses source's bottom edge as reference).
        # So from_end.top may be <= from_start.top (within floating point tolerance)
        assert region_from_end.top <= region_from_start.top + 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
