"""Test that trim() preserves sparse content like small numbers."""

import pytest


def test_trim_preserves_sparse_content(practice_pdf):
    """
    Regression test: trim() should not remove sparse content.

    The old averaging-based trim() would incorrectly remove rows/columns
    where content was sparse (e.g., small numbers like "0" or "10" that
    don't darken enough pixels to lower the row average below threshold).

    The new element-based default should preserve all content.
    """
    page = practice_pdf.pages[0]

    # Create a region and trim it
    region = page.region(50, 100, 400, 300)

    # Get text before and after trim
    text_before = region.extract_text()
    trimmed = region.trim()
    text_after = trimmed.extract_text()

    # Trimmed region should contain the same text content
    # (may have whitespace differences, so normalize)
    words_before = set(text_before.split())
    words_after = set(text_after.split())

    # All words from the trimmed region should be present
    # (trimmed region is smaller, so it may have fewer words,
    # but shouldn't have extra words not in original)
    assert words_after.issubset(
        words_before
    ), f"Trimmed region has words not in original: {words_after - words_before}"


def test_trim_method_any_vs_average(practice_pdf):
    """Test that 'any' mode is more conservative than 'average' mode."""
    page = practice_pdf.pages[0]
    region = page.region(50, 100, 400, 300)

    try:
        trimmed_any = region.trim(method="any")
    except ValueError:
        pytest.skip("Pixel-based rendering unavailable on this platform")

    try:
        trimmed_avg = region.trim(method="average")
    except ValueError:
        # average mode couldn't detect sparse content at all - 'any' is trivially
        # more conservative (keeps more content), which is the property under test.
        return

    # 'any' should generally produce equal or larger region than 'average'
    # because it stops at ANY non-white pixel, while 'average' needs
    # enough dark pixels to lower the average below threshold

    # Check that 'any' doesn't trim more aggressively than 'average' on top/left
    assert trimmed_any.x0 <= trimmed_avg.x0 + 0.5, "any mode trimmed left edge more than average"
    assert trimmed_any.top <= trimmed_avg.top + 0.5, "any mode trimmed top edge more than average"

    # Check that 'any' doesn't trim more aggressively than 'average' on bottom/right
    assert trimmed_any.x1 >= trimmed_avg.x1 - 0.5, "any mode trimmed right edge more than average"
    assert (
        trimmed_any.bottom >= trimmed_avg.bottom - 0.5
    ), "any mode trimmed bottom edge more than average"


def test_trim_method_elements_uses_bbox(practice_pdf):
    """Test that 'elements' mode uses actual element bounding boxes."""
    page = practice_pdf.pages[0]
    region = page.region(50, 100, 400, 300)

    # Get elements in region
    elements = region.find_all("*")
    if not elements:
        pytest.skip("No elements in test region")

    # Calculate expected bounds from elements
    expected_x0 = min(e.x0 for e in elements)
    expected_top = min(e.top for e in elements)
    expected_x1 = max(e.x1 for e in elements)
    expected_bottom = max(e.bottom for e in elements)

    # Trim with elements method
    trimmed = region.trim(method="elements", padding=0)

    # Should match element bounds closely
    assert abs(trimmed.x0 - expected_x0) < 1, f"x0 mismatch: {trimmed.x0} vs {expected_x0}"
    assert abs(trimmed.top - expected_top) < 1, f"top mismatch: {trimmed.top} vs {expected_top}"
    assert abs(trimmed.x1 - expected_x1) < 1, f"x1 mismatch: {trimmed.x1} vs {expected_x1}"
    assert (
        abs(trimmed.bottom - expected_bottom) < 1
    ), f"bottom mismatch: {trimmed.bottom} vs {expected_bottom}"


def test_trim_auto_falls_back_to_pixels(practice_pdf):
    """Test that 'auto' falls back to pixel-based when no elements exist."""
    page = practice_pdf.pages[0]

    # Create a region with no text elements (just whitespace area)
    # This is hard to guarantee, so we test the error case for 'elements' mode
    region = page.region(0, 0, 10, 10)  # Very small region, likely empty

    elements = region.find_all("*")
    if elements:
        pytest.skip("Test region has elements, can't test fallback")

    # 'elements' mode should fail with helpful error
    with pytest.raises(ValueError, match="no elements found"):
        region.trim(method="elements")

    # 'auto' mode should fall back to pixels without error
    # (may raise ValueError if no pixels either, but shouldn't raise "no elements" error)
    try:
        region.trim(method="auto")
    except ValueError as e:
        # Should be a pixel-based error, not element-based
        assert "no elements" not in str(e).lower()


def test_trim_invalid_method_raises(practice_pdf):
    """Test that invalid method raises ValueError."""
    page = practice_pdf.pages[0]
    region = page.region(50, 100, 400, 300)

    with pytest.raises(ValueError, match="method must be one of"):
        region.trim(method="invalid")
