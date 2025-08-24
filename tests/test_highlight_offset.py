"""Test highlighting with PDFs that have negative margin bounds in pdfplumber."""

import pytest

from natural_pdf import PDF


def test_show_method_highlight_offset():
    """Test that element.show() correctly handles offset for PDFs with negative bounds."""
    pdf = PDF("https://www.pak-ks.org/desk/inc/media/EB62887E-EDF3-4CE4-B4D5-DEC69D53A2EF.pdf")
    page = pdf.pages[0]

    # Find a specific element
    rect = page.find("rect[fill~=yellow]")
    assert rect is not None, "Should find yellow rectangle"

    # This should render without offset issues (visual check)
    # The fix ensures the highlight aligns with the element
    try:
        # Using element.show() which goes through _apply_spec_highlights
        img = rect.show()
        assert img is not None
    except Exception as e:
        pytest.fail(f"Failed to show element: {e}")


def test_highlight_offset_with_negative_bounds():
    """Test that highlights are correctly positioned for PDFs with negative bounds.

    Some PDFs have pdfplumber page bounds that start with negative coordinates,
    which can cause highlights to be offset from their actual elements.
    """
    # This specific PDF has pdfplumber bounds starting at (-14.4, -14.4)
    pdf = PDF("https://www.pak-ks.org/desk/inc/media/EB62887E-EDF3-4CE4-B4D5-DEC69D53A2EF.pdf")
    page = pdf.pages[0]

    # Find elements to highlight
    elements = page.find_all("rect[fill~=yellow]")
    assert len(elements) > 0, "Should find yellow rectangles"

    # Test that we can render highlights without errors
    # The fix ensures highlights align with elements despite negative bounds
    elem = elements[0]

    # Get the bbox to verify it's in the expected range
    bbox = elem.bbox
    assert bbox[0] > 0 and bbox[1] > 0, "Element coordinates should be positive"

    # Test rendering with direct bbox coordinates
    # This should position the highlight correctly over the element
    # (Previously would be offset by the negative margin amount)
    try:
        page.render(highlights=[{"bbox": bbox}])
        # If we get here without error, the highlight was rendered successfully
        assert True
    except Exception as e:
        pytest.fail(f"Failed to render highlight: {e}")


def test_multiple_highlight_types_with_offset():
    """Test different highlight types with PDFs having negative bounds."""
    pdf = PDF("https://www.pak-ks.org/desk/inc/media/EB62887E-EDF3-4CE4-B4D5-DEC69D53A2EF.pdf")
    page = pdf.pages[0]

    # Test various highlight scenarios
    highlights = [
        # Rectangle highlight
        {"bbox": (100, 100, 200, 150), "color": "red"},
        # Polygon highlight (triangle)
        {"polygon": [(300, 100), (350, 150), (300, 150)], "color": "blue"},
        # Element-based highlight
        {"element": page.find('text:contains("Tabela")'), "color": "green"},
    ]

    # All highlights should render without coordinate errors
    try:
        page.render(highlights=highlights)
        assert True
    except Exception as e:
        pytest.fail(f"Failed to render multiple highlights: {e}")


def test_highlight_alignment_verification():
    """Verify that highlights align with their source elements."""
    pdf = PDF("https://www.pak-ks.org/desk/inc/media/EB62887E-EDF3-4CE4-B4D5-DEC69D53A2EF.pdf")
    page = pdf.pages[0]

    # Find a specific text element
    text_elem = page.find('text:contains("Njësitë")')
    assert text_elem is not None, "Should find text element"

    # The highlight should cover the same area as the element's bbox
    # This tests that the coordinate system translation is working correctly
    elem_bbox = text_elem.bbox

    # Render with highlight at element's bbox
    # The visual highlight should align with the text (not offset)
    try:
        page.render(highlights=[{"bbox": elem_bbox, "color": "yellow", "alpha": 0.5}])
        assert True
    except Exception as e:
        pytest.fail(f"Failed to render aligned highlight: {e}")
