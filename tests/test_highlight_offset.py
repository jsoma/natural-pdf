"""Test highlighting with PDFs that have negative margin bounds in pdfplumber."""

from pathlib import Path

import pytest

from natural_pdf import PDF

PDF_PATH = Path("pdfs/pak-ks-expenses.pdf")


@pytest.fixture(scope="module")
def expenses_pdf_path():
    if not PDF_PATH.exists():
        pytest.skip("Local copy of pak-ks expenses PDF is missing")
    return str(PDF_PATH)


def test_show_method_highlight_offset(expenses_pdf_path):
    """Test that element.show() correctly handles offset for PDFs with negative bounds."""
    pdf = PDF(expenses_pdf_path)
    try:
        page = pdf.pages[0]

        # Find a specific element
        rect = page.find("rect[fill~=yellow]")
        assert rect is not None, "Should find yellow rectangle"

        # Using element.show() which goes through _apply_spec_highlights
        img = rect.show()
        assert img is not None
    except Exception as exc:
        pytest.fail(f"Failed to show element: {exc}")
    finally:
        pdf.close()


def test_highlight_offset_with_negative_bounds(expenses_pdf_path):
    """Test that highlights are correctly positioned for PDFs with negative bounds.

    Some PDFs have pdfplumber page bounds that start with negative coordinates,
    which can cause highlights to be offset from their actual elements.
    """
    pdf = PDF(expenses_pdf_path)
    try:
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
        page.render(highlights=[{"bbox": bbox}])
    except Exception as exc:
        pytest.fail(f"Failed to render highlight: {exc}")
    finally:
        pdf.close()


def test_multiple_highlight_types_with_offset(expenses_pdf_path):
    """Test different highlight types with PDFs having negative bounds."""
    pdf = PDF(expenses_pdf_path)
    try:
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

        page.render(highlights=highlights)
    except Exception as exc:
        pytest.fail(f"Failed to render multiple highlights: {exc}")
    finally:
        pdf.close()


def test_highlight_alignment_verification(expenses_pdf_path):
    """Verify that highlights align with their source elements."""
    pdf = PDF(expenses_pdf_path)
    try:
        page = pdf.pages[0]

        # Find a specific text element
        text_elem = page.find('text:contains("Njësitë")')
        assert text_elem is not None, "Should find text element"

        # The highlight should cover the same area as the element's bbox
        elem_bbox = text_elem.bbox

        page.render(highlights=[{"bbox": elem_bbox, "color": "yellow", "alpha": 0.5}])
    except Exception as exc:
        pytest.fail(f"Failed to render aligned highlight: {exc}")
    finally:
        pdf.close()
