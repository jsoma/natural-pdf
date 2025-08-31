"""Test documenting pdfplumber color parsing bug."""

import pytest

from natural_pdf import PDF


def test_pdfminer_highlight_color_bug():
    """
    Document a bug in pdfminer.six where it incorrectly parses colors.

    In the test PDF:
    - The PDF commands show: '1 1 0 sc' (RGB yellow) before drawing the rectangle
    - Illustrator correctly displays the highlight as yellow
    - But pdfminer.six reports the color as 0.0 (grayscale black)

    Root cause: When no explicit color space is set, pdfminer defaults to
    DeviceGray (1 component) instead of DeviceRGB (3 components). It then
    only reads 1 value from the stack, and due to LIFO order, gets the
    last value (0) instead of all three (1,1,0).
    """
    pdf = PDF("pdfs/types-of-type.pdf")
    page = pdf.pages[0]

    # Find highlighted text
    highlighted = page.find('text:contains("Highlighted text")')
    assert highlighted is not None
    assert highlighted.is_highlighted is True

    # Without patch: color is 0.0 (black) due to pdfminer bug
    # With patch: color is (1.0, 1.0, 0.0) (yellow) as it should be
    color = highlighted.highlight_color
    assert color in [0.0, (1.0, 1.0, 0.0)]

    # Check if patch is active
    from natural_pdf.utils.pdfminer_patches import _patches_applied

    if _patches_applied:
        assert color == (1.0, 1.0, 0.0), "With patch, should be yellow"
    else:
        assert color == 0.0, "Without patch, incorrectly shows as black"

    # Document the issue
    # In reality, this highlight appears yellow in PDF viewers and Illustrator
    # because the PDF uses a custom color space (likely ICCBased or similar)
    # where 0.0 maps to yellow, but pdfplumber doesn't handle this correctly


def test_highlight_color_in_styles_column():
    """Test that the styles column handles highlight colors correctly."""
    pdf = PDF("pdfs/types-of-type.pdf")
    page = pdf.pages[0]

    highlighted = page.find_all("text:highlighted")
    assert len(highlighted) > 0

    # The styles column should show highlight info even with color space issues
    from natural_pdf.describe.base import _extract_element_value

    for elem in highlighted:
        styles = _extract_element_value(elem, "styles")
        assert "highlight" in styles.lower()

        # For grayscale 0.0, it might show as highlight(#000000)
        # This is due to the pdfplumber limitation, not a bug in our code
