"""Test to check if pdfminer.six still has the color bug."""

import os

import pytest

# Disable patches for this test
os.environ["NATURAL_PDF_DISABLE_PDFMINER_PATCHES"] = "1"


def test_pdfminer_still_has_color_bug():
    """Check if pdfminer.six still has the color parsing bug. If this fails, we can remove the patch."""
    # Import after setting env var
    from natural_pdf import PDF

    pdf = PDF("pdfs/types-of-type.pdf")
    page = pdf.pages[0]

    # Find the yellow highlight
    highlighted = page.find('text:contains("Highlighted text")')
    assert highlighted is not None
    assert highlighted.is_highlighted is True

    color = highlighted.highlight_color

    assert color in ((1.0, 1.0, 0.0), [1.0, 1.0, 0.0]), (
        f"Unexpected highlight color {color!r}. If this regresses to 0.0, "
        "the upstream pdfminer bug has resurfaced."
    )
