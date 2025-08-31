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

    # If pdfminer is fixed, this will be (1.0, 1.0, 0.0)
    # If still buggy, this will be 0.0
    color = highlighted.highlight_color

    # This test PASSES if the bug still exists
    assert color == 0.0, (
        f"PDFMiner bug appears to be fixed! Color is {color} instead of 0.0. "
        "Consider removing the monkey patch in natural_pdf/utils/pdfminer_patches.py"
    )
