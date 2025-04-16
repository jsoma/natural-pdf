import os

import pytest

from natural_pdf import PDF

# URL for the test PDF used in the tutorial
TEST_PDF_URL = "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf"


def test_pdf_loading_from_url():
    """Tests if a PDF can be loaded successfully from a URL."""
    try:
        pdf = PDF(TEST_PDF_URL)
        # Basic assertions after loading
        assert pdf is not None
        assert len(pdf.pages) > 0, "PDF should have at least one page"
        assert os.path.exists(pdf.path), "PDF file should be downloaded locally"
        # Check if metadata (like Title) is accessible, even if None
        assert "Title" in pdf.metadata or pdf.metadata.get("Title") is None

    except Exception as e:
        pytest.fail(f"PDF loading from URL failed: {e}")


def test_page_text_extraction():
    """Tests if text can be extracted from the first page."""
    try:
        pdf = PDF(TEST_PDF_URL)
        assert len(pdf.pages) > 0, "PDF has no pages"
        page = pdf.pages[0]
        text = page.extract_text()
        assert isinstance(text, str), "Extracted text should be a string"
        assert len(text) > 50, "Extracted text seems too short or empty"
        # Add a more specific assertion if you know some expected text
        # assert "Expected sample text" in text

    except Exception as e:
        pytest.fail(f"Text extraction failed: {e}")


# Clean up downloaded file if necessary (optional, depends on PDF class behavior)
# You might want a fixture to handle setup/teardown of the downloaded file
# @pytest.fixture(scope="module")
# def downloaded_pdf():
#     pdf = PDF(TEST_PDF_URL)
#     yield pdf
#     # Cleanup code here if PDF() doesn't handle it
#     if os.path.exists(pdf.path):
#         os.remove(pdf.path)
