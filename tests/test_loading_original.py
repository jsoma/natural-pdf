from pathlib import Path

import pytest

from natural_pdf import PDF

TEST_PDF_PATH = Path("pdfs/01-practice.pdf")


@pytest.fixture(scope="module")
def practice_path():
    if not TEST_PDF_PATH.exists():
        pytest.skip("Local practice PDF is missing")
    return str(TEST_PDF_PATH)


def test_pdf_loading_from_path(practice_path):
    """Tests if a PDF can be loaded successfully from a local path."""
    pdf = PDF(practice_path)
    try:
        assert len(pdf.pages) > 0, "PDF should have at least one page"
        # Check if metadata (like Title) is accessible, even if None
        assert "Title" in pdf.metadata or pdf.metadata.get("Title") is None
    finally:
        pdf.close()


def test_page_text_extraction(practice_path):
    """Tests if text can be extracted from the first page."""
    pdf = PDF(practice_path)
    try:
        assert len(pdf.pages) > 0, "PDF has no pages"
        page = pdf.pages[0]
        text = page.extract_text()
        assert isinstance(text, str), "Extracted text should be a string"
        assert len(text) > 50, "Extracted text seems too short or empty"
    finally:
        pdf.close()


# You might want a fixture to handle setup/teardown of the downloaded file
# @pytest.fixture(scope="module")
# def downloaded_pdf():
#     pdf = PDF(TEST_PDF_URL)
#     yield pdf
#     # Cleanup code here if PDF() doesn't handle it
#     if os.path.exists(pdf.path):
#         os.remove(pdf.path)
