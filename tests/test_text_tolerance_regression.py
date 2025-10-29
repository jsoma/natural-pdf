from pathlib import Path

import pytest

import natural_pdf as npdf

TEST_PDF_PATH = Path("pdfs/candeubadam-2025-annual-oge-278e-oge-certified.pdf")


@pytest.mark.skipif(not TEST_PDF_PATH.exists(), reason="Issue PDF not available")
def test_page_find_respects_text_tolerance_override():
    pdf = npdf.PDF(str(TEST_PDF_PATH), auto_text_tolerance=False)
    page = pdf.pages[0]
    selector = 'text:contains("Public Financial Disclosure")'

    baseline = page.find(selector)
    assert baseline is None

    match = page.find(selector, text_tolerance={"x_tolerance": 20})
    assert match is not None


@pytest.mark.skipif(not TEST_PDF_PATH.exists(), reason="Issue PDF not available")
def test_pdf_find_respects_text_tolerance_override():
    pdf = npdf.PDF(str(TEST_PDF_PATH), auto_text_tolerance=False)
    selector = 'text:contains("Public Financial Disclosure")'

    baseline = pdf.find(selector)
    assert baseline is None

    match = pdf.find(selector, text_tolerance={"x_tolerance": 20})
    assert match is not None
