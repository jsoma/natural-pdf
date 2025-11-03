#!/usr/bin/env python3
"""Smoke tests for Arabic RTL PDF processing."""

from __future__ import annotations

from pathlib import Path

import pytest

from natural_pdf import PDF

pytestmark = [pytest.mark.slow]

ARABIC_PDF = Path("pdfs/arabic.pdf")
REGULAR_PDF = Path("pdfs/types-of-type.pdf")


def _extract_first_page_text(pdf_path: Path) -> str:
    """Helper that extracts text from the first page and ensures resources close."""
    pdf = PDF(str(pdf_path))
    try:
        page = pdf.pages[0]
        return page.extract_text()
    finally:
        pdf.close()


def test_arabic_and_regular_pdfs_extract_text():
    """Verify Arabic RTL and regular PDFs can be processed without profiling noise."""
    assert ARABIC_PDF.exists(), "Missing arabic.pdf fixture"
    assert REGULAR_PDF.exists(), "Missing types-of-type.pdf fixture"

    arabic_text = _extract_first_page_text(ARABIC_PDF)
    regular_text = _extract_first_page_text(REGULAR_PDF)

    assert len(arabic_text) > 0, "Arabic PDF should contain text"
    assert len(regular_text) > 0, "Reference PDF should contain text"
