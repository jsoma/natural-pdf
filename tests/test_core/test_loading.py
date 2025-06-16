import os
from pathlib import Path

import pytest

from natural_pdf import PDF


def test_pdf_loading_from_url():
    """Tests if a PDF can be loaded successfully from a URL."""
    url = "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf"

    pdf = PDF(url)
    try:
        assert pdf is not None
        assert len(pdf.pages) > 0, "PDF should have at least one page"
        # Check if metadata is accessible
        assert pdf.metadata is not None
    finally:
        pdf.close()


def test_pdf_loading_from_file(practice_pdf):
    """Tests if a PDF can be loaded successfully from a file."""
    assert practice_pdf is not None
    assert len(practice_pdf.pages) > 0, "PDF should have at least one page"
    assert hasattr(practice_pdf, "metadata"), "PDF should have metadata attribute"


def test_pdf_metadata(practice_pdf):
    """Tests if PDF metadata can be accessed correctly."""
    metadata = practice_pdf.metadata

    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    # Check for commonly available metadata fields
    assert "Title" in metadata or metadata.get("Title") is None
    assert "Author" in metadata or metadata.get("Author") is None


def test_pdf_pages_access(practice_pdf):
    """Tests if PDF pages can be accessed correctly."""
    pages = practice_pdf.pages

    assert len(pages) > 0, "PDF should have at least one page"

    # Access first page
    first_page = pages[0]
    assert hasattr(first_page, "number"), "Page should have a number attribute"
    assert first_page.number == 1, "First page should have number 1 (pages are 1-indexed)"

    # Check page dimensions
    assert hasattr(first_page, "width"), "Page should have width"
    assert hasattr(first_page, "height"), "Page should have height"
    assert first_page.width > 0, "Page width should be positive"
    assert first_page.height > 0, "Page height should be positive"


def test_pdf_close():
    """Tests if PDF can be closed properly."""
    url = "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf"

    pdf = PDF(url)
    pdf.close()

    # Additional checks could be added if the PDF class exposes a way
    # to check if it's closed (e.g., pdf.is_closed attribute)


def test_pdf_context_manager():
    """Tests if PDF works with context manager."""
    url = "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf"

    with PDF(url) as pdf:
        assert pdf is not None
        assert len(pdf.pages) > 0


def test_pdf_collection_loading(pdf_collection):
    """Tests if a collection of PDFs can be loaded successfully."""
    assert pdf_collection is not None
    assert len(pdf_collection.pdfs) > 0, "Collection should contain PDFs"

    # Check if we can access the first PDF in the collection
    first_pdf = pdf_collection.pdfs[0]
    assert first_pdf is not None
    assert len(first_pdf.pages) > 0, "First PDF should have pages"
