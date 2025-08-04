"""Test that exclusions are properly applied when accessing pages through slices."""

import io
from pathlib import Path
from unittest.mock import patch

import pytest

import natural_pdf as npdf


def create_test_pdf():
    """Create a simple test PDF using pdfplumber's capabilities."""
    # Create a minimal PDF-like content
    content = b"%PDF-1.4\n"  # Minimal PDF header
    # We'll mock this since we can't easily create PDFs without reportlab
    return io.BytesIO(content)


def test_slice_exclusion_consistency():
    """Test that exclusions work consistently whether accessing pages directly or via slices."""
    # Find a test PDF
    test_pdfs = list(Path("tests/samples").glob("*.pdf"))
    if not test_pdfs:
        pytest.skip("No test PDFs found in tests/samples/")

    pdf_path = test_pdfs[0]

    # Load PDF
    pdf = npdf.PDF(pdf_path)

    # Skip if not enough pages
    if len(pdf.pages) < 2:
        pytest.skip("Test PDF needs at least 2 pages")

    # Access some pages to cache them BEFORE adding exclusion
    pages_slice = pdf.pages[:2]  # This creates and caches pages 0-1
    first_page = pages_slice[0]  # Access to ensure it's cached

    # Add an exclusion that should affect all pages
    # Using a simple exclusion that removes the first text element
    def exclude_first_text(page):
        first_text = page.find("text")
        if first_text:
            return first_text.to_region()
        return None

    pdf.add_exclusion(exclude_first_text, label="test_exclusion")

    # Test 1: Direct access to page (should have exclusion)
    direct_page = pdf.pages[0]
    direct_text = direct_page.extract_text()

    # Test 2: Access via slice (should also have exclusion)
    slice_page = pages_slice[0]
    slice_text = slice_page.extract_text()

    # Test 3: Create a new slice after adding exclusion
    new_slice = pdf.pages[:2]
    new_slice_page = new_slice[0]
    new_slice_text = new_slice_page.extract_text()

    # All three should have the same text (with exclusion applied)
    assert direct_text == slice_text, "Direct access and slice access should have same exclusions"
    assert direct_text == new_slice_text, "New slice should also have exclusions"

    # Verify the exclusion was actually applied by checking it's shorter than without exclusions
    # We can't easily test without exclusions after they're added, so just verify they're non-empty
    assert len(direct_text.strip()) > 0, "Text extraction should still return some content"

    # Test that the page objects are the same (reused from cache)
    assert direct_page is slice_page, "Should reuse the same page object from cache"
    assert direct_page is new_slice_page, "New slices should also reuse cached pages"


def test_exclusion_with_late_page_access():
    """Test exclusions work correctly when pages are accessed after exclusion is added."""
    # Find a test PDF
    test_pdfs = list(Path("tests/samples").glob("*.pdf"))
    if not test_pdfs:
        pytest.skip("No test PDFs found in tests/samples/")

    pdf_path = test_pdfs[0]

    # Load PDF
    pdf = npdf.PDF(pdf_path)

    if len(pdf.pages) < 2:
        pytest.skip("Test PDF needs at least 2 pages")

    # Add exclusion BEFORE accessing any pages
    def exclude_first_text(page):
        first_text = page.find("text")
        if first_text:
            return first_text.to_region()
        return None

    pdf.add_exclusion(exclude_first_text, label="test_exclusion")

    # Now access pages through different methods
    direct_last = pdf.pages[-1]
    direct_text = direct_last.extract_text()

    # Access via slice
    all_pages = pdf.pages[:]
    slice_last = all_pages[-1]
    slice_text = slice_last.extract_text()

    # They should be identical
    assert direct_text == slice_text, "Exclusions should work the same regardless of access method"
    assert direct_last is slice_last, "Should be the same page object"


def test_exclusion_lambda_with_element_collection():
    """Test that lambda exclusions returning ElementCollections work correctly."""
    # Find a test PDF with some text
    test_pdfs = list(Path("tests/samples").glob("*.pdf"))
    if not test_pdfs:
        pytest.skip("No test PDFs found in tests/samples/")

    pdf_path = test_pdfs[0]
    pdf = npdf.PDF(pdf_path)

    # Get the first word on the first page to exclude
    first_page = pdf.pages[0]
    first_word = first_page.find("text")
    if not first_word:
        pytest.skip("Test PDF has no text on first page")

    first_word_text = first_word.extract_text()

    # Add exclusion that returns an ElementCollection
    pdf.add_exclusion(
        lambda page: page.find_all(f'text:contains("{first_word_text}")'),
        label="exclude_first_word",
    )

    # Access page through slice
    pages = pdf.pages[:1]
    page_from_slice = pages[0]

    # Extract text and verify exclusion worked
    extracted_text = page_from_slice.extract_text()
    assert first_word_text not in extracted_text, f"Text '{first_word_text}' should be excluded"


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
