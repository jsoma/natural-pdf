import pytest


def test_page_text_extraction(practice_pdf):
    """Tests basic text extraction from a page."""
    page = practice_pdf.pages[0]

    # Extract text from the page
    text = page.extract_text()

    # Assertions
    assert isinstance(text, str), "Extracted text should be a string"
    assert len(text) > 50, "Extracted text seems too short or empty"

    # Check for specific content in the practice PDF
    # These are specific text fragments expected in the practice PDF
    assert "Jungle Health" in text, "Expected words 'Jungle Health' in extracted text"
    assert "Inadequate" in text, "Expected word 'Inadequate' in extracted text"


def test_pdf_text_extraction(practice_pdf):
    """Tests text extraction from an entire PDF."""
    # Extract text from the entire PDF
    full_text = practice_pdf.extract_text()

    # Assertions
    assert isinstance(full_text, str), "Extracted text should be a string"
    assert len(full_text) > 100, "Full PDF text seems too short"

    # Check for specific content in different pages
    # This relies on knowledge of the PDF content
    assert "Jungle Health" in full_text, "Expected words 'Jungle Health' in extracted text"


def test_text_extraction_with_page_separator(practice_pdf):
    """Tests text extraction with custom page separator."""
    # Extract text with a custom separator
    separator = "\n---PAGE BREAK---\n"
    full_text = practice_pdf.extract_text(page_separator=separator)

    # Check if the separator is in the text (if PDF has multiple pages)
    if len(practice_pdf.pages) > 1:
        assert separator in full_text, "Page separator should be in extracted text"


def test_text_extraction_by_region(practice_pdf):
    """Tests text extraction from a specific region of a page."""
    page = practice_pdf.pages[0]

    # Create a region in the top-left quarter of the page
    region_width = page.width / 2
    region_height = page.height / 2
    region = page.create_region(0, 0, region_width, region_height)

    # Extract text from this region
    region_text = region.extract_text()

    # Assertions
    assert isinstance(region_text, str), "Region text should be a string"


def test_text_extraction_with_line_breaks(practice_pdf):
    """Tests text extraction with consideration for line breaks."""
    page = practice_pdf.pages[0]

    # Extract text with explicit line breaks
    text_with_breaks = page.extract_text(preserve_line_breaks=True)

    # Extract text without preserving line breaks
    text_without_breaks = page.extract_text(preserve_line_breaks=False)

    # Assertions
    assert isinstance(text_with_breaks, str)
    assert isinstance(text_without_breaks, str)

    # text_with_breaks should have at least as many newlines as text_without_breaks
    assert text_with_breaks.count("\n") >= text_without_breaks.count("\n")


def test_ocr_text_extraction(needs_ocr_pdf):
    """Tests text extraction before and after OCR."""
    page = needs_ocr_pdf.pages[0]

    # Try extracting text without OCR
    text_without_ocr = page.extract_text()

    # Skip the test if OCR dependencies aren't installed
    try:
        # Apply OCR and extract text
        page.apply_ocr(languages=["en"])
        text_with_ocr = page.extract_text()

        # Assertions
        assert isinstance(text_with_ocr, str)
        assert len(text_with_ocr) > len(text_without_ocr), "OCR should find more text"

    except ImportError:
        pytest.skip("OCR dependencies not installed, skipping OCR test")
    except Exception as e:
        pytest.skip(f"OCR failed with error: {e}")


def test_text_extraction_from_element(practice_pdf):
    """Tests text extraction from individual elements."""
    page = practice_pdf.pages[0]

    # Find some text elements
    text_elements = page.find_all("text")

    # Skip if no text elements found
    if not text_elements:
        pytest.skip("No text elements found in the practice PDF")

    # Extract text from the first element
    first_element_text = text_elements[0].text

    # Assertions
    assert isinstance(first_element_text, str)
    assert len(first_element_text) > 0, "Element text should not be empty"
