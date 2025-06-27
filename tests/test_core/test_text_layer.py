"""
Test text layer functionality including text_layer=False parameter and remove_text_layer() method.
"""

import pytest

from natural_pdf import PDF


def test_pdf_has_text_layer():
    """Test that pdfs/01-practice.pdf has a text layer by default."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Check that text elements exist
    assert len(page.words) > 0, "PDF should have word elements"
    assert len(page.chars) > 0, "PDF should have character elements"

    # Check that we can extract text
    text = page.extract_text()
    assert text.strip() != "", "Should be able to extract text from PDF"
    assert "Jungle Health and Safety" in text, "Expected text should be in the extracted text"

    pdf.close()


def test_text_layer_false_parameter():
    """Test that text_layer=False prevents loading text elements."""
    pdf = PDF("pdfs/01-practice.pdf", text_layer=False)
    page = pdf.pages[0]

    # Check that no text elements exist
    assert len(page.words) == 0, "PDF should have no word elements when text_layer=False"
    assert len(page.chars) == 0, "PDF should have no character elements when text_layer=False"

    # Check that extract_text returns empty
    text = page.extract_text()
    assert text.strip() == "", "extract_text should return empty string when text_layer=False"

    # But other elements should still exist
    assert len(page.rects) > 0, "Rectangle elements should still be loaded"
    assert len(page.lines) >= 0, "Line elements should still be accessible"

    pdf.close()


def test_remove_text_layer_method():
    """Test that remove_text_layer() removes existing text elements."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # First verify text exists
    initial_words = len(page.words)
    initial_chars = len(page.chars)
    assert initial_words > 0, "Should have words initially"
    assert initial_chars > 0, "Should have chars initially"

    # Extract text before removal
    text_before = page.extract_text()
    assert text_before.strip() != "", "Should have text before removal"

    # Remove text layer
    page.remove_text_layer()

    # Check that text elements are gone
    assert len(page.words) == 0, "Should have no words after remove_text_layer()"
    assert len(page.chars) == 0, "Should have no chars after remove_text_layer()"

    # Check that extract_text returns empty
    text_after = page.extract_text()
    assert text_after.strip() == "", "extract_text should return empty after remove_text_layer()"

    # But other elements should still exist
    assert len(page.rects) > 0, "Rectangle elements should still exist after text removal"

    pdf.close()


def test_text_layer_false_then_ocr():
    """Test that we can apply OCR to a PDF loaded with text_layer=False."""
    # Skip if OCR dependencies are not available
    pytest.importorskip("easyocr")

    pdf = PDF("pdfs/01-practice.pdf", text_layer=False)
    page = pdf.pages[0]

    # Verify no text initially
    assert len(page.words) == 0, "Should have no words with text_layer=False"

    # Apply OCR
    page.apply_ocr(engine="easyocr", languages=["en"])

    # Now we should have OCR text
    assert len(page.words) > 0, "Should have words after OCR"

    # Check that all words are from OCR
    for word in page.words:
        assert word.source == "ocr", "All words should have source='ocr'"

    pdf.close()


def test_multiple_pages_text_layer():
    """Test text_layer=False works across multiple pages."""
    pdf = PDF("pdfs/01-practice.pdf", text_layer=False)

    # Check all pages have no text
    for page in pdf.pages:
        assert len(page.words) == 0, f"Page {page.number} should have no words"
        assert len(page.chars) == 0, f"Page {page.number} should have no chars"
        assert page.extract_text().strip() == "", f"Page {page.number} should extract empty text"

    pdf.close()


def test_text_layer_parameter_types():
    """Test that text_layer parameter accepts proper boolean values."""
    # Test with explicit True (default behavior)
    pdf_true = PDF("pdfs/01-practice.pdf", text_layer=True)
    assert len(pdf_true.pages[0].words) > 0, "text_layer=True should load text"
    pdf_true.close()

    # Test with explicit False
    pdf_false = PDF("pdfs/01-practice.pdf", text_layer=False)
    assert len(pdf_false.pages[0].words) == 0, "text_layer=False should not load text"
    pdf_false.close()
