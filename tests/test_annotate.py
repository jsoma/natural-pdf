"""Test annotate parameter for element visualization."""

import pytest

from natural_pdf import PDF


def test_annotate_basic():
    """Test basic annotate functionality with single attribute."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf[0]

    # Find all text elements
    text_elements = page.find_all("text")

    # Test annotate with single attribute
    image = text_elements.show(annotate=["x0"], width=600)
    assert image is not None
    assert image.size[0] == 600


def test_annotate_string_to_list():
    """Test that annotate converts string to list automatically."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf[0]

    text_elements = page.find_all("text")

    # Test annotate with string (should be converted to list)
    image = text_elements.show(annotate="x0", width=600)
    assert image is not None
    assert image.size[0] == 600


def test_annotate_multiple_attributes():
    """Test annotate with multiple attributes."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf[0]

    text_elements = page.find_all("text")

    # Test annotate with multiple attributes
    image = text_elements.show(annotate=["x0", "fontname"], width=600)
    assert image is not None
    assert image.size[0] == 600


def test_annotate_with_group_by():
    """Test annotate combined with group_by."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf[0]

    text_elements = page.find_all("text")

    # Test annotate with group_by
    image = text_elements.show(group_by="fontname", annotate=["size"], width=600)
    assert image is not None
    # When group_by is used, a legend is added which increases width
    assert image.size[0] >= 600


def test_annotate_with_ocr():
    """Test annotate with OCR elements showing confidence scores."""
    # Use a PDF that needs OCR
    pdf = PDF("pdfs/tiny-ocr.pdf")
    page = pdf[0]

    # Run OCR - this adds OCR elements to the page
    page.apply_ocr(engine="paddle")  # Use paddle as it's fast

    # Find OCR elements
    ocr_elements = page.find_all("text[source=ocr]")

    if len(ocr_elements) > 0:
        # Test showing confidence scores
        image = ocr_elements.show(annotate=["confidence"], width=600)
        assert image is not None
        # The actual width might be less than 600 if the content is narrower
        assert image.size[0] > 0
    else:
        # Skip test if no OCR elements found
        pytest.skip("No OCR elements found in test PDF")


def test_annotate_nonexistent_attribute():
    """Test annotate with non-existent attribute (should handle gracefully)."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf[0]

    text_elements = page.find_all("text")

    # This should not crash but log warnings
    image = text_elements.show(annotate=["nonexistent_attr"], width=600)
    assert image is not None
    assert image.size[0] == 600


def test_annotate_with_regions():
    """Test annotate with regions."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf[0]

    # Run layout analysis using the correct API
    page.analyze_layout()
    detected_regions = page.find_all("region")

    if len(detected_regions) > 0:
        # Test showing region attributes
        image = detected_regions.show(group_by="region_type", annotate=["confidence"], width=600)
        assert image is not None
        # When group_by is used, a legend is added which increases width
        assert image.size[0] >= 600
    else:
        # Skip test if no regions found
        pytest.skip("No layout regions detected in test PDF")
