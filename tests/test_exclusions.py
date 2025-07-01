import os

import pytest

import natural_pdf as npdf


def _get_pdf_path(filename: str) -> str:
    """Helper to get absolute path to a sample PDF in the repository."""
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, os.pardir, "pdfs", filename))


def _get_bold_elements_texts(page):
    selector = "text:bold"
    bold_elements = page.find_all(selector, apply_exclusions=False)
    return [el.text.strip() for el in bold_elements if hasattr(el, "text")]


def test_page_add_exclusion_with_selector():
    pdf_path = _get_pdf_path("01-practice.pdf")
    pdf = npdf.PDF(pdf_path)
    page = pdf.pages[0]

    # Ensure bold elements exist on the page
    bold_texts = _get_bold_elements_texts(page)
    assert bold_texts, "Expected at least one bold text element on the page"

    # Verify bold texts appear in the raw extraction before exclusions
    original_text = page.extract_text(use_exclusions=False)
    for txt in bold_texts:
        assert txt in original_text, f"'{txt}' should be present before exclusions"

    # Add exclusion using selector string
    page.add_exclusion("text:bold", label="bold_text")
    filtered_text = page.extract_text()

    # Verify that bold texts are no longer present after exclusion
    for txt in bold_texts:
        assert txt not in filtered_text, f"'{txt}' should be excluded from extraction"


def test_pdf_add_exclusion_with_selector():
    pdf_path = _get_pdf_path("01-practice.pdf")
    pdf = npdf.PDF(pdf_path)
    page = pdf.pages[0]

    bold_texts = _get_bold_elements_texts(page)
    assert bold_texts, "Expected at least one bold text element on the page"

    # Add exclusion at PDF level using selector
    pdf.add_exclusion("text:bold", label="bold_text")

    filtered_text = page.extract_text()
    for txt in bold_texts:
        assert txt not in filtered_text, f"'{txt}' should be excluded via PDF-level exclusion"


class TestExclusionMethods:
    """Test different exclusion methods for filtering content."""

    def test_region_vs_element_exclusion(self):
        """Test the difference between region and element exclusion methods."""
        # Load the test PDF
        pdf = npdf.PDF("pdfs/confidential.pdf")
        page = pdf.pages[0]

        # Expected text snippets
        free_text = "This text is free and not connected to anything"
        behind_text = "This text is behind some other text"
        confidential_text = "CONFIDENTIAL"

        # Test 1: No exclusions - all text should be present
        full_text = page.extract_text()
        assert free_text in full_text
        assert behind_text in full_text
        assert confidential_text in full_text

        # Test 2: Region-based exclusion (default)
        # This should exclude everything in the CONFIDENTIAL bounding box
        page.clear_exclusions()
        page.add_exclusion('text:contains("CONFIDENTIAL")', method="region")
        region_text = page.extract_text()

        assert free_text in region_text  # Free text should still be there
        assert behind_text not in region_text  # Text behind CONFIDENTIAL should be excluded
        assert confidential_text not in region_text  # CONFIDENTIAL itself excluded

        # Test 3: Element-based exclusion
        # This should only exclude the CONFIDENTIAL element itself
        page.clear_exclusions()
        page.add_exclusion('text:contains("CONFIDENTIAL")', method="element")
        element_text = page.extract_text()

        assert free_text in element_text  # Free text should still be there
        assert behind_text in element_text  # Text behind CONFIDENTIAL should NOT be excluded
        assert confidential_text not in element_text  # CONFIDENTIAL itself excluded

    def test_element_collection_exclusion(self):
        """Test exclusions with ElementCollection objects."""
        pdf = npdf.PDF("pdfs/confidential.pdf")
        page = pdf.pages[0]

        # Find all CONFIDENTIAL elements
        confidential_elements = page.find_all('text:contains("CONFIDENTIAL")')

        # Test with element method
        page.clear_exclusions()
        page.add_exclusion(confidential_elements, method="element")
        text = page.extract_text()

        assert "This text is behind some other text" in text
        assert "CONFIDENTIAL" not in text

    def test_mixed_exclusion_methods(self):
        """Test using both region and element exclusions together."""
        pdf = npdf.PDF("pdfs/confidential.pdf")
        page = pdf.pages[0]

        # Add both types of exclusions
        page.clear_exclusions()

        # First add element-based exclusion for elements containing "free"
        # Note: In this PDF, the entire sentence is one word element
        page.add_exclusion('text:contains("free")', method="element", label="free_element")

        # Then add region-based exclusion for another
        page.add_exclusion('text:contains("CONFIDENTIAL")', method="region", label="conf_region")

        text = page.extract_text()

        # The entire sentence containing "free" should be excluded as it's one element
        assert "This text is free and not connected to anything" not in text
        assert "free" not in text

        # CONFIDENTIAL region should exclude everything in its bbox
        assert "This text is behind some other text" not in text
        assert "CONFIDENTIAL" not in text

    def test_backward_compatibility(self):
        """Test that old code without method parameter still works."""
        pdf = npdf.PDF("pdfs/confidential.pdf")
        page = pdf.pages[0]

        # Old style exclusion without method parameter
        # Should default to region-based exclusion
        page.clear_exclusions()
        page.add_exclusion('text:contains("CONFIDENTIAL")')

        text = page.extract_text()

        # Should behave like region exclusion
        assert "This text is behind some other text" not in text
        assert "CONFIDENTIAL" not in text

    def test_invalid_method_raises_error(self):
        """Test that invalid method values raise an error."""
        pdf = npdf.PDF("pdfs/confidential.pdf")
        page = pdf.pages[0]

        with pytest.raises(ValueError, match="Invalid exclusion method"):
            page.add_exclusion('text:contains("test")', method="invalid")

    def test_debug_output(self):
        """Test debug output for exclusions."""
        pdf = npdf.PDF("pdfs/confidential.pdf")
        page = pdf.pages[0]

        # Add mixed exclusions
        page.clear_exclusions()
        page.add_exclusion('text:contains("CONFIDENTIAL")', method="element", label="conf_elem")
        page.add_exclusion('text:contains("free")', method="region", label="free_region")

        # Extract with debug enabled - should print diagnostic info
        text = page.extract_text(debug_exclusions=True)

        # Just verify it runs without error
        assert isinstance(text, str)
