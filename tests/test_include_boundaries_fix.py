"""
Test that include_boundaries parameter works correctly in get_sections.
"""

from pathlib import Path

import pytest

import natural_pdf as npdf

# Get path to test PDF
TEST_PDF_PATH = Path(__file__).parent.parent / "pdfs" / "types-of-type.pdf"


def test_include_boundaries_parameter():
    """Test that include_boundaries parameter actually affects section boundaries."""
    pdf = npdf.PDF(str(TEST_PDF_PATH))

    # Find elements that contain "Type" (should be multiple headings)
    type_elements = pdf.find_all("text:contains(Type)")

    if len(type_elements) < 2:
        pytest.skip("Not enough 'Type' elements found for meaningful test")

    # Test with different include_boundaries settings
    sections_both = pdf.get_sections("text:contains(Type)", include_boundaries="both")
    sections_start = pdf.get_sections("text:contains(Type)", include_boundaries="start")
    sections_end = pdf.get_sections("text:contains(Type)", include_boundaries="end")
    sections_none = pdf.get_sections("text:contains(Type)", include_boundaries="none")

    # All should have the same number of sections
    assert len(sections_both) == len(sections_start)
    assert len(sections_both) == len(sections_end)
    assert len(sections_both) == len(sections_none)

    # Extract text from first section with each setting
    if len(sections_both) > 0:
        text_both = sections_both[0].extract_text().strip()
        text_start = sections_start[0].extract_text().strip()
        text_end = sections_end[0].extract_text().strip()
        text_none = sections_none[0].extract_text().strip()

        # The texts should be different based on boundary inclusion
        # 'both' should have the most text
        # 'none' should have the least text
        # 'start' and 'end' should be in between

        # Print for debugging
        print(f"Text with 'both': {len(text_both)} chars")
        print(f"Text with 'start': {len(text_start)} chars")
        print(f"Text with 'end': {len(text_end)} chars")
        print(f"Text with 'none': {len(text_none)} chars")

        # Verify that different settings produce different results
        # At least some of them should be different
        texts = [text_both, text_start, text_end, text_none]
        unique_texts = set(texts)
        assert len(unique_texts) > 1, "All include_boundaries settings produced the same text"


def test_include_boundaries_visual_verification():
    """Visual test to verify include_boundaries behavior."""
    pdf = npdf.PDF(str(TEST_PDF_PATH))

    # Find a specific section marker
    sections_both = pdf.get_sections("text:contains(Type)", include_boundaries="both")
    sections_none = pdf.get_sections("text:contains(Type)", include_boundaries="none")

    if len(sections_both) > 0 and len(sections_none) > 0:
        # Get the bounding boxes
        bbox_both = sections_both[0].bbox
        bbox_none = sections_none[0].bbox

        print(f"Section with 'both': {bbox_both}")
        print(f"Section with 'none': {bbox_none}")

        # With 'none', the section should be smaller (start lower, end higher)
        # In PDF coordinates, larger 'top' means higher on page
        assert (
            bbox_none[1] < bbox_both[1]
        ), "Section with 'none' should start lower (exclude start boundary)"
        assert (
            bbox_none[3] > bbox_both[3]
        ), "Section with 'none' should end higher (exclude end boundary)"


def test_include_boundaries_cross_page():
    """Test include_boundaries with sections that span multiple pages."""
    pdf = npdf.PDF(str(TEST_PDF_PATH))

    # For cross-page sections, we need to check if the PDF has multiple pages
    if len(pdf.pages) < 2:
        pytest.skip("PDF doesn't have multiple pages for cross-page test")

    # Try to find sections that might span pages
    sections_both = pdf.get_sections("text:contains(Type)", include_boundaries="both")

    # Check if any section spans multiple pages
    multi_page_sections = []
    for section in sections_both:
        if hasattr(section, "constituent_regions") and len(section.constituent_regions) > 1:
            multi_page_sections.append(section)

    if multi_page_sections:
        # Test with different boundaries on multi-page section
        # This verifies that _build_flow_region respects include_boundaries
        print(f"Found {len(multi_page_sections)} multi-page sections")

        # Get sections with different boundary settings
        sections_none = pdf.get_sections("text:contains(Type)", include_boundaries="none")

        # The text content should be different
        for i, section in enumerate(multi_page_sections):
            if i < len(sections_none):
                text_both = section.extract_text()
                text_none = sections_none[i].extract_text()
                assert (
                    text_both != text_none
                ), "Multi-page sections should have different text with different boundaries"


if __name__ == "__main__":
    test_include_boundaries_parameter()
    test_include_boundaries_visual_verification()
    test_include_boundaries_cross_page()
    print("All tests passed!")
