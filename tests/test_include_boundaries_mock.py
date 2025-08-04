"""
Test include_boundaries with a mock setup to verify the fix works.
"""

from unittest.mock import MagicMock, Mock, patch

import natural_pdf as npdf
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement


def create_mock_element(page, text, top, bottom, x0=0, x1=100):
    """Create a mock text element."""
    obj = {
        "text": text,
        "x0": x0,
        "top": top,
        "x1": x1,
        "bottom": bottom,
        "height": bottom - top,
        "page_number": page.number,
    }
    element = TextElement(obj, page)
    return element


def test_get_sections_include_boundaries():
    """Test that include_boundaries parameter works correctly in get_sections."""
    # Create mock PDF and pages
    pdf = Mock()
    pdf.pages = []

    # Create mock page
    page = Mock()
    page.number = 1
    page.index = 0
    page.width = 612
    page.height = 792
    page.pdf = pdf

    # Create mock elements on the page
    # In PDF coordinates, top of page has higher Y value
    # Header at top of page (high Y value)
    header_element = create_mock_element(page, "Section 1", top=100, bottom=120)

    # Content in middle
    content_elements = [
        create_mock_element(page, "Content line 1", top=150, bottom=170),
        create_mock_element(page, "Content line 2", top=200, bottom=220),
        create_mock_element(page, "Content line 3", top=250, bottom=270),
    ]

    # Next header (lower on page, higher Y value)
    next_header = create_mock_element(page, "Section 2", top=300, bottom=320)

    # Set up the page's element finding
    all_elements = [header_element] + content_elements + [next_header]

    def mock_find_all(selector, **kwargs):
        if "Section" in selector:
            return ElementCollection([header_element, next_header])
        return ElementCollection(all_elements)

    page.find_all = mock_find_all

    # Mock get_elements to return all elements
    page.get_elements = Mock(return_value=all_elements)

    # Mock get_section_between to return regions with correct boundaries
    def mock_get_section_between(start, end, include_boundaries="both", orientation="vertical"):
        # Ensure start and end are in the right order
        # In this test setup, start should come before end (lower top value)
        if not end:
            end_top = page.height
            end_bottom = page.height
        else:
            end_top = end.top
            end_bottom = end.bottom

        if include_boundaries == "both":
            top = start.top
            bottom = end_bottom
        elif include_boundaries == "start":
            top = start.top
            bottom = end_top
        elif include_boundaries == "end":
            top = start.bottom
            bottom = end_bottom
        else:  # none
            top = start.bottom
            bottom = end_top

        # Ensure top < bottom for valid region
        if top > bottom:
            top, bottom = bottom, top

        region = Region(page, (0, top, page.width, bottom))
        return region

    page.get_section_between = mock_get_section_between

    # Create PageCollection with mocked pages
    pages = [page]

    # Import PageCollection and patch its initialization
    from natural_pdf.core.page_collection import PageCollection

    collection = PageCollection(pages)
    collection.pages = pages

    # Test get_sections with different include_boundaries settings
    print("\nTesting get_sections with mock data...")

    # Mock the find_all method on collection
    collection.find_all = lambda selector, **kwargs: ElementCollection(
        [header_element, next_header]
    )

    # Test each include_boundaries option
    for boundaries in ["both", "start", "end", "none"]:
        sections = collection.get_sections("text:contains(Section)", include_boundaries=boundaries)

        print(f"\ninclude_boundaries='{boundaries}':")
        print(f"  Number of sections: {len(sections)}")

        if len(sections) > 0:
            section = sections[0]
            print(f"  Section bbox: {section.bbox}")
            print(f"  Top: {section.bbox[1]}, Bottom: {section.bbox[3]}")

            # When we have only start elements, sections go from start to next start
            # The section always ends at the TOP of the next start element
            # include_boundaries only affects whether we include the START element
            if boundaries == "both" or boundaries == "start":
                # Should include the start element
                assert (
                    section.bbox[1] == header_element.top
                ), f"'{boundaries}' should start at first element top"
                assert (
                    section.bbox[3] == next_header.top
                ), f"Section should always end at next element top"
            else:  # "end" or "none"
                # Should exclude the start element
                assert (
                    section.bbox[1] == header_element.bottom
                ), f"'{boundaries}' should start after first element"
                assert (
                    section.bbox[3] == next_header.top
                ), f"Section should always end at next element top"

    print("\n✅ All mock tests passed! include_boundaries parameter is working correctly.")


def test_real_pdf_simple():
    """Test with a real PDF using simple boundaries."""
    from pathlib import Path

    # Use the types PDF which is simpler
    pdf_path = Path(__file__).parent.parent / "pdfs" / "types-of-type.pdf"
    if not pdf_path.exists():
        print(f"Skipping real PDF test - {pdf_path} not found")
        return

    pdf = npdf.PDF(str(pdf_path))

    # Find any text elements
    all_text = pdf.find_all("text")
    if len(all_text) < 2:
        print("Not enough text elements for real PDF test")
        return

    # Use first and last text elements as boundaries
    first_text = all_text[0].extract_text().strip()[:20]

    print(f"\nTesting with real PDF using '{first_text}' as boundary...")

    # Get sections with different boundaries
    sections_both = pdf.get_sections(f"text:contains({first_text})", include_boundaries="both")
    sections_none = pdf.get_sections(f"text:contains({first_text})", include_boundaries="none")

    if len(sections_both) > 0 and len(sections_none) > 0:
        # Compare bounding boxes
        bbox_both = sections_both[0].bbox
        bbox_none = sections_none[0].bbox

        print(f"Section with 'both': {bbox_both}")
        print(f"Section with 'none': {bbox_none}")

        # Basic check - they should be different
        assert (
            bbox_both != bbox_none
        ), "Bounding boxes should be different with different include_boundaries"
        print("✅ Real PDF test passed!")


if __name__ == "__main__":
    test_get_sections_include_boundaries()
    test_real_pdf_simple()
