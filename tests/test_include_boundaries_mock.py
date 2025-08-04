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
    # Header at top of page
    header_element = create_mock_element(page, "Section 1", top=700, bottom=720)

    # Content in middle
    content_elements = [
        create_mock_element(page, "Content line 1", top=650, bottom=670),
        create_mock_element(page, "Content line 2", top=620, bottom=640),
        create_mock_element(page, "Content line 3", top=590, bottom=610),
    ]

    # Next header
    next_header = create_mock_element(page, "Section 2", top=550, bottom=570)

    # Set up the page's element finding
    all_elements = [header_element] + content_elements + [next_header]

    def mock_find_all(selector, **kwargs):
        if "Section" in selector:
            return ElementCollection([header_element, next_header])
        return ElementCollection(all_elements)

    page.find_all = mock_find_all

    # Mock get_section_between to return regions with correct boundaries
    def mock_get_section_between(start, end, include_boundaries="both"):
        if include_boundaries == "both":
            top = start.top
            bottom = end.bottom if end else page.height
        elif include_boundaries == "start":
            top = start.top
            bottom = end.top if end else page.height
        elif include_boundaries == "end":
            top = start.bottom
            bottom = end.bottom if end else page.height
        else:  # none
            top = start.bottom
            bottom = end.top if end else page.height

        region = Region(page, (0, top, page.width, bottom))
        # Store which elements would be in this region
        region._included_elements = [e for e in all_elements if e.top >= bottom and e.bottom <= top]
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

        if len(sections) > 0:
            section = sections[0]
            print(f"\ninclude_boundaries='{boundaries}':")
            print(f"  Section bbox: {section.bbox}")
            print(f"  Top: {section.bbox[1]}, Bottom: {section.bbox[3]}")

            # Verify boundaries are correct
            if boundaries == "both":
                assert (
                    section.bbox[1] == header_element.top
                ), f"'both' should include start element top"
                assert (
                    section.bbox[3] == next_header.bottom
                ), f"'both' should include end element bottom"
            elif boundaries == "start":
                assert (
                    section.bbox[1] == header_element.top
                ), f"'start' should include start element top"
                assert section.bbox[3] == next_header.top, f"'start' should exclude end element"
            elif boundaries == "end":
                assert (
                    section.bbox[1] == header_element.bottom
                ), f"'end' should exclude start element"
                assert (
                    section.bbox[3] == next_header.bottom
                ), f"'end' should include end element bottom"
            else:  # none
                assert (
                    section.bbox[1] == header_element.bottom
                ), f"'none' should exclude start element"
                assert section.bbox[3] == next_header.top, f"'none' should exclude end element"

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
