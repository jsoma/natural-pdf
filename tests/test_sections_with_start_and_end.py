"""
Test get_sections with both start and end elements to ensure the fix
doesn't break normal functionality.
"""

import pytest

from natural_pdf import PDF
from natural_pdf.elements.region import Region
from natural_pdf.flows.region import FlowRegion


def test_sections_with_both_start_and_end():
    """Test normal case with both start and end elements."""
    pdf = PDF("pdfs/2020.pdf")
    pages = pdf.pages[1:4]

    # Find some rectangles to use as boundaries
    all_rects = pages.find_all("rect[fill=#f3f1f1]")

    # Use some as starts, others as ends
    start_elements = [r for i, r in enumerate(all_rects) if i % 3 == 0][:3]
    end_elements = [r for i, r in enumerate(all_rects) if i % 3 == 1][:3]

    print(f"\nUsing {len(start_elements)} start elements and {len(end_elements)} end elements")

    sections = pages.get_sections(start_elements=start_elements, end_elements=end_elements)

    print(f"Created {len(sections)} sections")

    # All sections should be valid
    for i, section in enumerate(sections):
        assert isinstance(section, (Region, FlowRegion))

        if isinstance(section, Region):
            print(f"  Section {i}: Region with height={section.height:.1f}")
            assert section.width > 0
            assert section.height > 0


def test_sections_with_only_starts():
    """Test case with only start elements."""
    pdf = PDF("pdfs/2020.pdf")
    pages = pdf.pages[1:3]

    # Find start elements - use larger rectangles to get different positions
    all_rects = pages.find_all("rect[fill=#f3f1f1][width>200]")

    # If we don't have enough large rects, fall back to any rects
    if len(all_rects) < 3:
        all_rects = pages.find_all("rect[fill=#f3f1f1]")

    start_elements = all_rects[:3]

    print(f"\nUsing {len(start_elements)} start elements (no end elements)")
    for i, elem in enumerate(start_elements):
        print(f"  Start {i}: page={elem.page.index}, top={elem.top}")

    sections = pages.get_sections(start_elements=start_elements)

    print(f"Created {len(sections)} sections")

    # The number of sections depends on the positions of start elements
    # If all starts are at the same position, we might get fewer sections
    assert len(sections) > 0

    for i, section in enumerate(sections):
        if isinstance(section, Region):
            print(f"  Section {i}: height={section.height:.1f}")
            assert section.height > 0


def test_sections_page_boundaries():
    """Test sections with page break handling."""
    pdf = PDF("pdfs/2020.pdf")
    pages = pdf.pages[1:4]

    # Get one element per page
    elements_per_page = []
    for page in pages:
        elem = page.find("rect[fill=#f3f1f1]")
        if elem:
            elements_per_page.append(elem)

    print(f"\nFound elements on {len(elements_per_page)} pages")

    # Test with page breaks
    sections = pages.get_sections(end_elements=elements_per_page, new_section_on_page_break=True)

    print(f"Created {len(sections)} sections with page breaks")

    # Should have at least one section
    assert len(sections) >= 1

    # All sections should have positive height
    for i, section in enumerate(sections):
        if isinstance(section, Region):
            print(f"  Section {i}: height={section.height:.1f}")
            assert section.height > 0


if __name__ == "__main__":
    test_sections_with_both_start_and_end()
    print("\n" + "=" * 60)
    test_sections_with_only_starts()
    print("\n" + "=" * 60)
    test_sections_page_boundaries()
