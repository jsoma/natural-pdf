"""
Test to verify the fix for zero-height sections in get_sections.
"""

from natural_pdf import PDF
from natural_pdf.elements.region import Region


def test_get_sections_with_end_only_fixed():
    """
    Test that demonstrates the fix for zero-height sections.

    The issue: When using only end_elements, implicit starts are created
    at end_elem.bottom, but then might be paired with the same end_elem,
    creating zero-height regions.

    The fix: Ensure implicit starts created after an end element are not
    paired with that same end element.
    """
    pdf = PDF("pdfs/2020.pdf")
    pages = pdf.pages[1:4]

    # Get end elements
    end_elements = pages.find_all("rect[fill=#f3f1f1][width>200]")

    print(f"\nEnd elements found: {len(end_elements)}")
    for i, elem in enumerate(end_elements):
        print(f"  End[{i}]: page={elem.page.index}, top={elem.top:.1f}, bottom={elem.bottom:.1f}")

    # This should create sections without zero height
    sections = pages.get_sections(end_elements=end_elements)

    print(f"\nSections created: {len(sections)}")

    # Verify no zero-height sections
    min_height = float("inf")
    for i, section in enumerate(sections):
        if isinstance(section, Region):
            print(f"  Section[{i}]: page={section.page.index}, height={section.height:.1f}")
            min_height = min(min_height, section.height)

            # Debug info for small sections
            if section.height < 10:
                print(f"    Small section detected!")
                print(f"    BBox: {section.bbox}")
                if hasattr(section, "start_element"):
                    print(f"    Start element: {section.start_element}")
                if hasattr(section, "boundary_element_found"):
                    print(f"    End element: {section.boundary_element_found}")

    print(f"\nMinimum section height: {min_height:.1f}")

    # With the fix, all sections should have reasonable height
    assert min_height > 1.0, f"Found section with height {min_height}"


def test_edge_case_single_end_element():
    """
    Test edge case with a single end element.
    """
    pdf = PDF("pdfs/2020.pdf")
    page = pdf.pages[1]

    # Get just one end element
    end_elem = page.find("rect[fill=#f3f1f1][width>200]")

    if end_elem:
        print(f"\nSingle end element: bottom={end_elem.bottom}")

        # Create sections with single end element
        # When using only end elements, we typically want to include the end boundary
        sections = page.get_sections(end_elements=[end_elem], include_boundaries="end")

        print(f"Sections created: {len(sections)}")

        # Should create one section from top of page to end element
        assert len(sections) == 1
        section = sections[0]

        print(f"Section height: {section.height}")
        print(f"Expected height: {end_elem.bottom}")

        # Height should be approximately end_elem.bottom (from top of page)
        # Allow for small rounding differences
        assert abs(section.height - end_elem.bottom) <= 1.0


def test_mixed_start_end_elements():
    """
    Test with both start and end elements to ensure fix doesn't break normal case.
    """
    pdf = PDF("pdfs/2020.pdf")
    pages = pdf.pages[1:3]

    # Get some rectangles as boundaries
    all_rects = pages.find_all("rect[fill=#f3f1f1]")[:4]

    if len(all_rects) >= 4:
        # Use alternating as start/end
        start_elements = [all_rects[0], all_rects[2]]
        end_elements = [all_rects[1], all_rects[3]]

        print(f"\nStart elements: {len(start_elements)}")
        print(f"End elements: {len(end_elements)}")

        sections = pages.get_sections(start_elements=start_elements, end_elements=end_elements)

        print(f"Sections created: {len(sections)}")

        # All sections should have positive height
        for i, section in enumerate(sections):
            if isinstance(section, Region):
                print(f"  Section[{i}]: height={section.height:.1f}")
                assert section.height > 0, f"Section {i} has zero height"


if __name__ == "__main__":
    test_get_sections_with_end_only_fixed()
    print("\n" + "=" * 60)
    test_edge_case_single_end_element()
    print("\n" + "=" * 60)
    test_mixed_start_end_elements()
