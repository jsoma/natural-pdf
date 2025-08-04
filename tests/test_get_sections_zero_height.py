"""
Test to identify and reproduce the zero-size region bug in get_sections.

The issue occurs when get_sections creates implicit start elements and
they get paired incorrectly with end elements, resulting in zero-height regions.
"""

from natural_pdf import PDF
from natural_pdf.elements.region import Region
from natural_pdf.flows.region import FlowRegion


def test_zero_height_sections_with_end_only():
    """
    Test that sections created with only end_elements don't have zero height.

    The bug occurs when implicit starts are created at end_elem.bottom and
    then paired with the same end_elem, resulting in zero height.
    """
    pdf = PDF("pdfs/2020.pdf")

    # Use multiple pages to test cross-page scenarios
    pages = pdf.pages[1:4]

    # Find end elements (gray rectangles)
    end_elements = pages.find_all("rect[fill=#f3f1f1][width>200]")

    print(f"\nFound {len(end_elements)} end elements across pages")
    for i, elem in enumerate(end_elements):
        print(f"  End[{i}]: page {elem.page.index}, top={elem.top:.2f}, bottom={elem.bottom:.2f}")

    # Create sections with only end elements
    # This should create implicit starts
    sections = pages.get_sections(end_elements=end_elements)

    print(f"\nCreated {len(sections)} sections")

    # Check for zero or very small height sections
    issues = []
    for i, section in enumerate(sections):
        if isinstance(section, Region):
            print(f"\nSection {i}: Region on page {section.page.index}")
            print(f"  BBox: {section.bbox}")
            print(f"  Height: {section.height:.6f}")

            # Check for suspiciously small height (1 or less)
            # Implicit starts add 1 to the bottom coordinate
            if section.height <= 1.0:
                issues.append(
                    {
                        "index": i,
                        "height": section.height,
                        "bbox": section.bbox,
                        "page": section.page.index,
                    }
                )
                print(f"  ⚠️  ISSUE: Very small height detected!")

                # Debug info
                if hasattr(section, "start_element"):
                    start = section.start_element
                    if hasattr(start, "top"):
                        print(f"  Start element top: {start.top}")
                    if hasattr(start, "is_implicit_start"):
                        print(f"  Start is implicit: {start.is_implicit_start}")

                if hasattr(section, "boundary_element_found"):
                    end = section.boundary_element_found
                    if hasattr(end, "bottom"):
                        print(f"  End element bottom: {end.bottom}")

        elif isinstance(section, FlowRegion):
            print(f"\nSection {i}: FlowRegion spanning multiple pages")

    # Report findings
    if issues:
        print(f"\n❌ FOUND {len(issues)} sections with height <= 1.0:")
        for issue in issues:
            print(
                f"  Section {issue['index']}: height={issue['height']:.6f} on page {issue['page']}"
            )
    else:
        print(f"\n✅ All sections have reasonable height")

    # Test should fail if we find any very small sections
    assert len(issues) == 0, f"Found {len(issues)} sections with height <= 1.0"


def test_manual_implicit_start_logic():
    """
    Manually test the implicit start creation logic to understand the bug.
    """
    pdf = PDF("pdfs/2020.pdf")
    pages = pdf.pages[1:3]

    # Get end elements
    end_elements = []
    for page in pages:
        rect = page.find("rect[fill=#f3f1f1][width>200]")
        if rect:
            end_elements.append(rect)

    if len(end_elements) >= 2:
        print("\nEnd elements:")
        for i, elem in enumerate(end_elements):
            print(f"  End[{i}]: page {elem.page.index}, bottom={elem.bottom}")

        # Simulate implicit start creation (from PageCollection.get_sections lines 538-545)
        print("\nImplicit starts that would be created:")

        # First implicit start at beginning of first page
        print(f"  Start[0]: page {pages[0].index}, top=0")

        # For each end element except the last
        sorted_ends = sorted(end_elements, key=lambda e: (e.page.index, e.top))
        for i, end_elem in enumerate(sorted_ends[:-1]):
            # Implicit start is created at bottom of this end element
            print(f"  Start[{i+1}]: page {end_elem.page.index}, top={end_elem.bottom}")

            # Check if this could cause a problem
            # If the next section uses this same end_elem as its end...
            if i == 0:  # First end element
                print(
                    f"    -> If paired with End[0] (bottom={end_elem.bottom}), height would be ~0!"
                )

        # Now run actual get_sections
        sections = pages.get_sections(end_elements=end_elements)

        print(f"\nActual sections created: {len(sections)}")
        for i, section in enumerate(sections):
            if isinstance(section, Region):
                print(f"  Section {i}: height={section.height:.2f}")


def test_same_page_zero_height():
    """
    Test a specific case where start and end are on the same page
    but might create zero height.
    """
    pdf = PDF("pdfs/2020.pdf")
    page = pdf.pages[1]

    # Get all gray rectangles on this page
    rects = page.find_all("rect[fill=#f3f1f1]")

    if len(rects) >= 2:
        # Create a scenario similar to implicit starts
        # Start at bottom of first rect, end at same rect
        start_y = rects[0].bottom

        print(f"\nFirst rect bottom: {rects[0].bottom}")
        print(f"Creating region from y={start_y} to rect bottom={rects[0].bottom}")

        # This would create zero height
        test_region = Region(page, (0, start_y, page.width, rects[0].bottom))
        print(f"Test region height: {test_region.height}")

        assert test_region.height == 0, "This should be zero height"


if __name__ == "__main__":
    print("=" * 60)
    print("Testing zero-height sections with end_only...")
    print("=" * 60)
    try:
        test_zero_height_sections_with_end_only()
    except AssertionError as e:
        print(f"\n⚠️  Test failed as expected: {e}")

    print("\n" + "=" * 60)
    print("Testing manual implicit start logic...")
    print("=" * 60)
    test_manual_implicit_start_logic()

    print("\n" + "=" * 60)
    print("Testing same page zero height...")
    print("=" * 60)
    test_same_page_zero_height()
