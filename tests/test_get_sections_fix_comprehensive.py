"""
Comprehensive test for the get_sections zero-height fix.
"""

import pytest

from natural_pdf import PDF
from natural_pdf.elements.region import Region
from natural_pdf.flows.region import FlowRegion


def test_no_zero_height_sections_various_scenarios():
    """
    Test that no zero-height sections are created in various scenarios.
    """
    pdf = PDF("pdfs/2020.pdf")

    test_cases = [
        {
            "name": "End elements only - single page",
            "pages": pdf.pages[1:2],
            "start_elements": None,
            "end_elements": "rect[fill=#f3f1f1]",
        },
        {
            "name": "End elements only - multiple pages",
            "pages": pdf.pages[1:4],
            "start_elements": None,
            "end_elements": "rect[fill=#f3f1f1][width>200]",
        },
        {
            "name": "Start elements only",
            "pages": pdf.pages[1:3],
            "start_elements": "rect[fill=#f3f1f1]",
            "end_elements": None,
        },
        # Skip the 'Both start and end elements' test case for now
        # as it would require more complex element selection
        {
            "name": "With page breaks",
            "pages": pdf.pages[1:4],
            "start_elements": None,
            "end_elements": "rect[fill=#f3f1f1][width>200]",
            "new_section_on_page_break": True,
        },
    ]

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test case: {test_case['name']}")
        print(f"{'='*60}")

        pages = test_case["pages"]
        kwargs = {
            "start_elements": test_case.get("start_elements"),
            "end_elements": test_case.get("end_elements"),
        }
        if "new_section_on_page_break" in test_case:
            kwargs["new_section_on_page_break"] = test_case["new_section_on_page_break"]

        sections = pages.get_sections(**kwargs)

        print(f"Created {len(sections)} sections")

        # Check all sections
        min_height = float("inf")
        zero_height_count = 0

        for i, section in enumerate(sections):
            if isinstance(section, Region):
                height = section.height
                min_height = min(min_height, height)

                if height <= 1.0:  # Consider <= 1.0 as problematic
                    zero_height_count += 1
                    print(f"  ⚠️  Section {i}: height={height:.6f} (TOO SMALL)")
                    print(f"     BBox: {section.bbox}")
                elif i < 3:  # Print first 3 for brevity
                    print(f"  ✓ Section {i}: height={height:.1f}")
            elif isinstance(section, FlowRegion) and i < 3:
                print(f"  ✓ Section {i}: FlowRegion")

        if len(sections) > 3:
            print(f"  ... and {len(sections) - 3} more sections")

        print(f"\nMinimum region height: {min_height:.1f}")

        # Assert no zero-height sections
        assert (
            zero_height_count == 0
        ), f"Found {zero_height_count} sections with height <= 1.0 in '{test_case['name']}'"
        assert (
            min_height > 1.0
        ), f"Minimum height {min_height} is too small in '{test_case['name']}'"


def test_implicit_start_not_paired_with_source_end():
    """
    Specifically test that implicit starts created from an end element
    are not paired with that same end element.
    """
    pdf = PDF("pdfs/2020.pdf")
    page = pdf.pages[1]

    # Get two end elements on the same page
    end_elements = page.find_all("rect[fill=#f3f1f1]")[:2]

    if len(end_elements) >= 2:
        print(f"\nEnd elements:")
        for i, elem in enumerate(end_elements):
            print(f"  End {i}: top={elem.top:.1f}, bottom={elem.bottom:.1f}")

        # Create sections with only end elements
        sections = page.get_sections(end_elements=end_elements)

        print(f"\nSections created: {len(sections)}")

        # With default include_boundaries="start", sections exclude the end boundary
        # So the first section should go from top of page to TOP of first end element
        # There should NOT be a zero-height section at first end

        # Sort end elements like the implementation does
        sorted_ends = sorted(end_elements, key=lambda e: (e.page.index, e.top, e.bottom, e.x0))

        expected_sections = [
            (0, sorted_ends[0].top),  # Top to TOP of first sorted end (exclude end boundary)
            # Second section continues from there - we don't check its end
        ]

        for i, section in enumerate(sections):
            print(f"\nSection {i}:")
            print(f"  Top: {section.top:.1f}")
            print(f"  Bottom: {section.bottom:.1f}")
            print(f"  Height: {section.height:.1f}")

            # Check height is reasonable
            assert section.height > 1.0, f"Section {i} has height {section.height}"

            # Verify approximate bounds
            if i < len(expected_sections):
                exp_top, exp_bottom = expected_sections[i]
                print(f"  Expected: top≈{exp_top:.1f}, bottom≈{exp_bottom:.1f}")

                # Allow some tolerance
                assert abs(section.top - exp_top) < 2.0
                assert abs(section.bottom - exp_bottom) < 2.0


def test_cross_page_sections_no_zero_height():
    """
    Test that cross-page sections don't create zero-height regions.
    """
    pdf = PDF("pdfs/2020.pdf")
    pages = pdf.pages[1:4]

    # Get end elements across pages
    end_elements = []
    for i, page in enumerate(pages):
        elem = page.find("rect[fill=#f3f1f1][width>200]")
        if elem:
            end_elements.append(elem)
            print(f"Page {page.index}: found end element at bottom={elem.bottom:.1f}")

    if len(end_elements) >= 2:
        sections = pages.get_sections(end_elements=end_elements)

        print(f"\nCreated {len(sections)} sections")

        for i, section in enumerate(sections):
            if isinstance(section, Region):
                print(
                    f"  Section {i}: Region on page {section.page.index}, height={section.height:.1f}"
                )
                assert section.height > 1.0
            elif isinstance(section, FlowRegion):
                print(f"  Section {i}: FlowRegion across pages")
                # FlowRegions should also be valid
                assert len(section.constituent_regions) > 0


if __name__ == "__main__":
    test_no_zero_height_sections_various_scenarios()
    print("\n" + "=" * 80 + "\n")
    test_implicit_start_not_paired_with_source_end()
    print("\n" + "=" * 80 + "\n")
    test_cross_page_sections_no_zero_height()
