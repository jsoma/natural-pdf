"""
Verify that include_boundaries works by creating a controlled test scenario.
"""

from pathlib import Path

import natural_pdf as npdf


def create_test_text():
    """Create a simple test document to verify include_boundaries."""
    # Find any PDF
    pdfs_dir = Path(__file__).parent.parent / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found")
        return

    # Use the first PDF
    pdf = npdf.PDF(str(pdf_files[0]))
    page = pdf.pages[0]

    print(f"Using {pdf_files[0].name}")
    print(f"Page size: {page.width} x {page.height}")

    # Get all text elements
    all_text = page.find_all("text")
    print(f"Found {len(all_text)} text elements")

    if len(all_text) < 2:
        print("Not enough text for testing")
        return

    # Show the first few elements and their positions
    print("\nText elements and their positions:")
    for i, elem in enumerate(all_text[:5]):
        text = elem.extract_text().strip()[:50]
        print(f"  {i}: '{text}' at y={elem.top:.1f}-{elem.bottom:.1f}")

    # Test get_section_between directly
    if len(all_text) >= 2:
        print("\n" + "=" * 60)
        print("Testing get_section_between with first two elements")
        print("=" * 60)

        elem1 = all_text[0]
        elem2 = all_text[1]

        print(
            f"\nElement 1: '{elem1.extract_text().strip()[:30]}' at y={elem1.top:.1f}-{elem1.bottom:.1f}"
        )
        print(
            f"Element 2: '{elem2.extract_text().strip()[:30]}' at y={elem2.top:.1f}-{elem2.bottom:.1f}"
        )

        # Test each include_boundaries option
        for boundaries in ["both", "start", "end", "none"]:
            section = page.get_section_between(elem1, elem2, include_boundaries=boundaries)

            if section:
                print(f"\ninclude_boundaries='{boundaries}':")
                print(f"  Section bbox: {section.bbox}")
                print(f"  Top: {section.top:.1f}, Bottom: {section.bottom:.1f}")

                # Expected behavior:
                if boundaries == "both":
                    expected_top = elem1.top
                    expected_bottom = elem2.bottom
                elif boundaries == "start":
                    expected_top = elem1.top
                    expected_bottom = elem2.top
                elif boundaries == "end":
                    expected_top = elem1.bottom
                    expected_bottom = elem2.bottom
                else:  # 'none'
                    expected_top = elem1.bottom
                    expected_bottom = elem2.top

                print(f"  Expected: top={expected_top:.1f}, bottom={expected_bottom:.1f}")

                # Check if it matches
                if (
                    abs(section.top - expected_top) < 1
                    and abs(section.bottom - expected_bottom) < 1
                ):
                    print("  ✅ Correct!")
                else:
                    print("  ❌ Incorrect boundaries")

    # Now test get_sections with a selector
    print("\n" + "=" * 60)
    print("Testing get_sections with selector")
    print("=" * 60)

    # Use any text that appears
    if all_text:
        test_text = all_text[0].extract_text().strip()[:20]
        print(f"\nUsing selector: 'text:contains({test_text})'")

        for boundaries in ["both", "start", "end", "none"]:
            try:
                sections = pdf.get_sections(
                    f"text:contains({test_text})", include_boundaries=boundaries
                )

                if sections:
                    section = sections[0]
                    text_content = section.extract_text()
                    has_boundary_text = test_text in text_content

                    print(f"\ninclude_boundaries='{boundaries}':")
                    print(f"  Sections found: {len(sections)}")
                    print(f"  First section bbox: {section.bbox}")
                    print(f"  Contains boundary text: {has_boundary_text}")

                    # Expected behavior for 'none' and 'end' - should NOT contain start boundary
                    if boundaries in ["none", "end"]:
                        if not has_boundary_text:
                            print("  ✅ Correctly excludes start boundary")
                        else:
                            print("  ❌ Should exclude start boundary")
                    else:  # 'both' and 'start'
                        if has_boundary_text:
                            print("  ✅ Correctly includes start boundary")
                        else:
                            print("  ❌ Should include start boundary")

            except Exception as e:
                print(f"\nError with '{boundaries}': {e}")


if __name__ == "__main__":
    create_test_text()
