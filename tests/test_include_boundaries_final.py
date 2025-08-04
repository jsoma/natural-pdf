"""
Final test to verify include_boundaries is working correctly.
"""

from pathlib import Path

import natural_pdf as npdf


def test_include_boundaries():
    """Test that include_boundaries parameter actually affects the section boundaries."""

    # Create a simple test case
    pdfs_dir = Path(__file__).parent.parent / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found")
        return False

    pdf = npdf.PDF(str(pdf_files[0]))
    print(f"\nTesting with: {pdf_files[0].name}")

    # Find any text element to use as a boundary
    all_text = pdf.find_all("text")

    if len(all_text) < 3:
        print("Not enough text elements")
        return False

    # Use first few text elements
    first_text = all_text[0].extract_text().strip()[:30]

    print(f"\nUsing boundary text: '{first_text}'")

    # Get sections with different include_boundaries settings
    try:
        sections_both = pdf.get_sections(f"text:contains({first_text})", include_boundaries="both")
        sections_none = pdf.get_sections(f"text:contains({first_text})", include_boundaries="none")
        sections_start = pdf.get_sections(
            f"text:contains({first_text})", include_boundaries="start"
        )
        sections_end = pdf.get_sections(f"text:contains({first_text})", include_boundaries="end")

        if not sections_both:
            print("No sections found")
            return False

        # Get first section from each
        s_both = sections_both[0]
        s_none = sections_none[0] if sections_none else None
        s_start = sections_start[0] if sections_start else None
        s_end = sections_end[0] if sections_end else None

        print("\nBounding boxes:")
        print(f"  'both':  {s_both.bbox}")
        if s_none:
            print(f"  'none':  {s_none.bbox}")
        if s_start:
            print(f"  'start': {s_start.bbox}")
        if s_end:
            print(f"  'end':   {s_end.bbox}")

        # Check if boundary text is included/excluded as expected
        text_both = s_both.extract_text()

        print(f"\nBoundary text '{first_text}' in section:")
        print(f"  'both':  {first_text in text_both}")

        if s_none:
            text_none = s_none.extract_text()
            print(f"  'none':  {first_text in text_none}")

            # Key test: 'none' should NOT include the boundary text
            if first_text not in text_none:
                print("\n✅ SUCCESS: 'none' correctly excludes boundary text!")
                return True
            else:
                print("\n❌ FAIL: 'none' still includes boundary text")
                return False

        # Also check bbox differences
        if s_both.bbox != s_none.bbox if s_none else True:
            print("\n✅ Bounding boxes are different!")
            return True
        else:
            print("\n❌ Bounding boxes are the same")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_occurrence_example():
    """Test with the specific 'Occurrence' example from the user."""
    # Look for a PDF that might have "Occurrence" text
    pdf_path = Path(__file__).parent.parent / "pdfs" / "24480polcompleted.pdf"

    if not pdf_path.exists():
        print("\nSkipping Occurrence test - PDF not found")
        return

    print("\n" + "=" * 60)
    print("Testing with 'Occurrence' example")
    print("=" * 60)

    pdf = npdf.PDF(str(pdf_path))

    # Try to find Occurrence text
    occurrences = pdf.find_all("text:contains(Occurrence)")

    if occurrences:
        print(f"Found {len(occurrences)} occurrences of 'Occurrence'")

        # Test sections
        sections_both = pdf.get_sections("text:contains(Occurrence)", include_boundaries="both")
        sections_start = pdf.get_sections("text:contains(Occurrence)", include_boundaries="start")
        sections_none = pdf.get_sections("text:contains(Occurrence)", include_boundaries="none")

        print(f"\nSections found:")
        print(f"  'both':  {len(sections_both)}")
        print(f"  'start': {len(sections_start)}")
        print(f"  'none':  {len(sections_none)}")

        if sections_both and sections_none:
            # Check first section
            s1_both = sections_both[0]
            s1_none = sections_none[0]

            text_both = s1_both.extract_text()
            text_none = s1_none.extract_text()

            has_occurrence_both = "Occurrence" in text_both
            has_occurrence_none = "Occurrence" in text_none

            print(f"\n'Occurrence' in first section:")
            print(f"  'both':  {has_occurrence_both}")
            print(f"  'none':  {has_occurrence_none}")

            if has_occurrence_both and not has_occurrence_none:
                print("\n✅ SUCCESS: include_boundaries is working correctly!")
            else:
                print("\n❌ FAIL: include_boundaries not working as expected")
    else:
        print("No 'Occurrence' text found in PDF")


if __name__ == "__main__":
    success = test_include_boundaries()
    test_occurrence_example()

    if success:
        print("\n🎉 include_boundaries parameter is now working correctly!")
    else:
        print("\n⚠️  More investigation needed")
