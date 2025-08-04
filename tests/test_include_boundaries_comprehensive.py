"""
Comprehensive test for include_boundaries parameter in get_sections.
"""

from pathlib import Path

import natural_pdf as npdf

# Get path to test PDF
TEST_PDF_PATH = Path(__file__).parent.parent / "pdfs" / "24480polcompleted.pdf"


def test_include_boundaries_comprehensive():
    """Test that include_boundaries parameter works with a real-world PDF."""
    pdf = npdf.PDF(str(TEST_PDF_PATH))

    # Use "Occurrence" as the section marker (from user's example)
    print("\nTesting with 'Occurrence' sections...")

    # Get sections with different boundary settings
    sections_both = pdf.get_sections("text:contains(Occurrence)", include_boundaries="both")
    sections_start = pdf.get_sections("text:contains(Occurrence)", include_boundaries="start")
    sections_end = pdf.get_sections("text:contains(Occurrence)", include_boundaries="end")
    sections_none = pdf.get_sections("text:contains(Occurrence)", include_boundaries="none")

    print(f"Found {len(sections_both)} sections")

    if len(sections_both) > 0:
        # Compare the first section with different settings
        section_both = sections_both[0]
        section_start = sections_start[0]
        section_end = sections_end[0]
        section_none = sections_none[0]

        # Extract text from each
        text_both = section_both.extract_text()
        text_start = section_start.extract_text()
        text_end = section_end.extract_text()
        text_none = section_none.extract_text()

        # Check if "Occurrence" appears in the text
        has_occurrence_both = "Occurrence" in text_both
        has_occurrence_start = "Occurrence" in text_start
        has_occurrence_end = "Occurrence" in text_end
        has_occurrence_none = "Occurrence" in text_none

        print(f"\n'both' includes 'Occurrence': {has_occurrence_both}")
        print(f"'start' includes 'Occurrence': {has_occurrence_start}")
        print(f"'end' includes 'Occurrence': {has_occurrence_end}")
        print(f"'none' includes 'Occurrence': {has_occurrence_none}")

        # Verify expected behavior
        assert has_occurrence_both, "'both' should include the boundary text"
        assert has_occurrence_start, "'start' should include the start boundary"
        assert not has_occurrence_end, "'end' should NOT include the start boundary"
        assert not has_occurrence_none, "'none' should NOT include any boundary"

        # Compare text lengths
        print(f"\nText lengths:")
        print(f"  'both': {len(text_both)} chars")
        print(f"  'start': {len(text_start)} chars")
        print(f"  'end': {len(text_end)} chars")
        print(f"  'none': {len(text_none)} chars")

        # Verify that they're different
        assert len(text_both) > len(text_none), "'both' should have more text than 'none'"
        assert len(text_start) > len(
            text_end
        ), "'start' should have more text than 'end' (includes start boundary)"

        # Check bounding boxes
        print(f"\nBounding boxes:")
        print(f"  'both': {section_both.bbox}")
        print(f"  'start': {section_start.bbox}")
        print(f"  'end': {section_end.bbox}")
        print(f"  'none': {section_none.bbox}")

        # In PDF coordinates, smaller top means higher on page
        # 'none' should exclude both boundaries, so it should start lower and end higher
        assert section_none.bbox[1] < section_both.bbox[1], "'none' should start lower than 'both'"
        assert section_none.bbox[3] > section_both.bbox[3], "'none' should end higher than 'both'"

        print("\n✅ All assertions passed! include_boundaries is working correctly.")
    else:
        print("⚠️  No 'Occurrence' sections found in the test PDF")


def test_visual_comparison():
    """Create visual comparison of different include_boundaries settings."""
    pdf = npdf.PDF(str(TEST_PDF_PATH))

    # Get sections with different settings
    sections_both = pdf.get_sections("text:contains(Occurrence)", include_boundaries="both")
    sections_start = pdf.get_sections("text:contains(Occurrence)", include_boundaries="start")
    sections_end = pdf.get_sections("text:contains(Occurrence)", include_boundaries="end")
    sections_none = pdf.get_sections("text:contains(Occurrence)", include_boundaries="none")

    print("\nVisual comparison of include_boundaries settings:")
    print("=" * 60)

    if len(sections_both) > 0:
        print("Showing first section with each setting...")

        # Show each with a different highlight color
        print("\n1. include_boundaries='both' (RED):")
        sections_both[0].highlight(color="red")

        print("\n2. include_boundaries='start' (BLUE):")
        sections_start[0].highlight(color="blue")

        print("\n3. include_boundaries='end' (GREEN):")
        sections_end[0].highlight(color="green")

        print("\n4. include_boundaries='none' (YELLOW):")
        sections_none[0].highlight(color="yellow")

        print("\nCheck the PDF to see the different highlighted regions!")
        print("The highlighted areas should be different sizes based on boundary inclusion.")


if __name__ == "__main__":
    test_include_boundaries_comprehensive()
    print("\n" + "=" * 60)
    test_visual_comparison()
