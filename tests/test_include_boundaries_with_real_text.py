"""
Test include_boundaries with actual text content.
"""

from pathlib import Path

import natural_pdf as npdf


def test_with_real_text():
    """Test with text that has actual content."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "types-of-type.pdf"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    pdf = npdf.PDF(str(pdf_path))

    # Use text-based sections
    print("Testing get_sections with text selectors")
    print("=" * 60)

    # Try with "Normal text" as boundary
    selector = "text:contains(Normal)"

    print(f"\nUsing selector: '{selector}'")

    # Test all boundary options
    for boundaries in ["both", "start", "end", "none"]:
        sections = pdf.get_sections(selector, include_boundaries=boundaries)

        if sections:
            print(f"\n{boundaries}:")
            print(f"  Found {len(sections)} sections")

            section = sections[0]
            print(f"  Section bbox: {section.bbox}")

            # Extract and check text
            text = section.extract_text()
            has_normal = "Normal" in text

            print(f"  Contains 'Normal': {has_normal}")
            print(f"  Text preview: '{text.strip()[:100]}'...")

            # For 'none' and 'end', should NOT have "Normal"
            # For 'both' and 'start', should have "Normal"
            expected = boundaries in ["both", "start"]
            if has_normal == expected:
                print(f"  ✅ Correct!")
            else:
                print(f"  ❌ Expected {'to include' if expected else 'NOT to include'} 'Normal'")

    # Also test with a different approach - using page sections
    print("\n" + "=" * 60)
    print("Testing with multiple text boundaries")
    print("=" * 60)

    # Find all non-empty text
    all_text = pdf.find_all("text")
    non_empty = [t for t in all_text if t.extract_text().strip()]

    if len(non_empty) >= 2:
        # Use "Bold" and "Highlighted" as boundaries
        bold_text = [t for t in non_empty if "Bold" in t.extract_text()]
        highlighted_text = [t for t in non_empty if "Highlighted" in t.extract_text()]

        if bold_text and highlighted_text:
            print(f"\nUsing 'Bold' as start and 'Highlighted' as end")

            for boundaries in ["both", "none"]:
                sections = pdf.get_sections(
                    start_elements="text:contains(Bold)",
                    end_elements="text:contains(Highlighted)",
                    include_boundaries=boundaries,
                )

                if sections:
                    section = sections[0]
                    text = section.extract_text()

                    has_bold = "Bold" in text
                    has_highlighted = "Highlighted" in text

                    print(f"\n{boundaries}:")
                    print(f"  Has 'Bold': {has_bold}")
                    print(f"  Has 'Highlighted': {has_highlighted}")
                    print(f"  Section text: '{text.strip()}'")

                    if boundaries == "both":
                        if has_bold and has_highlighted:
                            print("  ✅ Correctly includes both boundaries")
                        else:
                            print("  ❌ Should include both boundaries")
                    elif boundaries == "none":
                        if not has_bold and not has_highlighted:
                            print("  ✅ Correctly excludes both boundaries")
                        else:
                            print("  ❌ Should exclude both boundaries")


if __name__ == "__main__":
    test_with_real_text()
