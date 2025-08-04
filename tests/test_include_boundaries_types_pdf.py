"""
Test include_boundaries with types-of-type.pdf specifically.
"""

from pathlib import Path

import natural_pdf as npdf


def test_with_types_pdf():
    """Test with the types-of-type.pdf file."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "types-of-type.pdf"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    pdf = npdf.PDF(str(pdf_path))
    page = pdf.pages[0]

    print(f"Testing with: {pdf_path.name}")
    print(f"Page size: {page.width} x {page.height}")

    # Get all text elements
    all_text = page.find_all("text")
    print(f"\nFound {len(all_text)} text elements:")

    # Show all text elements
    for i, elem in enumerate(all_text[:10]):
        text = elem.extract_text()
        print(f"  {i}: '{text}' at y={elem.top:.1f}-{elem.bottom:.1f}")

    if len(all_text) >= 3:
        # Test get_section_between
        print("\n" + "=" * 60)
        print("Testing get_section_between")
        print("=" * 60)

        # Use first and third elements to have something in between
        start_elem = all_text[0]
        end_elem = all_text[2]

        print(
            f"\nStart: '{start_elem.extract_text()}' at y={start_elem.top:.1f}-{start_elem.bottom:.1f}"
        )
        print(f"End: '{end_elem.extract_text()}' at y={end_elem.top:.1f}-{end_elem.bottom:.1f}")

        # Test each boundary option
        results = {}
        for boundaries in ["both", "start", "end", "none"]:
            section = page.get_section_between(start_elem, end_elem, include_boundaries=boundaries)

            if section:
                text = section.extract_text()
                results[boundaries] = {
                    "bbox": section.bbox,
                    "text": text,
                    "has_start": start_elem.extract_text() in text,
                    "has_end": end_elem.extract_text() in text,
                }

                print(f"\n{boundaries}:")
                print(f"  Bbox: {section.bbox}")
                print(f"  Top: {section.top:.1f}, Bottom: {section.bottom:.1f}")
                print(f"  Has start text: {results[boundaries]['has_start']}")
                print(f"  Has end text: {results[boundaries]['has_end']}")

        # Verify the behavior
        print("\n" + "=" * 60)
        print("VERIFICATION:")
        print("=" * 60)

        # Check 'both' - should include both
        if "both" in results:
            if results["both"]["has_start"] and results["both"]["has_end"]:
                print("✅ 'both' correctly includes both boundaries")
            else:
                print("❌ 'both' should include both boundaries")

        # Check 'none' - should exclude both
        if "none" in results:
            if not results["none"]["has_start"] and not results["none"]["has_end"]:
                print("✅ 'none' correctly excludes both boundaries")
            else:
                print("❌ 'none' should exclude both boundaries")

        # Check 'start' - should include start, exclude end
        if "start" in results:
            if results["start"]["has_start"] and not results["start"]["has_end"]:
                print("✅ 'start' correctly includes start only")
            else:
                print("❌ 'start' should include start and exclude end")

        # Check 'end' - should exclude start, include end
        if "end" in results:
            if not results["end"]["has_start"] and results["end"]["has_end"]:
                print("✅ 'end' correctly includes end only")
            else:
                print("❌ 'end' should exclude start and include end")

        # Also check that bounding boxes are different
        bboxes = [r["bbox"] for r in results.values()]
        unique_bboxes = set(bboxes)

        print(f"\nUnique bounding boxes: {len(unique_bboxes)} out of {len(bboxes)}")
        if len(unique_bboxes) > 1:
            print("✅ Different include_boundaries produce different regions")
        else:
            print("❌ All include_boundaries produce the same region")


if __name__ == "__main__":
    test_with_types_pdf()
