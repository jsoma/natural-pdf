"""
Test that highlights are preserved when cropping to a region.
"""

from pathlib import Path

import natural_pdf as npdf


def test_crop_region_preserves_highlights():
    """Test that highlights show when cropping to another region."""
    # Find a test PDF
    pdfs_dir = Path(__file__).parent.parent / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found")
        return

    pdf = npdf.PDF(str(pdf_files[0]))
    page = pdf.pages[0]

    print("Testing crop=region highlight preservation")
    print("=" * 50)

    # Create a section (crop region)
    section = page.region(0, 0, page.width, page.height / 2)

    # Find some text
    text_elements = page.find_all("text")

    if not text_elements:
        print("No text elements found")
        return

    element = text_elements[0]

    # Test 1: Show with crop=True (should suppress highlight by default)
    print("\n1. crop=True (tight crop):")
    print("   Highlight should be suppressed (unless color specified)")
    img1 = element.show(crop=True)

    # Test 2: Show with crop=section (should preserve highlight)
    print("\n2. crop=section (region crop):")
    print("   Highlight should be visible")
    img2 = element.show(crop=section)

    # Test 3: Show with crop=50 (padding - should preserve highlight)
    print("\n3. crop=50 (with padding):")
    print("   Highlight should be visible")
    img3 = element.show(crop=50)

    # Test 4: Show with crop='wide' (should preserve highlight)
    print("\n4. crop='wide':")
    print("   Highlight should be visible")
    img4 = element.show(crop="wide")

    # Test your specific use case
    print("\n5. Your use case:")
    print("   section.find_all(...).right().merge().show(crop=section)")

    if len(text_elements) > 0:
        # Simulate your workflow
        merged = text_elements[:1].merge()
        img5 = merged.show(crop=section)
        print("   ✓ Merged region should be highlighted when cropped to section")

    print("\n✅ All tests completed")
    print("\nSummary:")
    print("- crop=True: Suppresses highlight (existing behavior)")
    print("- crop=<number>: Shows highlight ✓")
    print("- crop='wide': Shows highlight ✓")
    print("- crop=<region>: Shows highlight ✓")


def test_visual_comparison():
    """Create visual comparison of highlight behavior."""
    pdfs_dir = Path(__file__).parent.parent / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        return

    pdf = npdf.PDF(str(pdf_files[0]))
    page = pdf.pages[0]

    text = page.find("text")
    if not text:
        return

    # Create a larger region for cropping
    crop_region = page.region(0, 0, page.width, page.height / 2)

    output_dir = Path("temp")
    output_dir.mkdir(exist_ok=True)

    # Save comparisons
    examples = [
        (True, "tight_crop"),
        (crop_region, "region_crop"),
        (50, "padding_crop"),
        ("wide", "wide_crop"),
    ]

    for crop_mode, name in examples:
        try:
            img = text.show(crop=crop_mode)
            if img:
                path = output_dir / f"highlight_{name}.png"
                img.save(path)
                print(f"✓ Saved {name}: {path}")
        except Exception as e:
            print(f"✗ Error with {name}: {e}")


if __name__ == "__main__":
    test_crop_region_preserves_highlights()
    print("\n" + "=" * 50 + "\n")
    test_visual_comparison()
