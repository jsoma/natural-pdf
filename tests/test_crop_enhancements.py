"""
Test the enhanced crop functionality for show() method.
"""

from pathlib import Path

import natural_pdf as npdf


def test_crop_modes():
    """Test different crop modes with various elements."""
    # Find a test PDF
    pdfs_dir = Path(__file__).parent.parent / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found for testing")
        return

    pdf = npdf.PDF(str(pdf_files[0]))
    page = pdf.pages[0]

    print(f"Testing crop modes with: {pdf_files[0].name}")
    print(f"Page size: {page.width} x {page.height}")

    # Find some text elements
    text_elements = page.find_all("text")

    if not text_elements:
        print("No text elements found")
        return

    # Get first text element for testing
    element = text_elements[0]
    print(f"\nTest element: '{element.extract_text()}'")
    print(f"Element bbox: {element.bbox}")

    # Test 1: Tight crop (default True)
    print("\n1. Testing crop=True (tight crop)")
    try:
        img = element.show(crop=True)
        if img:
            print(f"   ✓ Success: Image size {img.size}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 2: Crop with padding
    print("\n2. Testing crop=50 (50px padding)")
    try:
        img = element.show(crop=50)
        if img:
            print(f"   ✓ Success: Image size {img.size}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 3: Wide crop
    print("\n3. Testing crop='wide' (full page width)")
    try:
        img = element.show(crop="wide")
        if img:
            print(f"   ✓ Success: Image size {img.size}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 4: Crop to another region
    if len(text_elements) > 1:
        print("\n4. Testing crop=region (crop to another element's bounds)")
        try:
            other_element = text_elements[1]
            img = element.show(crop=other_element)
            if img:
                print(f"   ✓ Success: Image size {img.size}")
        except Exception as e:
            print(f"   ✗ Error: {e}")

    # Test with regions
    print("\n5. Testing with regions")
    region = page.region(50, 50, 200, 200)

    # Region with padding
    try:
        img = region.show(crop=30)  # 30px padding
        if img:
            print(f"   ✓ Region with padding: Image size {img.size}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test with element collection
    if len(text_elements) >= 3:
        print("\n6. Testing with ElementCollection")
        collection = text_elements[:3]

        try:
            # Collection with wide crop
            img = collection.show(crop="wide")
            if img:
                print(f"   ✓ Collection wide crop: Image size {img.size}")
        except Exception as e:
            print(f"   ✗ Error: {e}")


def test_crop_visual_comparison():
    """Create visual comparison of different crop modes."""
    pdfs_dir = Path(__file__).parent.parent / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        return

    pdf = npdf.PDF(str(pdf_files[0]))

    # Find an element with some content around it
    elements = pdf.find_all("text")

    if not elements:
        return

    # Use middle element to have content above and below
    mid_idx = len(elements) // 2
    element = elements[mid_idx]

    print("\nVisual comparison of crop modes:")
    print("=" * 50)

    # Save examples
    output_dir = Path("temp")
    output_dir.mkdir(exist_ok=True)

    modes = [
        (True, "tight"),
        (50, "padding_50"),
        (100, "padding_100"),
        ("wide", "wide"),
    ]

    for crop_mode, name in modes:
        try:
            img = element.render(crop=crop_mode)
            if img:
                path = output_dir / f"crop_{name}.png"
                img.save(path)
                print(f"✓ Saved {name}: {path}")
        except Exception as e:
            print(f"✗ Error with {name}: {e}")


if __name__ == "__main__":
    test_crop_modes()
    test_crop_visual_comparison()
