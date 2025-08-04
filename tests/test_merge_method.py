"""
Test the merge() method for ElementCollection.
"""

from pathlib import Path

import pytest

import natural_pdf as npdf
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region


def test_merge_basic():
    """Test basic merge functionality."""
    # Find a test PDF
    pdfs_dir = Path(__file__).parent.parent / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        pytest.skip("No PDF files found for testing")

    pdf = npdf.PDF(str(pdf_files[0]))
    page = pdf.pages[0]

    # Get some text elements
    elements = page.find_all("text")

    if len(elements) < 1:
        pytest.skip("No elements for merge test")

    # Test merging elements (even if just one)
    collection = elements[: min(3, len(elements))]
    merged = collection.merge()

    # Verify it returns a Region
    from natural_pdf.elements.region import Region

    assert isinstance(merged, Region), "merge() should return a Region"

    # Verify the bbox encompasses all elements
    for elem in collection:
        assert merged.x0 <= elem.x0, "Merged region should encompass element's left"
        assert merged.x1 >= elem.x1, "Merged region should encompass element's right"
        assert merged.top <= elem.top, "Merged region should encompass element's top"
        assert merged.bottom >= elem.bottom, "Merged region should encompass element's bottom"

    print(f"✓ Basic merge test passed")


def test_merge_scattered_elements():
    """Test merging non-adjacent elements."""
    pdfs_dir = Path(__file__).parent.parent / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        pytest.skip("No PDF files found")

    pdf = npdf.PDF(str(pdf_files[0]))
    page = pdf.pages[0]

    # Get all text elements
    all_elements = page.find_all("text")

    if len(all_elements) < 3:
        pytest.skip("Not enough elements")

    # Pick scattered elements (first, middle, last)
    scattered = ElementCollection(
        [all_elements[0], all_elements[len(all_elements) // 2], all_elements[-1]]
    )

    merged = scattered.merge()

    # The merged region should span from first to last element
    assert merged.x0 <= min(e.x0 for e in scattered)
    assert merged.x1 >= max(e.x1 for e in scattered)
    assert merged.top <= min(e.top for e in scattered)
    assert merged.bottom >= max(e.bottom for e in scattered)

    print(f"✓ Scattered elements merge test passed")


def test_merge_errors():
    """Test error conditions for merge."""
    # Empty collection
    empty = ElementCollection([])

    with pytest.raises(ValueError, match="empty"):
        empty.merge()

    print(f"✓ Error handling test passed")


def test_merge_vs_dissolve():
    """Compare merge() vs dissolve() behavior."""
    pdfs_dir = Path(__file__).parent.parent / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        pytest.skip("No PDF files found")

    pdf = npdf.PDF(str(pdf_files[0]))

    # Find elements that are NOT connected
    elements = pdf.find_all("text")

    if len(elements) < 2:
        pytest.skip("Not enough elements")

    # Take first and last element (likely not connected)
    collection = ElementCollection([elements[0], elements[-1]])

    # Merge should create one region
    merged = collection.merge()
    assert isinstance(merged, Region)

    # Dissolve might create multiple regions if not connected
    dissolved = collection.dissolve()

    # Merged region should encompass all dissolved regions
    if isinstance(dissolved, ElementCollection):
        for region in dissolved:
            assert merged.x0 <= region.x0
            assert merged.x1 >= region.x1
            assert merged.top <= region.top
            assert merged.bottom >= region.bottom

    print(f"✓ Merge vs dissolve comparison passed")


def test_merge_visual():
    """Visual test showing merge behavior."""
    pdfs_dir = Path(__file__).parent.parent / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        return

    pdf = npdf.PDF(str(pdf_files[0]))

    # Find some scattered text
    matches = pdf.find_all("text")

    if len(matches) < 3:
        return

    # Take every other element to ensure gaps
    scattered = ElementCollection([matches[i] for i in range(0, len(matches), 2)])

    print("\nVisual merge demonstration:")
    print(f"Merging {len(scattered)} scattered elements")

    # Show individual elements
    print("\nIndividual elements:")
    for i, elem in enumerate(scattered[:5]):  # Show first 5
        print(f"  {i}: '{elem.extract_text()[:20]}...' at {elem.bbox}")

    # Merge them
    merged = scattered.merge()

    print(f"\nMerged region bbox: {merged.bbox}")
    print(f"Merged region size: {merged.width} x {merged.height}")

    # Visual comparison
    output_dir = Path("temp")
    output_dir.mkdir(exist_ok=True)

    # Show scattered elements
    img1 = scattered.show(color="red")
    if img1:
        img1.save(output_dir / "merge_before.png")
        print(f"✓ Saved scattered elements: temp/merge_before.png")

    # Show merged region
    img2 = merged.show(color="blue")
    if img2:
        img2.save(output_dir / "merge_after.png")
        print(f"✓ Saved merged region: temp/merge_after.png")


if __name__ == "__main__":
    test_merge_basic()
    test_merge_scattered_elements()
    test_merge_errors()
    test_merge_vs_dissolve()
    test_merge_visual()
