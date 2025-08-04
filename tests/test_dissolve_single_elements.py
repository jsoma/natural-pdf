"""Test to verify that dissolve() preserves single elements as regions."""

import pytest

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement


class MockPage:
    """Mock page for testing."""

    def __init__(self, page_number=1):
        self.page_number = page_number
        self.width = 612
        self.height = 792


def test_dissolve_preserves_single_elements():
    """Test that single elements that don't connect are preserved as regions."""
    # Create a mock page
    page = MockPage()

    # Create three elements that are far apart (won't connect)
    elem1 = TextElement(
        {"x0": 10, "top": 10, "x1": 50, "bottom": 30, "text": "Element 1", "size": 12.0}, page
    )

    elem2 = TextElement(
        {"x0": 200, "top": 10, "x1": 250, "bottom": 30, "text": "Element 2", "size": 12.0}, page
    )

    elem3 = TextElement(
        {"x0": 10, "top": 200, "x1": 50, "bottom": 220, "text": "Element 3", "size": 12.0}, page
    )

    # Create collection
    collection = ElementCollection([elem1, elem2, elem3])

    # Dissolve with small padding (elements won't connect)
    dissolved = collection.dissolve(padding=2.0)

    # Verify all elements are preserved as regions
    assert len(dissolved) == 3, f"Expected 3 regions, got {len(dissolved)}"
    assert all(isinstance(r, Region) for r in dissolved), "All results should be regions"

    # Verify bboxes are preserved
    bboxes = [r.bbox for r in dissolved]
    assert (10, 10, 50, 30) in bboxes
    assert (200, 10, 250, 30) in bboxes
    assert (10, 200, 50, 220) in bboxes


def test_dissolve_single_element_collection():
    """Test dissolve on a collection with only one element."""
    page = MockPage()

    elem = TextElement(
        {"x0": 10, "top": 10, "x1": 50, "bottom": 30, "text": "Single Element", "size": 12.0}, page
    )

    collection = ElementCollection([elem])
    dissolved = collection.dissolve(padding=2.0)

    assert len(dissolved) == 1, f"Expected 1 region, got {len(dissolved)}"
    assert isinstance(dissolved[0], Region), "Result should be a region"
    assert dissolved[0].bbox == (10, 10, 50, 30), "Bbox should be preserved"
    assert "Dissolved (1 text)" in dissolved[0].label


def test_dissolve_mixed_connected_and_single():
    """Test dissolve with some connected elements and some singles."""
    page = MockPage()

    # Two elements close together (will connect)
    elem1 = TextElement(
        {"x0": 10, "top": 10, "x1": 50, "bottom": 30, "text": "Connected 1", "size": 12.0}, page
    )

    elem2 = TextElement(
        {"x0": 52, "top": 10, "x1": 90, "bottom": 30, "text": "Connected 2", "size": 12.0}, page
    )

    # One element far away (won't connect)
    elem3 = TextElement(
        {"x0": 200, "top": 200, "x1": 250, "bottom": 220, "text": "Isolated", "size": 12.0}, page
    )

    collection = ElementCollection([elem1, elem2, elem3])
    dissolved = collection.dissolve(padding=3.0)

    assert len(dissolved) == 2, f"Expected 2 regions (1 merged, 1 single), got {len(dissolved)}"

    # Find the merged region and single region
    merged = None
    single = None
    for r in dissolved:
        if "2 texts" in r.label:
            merged = r
        elif "1 text" in r.label:
            single = r

    assert merged is not None, "Should have a merged region"
    assert single is not None, "Should have a single element region"

    # Verify the merged region has the combined bbox
    assert merged.bbox == (10, 10, 90, 30), "Merged bbox should span both elements"

    # Verify the single region has the original bbox
    assert single.bbox == (200, 200, 250, 220), "Single element bbox should be preserved"


def test_dissolve_with_group_by_preserves_singles():
    """Test that dissolve with group_by still preserves single elements."""
    page = MockPage()

    # Three elements with different sizes (won't group together)
    elem1 = TextElement(
        {"x0": 10, "top": 10, "x1": 50, "bottom": 30, "text": "Size 10", "size": 10.0}, page
    )

    elem2 = TextElement(
        {"x0": 10, "top": 50, "x1": 50, "bottom": 70, "text": "Size 12", "size": 12.0}, page
    )

    elem3 = TextElement(
        {"x0": 10, "top": 90, "x1": 50, "bottom": 110, "text": "Size 14", "size": 14.0}, page
    )

    collection = ElementCollection([elem1, elem2, elem3])

    # Dissolve grouping by size - each element should be in its own group
    dissolved = collection.dissolve(padding=100.0, group_by=["size"])

    assert len(dissolved) == 3, f"Expected 3 regions (one per size group), got {len(dissolved)}"
    assert all(isinstance(r, Region) for r in dissolved), "All results should be regions"

    # Verify all original bboxes are preserved
    bboxes = [r.bbox for r in dissolved]
    assert (10, 10, 50, 30) in bboxes
    assert (10, 50, 50, 70) in bboxes
    assert (10, 90, 50, 110) in bboxes


if __name__ == "__main__":
    # Run the tests
    test_dissolve_preserves_single_elements()
    print("✓ test_dissolve_preserves_single_elements passed")

    test_dissolve_single_element_collection()
    print("✓ test_dissolve_single_element_collection passed")

    test_dissolve_mixed_connected_and_single()
    print("✓ test_dissolve_mixed_connected_and_single passed")

    test_dissolve_with_group_by_preserves_singles()
    print("✓ test_dissolve_with_group_by_preserves_singles passed")

    print("\nAll tests passed!")
