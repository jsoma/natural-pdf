"""Debug test to investigate dissolve issue with specific scenarios."""

from unittest.mock import MagicMock

import pytest
from pdfplumber.utils.geometry import get_bbox_overlap, merge_bboxes

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement


def test_dissolve_detailed_debug():
    """Detailed debug test for dissolve with overlapping elements."""
    mock_page = MagicMock()
    mock_page.pdf = MagicMock()

    # Create two regions that overlap with vertical offset
    # Using coordinates that might reveal issues
    region1 = Region(page=mock_page, bbox=(100, 500, 200, 520), label="Region 1")
    region2 = Region(page=mock_page, bbox=(150, 510, 250, 530), label="Region 2")

    print("Initial regions:")
    print(f"  Region 1: {region1.bbox} (height: {region1.bbox[3] - region1.bbox[1]})")
    print(f"  Region 2: {region2.bbox} (height: {region2.bbox[3] - region2.bbox[1]})")

    # Check if they overlap
    overlap = get_bbox_overlap(region1.bbox, region2.bbox)
    print(f"\nOverlap: {overlap}")

    # Manual merge calculation
    bboxes = [region1.bbox, region2.bbox]
    manual_merged = (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )
    print(f"\nManual merge: {manual_merged} (height: {manual_merged[3] - manual_merged[1]})")

    # pdfplumber merge
    pdfplumber_merged = merge_bboxes(bboxes)
    print(
        f"Pdfplumber merge: {pdfplumber_merged} (height: {pdfplumber_merged[3] - pdfplumber_merged[1]})"
    )

    # Test dissolve
    collection = ElementCollection([region1, region2])

    # Test connectivity check
    connected = collection._are_elements_connected(region1, region2, 0)
    print(f"\nAre elements connected (padding=0)? {connected}")

    connected_with_padding = collection._are_elements_connected(region1, region2, 10)
    print(f"Are elements connected (padding=10)? {connected_with_padding}")

    # Perform dissolve
    dissolved = collection.dissolve(padding=2)

    print(f"\nDissolved result: {len(dissolved)} region(s)")
    for i, region in enumerate(dissolved):
        print(f"  Region {i}: {region.bbox} (height: {region.bbox[3] - region.bbox[1]})")
        print(f"    Label: {region.label}")


def test_specific_problematic_case():
    """Test a specific case that might be problematic."""
    mock_page = MagicMock()
    mock_page.pdf = MagicMock()

    # Simulate elements that when merged might create unexpected height
    # These could represent real PDF elements with specific characteristics
    elements = []

    # Element 1: Top element
    obj1 = {"x0": 100, "top": 495, "x1": 200, "bottom": 505, "text": "Top", "size": 10}
    elem1 = TextElement(obj=obj1, page=mock_page)
    elements.append(elem1)

    # Element 2: Overlapping with offset
    obj2 = {"x0": 150, "top": 502, "x1": 250, "bottom": 512, "text": "Middle", "size": 10}
    elem2 = TextElement(obj=obj2, page=mock_page)
    elements.append(elem2)

    # Element 3: Further offset
    obj3 = {"x0": 200, "top": 509, "x1": 300, "bottom": 519, "text": "Bottom", "size": 10}
    elem3 = TextElement(obj=obj3, page=mock_page)
    elements.append(elem3)

    print("Elements to dissolve:")
    for elem in elements:
        print(f"  {elem._obj['text']}: {elem.bbox} (height: {elem.bbox[3] - elem.bbox[1]})")

    collection = ElementCollection(elements)

    # Check connectivity between pairs
    print("\nConnectivity check (padding=2):")
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            connected = collection._are_elements_connected(elements[i], elements[j], 2)
            print(f"  {elements[i]._obj['text']} <-> {elements[j]._obj['text']}: {connected}")

    # Dissolve
    dissolved = collection.dissolve(padding=2)

    print(f"\nDissolved: {len(dissolved)} region(s)")
    for region in dissolved:
        print(f"  {region.label}: {region.bbox}")
        print(f"    Height: {region.bbox[3] - region.bbox[1]}")
        print(f"    Expected height: {519 - 495} = 24")


def test_edge_case_vertical_spread():
    """Test case with elements that have significant vertical spread."""
    mock_page = MagicMock()
    mock_page.pdf = MagicMock()

    # Create elements with increasing vertical positions
    elements = []
    base_y = 500

    for i in range(10):
        # Each element is offset vertically and horizontally
        x0 = 100 + i * 20
        y0 = base_y + i * 5
        x1 = x0 + 50
        y1 = y0 + 8  # Small height for each element

        region = Region(page=mock_page, bbox=(x0, y0, x1, y1), label=f"Element {i}")
        elements.append(region)

    print("Elements:")
    for elem in elements:
        print(f"  {elem.label}: {elem.bbox}")

    collection = ElementCollection(elements)
    dissolved = collection.dissolve(padding=15)  # Large padding to connect all

    print(f"\nDissolved: {len(dissolved)} region(s)")
    for region in dissolved:
        print(f"  {region.label}: {region.bbox}")
        height = region.bbox[3] - region.bbox[1]
        print(f"    Height: {height}")

        # Calculate expected height
        min_y = min(e.bbox[1] for e in elements)
        max_y = max(e.bbox[3] for e in elements)
        expected_height = max_y - min_y
        print(f"    Expected height: {expected_height}")

        assert (
            height == expected_height
        ), f"Height mismatch: got {height}, expected {expected_height}"


def test_investigate_merge_algorithm():
    """Deep dive into the merge algorithm to understand potential issues."""
    mock_page = MagicMock()
    mock_page.pdf = MagicMock()

    # Create a specific scenario
    r1 = Region(page=mock_page, bbox=(100, 100, 200, 120), label="R1")
    r2 = Region(page=mock_page, bbox=(180, 115, 280, 135), label="R2")
    r3 = Region(page=mock_page, bbox=(260, 130, 360, 150), label="R3")

    elements = [r1, r2, r3]
    collection = ElementCollection(elements)

    # Manually trace through the algorithm
    print("Manual algorithm trace:")
    print(f"Elements: {[e.label for e in elements]}")
    print(f"Bboxes: {[e.bbox for e in elements]}")

    # Check connectivity with different proximities
    for padding in [0, 5, 10, 20]:
        print(f"\npadding = {padding}:")
        components = collection._find_connected_components_elements(elements, padding)
        print(f"  Components: {len(components)}")
        for i, comp in enumerate(components):
            print(f"    Component {i}: {[e.label for e in comp]}")
            if len(comp) > 1:
                bboxes = [e.bbox for e in comp]
                merged = merge_bboxes(bboxes)
                print(f"      Merged bbox: {merged}")
                print(f"      Height: {merged[3] - merged[1]}")


if __name__ == "__main__":
    test_dissolve_detailed_debug()
    print("\n" + "=" * 60 + "\n")
    test_specific_problematic_case()
    print("\n" + "=" * 60 + "\n")
    test_edge_case_vertical_spread()
    print("\n" + "=" * 60 + "\n")
    test_investigate_merge_algorithm()
