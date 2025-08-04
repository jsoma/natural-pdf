"""Test real-world dissolve issues with overlapping elements."""

from unittest.mock import MagicMock

import pdfplumber
import pytest

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement


def test_dissolve_text_elements_with_overlap():
    """Test dissolve with actual text elements that overlap."""
    # Create mock page and elements
    mock_page = MagicMock()
    mock_page.pdf = MagicMock()

    # Create text elements that might represent real PDF text with slight overlaps
    # Simulating text on same line with slight vertical variations
    elements = []

    # First word "Hello" - baseline at y=100
    obj1 = {
        "x0": 50,
        "top": 98,
        "x1": 80,
        "bottom": 102,
        "text": "Hello",
        "size": 12,
        "fontname": "Arial",
    }
    elem1 = TextElement(obj=obj1, page=mock_page)
    elements.append(elem1)

    # Second word "World" - slightly lower baseline (y=101)
    obj2 = {
        "x0": 85,
        "top": 99,
        "x1": 120,
        "bottom": 103,
        "text": "World",
        "size": 12,
        "fontname": "Arial",
    }
    elem2 = TextElement(obj=obj2, page=mock_page)
    elements.append(elem2)

    # Third element with more vertical offset
    obj3 = {
        "x0": 125,
        "top": 100,
        "x1": 160,
        "bottom": 104,
        "text": "Test",
        "size": 12,
        "fontname": "Arial",
    }
    elem3 = TextElement(obj=obj3, page=mock_page)
    elements.append(elem3)

    collection = ElementCollection(elements)
    dissolved = collection.dissolve(padding=5)

    print(f"Original elements:")
    for elem in elements:
        print(f"  {elem._obj['text']}: bbox={elem.bbox}")

    print(f"\nDissolved regions: {len(dissolved)}")
    for region in dissolved:
        print(f"  {region.label}: bbox={region.bbox}")
        height = region.bbox[3] - region.bbox[1]
        print(f"    Height: {height}")

    # Check that we get one merged region
    assert len(dissolved) == 1

    # Check the height is reasonable (not excessively tall)
    merged_bbox = dissolved[0].bbox
    height = merged_bbox[3] - merged_bbox[1]

    # Expected height should be roughly the max bottom - min top
    expected_min_top = min(elem.bbox[1] for elem in elements)
    expected_max_bottom = max(elem.bbox[3] for elem in elements)
    expected_height = expected_max_bottom - expected_min_top

    print(f"\nExpected height: {expected_height}")
    print(f"Actual height: {height}")

    assert height == expected_height, f"Height mismatch: expected {expected_height}, got {height}"


def test_dissolve_with_floating_point_precision():
    """Test dissolve with floating point coordinates that might cause precision issues."""
    mock_page = MagicMock()
    mock_page.pdf = MagicMock()

    # Create regions with floating point coordinates
    region1 = Region(page=mock_page, bbox=(100.1, 100.2, 200.3, 150.4), label="R1")
    region2 = Region(page=mock_page, bbox=(150.5, 130.6, 250.7, 180.8), label="R2")

    collection = ElementCollection([region1, region2])
    dissolved = collection.dissolve(padding=60)

    assert len(dissolved) == 1

    merged_bbox = dissolved[0].bbox
    print(f"Region 1: {region1.bbox}")
    print(f"Region 2: {region2.bbox}")
    print(f"Merged: {merged_bbox}")

    # Check bounds
    assert merged_bbox[0] <= min(region1.bbox[0], region2.bbox[0])
    assert merged_bbox[1] <= min(region1.bbox[1], region2.bbox[1])
    assert merged_bbox[2] >= max(region1.bbox[2], region2.bbox[2])
    assert merged_bbox[3] >= max(region1.bbox[3], region2.bbox[3])


def test_dissolve_coordinate_system_interpretation():
    """Test if there's a coordinate system interpretation issue."""
    mock_page = MagicMock()
    mock_page.pdf = MagicMock()

    # In PDFs, typically:
    # - x increases from left to right
    # - y can increase from bottom to top OR top to bottom depending on the tool

    # Let's test with regions that would reveal coordinate system issues
    # If y increases downward (common in many tools):
    # - smaller y = higher on page
    # - larger y = lower on page

    # Top region (smaller y values)
    top_region = Region(page=mock_page, bbox=(100, 50, 200, 70), label="Top")

    # Bottom region (larger y values) with overlap
    bottom_region = Region(page=mock_page, bbox=(120, 65, 220, 85), label="Bottom")

    collection = ElementCollection([top_region, bottom_region])
    dissolved = collection.dissolve(padding=20)

    assert len(dissolved) == 1

    merged_bbox = dissolved[0].bbox
    expected_bbox = (100, 50, 220, 85)

    print(f"Top region: {top_region.bbox}")
    print(f"Bottom region: {bottom_region.bbox}")
    print(f"Expected merged: {expected_bbox}")
    print(f"Actual merged: {merged_bbox}")

    assert merged_bbox == expected_bbox


def test_dissolve_edge_case_thin_overlaps():
    """Test dissolve with very thin overlapping regions."""
    mock_page = MagicMock()
    mock_page.pdf = MagicMock()

    # Create multiple thin horizontal regions with slight vertical overlaps
    regions = []
    for i in range(5):
        y_start = 100 + i * 2  # Each region starts 2 units below the previous
        y_end = y_start + 3  # Each region is 3 units tall (1 unit overlap)
        region = Region(page=mock_page, bbox=(100, y_start, 300, y_end), label=f"Line {i+1}")
        regions.append(region)

    collection = ElementCollection(regions)
    dissolved = collection.dissolve(padding=2)

    print(f"Original regions:")
    for r in regions:
        print(f"  {r.label}: {r.bbox}, height={r.bbox[3]-r.bbox[1]}")

    print(f"\nDissolved: {len(dissolved)} regions")
    for r in dissolved:
        print(f"  {r.label}: {r.bbox}, height={r.bbox[3]-r.bbox[1]}")

    # Should merge into one region
    assert len(dissolved) == 1

    # Check the total height
    merged_bbox = dissolved[0].bbox
    actual_height = merged_bbox[3] - merged_bbox[1]

    # Expected: from y=100 to y=111 (last region ends at 108+3)
    expected_height = 11

    assert (
        actual_height == expected_height
    ), f"Expected height {expected_height}, got {actual_height}"


if __name__ == "__main__":
    test_dissolve_text_elements_with_overlap()
    print("\n" + "=" * 50 + "\n")
    test_dissolve_with_floating_point_precision()
    print("\n" + "=" * 50 + "\n")
    test_dissolve_coordinate_system_interpretation()
    print("\n" + "=" * 50 + "\n")
    test_dissolve_edge_case_thin_overlaps()
