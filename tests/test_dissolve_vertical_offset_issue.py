"""Test to reproduce dissolve vertical offset issue."""

from unittest.mock import MagicMock

import pytest
from pdfplumber.utils.geometry import merge_bboxes

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region


def test_dissolve_vertical_offset_manual_calculation():
    """Test dissolve with overlapping elements with vertical offset - manual calculation."""
    # Create mock page
    mock_page = MagicMock()
    mock_page.pdf = MagicMock()

    # Create two overlapping regions with vertical offset
    # Region 1: x0=100, y0=100, x1=200, y1=150 (top region)
    # Region 2: x0=150, y0=130, x1=250, y1=180 (overlapping with vertical offset)
    region1 = Region(page=mock_page, bbox=(100, 100, 200, 150), label="Region 1")
    region2 = Region(page=mock_page, bbox=(150, 130, 250, 180), label="Region 2")

    # Create collection and dissolve
    collection = ElementCollection([region1, region2])
    dissolved = collection.dissolve(padding=50)  # High padding to ensure they merge

    # Check the result
    assert len(dissolved) == 1, f"Expected 1 dissolved region, got {len(dissolved)}"

    merged_region = dissolved[0]
    merged_bbox = merged_region.bbox

    # Expected bbox should be the union: (100, 100, 250, 180)
    expected_bbox = (100, 100, 250, 180)

    print(f"Region 1 bbox: {region1.bbox}")
    print(f"Region 2 bbox: {region2.bbox}")
    print(f"Expected merged bbox: {expected_bbox}")
    print(f"Actual merged bbox: {merged_bbox}")

    # Check if the merged bbox is correct
    assert merged_bbox == expected_bbox, f"Expected bbox {expected_bbox}, got {merged_bbox}"

    # Additional check: ensure the merged bbox is not taller than expected
    expected_height = expected_bbox[3] - expected_bbox[1]  # 180 - 100 = 80
    actual_height = merged_bbox[3] - merged_bbox[1]

    assert (
        actual_height == expected_height
    ), f"Expected height {expected_height}, got {actual_height}"


def test_compare_merge_methods():
    """Compare manual bbox calculation vs pdfplumber's merge_bboxes."""
    # Two overlapping bboxes with vertical offset
    bbox1 = (100, 100, 200, 150)
    bbox2 = (150, 130, 250, 180)

    # Manual calculation (current method in the code)
    x0s = [bbox1[0], bbox2[0]]
    tops = [bbox1[1], bbox2[1]]
    x1s = [bbox1[2], bbox2[2]]
    bottoms = [bbox1[3], bbox2[3]]

    manual_merged = (min(x0s), min(tops), max(x1s), max(bottoms))

    # pdfplumber's merge_bboxes
    pdfplumber_merged = merge_bboxes([bbox1, bbox2])

    print(f"Bbox 1: {bbox1}")
    print(f"Bbox 2: {bbox2}")
    print(f"Manual merge result: {manual_merged}")
    print(f"Pdfplumber merge result: {pdfplumber_merged}")

    # They should be the same
    assert (
        manual_merged == pdfplumber_merged
    ), f"Manual merge {manual_merged} != pdfplumber merge {pdfplumber_merged}"


def test_dissolve_with_extreme_vertical_offset():
    """Test dissolve with extreme vertical offset case."""
    # Create mock page
    mock_page = MagicMock()
    mock_page.pdf = MagicMock()

    # Create three regions forming a triangle pattern
    # This tests if there's any issue with coordinate system or calculation
    region1 = Region(page=mock_page, bbox=(100, 100, 150, 120), label="Top")
    region2 = Region(page=mock_page, bbox=(120, 110, 170, 130), label="Middle")
    region3 = Region(page=mock_page, bbox=(140, 120, 190, 140), label="Bottom")

    collection = ElementCollection([region1, region2, region3])
    dissolved = collection.dissolve(padding=50)

    assert len(dissolved) == 1, f"Expected 1 dissolved region, got {len(dissolved)}"

    merged_bbox = dissolved[0].bbox

    # Expected bbox should encompass all three regions
    expected_bbox = (100, 100, 190, 140)

    print(f"Region bboxes: {[r.bbox for r in [region1, region2, region3]]}")
    print(f"Expected merged bbox: {expected_bbox}")
    print(f"Actual merged bbox: {merged_bbox}")

    assert merged_bbox == expected_bbox, f"Expected bbox {expected_bbox}, got {merged_bbox}"


def test_pdfplumber_coordinate_system():
    """Test to understand pdfplumber's coordinate system."""
    # In PDFs, y-coordinates typically increase from bottom to top
    # Let's verify how merge_bboxes handles this

    # Two bboxes where bbox2 is below bbox1 (higher y values)
    bbox1 = (100, 100, 200, 150)  # Top box
    bbox2 = (100, 200, 200, 250)  # Bottom box (higher y values)

    merged = merge_bboxes([bbox1, bbox2])

    print(f"Top bbox: {bbox1}")
    print(f"Bottom bbox: {bbox2}")
    print(f"Merged bbox: {merged}")

    # The merged bbox should span from the topmost y to the bottommost y
    expected = (100, 100, 200, 250)
    assert merged == expected, f"Expected {expected}, got {merged}"


if __name__ == "__main__":
    # Run tests with verbose output
    test_dissolve_vertical_offset_manual_calculation()
    print("\n" + "=" * 50 + "\n")
    test_compare_merge_methods()
    print("\n" + "=" * 50 + "\n")
    test_dissolve_with_extreme_vertical_offset()
    print("\n" + "=" * 50 + "\n")
    test_pdfplumber_coordinate_system()
