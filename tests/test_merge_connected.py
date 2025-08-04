"""Test the merge_connected method of ElementCollection."""

import pytest

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region


class MockPage:
    """Mock page for testing."""

    def __init__(self, index=0):
        self.index = index
        self.width = 612
        self.height = 792


def test_merge_connected_empty_collection():
    """Test merge_connected with empty collection."""
    collection = ElementCollection([])
    result = collection.merge_connected()
    assert len(result) == 0


def test_merge_connected_single_region():
    """Test merge_connected with single region."""
    page = MockPage()
    region = Region(page, (100, 100, 200, 150))
    collection = ElementCollection([region])
    result = collection.merge_connected()
    assert len(result) == 1
    assert result[0] is region  # Should be unchanged


def test_merge_connected_overlapping_regions():
    """Test merge_connected with overlapping regions."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (150, 120, 250, 170))  # Overlaps with region1
    collection = ElementCollection([region1, region2])

    result = collection.merge_connected()
    assert len(result) == 1
    merged = result[0]
    assert merged.bbox == (100, 100, 250, 170)  # Should encompass both


def test_merge_connected_adjacent_regions():
    """Test merge_connected with adjacent regions within threshold."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (202, 100, 300, 150))  # 2 points gap
    collection = ElementCollection([region1, region2])

    # Should merge with default threshold (5.0)
    result = collection.merge_connected()
    assert len(result) == 1
    merged = result[0]
    assert merged.bbox == (100, 100, 300, 150)

    # Should not merge with threshold 1.0
    result = collection.merge_connected(proximity_threshold=1.0)
    assert len(result) == 2


def test_merge_connected_distant_regions():
    """Test merge_connected with distant regions."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (300, 300, 400, 350))  # Far apart
    collection = ElementCollection([region1, region2])

    result = collection.merge_connected()
    assert len(result) == 2  # Should not merge


def test_merge_connected_multiple_groups():
    """Test merge_connected with multiple connected groups."""
    page = MockPage()
    # Group 1
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (195, 100, 295, 150))  # Connected to region1
    # Group 2
    region3 = Region(page, (400, 100, 500, 150))
    region4 = Region(page, (495, 100, 595, 150))  # Connected to region3

    collection = ElementCollection([region1, region2, region3, region4])
    result = collection.merge_connected()

    assert len(result) == 2  # Two merged groups
    # Check that groups are properly merged
    bboxes = sorted([r.bbox for r in result])
    assert bboxes[0] == (100, 100, 295, 150)  # Group 1
    assert bboxes[1] == (400, 100, 595, 150)  # Group 2


def test_merge_connected_chain():
    """Test merge_connected with chain of connected regions."""
    page = MockPage()
    # Create a chain: A-B-C where A connects to B and B connects to C
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (195, 100, 295, 150))  # Connected to region1
    region3 = Region(page, (290, 100, 390, 150))  # Connected to region2

    collection = ElementCollection([region1, region2, region3])
    result = collection.merge_connected()

    assert len(result) == 1  # All should merge into one
    merged = result[0]
    assert merged.bbox == (100, 100, 390, 150)


def test_merge_connected_across_pages():
    """Test merge_connected with regions on different pages."""
    page1 = MockPage(index=0)
    page2 = MockPage(index=1)

    region1 = Region(page1, (100, 100, 200, 150))
    region2 = Region(page1, (195, 100, 295, 150))  # Same page, connected
    region3 = Region(page2, (400, 400, 500, 450))  # Different page, far away

    collection = ElementCollection([region1, region2, region3])

    # Default: don't merge across pages
    result = collection.merge_connected()
    assert len(result) == 2  # region1+region2 merged, region3 separate

    # Allow merging across pages - still should be 2 groups due to distance
    result = collection.merge_connected(merge_across_pages=True)
    assert len(result) == 2  # Still 2 because region3 is not connected spatially

    # Test merging across pages when regions are actually adjacent
    region4 = Region(page2, (195, 100, 295, 150))  # Same position as region2 but on page2
    collection2 = ElementCollection([region1, region2, region4])

    # Without merge_across_pages: should get 2 groups
    result = collection2.merge_connected()
    assert len(result) == 2

    # With merge_across_pages and overlapping positions: all should merge
    result = collection2.merge_connected(merge_across_pages=True)
    assert len(result) == 1  # All connected due to overlapping positions


def test_merge_connected_preserve_metadata():
    """Test that merge_connected preserves metadata from first region."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region1.metadata = {"source": "ocr", "confidence": 0.95}
    region1.region_type = "text"

    region2 = Region(page, (195, 100, 295, 150))
    region2.metadata = {"source": "layout", "confidence": 0.85}
    region2.region_type = "text"

    collection = ElementCollection([region1, region2])
    result = collection.merge_connected()

    assert len(result) == 1
    merged = result[0]
    # Should preserve metadata from first region
    assert merged.metadata["source"] == "ocr"
    assert merged.metadata["confidence"] == 0.95
    # Should add merge info
    assert "merge_info" in merged.metadata
    assert merged.metadata["merge_info"]["source_count"] == 2
    # Should preserve region_type since both are 'text'
    assert merged.region_type == "text"


def test_merge_connected_mixed_region_types():
    """Test merge_connected with different region types."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region1.region_type = "text"

    region2 = Region(page, (195, 100, 295, 150))
    region2.region_type = "table"

    collection = ElementCollection([region1, region2])
    result = collection.merge_connected()

    assert len(result) == 1
    merged = result[0]
    # Should not have region_type since they differ
    assert not hasattr(merged, "region_type") or merged.region_type is None


def test_merge_connected_with_text_extraction():
    """Test merge_connected preserves text in merge_info."""
    page = MockPage()

    # Mock extract_text method
    region1 = Region(page, (100, 100, 200, 150))
    region1.extract_text = lambda: "Hello"

    region2 = Region(page, (195, 100, 295, 150))
    region2.extract_text = lambda: "World"

    collection = ElementCollection([region1, region2])
    result = collection.merge_connected()

    assert len(result) == 1
    merged = result[0]
    assert merged.metadata["merge_info"]["merged_text"] == "Hello World"


def test_merge_connected_text_ordering():
    """Test merge_connected orders text by reading order."""
    page = MockPage()

    # Create regions in non-reading order
    region1 = Region(page, (200, 100, 300, 150))  # Right
    region1.extract_text = lambda: "World"

    region2 = Region(page, (100, 100, 195, 150))  # Left
    region2.extract_text = lambda: "Hello"

    collection = ElementCollection([region1, region2])
    result = collection.merge_connected(preserve_order=True)

    assert len(result) == 1
    merged = result[0]
    # Should order left-to-right
    assert merged.metadata["merge_info"]["merged_text"] == "Hello World"

    # Test without ordering
    result = collection.merge_connected(preserve_order=False)
    merged = result[0]
    # Should keep original order
    assert merged.metadata["merge_info"]["merged_text"] == "World Hello"


def test_merge_connected_vertical_adjacency():
    """Test merge_connected with vertically adjacent regions."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (100, 152, 200, 200))  # 2 points gap vertically

    collection = ElementCollection([region1, region2])
    result = collection.merge_connected()

    assert len(result) == 1
    merged = result[0]
    assert merged.bbox == (100, 100, 200, 200)


def test_merge_connected_custom_separator():
    """Test merge_connected with custom text separator."""
    page = MockPage()

    region1 = Region(page, (100, 100, 200, 150))
    region1.extract_text = lambda: "Part1"

    region2 = Region(page, (195, 100, 295, 150))
    region2.extract_text = lambda: "Part2"

    collection = ElementCollection([region1, region2])
    result = collection.merge_connected(text_separator="|")

    assert len(result) == 1
    merged = result[0]
    assert merged.metadata["merge_info"]["merged_text"] == "Part1|Part2"


def test_merge_connected_threshold_zero():
    """Test merge_connected with threshold=0 (only overlapping)."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (301, 100, 400, 150))  # Far away, no overlap
    region3 = Region(page, (150, 120, 199, 170))  # Overlaps only region1

    collection = ElementCollection([region1, region2, region3])
    result = collection.merge_connected(proximity_threshold=0)

    assert len(result) == 2  # region1+region3 merged, region2 separate
    bboxes = sorted([r.bbox for r in result], key=lambda b: b[0])
    assert bboxes[0] == (100, 100, 200, 170)  # Merged region1+region3
    assert bboxes[1] == (301, 100, 400, 150)  # region2 alone


def test_merge_connected_chebyshev_distance():
    """Test that merge_connected uses Chebyshev distance (max of dx, dy)."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    # Region2 is 3 points away horizontally and 2 points away vertically
    # Chebyshev distance = max(3, 2) = 3
    region2 = Region(page, (203, 152, 303, 202))

    collection = ElementCollection([region1, region2])

    # Should merge with threshold >= 3
    result = collection.merge_connected(proximity_threshold=3.0)
    assert len(result) == 1

    # Should not merge with threshold < 3
    result = collection.merge_connected(proximity_threshold=2.9)
    assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
