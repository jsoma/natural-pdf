"""Test the dissolve method of ElementCollection."""

import pytest

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement


class MockPage:
    """Mock page for testing."""

    def __init__(self, index=0):
        self.index = index
        self.width = 612
        self.height = 792


def test_dissolve_empty_collection():
    """Test dissolve with empty collection."""
    collection = ElementCollection([])
    result = collection.dissolve()
    assert len(result) == 0


def test_dissolve_no_regions():
    """Test dissolve with collection containing no Region elements."""
    page = MockPage()
    # Create some non-Region elements - TextElement expects (obj, page) not (page, obj)
    text1 = TextElement({"x0": 100, "top": 100, "x1": 200, "bottom": 150, "text": "Hello"}, page)
    text2 = TextElement({"x0": 200, "top": 100, "x1": 300, "bottom": 150, "text": "World"}, page)

    collection = ElementCollection([text1, text2])
    result = collection.dissolve()
    assert len(result) == 1  # TextElements should be dissolved into a Region
    assert result[0].bbox == (100, 100, 300, 150)  # Merged bbox
    assert "2 text" in result[0].label  # Label shows element types


def test_dissolve_single_region():
    """Test dissolve with single region."""
    page = MockPage()
    region = Region(page, (100, 100, 200, 150))
    collection = ElementCollection([region])
    result = collection.dissolve()
    assert len(result) == 1
    assert result[0].bbox == region.bbox  # Should have same bbox
    assert "1 region" in result[0].label  # Label indicates single region


def test_dissolve_overlapping_regions():
    """Test dissolve with overlapping regions."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (150, 120, 250, 170))  # Overlaps with region1
    collection = ElementCollection([region1, region2])

    result = collection.dissolve()
    assert len(result) == 1
    merged = result[0]
    assert merged.bbox == (100, 100, 250, 170)  # Should encompass both
    assert merged.metadata["dissolve_info"]["source_count"] == 2


def test_dissolve_adjacent_regions_within_padding():
    """Test dissolve with adjacent regions within padding threshold."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (201, 100, 300, 150))  # 1 point gap
    collection = ElementCollection([region1, region2])

    # Should merge with default padding (2.0)
    result = collection.dissolve()
    assert len(result) == 1
    merged = result[0]
    assert merged.bbox == (100, 100, 300, 150)

    # Should not merge with padding 0.5
    result = collection.dissolve(padding=0.5)
    assert len(result) == 2


def test_dissolve_distant_regions():
    """Test dissolve with distant regions."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (300, 300, 400, 350))  # Far apart
    collection = ElementCollection([region1, region2])

    result = collection.dissolve()
    assert len(result) == 2  # Should not merge


def test_dissolve_multiple_groups():
    """Test dissolve with multiple connected groups."""
    page = MockPage()
    # Group 1
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (199, 100, 299, 150))  # Connected to region1
    # Group 2
    region3 = Region(page, (400, 100, 500, 150))
    region4 = Region(page, (499, 100, 599, 150))  # Connected to region3

    collection = ElementCollection([region1, region2, region3, region4])
    result = collection.dissolve()

    assert len(result) == 2  # Two merged groups
    # Check that groups are properly merged
    bboxes = sorted([r.bbox for r in result], key=lambda b: b[0])
    assert bboxes[0] == (100, 100, 299, 150)  # Group 1
    assert bboxes[1] == (400, 100, 599, 150)  # Group 2


def test_dissolve_chain():
    """Test dissolve with chain of connected regions."""
    page = MockPage()
    # Create a chain: A-B-C where A connects to B and B connects to C
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (199, 100, 299, 150))  # Connected to region1
    region3 = Region(page, (298, 100, 398, 150))  # Connected to region2

    collection = ElementCollection([region1, region2, region3])
    result = collection.dissolve()

    assert len(result) == 1  # All should merge into one
    merged = result[0]
    assert merged.bbox == (100, 100, 398, 150)


def test_dissolve_with_group_by_single_attribute():
    """Test dissolve with grouping by single attribute."""
    page = MockPage()

    # Create regions with different font sizes
    region1 = Region(page, (100, 100, 200, 150))
    region1.font_size = 12.0

    region2 = Region(page, (199, 100, 299, 150))  # Adjacent to region1
    region2.font_size = 12.0

    region3 = Region(page, (198, 100, 298, 150))  # Adjacent to region1 and region2
    region3.font_size = 14.0

    collection = ElementCollection([region1, region2, region3])

    # Without grouping, all should merge
    result = collection.dissolve()
    assert len(result) == 1

    # With grouping by font_size, should get 2 groups
    result = collection.dissolve(group_by=["font_size"])
    assert len(result) == 2

    # Check the groups
    sizes = sorted(
        [
            getattr(r, "font_size", None)
            for r in [region1, region2, region3]
            if hasattr(r, "font_size")
        ]
    )
    # One merged region for font_size=12, one single region for font_size=14


def test_dissolve_with_group_by_multiple_attributes():
    """Test dissolve with grouping by multiple attributes."""
    page = MockPage()

    # Create regions with different attributes
    region1 = Region(page, (100, 100, 200, 150))
    region1.font_size = 12.0
    region1.font_name = "Arial"

    region2 = Region(page, (199, 100, 299, 150))
    region2.font_size = 12.0
    region2.font_name = "Arial"

    region3 = Region(page, (298, 100, 398, 150))
    region3.font_size = 12.0
    region3.font_name = "Times"

    region4 = Region(page, (397, 100, 497, 150))
    region4.font_size = 12.0
    region4.font_name = "Times"

    collection = ElementCollection([region1, region2, region3, region4])

    # With grouping by both attributes, should get 2 merged groups
    result = collection.dissolve(group_by=["font_size", "font_name"])
    assert len(result) == 2

    # Check that each group has consistent attributes
    for merged in result:
        # All source regions should have had the same font_name
        assert hasattr(merged, "metadata")
        assert "dissolve_info" in merged.metadata
        assert merged.metadata["dissolve_info"]["source_count"] == 2


def test_dissolve_with_metadata_attributes():
    """Test dissolve with grouping by metadata attributes."""
    page = MockPage()

    region1 = Region(page, (100, 100, 200, 150))
    region1.metadata["category"] = "header"

    region2 = Region(page, (199, 100, 299, 150))
    region2.metadata["category"] = "header"

    region3 = Region(page, (400, 100, 500, 150))  # Far from regions 1&2, won't merge
    region3.metadata["category"] = "body"

    collection = ElementCollection([region1, region2, region3])

    # Group by metadata attribute
    result = collection.dissolve(group_by=["category"])
    assert len(result) == 2  # Two groups: header (merged) and body (single)


def test_dissolve_float_rounding():
    """Test that float attributes are rounded to 2 decimals for grouping."""
    page = MockPage()

    region1 = Region(page, (100, 100, 200, 150))
    region1.font_size = 12.001

    region2 = Region(page, (199, 100, 299, 150))
    region2.font_size = 12.004

    region3 = Region(page, (298, 100, 398, 150))
    region3.font_size = 12.51

    collection = ElementCollection([region1, region2, region3])

    # With grouping, 12.001 and 12.004 should round to 12.00 and merge
    # 12.51 should round to 12.51 and stay separate
    result = collection.dissolve(group_by=["font_size"])
    assert len(result) == 2


def test_dissolve_preserve_region_type():
    """Test that dissolve preserves region_type when consistent."""
    page = MockPage()

    region1 = Region(page, (100, 100, 200, 150))
    region1.region_type = "text"

    region2 = Region(page, (199, 100, 299, 150))
    region2.region_type = "text"

    collection = ElementCollection([region1, region2])
    result = collection.dissolve()

    assert len(result) == 1
    merged = result[0]
    assert merged.region_type == "text"


def test_dissolve_mixed_region_types():
    """Test dissolve with different region types."""
    page = MockPage()

    region1 = Region(page, (100, 100, 200, 150))
    region1.region_type = "text"

    region2 = Region(page, (199, 100, 299, 150))
    region2.region_type = "table"

    collection = ElementCollection([region1, region2])
    result = collection.dissolve()

    assert len(result) == 1
    merged = result[0]
    # Should not have region_type since they differ
    assert not hasattr(merged, "region_type") or merged.region_type is None


def test_dissolve_vertical_adjacency():
    """Test dissolve with vertically adjacent regions."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (100, 151, 200, 200))  # 1 point gap vertically

    collection = ElementCollection([region1, region2])
    result = collection.dissolve()

    assert len(result) == 1
    merged = result[0]
    assert merged.bbox == (100, 100, 200, 200)


def test_dissolve_padding_zero():
    """Test dissolve with padding=0 (only overlapping)."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (201, 100, 300, 150))  # 1 point gap, no overlap
    region3 = Region(page, (150, 120, 199, 170))  # Overlaps only region1

    collection = ElementCollection([region1, region2, region3])
    result = collection.dissolve(padding=0)

    assert len(result) == 2  # region1+region3 merged, region2 separate
    bboxes = sorted([r.bbox for r in result], key=lambda b: b[0])
    assert bboxes[0] == (100, 100, 200, 170)  # Merged region1+region3
    assert bboxes[1] == (201, 100, 300, 150)  # region2 alone


def test_dissolve_chebyshev_distance():
    """Test that dissolve uses Chebyshev distance (max of dx, dy)."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    # Region2 is 2 points away horizontally and 1 point away vertically
    # Chebyshev distance = max(2, 1) = 2
    region2 = Region(page, (202, 151, 302, 201))

    collection = ElementCollection([region1, region2])

    # Should merge with padding >= 2
    result = collection.dissolve(padding=2.0)
    assert len(result) == 1

    # Should not merge with padding < 2
    result = collection.dissolve(padding=1.9)
    assert len(result) == 2


def test_dissolve_invalid_geometry():
    """Test dissolve with invalid geometry parameter."""
    page = MockPage()
    region = Region(page, (100, 100, 200, 150))
    collection = ElementCollection([region])

    # Should raise ValueError for invalid geometry
    with pytest.raises(ValueError, match="Invalid geometry type"):
        collection.dissolve(geometry="invalid")


def test_dissolve_polygon_not_implemented():
    """Test that polygon geometry raises NotImplementedError."""
    page = MockPage()
    region = Region(page, (100, 100, 200, 150))
    collection = ElementCollection([region])

    # Should raise NotImplementedError for polygon geometry
    with pytest.raises(NotImplementedError, match="Polygon geometry is not yet supported"):
        collection.dissolve(geometry="polygon")


def test_dissolve_mixed_elements():
    """Test dissolve with mixed Region and non-Region elements."""
    page = MockPage()

    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (199, 100, 299, 150))
    text = TextElement({"x0": 400, "top": 100, "x1": 500, "bottom": 150, "text": "Text"}, page)

    collection = ElementCollection([region1, text, region2])
    result = collection.dissolve()

    # Should process all elements and create regions
    assert len(result) == 2
    # One merged region from region1 and region2
    # One region from the text element

    # Check bounding boxes
    bboxes = sorted([r.bbox for r in result], key=lambda b: b[0])
    assert bboxes[0] == (100, 100, 299, 150)  # Merged regions
    assert bboxes[1] == (400, 100, 500, 150)  # Text element as region


def test_dissolve_text_elements_by_font_size():
    """Test dissolve with TextElements grouped by font size."""
    page = MockPage()

    # Create TextElements with different font sizes
    text1 = TextElement(
        {"x0": 100, "top": 100, "x1": 200, "bottom": 120, "text": "Title", "size": 16.0}, page
    )

    text2 = TextElement(
        {
            "x0": 199,
            "top": 100,
            "x1": 300,
            "bottom": 120,
            "text": "Header",
            "size": 16.0,  # Same size as text1, adjacent
        },
        page,
    )

    text3 = TextElement(
        {
            "x0": 100,
            "top": 150,
            "x1": 200,
            "bottom": 165,
            "text": "Body",
            "size": 12.0,  # Different size
        },
        page,
    )

    text4 = TextElement(
        {
            "x0": 199,
            "top": 150,
            "x1": 300,
            "bottom": 165,
            "text": "Text",
            "size": 12.0,  # Same size as text3, adjacent
        },
        page,
    )

    collection = ElementCollection([text1, text2, text3, text4])

    # Without grouping, all adjacent elements should merge
    result = collection.dissolve()
    assert len(result) == 2  # Two groups based on vertical separation

    # With grouping by size, should get 2 groups
    result = collection.dissolve(group_by=["size"])
    assert len(result) == 2

    # Check that each group has the correct font_size attribute
    font_sizes = sorted([getattr(r, "font_size", None) for r in result])
    assert font_sizes == [12.0, 16.0]

    # Check the bounding boxes
    for region in result:
        if getattr(region, "font_size", None) == 16.0:
            assert region.bbox == (100, 100, 300, 120)  # Merged text1 and text2
        elif getattr(region, "font_size", None) == 12.0:
            assert region.bbox == (100, 150, 300, 165)  # Merged text3 and text4


def test_dissolve_label():
    """Test that dissolved regions have appropriate labels."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (199, 100, 299, 150))

    collection = ElementCollection([region1, region2])
    result = collection.dissolve()

    assert len(result) == 1
    assert result[0].label == "Dissolved (2 regions)"


def test_dissolve_metadata_preservation():
    """Test that dissolve metadata is properly added."""
    page = MockPage()
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (199, 100, 299, 150))

    collection = ElementCollection([region1, region2])
    result = collection.dissolve()

    assert len(result) == 1
    merged = result[0]

    assert "dissolve_info" in merged.metadata
    assert merged.metadata["dissolve_info"]["source_count"] == 2
    assert len(merged.metadata["dissolve_info"]["source_bboxes"]) == 2
    assert merged.metadata["dissolve_info"]["source_bboxes"][0] == (100, 100, 200, 150)
    assert merged.metadata["dissolve_info"]["source_bboxes"][1] == (199, 100, 299, 150)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
