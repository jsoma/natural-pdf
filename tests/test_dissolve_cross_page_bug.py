"""Test to demonstrate the dissolve() cross-page merging bug."""

import pytest

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region


# Mock page class for testing
class MockPage:
    """Mock page for testing."""

    def __init__(self, number=1, width=612, height=792, id=None):
        self.number = number
        self.index = number - 1
        self.width = width
        self.height = height
        self.id = id or f"page{number}"


def test_dissolve_should_not_merge_across_pages():
    """Test that dissolve() does not merge elements from different pages."""

    # Create mock pages
    page1 = MockPage(number=1, width=612, height=792, id="page1")
    page2 = MockPage(number=2, width=612, height=792, id="page2")

    # Create regions with similar positions on different pages
    # These regions have the same bbox coordinates but are on different pages
    region1 = Region(page1, (100, 100, 200, 120))
    region1.label = "Header on page 1"
    region1.font_size = 14.0

    region2 = Region(page2, (100, 100, 200, 120))  # Same position as region1
    region2.label = "Header on page 2"
    region2.font_size = 14.0

    # Create additional regions that are close to the headers
    region3 = Region(page1, (100, 122, 200, 140))  # Just 2 points below region1
    region3.label = "Close to header on page 1"
    region3.font_size = 14.0

    region4 = Region(page2, (100, 122, 200, 140))  # Just 2 points below region2
    region4.label = "Close to header on page 2"
    region4.font_size = 14.0

    # Create collection with all regions
    collection = ElementCollection([region1, region2, region3, region4])

    # Dissolve with default padding (2.0 points)
    dissolved = collection.dissolve(padding=2.0)

    # We should have 2 dissolved regions (one per page), not 1 merged across pages
    assert len(dissolved) == 2, f"Expected 2 dissolved regions (one per page), got {len(dissolved)}"

    # Check that each dissolved region contains elements from only one page
    for region in dissolved:
        source_info = region.metadata.get("dissolve_info", {})
        source_count = source_info.get("source_count", 0)

        # Each page should have its 2 regions merged together
        assert source_count == 2, f"Expected 2 elements per dissolved region, got {source_count}"

        # Verify the region is assigned to a single page
        assert region.page in [page1, page2], "Region should be assigned to one of the pages"

    # Get the pages of dissolved regions
    dissolved_pages = [r.page for r in dissolved]

    # Should have one region per page
    assert page1 in dissolved_pages, "Should have a dissolved region on page 1"
    assert page2 in dissolved_pages, "Should have a dissolved region on page 2"


def test_dissolve_cross_page_bbox_issue():
    """Test that demonstrates the bbox issue when elements from different pages are merged."""

    # Create mock pages
    page1 = MockPage(number=1, width=612, height=792, id="page1")
    page2 = MockPage(number=2, width=612, height=792, id="page2")

    # Create regions that would be connected if page wasn't considered
    region1 = Region(page1, (100, 750, 200, 770))  # Near bottom of page 1
    region1.label = "Bottom of page 1"

    region2 = Region(page2, (100, 30, 200, 50))  # Near top of page 2
    region2.label = "Top of page 2"

    collection = ElementCollection([region1, region2])

    # Dissolve with large padding - this might incorrectly merge across pages
    # if the implementation doesn't check page boundaries
    dissolved = collection.dissolve(padding=50.0)

    # Should have 2 regions, not 1
    assert len(dissolved) == 2, f"Expected 2 dissolved regions (one per page), got {len(dissolved)}"

    # If incorrectly merged, the bbox would span from top=30 to bottom=770
    # which makes no sense for a single page element
    for region in dissolved:
        bbox = region.bbox
        height = bbox[3] - bbox[1]

        # Height should be reasonable for a single element, not spanning 740 points
        assert height < 100, f"Region height {height} suggests cross-page merge (bbox: {bbox})"


def test_dissolve_with_group_by_should_still_respect_pages():
    """Test that even with group_by, dissolve respects page boundaries."""

    page1 = MockPage(number=1, width=612, height=792, id="page1")
    page2 = MockPage(number=2, width=612, height=792, id="page2")

    # Create regions with same font size on different pages
    region1 = Region(page1, (100, 100, 150, 120))
    region1.font_size = 12.0
    region1.label = "A1"

    region2 = Region(page2, (100, 100, 150, 120))
    region2.font_size = 12.0
    region2.label = "A2"

    region3 = Region(page1, (100, 122, 150, 142))
    region3.font_size = 12.0
    region3.label = "B1"

    region4 = Region(page2, (100, 122, 150, 142))
    region4.font_size = 12.0
    region4.label = "B2"

    collection = ElementCollection([region1, region2, region3, region4])

    # Group by size - all regions have the same size
    dissolved = collection.dissolve(padding=2.0, group_by=["font_size"])

    # Should still have 2 regions (one per page), not 1
    assert len(dissolved) == 2, f"Expected 2 regions even with group_by, got {len(dissolved)}"

    # Verify each region is on a single page
    pages = {r.page for r in dissolved}
    assert pages == {page1, page2}, "Should have one region per page"


if __name__ == "__main__":
    # Run the tests
    test_dissolve_should_not_merge_across_pages()
    print("✓ Test 1 passed: dissolve should not merge across pages")

    test_dissolve_cross_page_bbox_issue()
    print("✓ Test 2 passed: cross-page bbox issue test")

    test_dissolve_with_group_by_should_still_respect_pages()
    print("✓ Test 3 passed: group_by should still respect pages")

    print("\nAll tests passed! The issue is confirmed.")
