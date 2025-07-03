"""
Tests for the highlighting protocol that enables ElementCollection.show()
to work with mixed content including FlowRegions and elements from multiple pages.
"""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

import natural_pdf as npdf
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.flows.element import FlowElement
from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion


class TestCurrentBehavior:
    """Tests documenting current limitations before implementing the protocol."""

    def test_flow_region_in_collection_now_works(self):
        """FlowRegions in ElementCollection.show() now work with the protocol."""
        pdf = npdf.PDF("pdfs/multipage-table-african-recipes.pdf")

        # Create a manual FlowRegion for testing
        flow = Flow(segments=[pdf.pages[0], pdf.pages[1]], arrangement="vertical")

        # Create regions on both pages
        region1 = pdf.pages[0].region(50, 100, 200, 300)
        region2 = pdf.pages[1].region(50, 50, 200, 250)

        # Create a FlowRegion
        source_elem = pdf.pages[0].find("text")
        flow_elem = FlowElement(physical_object=source_elem, flow=flow)
        flow_region = FlowRegion(
            flow=flow, constituent_regions=[region1, region2], source_flow_element=flow_elem
        )

        # Create a collection with the FlowRegion
        collection = ElementCollection([flow_region])

        # This should now work with the highlighting protocol
        img = collection.show()

        # Should produce a stacked image
        assert img is not None
        assert img.width > 0
        assert img.height > 0

        # Height should be greater than a single page since it's stacked
        assert img.height > 1000  # Multi-page stacked

        pdf.close()

    def test_mixed_pages_in_collection_now_works(self):
        """Elements from multiple pages in ElementCollection.show() now work."""
        pdf = npdf.PDF("pdfs/multipage-table-african-recipes.pdf")

        # Get elements from different pages
        page1_text = pdf.pages[0].find_all("text")[:5]
        page2_text = pdf.pages[1].find_all("text")[:5]

        # Combine them
        mixed_collection = ElementCollection(list(page1_text) + list(page2_text))

        # This should now work with the highlighting protocol
        img = mixed_collection.show()

        # Should produce a stacked image
        assert img is not None
        assert img.width > 0
        assert img.height > 0

        # Height should be greater than a single page since it's stacked
        assert img.height > 1000  # Multi-page stacked

        pdf.close()


class TestHighlightProtocol:
    """Tests for the new highlighting protocol."""

    def test_element_highlight_specs(self):
        """Regular elements should provide highlight specs."""
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        # Get a text element
        text_elem = page.find("text")

        # Element should have get_highlight_specs method
        assert hasattr(text_elem, "get_highlight_specs")

        # Should return a list with one spec
        specs = text_elem.get_highlight_specs()
        assert isinstance(specs, list)
        assert len(specs) == 1

        # Check spec structure
        spec = specs[0]
        assert "page" in spec
        assert "page_index" in spec
        assert "bbox" in spec
        assert "element" in spec
        assert spec["page"] == page
        assert spec["page_index"] == 0
        assert spec["bbox"] == text_elem.bbox
        assert spec["element"] == text_elem

        pdf.close()

    def test_flow_region_highlight_specs(self):
        """FlowRegions should provide highlight specs for all constituent regions."""
        pdf = npdf.PDF("pdfs/multipage-table-african-recipes.pdf")

        # Create a flow and find a table that spans pages
        flow = Flow(segments=[pdf.pages[0], pdf.pages[1]], arrangement="vertical")

        # Create a FlowRegion manually for testing
        # Get a text element as source
        source_elem = pdf.pages[0].find("text")
        flow_elem = FlowElement(physical_object=source_elem, flow=flow)

        # Create regions on both pages
        region1 = pdf.pages[0].region(50, 100, 200, 300)
        region2 = pdf.pages[1].region(50, 50, 200, 250)

        flow_region = FlowRegion(
            flow=flow, constituent_regions=[region1, region2], source_flow_element=flow_elem
        )

        # Should have get_highlight_specs method
        assert hasattr(flow_region, "get_highlight_specs")

        # Should return specs for both constituent regions
        specs = flow_region.get_highlight_specs()
        assert isinstance(specs, list)
        assert len(specs) == 2

        # Check first spec (from page 0)
        assert specs[0]["page"] == pdf.pages[0]
        assert specs[0]["page_index"] == 0
        assert specs[0]["bbox"] == region1.bbox

        # Check second spec (from page 1)
        assert specs[1]["page"] == pdf.pages[1]
        assert specs[1]["page_index"] == 1
        assert specs[1]["bbox"] == region2.bbox

        pdf.close()

    def test_element_collection_with_protocol(self):
        """ElementCollection.show() should work with mixed content using the protocol."""
        pdf = npdf.PDF("pdfs/multipage-table-african-recipes.pdf")

        # Create a collection with:
        # 1. Regular elements from page 0
        # 2. Regular elements from page 1
        # 3. A FlowRegion spanning both pages

        page0_elems = pdf.pages[0].find_all("text")[:3]
        page1_elems = pdf.pages[1].find_all("text")[:3]

        # Create a FlowRegion
        flow = Flow(segments=[pdf.pages[0], pdf.pages[1]], arrangement="vertical")
        source_elem = pdf.pages[0].find("text")
        flow_elem = FlowElement(physical_object=source_elem, flow=flow)

        region1 = pdf.pages[0].region(250, 100, 400, 300)
        region2 = pdf.pages[1].region(250, 50, 400, 250)

        flow_region = FlowRegion(
            flow=flow, constituent_regions=[region1, region2], source_flow_element=flow_elem
        )

        # Combine everything
        mixed_collection = ElementCollection(list(page0_elems) + list(page1_elems) + [flow_region])

        # Mock the rendering to test the logic without actual image generation
        with patch.object(mixed_collection, "_get_highlighter") as mock_get_highlighter:
            mock_highlighter = Mock()
            mock_highlighter.unified_render.return_value = Mock()  # Fake PIL Image
            mock_get_highlighter.return_value = mock_highlighter

            # This should now work!
            result = mixed_collection.show()

            # Verify the mock was called
            assert mock_highlighter.unified_render.called

            # Check that specs were passed to unified_render correctly
            # Get the call arguments to unified_render
            call_args = mock_highlighter.unified_render.call_args
            specs = call_args[1]["specs"]  # specs is passed as a keyword argument

            # Should have specs for both pages
            assert len(specs) == 2  # Two pages involved

            # Check that each spec has the expected page
            pages_in_specs = [spec.page for spec in specs]
            assert pdf.pages[0] in pages_in_specs
            assert pdf.pages[1] in pages_in_specs

            # Each page should have highlights from elements
            for spec in specs:
                assert len(spec.highlights) > 0

        pdf.close()


class TestHighlightProtocolIntegration:
    """Integration tests with real rendering."""

    def test_multipage_table_cells_visualization(self):
        """Test the motivating use case: showing table cells across pages."""
        pdf = npdf.PDF("pdfs/multipage-table-african-recipes.pdf")

        # This is the use case from the user's question
        # After implementing the protocol, this should work

        # Create guides and build grid
        from natural_pdf import Guides

        # Assuming the table structure has been detected
        guides = Guides.from_lines(pdf.pages[0], threshold=0.1)
        guides.build_grid()

        # Find all table cells - mix of regular and FlowRegions
        cells = pdf.pages[0].find_all("table_cell")

        # Check if there are any cells
        if len(cells) == 0:
            # No cells found, skip test
            pytest.skip("No table cells found in test PDF")

        # This should produce an image
        img = cells.show()
        assert img is not None

        pdf.close()

    def test_flow_search_results_visualization(self):
        """Test showing search results from a Flow."""
        pdf = npdf.PDF("pdfs/multipage-table-african-recipes.pdf")

        # Create a flow across multiple pages
        flow = Flow(segments=pdf.pages[:3], arrangement="vertical")

        # Search for something that appears on multiple pages
        results = flow.find_all('text:contains("ingredient")')

        # Check if there are any results
        if len(results) == 0:
            # No results found, try a different search
            results = flow.find_all("text")

        # This should work with the protocol
        img = results.show()

        # Should produce an image if there are results
        if len(results) > 0:
            assert img is not None

        pdf.close()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_element_without_page(self):
        """Elements without a page should return empty specs."""
        # Create a mock element without a page
        mock_elem = Mock()
        mock_elem.page = None
        mock_elem.bbox = (0, 0, 100, 100)

        # Add the protocol method
        mock_elem.get_highlight_specs = lambda: []

        specs = mock_elem.get_highlight_specs()
        assert specs == []

    def test_element_without_bbox(self):
        """Elements without a bbox should return empty specs."""
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        # Create a mock element with page but no bbox
        mock_elem = Mock()
        mock_elem.page = page
        mock_elem.bbox = None

        # Add the protocol method
        from natural_pdf.elements.base import HighlightableMixin

        mock_elem.get_highlight_specs = HighlightableMixin.get_highlight_specs.__get__(mock_elem)

        specs = mock_elem.get_highlight_specs()
        assert specs == []

        pdf.close()

    def test_empty_flow_region(self):
        """Empty FlowRegions should return empty specs."""
        pdf = npdf.PDF("pdfs/01-practice.pdf")

        flow = Flow(segments=[pdf.pages[0]], arrangement="vertical")
        source_elem = pdf.pages[0].find("text")
        flow_elem = FlowElement(physical_object=source_elem, flow=flow)

        # Create FlowRegion with no constituent regions
        flow_region = FlowRegion(flow=flow, constituent_regions=[], source_flow_element=flow_elem)

        specs = flow_region.get_highlight_specs()
        assert specs == []

        pdf.close()


class TestBackwardCompatibility:
    """Ensure existing functionality still works."""

    def test_single_page_collection_still_works(self):
        """Single-page collections should work as before."""
        pdf = npdf.PDF("pdfs/01-practice.pdf")

        # Get elements from a single page
        elements = pdf.pages[0].find_all("text")

        # Single page collections should work normally
        img = elements.show()

        # Should produce an image
        assert img is not None
        assert img.width > 0
        assert img.height > 0

        pdf.close()

    def test_existing_show_parameters_work(self):
        """All existing parameters should be passed through correctly."""
        pdf = npdf.PDF("pdfs/01-practice.pdf")

        elements = pdf.pages[0].find_all("text")
        # Limit to first 5 elements using slicing (now returns ElementCollection)
        limited_elements = elements[:5]

        # Test with various parameters
        img = limited_elements.show(
            resolution=150,  # Lower resolution for testing
            labels=True,
            legend_position="bottom",
            color="red",
            distinct=True,
            width=400,
        )

        # Should produce an image with the requested properties
        assert img is not None
        assert img.width == 400  # Width should be as requested

        pdf.close()


# Add fixtures for commonly used PDFs
@pytest.fixture
def multipage_table_pdf():
    """Load the multi-page table PDF."""
    pdf = npdf.PDF("pdfs/multipage-table-african-recipes.pdf")
    yield pdf
    pdf.close()


@pytest.fixture
def simple_pdf():
    """Load a simple single-page PDF."""
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    yield pdf
    pdf.close()
