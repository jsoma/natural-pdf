"""Test the expand() method with single positional argument support."""

import pytest

from natural_pdf import PDF


class TestExpand:
    """Test expand functionality using real PDFs."""

    def test_expand_single_argument(self):
        """Test expand() with single positional argument."""
        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf[0]

        # Test single element
        text_element = page.find("text")
        assert text_element is not None

        original_bbox = text_element.bbox
        expanded = text_element.expand(5)

        # Verify expansion in all directions by 5
        assert abs(expanded.x0 - (original_bbox[0] - 5)) < 0.01
        assert abs(expanded.x1 - (original_bbox[2] + 5)) < 0.01
        assert abs(expanded.top - (original_bbox[1] - 5)) < 0.01
        assert abs(expanded.bottom - (original_bbox[3] + 5)) < 0.01

    def test_expand_keyword_arguments(self):
        """Test expand() with keyword arguments still works."""
        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf[0]

        text_element = page.find("text")
        assert text_element is not None

        # Expand with different amounts in each direction
        expanded = text_element.expand(left=10, right=5, top=3, bottom=7)

        assert abs(expanded.x0 - (text_element.x0 - 10)) < 0.01
        assert abs(expanded.x1 - (text_element.x1 + 5)) < 0.01
        assert abs(expanded.top - (text_element.top - 3)) < 0.01
        assert abs(expanded.bottom - (text_element.bottom + 7)) < 0.01

    def test_collection_expand_single_argument(self):
        """Test ElementCollection.expand() with single positional argument."""
        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf[0]

        # Get first 3 text elements
        text_elements = page.find_all("text")[:3]
        assert len(text_elements) >= 3

        # Expand all elements by 5 in all directions
        expanded_collection = text_elements.expand(5)

        assert len(expanded_collection) == len(text_elements)

        for original, expanded in zip(text_elements, expanded_collection):
            assert abs(expanded.x0 - (original.x0 - 5)) < 0.01
            assert abs(expanded.x1 - (original.x1 + 5)) < 0.01
            assert abs(expanded.top - (original.top - 5)) < 0.01
            assert abs(expanded.bottom - (original.bottom + 5)) < 0.01

    def test_expand_on_rectangles(self):
        """Test expand on rectangle elements."""
        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf[0]

        # Find rectangles
        rects = page.find_all("rect")
        if len(rects) > 0:
            rect = rects[0]

            # Expand by 10
            expanded = rect.expand(10)

            assert abs(expanded.x0 - (rect.x0 - 10)) < 0.01
            assert abs(expanded.x1 - (rect.x1 + 10)) < 0.01
            assert abs(expanded.top - (rect.top - 10)) < 0.01
            assert abs(expanded.bottom - (rect.bottom + 10)) < 0.01

    def test_expand_boundary_clamping(self):
        """Test that expand() clamps to page boundaries."""
        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf[0]

        # Find an element and expand it by a huge amount
        element = page.find("text")
        assert element is not None

        # Expand by 1000 pixels - should be clamped to page boundaries
        expanded = element.expand(1000)

        # Should be clamped to page boundaries
        assert expanded.x0 == 0
        assert expanded.top == 0
        assert expanded.x1 == page.width
        assert expanded.bottom == page.height

    def test_expand_with_factors(self):
        """Test expand() with width and height factors."""
        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf[0]

        # Find a smaller element that won't hit page boundaries when expanded
        elements = page.find_all("text")
        # Find an element that's not too close to the edges
        element = None
        for el in elements:
            if el.x0 > 100 and el.x1 < 400 and el.top > 100 and el.bottom < 600:
                element = el
                break

        assert element is not None, "Could not find suitable element for factor test"

        # Get original dimensions
        original_width = element.x1 - element.x0
        original_height = element.bottom - element.top
        original_center_x = (element.x0 + element.x1) / 2
        original_center_y = (element.top + element.bottom) / 2

        # Use smaller factors to avoid hitting page boundaries
        expanded = element.expand(width_factor=1.5, height_factor=1.2)

        # Check new dimensions
        new_width = expanded.x1 - expanded.x0
        new_height = expanded.bottom - expanded.top
        new_center_x = (expanded.x0 + expanded.x1) / 2
        new_center_y = (expanded.top + expanded.bottom) / 2

        # Check that dimensions are approximately correct
        # Allow for small rounding errors
        assert abs(new_width - (original_width * 1.5)) < 0.1
        assert abs(new_height - (original_height * 1.2)) < 0.1

        # Center should remain approximately the same
        assert abs(new_center_x - original_center_x) < 0.1
        assert abs(new_center_y - original_center_y) < 0.1

    def test_expand_chaining(self):
        """Test chaining expand operations."""
        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf[0]

        # Chain operations: find text below something and expand
        expanded_regions = page.find("text:contains(Date)").below().find_all("rect").expand(5)

        # Should return expanded regions
        assert all(hasattr(r, "x0") for r in expanded_regions)
