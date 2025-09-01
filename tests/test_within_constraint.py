"""Test for within= parameter and .within() context manager for constraining directional operations."""

import pytest

from natural_pdf import PDF


class TestWithinConstraint:
    """Test suite for region-constrained directional operations."""

    @pytest.fixture
    def pdf(self):
        """Load the practice PDF."""
        return PDF("pdfs/01-practice.pdf")

    @pytest.fixture
    def first_page(self, pdf):
        """Get the first page."""
        return pdf.pages[0]

    def test_within_parameter_basic(self, first_page):
        """Test basic within= parameter functionality."""
        # Create a region for the left half of the page
        left_half = first_page.region(right=first_page.width / 2)

        # Find an element in the left region
        element = left_half.find("text")
        assert element is not None

        # Search below within the constrained region
        below_result = element.below(within=left_half)

        # The result should be within the left half
        assert below_result.x1 <= first_page.width / 2

    def test_within_parameter_with_until(self, first_page):
        """Test within= parameter with until clause."""
        # Create a narrow column region
        left_column = first_page.region(left=50, right=200)

        # Find a text element in the column
        start_element = left_column.find("text")
        assert start_element is not None

        # Search below until another text, constrained to column
        result = start_element.below(until="text", within=left_column)

        # Verify the endpoint is within the column bounds
        if result.endpoint:
            assert result.endpoint.x0 >= 50
            assert result.endpoint.x1 <= 200

    def test_within_context_manager_single_operation(self, first_page):
        """Test .within() context manager for a single operation."""
        # Create a region for the right half of the page
        right_half = first_page.region(left=first_page.width / 2)

        # Find an element anywhere on the page
        element = first_page.find("text")

        # Use context manager to constrain operations
        with right_half.within():
            below_result = element.below()

            # Result should be constrained to right half
            assert below_result.x0 >= first_page.width / 2

    def test_within_context_manager_multiple_operations(self, first_page):
        """Test .within() context manager affects multiple operations."""
        # Create a central region
        center_region = first_page.region(
            left=first_page.width * 0.25, right=first_page.width * 0.75
        )

        # Find elements
        element1 = first_page.find("text")
        element2 = (
            first_page.find_all("text")[1] if len(first_page.find_all("text")) > 1 else element1
        )

        with center_region.within():
            # Both operations should be constrained
            result1 = element1.below()
            result2 = element2.above()

            # Both results should be within center region
            assert result1.x0 >= first_page.width * 0.25
            assert result1.x1 <= first_page.width * 0.75
            assert result2.x0 >= first_page.width * 0.25
            assert result2.x1 <= first_page.width * 0.75

    def test_within_parameter_overrides_context_manager(self, first_page):
        """Test that explicit within= parameter overrides context manager."""
        left_region = first_page.region(right=first_page.width / 2)
        right_region = first_page.region(left=first_page.width / 2)

        element = first_page.find("text")

        # Context manager sets left region, but parameter overrides to right
        with left_region.within():
            result = element.below(within=right_region)

            # Should be in right region, not left
            assert result.x0 >= first_page.width / 2

    def test_within_horizontal_navigation(self, first_page):
        """Test within constraint for left/right navigation."""
        # Create a top region
        top_region = first_page.region(bottom=first_page.height / 2)

        element = top_region.find("text")

        # Navigate right within top region
        result = element.right(within=top_region)

        # Result should be constrained to top half
        assert result.bottom <= first_page.height / 2

    def test_within_no_results_in_constraint(self, first_page):
        """Test behavior when no elements found within constraint."""
        # Create a small region unlikely to have text
        tiny_region = first_page.region(left=0, right=10, top=0, bottom=10)

        # Find element outside the tiny region
        element = first_page.find("text")

        # Search within tiny region
        result = element.below(until="text", within=tiny_region)

        # Should return a region with no endpoint
        assert result.endpoint is None

    def test_within_multipage_not_supported(self, pdf):
        """Test that within constraint with multipage raises appropriate error."""
        if len(pdf.pages) < 2:
            pytest.skip("Need multi-page PDF for this test")

        first_page = pdf.pages[0]
        region = first_page.region(right=first_page.width / 2)
        element = first_page.find("text")

        # Should raise error or handle gracefully
        with pytest.raises(ValueError, match=".*multipage.*within.*"):
            element.below(within=region, multipage=True)

    def test_within_nested_context_managers(self, first_page):
        """Test nested within() context managers."""
        outer_region = first_page.region(left=50, right=500)
        inner_region = first_page.region(left=100, right=400)

        element = first_page.find("text")

        with outer_region.within():
            # First operation uses outer region
            result1 = element.below()
            assert result1.x0 >= 50
            assert result1.x1 <= 500

            with inner_region.within():
                # Nested operation uses inner region
                result2 = element.below()
                assert result2.x0 >= 100
                assert result2.x1 <= 400

            # Back to outer region
            result3 = element.below()
            assert result3.x0 >= 50
            assert result3.x1 <= 500

    def test_within_with_different_directional_methods(self, first_page):
        """Test within works with all directional methods."""
        region = first_page.region(left=100, right=400, top=100, bottom=500)

        element = region.find("text")
        if not element:
            pytest.skip("No text found in region")

        # Test all four directions with within parameter
        below_result = element.below(within=region)
        above_result = element.above(within=region)
        left_result = element.left(within=region)
        right_result = element.right(within=region)

        # All results should be within region bounds
        for result in [below_result, above_result, left_result, right_result]:
            assert result.x0 >= 100
            assert result.x1 <= 400
            assert result.top >= 100
            assert result.bottom <= 500

    def test_within_context_manager_as_syntax(self, first_page):
        """Test that within() context manager returns the region for 'as' syntax."""
        # Create a region
        region = first_page.region(left=100, right=400)

        # Test that context manager returns the region
        with region.within() as col:
            # Should get the same region back
            assert col is region

            # Can use it to find elements
            element = col.find("text")
            assert element is not None

            # Can access region properties
            assert col.width == 300  # 400 - 100
            assert col.x0 == 100
            assert col.x1 == 400

            # Directional operations are still constrained
            if element:
                result = element.below()
                assert result.x0 >= 100
                assert result.x1 <= 400
