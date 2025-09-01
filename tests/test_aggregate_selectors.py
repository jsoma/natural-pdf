"""Test aggregate selectors in natural-pdf."""

import pytest

from natural_pdf import PDF
from natural_pdf.elements.element_collection import ElementCollection


class TestAggregateSelectors:
    """Test aggregate functions in selectors like min(), max(), avg(), etc."""

    @pytest.fixture
    def sample_pdf(self):
        """Create a sample PDF for testing."""
        # This assumes we have a test PDF with various text sizes and positions
        pdf = PDF("tests/fixtures/sample.pdf")
        return pdf

    def test_min_max_coordinates(self, sample_pdf):
        """Test min/max functions with coordinates."""
        page = sample_pdf.pages[0]

        # Find leftmost text
        leftmost = page.find("text[x0=min()]")
        assert leftmost is not None

        # Verify it really is the leftmost
        all_text = page.find_all("text")
        min_x0 = min(t.x0 for t in all_text)
        assert leftmost.x0 == min_x0

        # Find rightmost text
        rightmost = page.find("text[x1=max()]")
        assert rightmost is not None
        max_x1 = max(t.x1 for t in all_text)
        assert rightmost.x1 == max_x1

        # Find topmost and bottommost
        topmost = page.find("text[y0=min()]")
        bottommost = page.find("text[y1=max()]")
        assert topmost is not None
        assert bottommost is not None

    def test_avg_median_functions(self, sample_pdf):
        """Test average and median functions."""
        page = sample_pdf.pages[0]

        # Find text with size equal to average
        avg_size_text = page.find_all("text[size=avg()]")

        # Calculate expected average
        all_text = page.find_all("text")
        if all_text:
            avg_size = sum(t.size for t in all_text if hasattr(t, "size") and t.size) / len(
                [t for t in all_text if hasattr(t, "size") and t.size]
            )

            # Should find text with that exact size (if any)
            for text in avg_size_text:
                assert abs(text.size - avg_size) < 0.01

        # Test median
        median_width_text = page.find_all("text[width=median()]")
        assert isinstance(median_width_text, ElementCollection)

    def test_mean_alias(self, sample_pdf):
        """Test that mean() is an alias for avg()."""
        page = sample_pdf.pages[0]

        avg_result = page.find_all("text[x0=avg()]")
        mean_result = page.find_all("text[x0=mean()]")

        # Should return same results
        assert len(avg_result) == len(mean_result)
        if avg_result and mean_result:
            assert avg_result.first.x0 == mean_result.first.x0

    def test_mode_most_common(self, sample_pdf):
        """Test mode/most_common functions."""
        page = sample_pdf.pages[0]

        # Find text with most common font size
        mode_size_text = page.find_all("text[size=mode()]")

        # Test alias
        most_common_size_text = page.find_all("text[size=most_common()]")

        assert len(mode_size_text) == len(most_common_size_text)

        # Mode should work on any attribute
        mode_color_text = page.find_all("text[color=mode()]")
        assert isinstance(mode_color_text, ElementCollection)

    def test_closest_color_function(self, sample_pdf):
        """Test closest() function for colors."""
        page = sample_pdf.pages[0]

        # Find text closest to red
        red_text = page.find_all('text[color=closest("red")]')
        assert isinstance(red_text, ElementCollection)

        # Find rects closest to blue
        blue_rects = page.find_all('rect[fill=closest("blue")]')
        assert isinstance(blue_rects, ElementCollection)

        # Test with hex color
        hex_text = page.find_all('text[color=closest("#FF0000")]')
        assert isinstance(hex_text, ElementCollection)

    def test_multiple_aggregates(self, sample_pdf):
        """Test multiple aggregate conditions (intersection)."""
        page = sample_pdf.pages[0]

        # Find element that is both leftmost AND has max size
        result = page.find("text[x0=min()][size=max()]")

        if result:
            # Verify it satisfies both conditions
            all_text = page.find_all("text")
            min_x0 = min(t.x0 for t in all_text)
            max_size = max(t.size for t in all_text if hasattr(t, "size") and t.size)

            assert result.x0 == min_x0
            assert result.size == max_size

    def test_aggregates_with_filters(self, sample_pdf):
        """Test aggregates combined with other filters."""
        page = sample_pdf.pages[0]

        # Find leftmost among bold text
        leftmost_bold = page.find("text[bold][x0=min()]")

        if leftmost_bold:
            # Verify it's bold
            assert leftmost_bold.bold

            # Verify it's leftmost among ALL text (not just bold)
            all_text = page.find_all("text")
            min_x0 = min(t.x0 for t in all_text)
            assert leftmost_bold.x0 == min_x0

    def test_aggregates_in_or_selectors(self, sample_pdf):
        """Test aggregates in OR selectors."""
        page = sample_pdf.pages[0]

        # Find either leftmost text OR largest rect
        result = page.find_all("text[x0=min()]|rect[area=max()]")
        assert isinstance(result, ElementCollection)

        # With comma separator
        result2 = page.find_all("text[x0=min()],rect[width=max()]")
        assert isinstance(result2, ElementCollection)

    def test_empty_collection_handling(self, sample_pdf):
        """Test aggregate functions on empty collections."""
        page = sample_pdf.pages[0]

        # Search for non-existent element type
        result = page.find("nonexistent[x0=min()]")
        assert result is None

        result_all = page.find_all("nonexistent[size=max()]")
        assert len(result_all) == 0

    def test_non_numeric_aggregates(self, sample_pdf):
        """Test aggregate functions on non-numeric attributes."""
        page = sample_pdf.pages[0]

        # Mode works on strings
        mode_font = page.find_all("text[fontname=mode()]")
        assert isinstance(mode_font, ElementCollection)

        # Min/max on strings should work (alphabetical)
        min_text = page.find("text[text=min()]")
        if min_text and hasattr(min_text, "text"):
            all_texts = [t.text for t in page.find_all("text") if hasattr(t, "text") and t.text]
            if all_texts:
                assert min_text.text == min(all_texts)

    def test_aggregates_with_spatial_methods(self, sample_pdf):
        """Test using aggregates in spatial navigation methods."""
        page = sample_pdf.pages[0]

        # Get some element
        element = page.find("text")
        if element:
            # Navigate right until leftmost text
            result = element.right(until="text[x0=min()]")
            assert isinstance(result, ElementCollection)

            # Navigate below to largest text
            result2 = element.below(until="text[size=max()]")
            assert isinstance(result2, ElementCollection)

    def test_aggregates_different_element_types(self, sample_pdf):
        """Test that aggregates are calculated per element type."""
        page = sample_pdf.pages[0]

        # Min x0 for text should be different from min x0 for rects
        min_text = page.find("text[x0=min()]")
        min_rect = page.find("rect[x0=min()]")

        if min_text and min_rect:
            # They might be different (unless coincidentally same)
            all_text_x0 = [t.x0 for t in page.find_all("text")]
            all_rect_x0 = [r.x0 for r in page.find_all("rect")]

            if all_text_x0 and all_rect_x0:
                assert min_text.x0 == min(all_text_x0)
                assert min_rect.x0 == min(all_rect_x0)


class TestAggregateEdgeCases:
    """Test edge cases and error conditions for aggregate selectors."""

    def test_invalid_aggregate_syntax(self):
        """Test invalid aggregate function syntax."""
        from natural_pdf.selectors.parser import _parse_aggregate_function, parse_selector

        # These should not be parsed as aggregates
        assert _parse_aggregate_function("minimum()") is None
        assert _parse_aggregate_function("max") is None  # Missing parentheses
        assert _parse_aggregate_function("max(]") is None  # Invalid syntax

        # These should parse correctly
        assert _parse_aggregate_function("min()") is not None
        assert _parse_aggregate_function("max()") is not None
        assert _parse_aggregate_function("closest('red')") is not None

    def test_closest_without_color_arg(self):
        """Test closest() without argument should fail."""
        from natural_pdf.selectors.parser import parse_selector

        selector = parse_selector("text[color=closest()]")
        # Should parse but have None args
        attr = selector["attributes"][0]
        assert attr["value"]["func"] == "closest"
        assert attr["value"]["args"] is None

    def test_aggregate_on_missing_attribute(self, sample_pdf):
        """Test aggregate functions on attributes that don't exist."""
        page = sample_pdf.pages[0]

        # Try to find max of non-existent attribute
        result = page.find("text[nonexistent=max()]")
        assert result is None

        result_all = page.find_all("text[fakeprop=min()]")
        assert len(result_all) == 0
