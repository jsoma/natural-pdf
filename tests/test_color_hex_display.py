"""
Test that colors are displayed as hex values in group_by operations.
"""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from natural_pdf.utils.color_utils import (
    COLOR_ATTRIBUTES,
    format_color_value,
    is_color_attribute,
    rgb_to_hex,
)


class TestColorUtils:
    """Test the color utility functions."""

    def test_rgb_to_hex_normalized(self):
        """Test RGB to hex conversion with normalized values [0,1]."""
        assert rgb_to_hex((1, 0, 0)) == "#ff0000"  # Red
        assert rgb_to_hex((0, 1, 0)) == "#00ff00"  # Green
        assert rgb_to_hex((0, 0, 1)) == "#0000ff"  # Blue
        assert rgb_to_hex((0.5, 0.5, 0.5)) == "#7f7f7f"  # Gray
        assert rgb_to_hex((1, 1, 1)) == "#ffffff"  # White
        assert rgb_to_hex((0, 0, 0)) == "#000000"  # Black

    def test_rgb_to_hex_255_scale(self):
        """Test RGB to hex conversion with 0-255 scale values."""
        assert rgb_to_hex((255, 0, 0)) == "#ff0000"  # Red
        assert rgb_to_hex((0, 255, 0)) == "#00ff00"  # Green
        assert rgb_to_hex((0, 0, 255)) == "#0000ff"  # Blue
        assert rgb_to_hex((128, 128, 128)) == "#808080"  # Gray

    def test_rgb_to_hex_with_alpha(self):
        """Test that RGBA values work (alpha is ignored)."""
        assert rgb_to_hex((1, 0, 0, 0.5)) == "#ff0000"  # Red with alpha
        assert rgb_to_hex((255, 128, 0, 128)) == "#ff8000"  # Orange with alpha

    def test_rgb_to_hex_edge_cases(self):
        """Test edge cases for RGB conversion."""
        # Values outside normal range get clamped
        # When max value > 1, assumes 0-255 scale
        assert rgb_to_hex((1.5, 0, -0.5)) == "#010000"  # 1.5 rounds to 1, -0.5 clamps to 0
        assert rgb_to_hex((300, -50, 128)) == "#ff0080"  # 300 clamps to 255, -50 clamps to 0

        # Lists work too
        assert rgb_to_hex([1, 0.5, 0]) == "#ff7f00"  # Max is 1, so 0-1 scale

        # More edge cases
        assert rgb_to_hex((0.5, 0.5, 0.5)) == "#7f7f7f"  # 0-1 scale
        assert rgb_to_hex((128, 128, 128)) == "#808080"  # 0-255 scale

    def test_rgb_to_hex_errors(self):
        """Test error cases for RGB conversion."""
        with pytest.raises(ValueError):
            rgb_to_hex((1, 0))  # Too few values
        with pytest.raises(ValueError):
            rgb_to_hex("red")  # Not a tuple/list
        with pytest.raises(ValueError):
            rgb_to_hex(None)  # None value

    def test_is_color_attribute(self):
        """Test color attribute detection."""
        # Known color attributes
        assert is_color_attribute("fill")
        assert is_color_attribute("stroke")
        assert is_color_attribute("color")
        assert is_color_attribute("COLOR")  # Case insensitive
        assert is_color_attribute("Fill")
        assert is_color_attribute("highlight_color")

        # Non-color attributes
        assert not is_color_attribute("size")
        assert not is_color_attribute("font")
        assert not is_color_attribute("text")
        assert not is_color_attribute("bbox")

    def test_format_color_value(self):
        """Test the color value formatting function."""
        # Color tuples get converted to hex
        assert format_color_value((1, 0, 0)) == "#ff0000"
        assert format_color_value((255, 128, 0)) == "#ff8000"
        assert format_color_value((0.5, 0.5, 0.5, 1.0)) == "#7f7f7f"  # RGBA

        # With attribute name
        assert format_color_value((1, 0, 0), "fill") == "#ff0000"
        assert format_color_value((1, 0, 0), "size") == "(1, 0, 0)"  # Non-color attr, no conversion

        # Non-color values pass through
        assert format_color_value("red") == "red"
        assert format_color_value(12) == "12"
        assert format_color_value(None) == "None"
        assert format_color_value([1, 2]) == "[1, 2]"  # Wrong length

        # Invalid color tuples fall back to string
        assert format_color_value((1, "green", 0)) == "(1, 'green', 0)"
        assert format_color_value((256, 256, 256)) == "(256, 256, 256)"  # Out of valid range


class TestGroupByColorDisplay:
    """Test that group_by displays colors as hex."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_page_groupby_hex_display(self, mock_stdout):
        """Test PageGroupBy shows hex colors."""
        from natural_pdf.core.page_collection import PageCollection
        from natural_pdf.core.page_groupby import PageGroupBy

        # Mock pages with different fill colors
        mock_pages = []
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for i, color in enumerate(colors):
            page = MagicMock()
            # PageGroupBy groups by the text content of the element found
            # So we need to return the color tuple as the extracted text
            page.find.return_value = MagicMock(extract_text=lambda c=color: c)
            mock_pages.append(page)

        collection = PageCollection(mock_pages)
        grouped = PageGroupBy(collection, "fill", show_progress=False)

        # Call show() which should display hex colors
        grouped.show()
        output = mock_stdout.getvalue()

        # Should show hex values, not RGB tuples
        assert "#ff0000" in output  # Red
        assert "#00ff00" in output  # Green
        assert "#0000ff" in output  # Blue

    @patch("sys.stdout", new_callable=StringIO)
    def test_page_groupby_info_hex_display(self, mock_stdout):
        """Test PageGroupBy.info() shows hex colors."""
        from natural_pdf.core.page_collection import PageCollection
        from natural_pdf.core.page_groupby import PageGroupBy

        # Mock pages
        mock_pages = []
        colors = [(255, 0, 0), (0, 255, 0)]
        for color in colors:
            page = MagicMock()
            page.find.return_value = MagicMock(extract_text=lambda c=color: c)
            mock_pages.append(page)

        collection = PageCollection(mock_pages)
        grouped = PageGroupBy(collection, "stroke", show_progress=False)

        # Call info()
        grouped.info()
        output = mock_stdout.getvalue()

        # Should contain hex values
        assert "#ff0000" in output
        assert "#00ff00" in output


class TestElementCollectionGroupBy:
    """Test ElementCollection group_by color formatting."""

    def test_format_group_label_with_colors(self):
        """Test _format_group_label converts colors to hex."""
        from natural_pdf.elements.element_collection import ElementCollection

        # Create a mock element
        mock_element = MagicMock()
        mock_element.__dict__ = {"size": 12, "font": "Arial"}

        collection = ElementCollection([mock_element])

        # Test color formatting
        label = collection._format_group_label(
            group_key=(1, 0, 0),
            label_format=None,
            sample_element=mock_element,
            group_by_attr="fill",
        )
        assert label == "#ff0000"

        # Test with custom format
        label = collection._format_group_label(
            group_key=(0, 0, 1),
            label_format="Color: {fill}",
            sample_element=mock_element,
            group_by_attr="fill",
        )
        assert label == "Color: #0000ff"

        # Test non-color attribute
        label = collection._format_group_label(
            group_key=12, label_format=None, sample_element=mock_element, group_by_attr="size"
        )
        assert label == "12"
