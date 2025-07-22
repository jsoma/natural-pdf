"""Regression test for page.highlight() method with regions.

This test ensures that the highlight() method from Visualizable
properly handles multiple regions as elements, not as color parameters.
"""

import logging

import pytest
from PIL import Image

from natural_pdf import PDF


class TestHighlightRegions:
    """Test highlighting multiple regions works correctly."""

    @pytest.fixture
    def sample_pdf(self):
        """Load a sample PDF for testing."""
        pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/multicolumn.pdf")
        return pdf

    def test_highlight_multiple_regions(self, sample_pdf, caplog):
        """Test that page.highlight() works with multiple regions."""
        page = sample_pdf.pages[0]

        # Create three regions dividing the page into columns
        left = page.region(left=0, right=page.width / 3, top=0, bottom=page.height)
        mid = page.region(left=page.width / 3, right=page.width / 3 * 2, top=0, bottom=page.height)
        right = page.region(left=page.width / 3 * 2, right=page.width, top=0, bottom=page.height)

        # Clear any existing log records
        caplog.clear()

        # This should work without warnings about "Invalid color input type"
        with caplog.at_level(logging.WARNING):
            img = page.highlight(left, mid, right)

        # Verify we get a PIL Image back
        assert isinstance(img, Image.Image)
        assert img.width > 0
        assert img.height > 0

        # Most importantly: verify no warnings about invalid color type
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        for msg in warning_messages:
            assert "Invalid color input type" not in msg, f"Got unexpected warning: {msg}"

    def test_highlight_regions_with_colors(self, sample_pdf):
        """Test highlighting regions with custom colors."""
        page = sample_pdf.pages[0]

        # Create regions
        top_half = page.region(left=0, right=page.width, top=0, bottom=page.height / 2)
        bottom_half = page.region(left=0, right=page.width, top=page.height / 2, bottom=page.height)

        # Highlight with colors
        img = page.highlight((top_half, "red"), (bottom_half, "blue"))

        assert isinstance(img, Image.Image)

    def test_highlight_regions_with_colors_and_labels(self, sample_pdf):
        """Test highlighting regions with colors and labels."""
        page = sample_pdf.pages[0]

        # Create quadrants
        top_left = page.region(left=0, right=page.width / 2, top=0, bottom=page.height / 2)
        top_right = page.region(
            left=page.width / 2, right=page.width, top=0, bottom=page.height / 2
        )
        bottom_left = page.region(
            left=0, right=page.width / 2, top=page.height / 2, bottom=page.height
        )
        bottom_right = page.region(
            left=page.width / 2, right=page.width, top=page.height / 2, bottom=page.height
        )

        # Highlight with colors and labels
        img = page.highlight(
            (top_left, "red", "Top Left"),
            (top_right, "green", "Top Right"),
            (bottom_left, "blue", "Bottom Left"),
            (bottom_right, "yellow", "Bottom Right"),
        )

        assert isinstance(img, Image.Image)

    def test_highlight_mixed_elements(self, sample_pdf):
        """Test highlighting a mix of regions and elements."""
        page = sample_pdf.pages[0]

        # Create a region
        header_region = page.region(left=0, right=page.width, top=0, bottom=100)

        # Find some text elements
        text_elements = page.find_all("text")[:3]  # First 3 text elements

        if text_elements:
            # Mix regions and elements
            img = page.highlight(header_region, *text_elements.elements)
            assert isinstance(img, Image.Image)

    def test_add_highlight_still_works(self, sample_pdf):
        """Test that the renamed add_highlight method still works."""
        page = sample_pdf.pages[0]

        # The old highlight method is now add_highlight
        result = page.add_highlight(bbox=(100, 100, 200, 200), color="red", label="Test Box")

        # add_highlight returns the page for chaining
        assert result is page

        # Now show the page with the highlight
        img = page.show()
        assert isinstance(img, Image.Image)

    def test_highlight_vs_add_highlight_difference(self, sample_pdf):
        """Test the difference between highlight() and add_highlight()."""
        page = sample_pdf.pages[0]

        # Create a region
        region = page.region(left=50, right=150, top=50, bottom=150)

        # highlight() returns an image immediately
        img1 = page.highlight(region)
        assert isinstance(img1, Image.Image)

        # add_highlight() returns the page for chaining
        page_result = page.add_highlight(bbox=region.bbox, color="blue")
        assert page_result is page

        # To see the result of add_highlight, we need to call show()
        img2 = page.show()
        assert isinstance(img2, Image.Image)

    def test_highlight_empty_args(self, sample_pdf):
        """Test highlight with no arguments."""
        page = sample_pdf.pages[0]

        # Should return an image even with no highlights
        img = page.highlight()
        assert isinstance(img, Image.Image)

    def test_highlight_with_kwargs(self, sample_pdf):
        """Test highlight method passes kwargs correctly."""
        page = sample_pdf.pages[0]

        region = page.region(left=0, right=100, top=0, bottom=100)

        # Pass additional parameters
        img = page.highlight(region, width=800, labels=True)
        assert isinstance(img, Image.Image)
        # Width parameter should affect the output
        assert img.width == 800


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
