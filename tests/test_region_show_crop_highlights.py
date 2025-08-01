"""Test for region.show(crop=True) disabling highlights by default.

This test ensures that when crop=True is specified on a region's show() method,
highlights are automatically disabled (unless explicitly enabled).
"""

import pytest
from PIL import Image

from natural_pdf import PDF
from natural_pdf.core.render_spec import RenderSpec


class TestRegionShowCropHighlights:
    """Test region.show() behavior with crop parameter."""

    @pytest.fixture
    def sample_pdf(self):
        """Load a sample PDF for testing."""
        pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/multicolumn.pdf")
        return pdf

    def test_show_default_behavior_no_highlights(self, sample_pdf):
        """Test that region.show() default behavior (crop=True) has no highlights."""
        page = sample_pdf.pages[0]
        region = page.region(left=50, right=200, top=50, bottom=200)

        # Get render specs with default behavior (crop=True by default)
        specs = region._get_render_specs(mode="show")

        assert len(specs) == 1
        spec = specs[0]

        # Should NOT have highlights (because default is crop=True)
        assert len(spec.highlights) == 0

        # Should be cropped (default is crop=True)
        assert spec.crop_bbox == region.bbox

    def test_show_with_crop_false_has_highlights(self, sample_pdf):
        """Test that region.show(crop=False) includes highlights."""
        page = sample_pdf.pages[0]
        region = page.region(left=50, right=200, top=50, bottom=200)

        # Get render specs with crop=False
        specs = region._get_render_specs(mode="show", crop=False)

        assert len(specs) == 1
        spec = specs[0]

        # Should have highlights
        assert len(spec.highlights) > 0
        assert spec.highlights[0].get("color") == "blue"  # Default color

        # Should not be cropped
        assert spec.crop_bbox is None

    def test_show_with_crop_true_no_highlights(self, sample_pdf):
        """Test that region.show(crop=True) disables highlights by default."""
        page = sample_pdf.pages[0]
        region = page.region(left=50, right=200, top=50, bottom=200)

        # Get render specs with crop=True
        specs = region._get_render_specs(mode="show", crop=True)

        assert len(specs) == 1
        spec = specs[0]

        # Should NOT have highlights
        assert len(spec.highlights) == 0

        # Should be cropped to region bounds
        assert spec.crop_bbox == region.bbox

    def test_show_with_crop_true_explicit_color_has_highlights(self, sample_pdf):
        """Test that region.show(crop=True, color='red') still shows highlights when explicitly requested."""
        page = sample_pdf.pages[0]
        region = page.region(left=50, right=200, top=50, bottom=200)

        # Get render specs with crop=True but explicit color
        specs = region._get_render_specs(mode="show", crop=True, color="red")

        assert len(specs) == 1
        spec = specs[0]

        # Should have highlights because color was explicitly specified
        assert len(spec.highlights) > 0
        assert spec.highlights[0].get("color") == "red"

        # Should be cropped to region bounds
        assert spec.crop_bbox == region.bbox

    def test_show_with_crop_true_additional_highlights(self, sample_pdf):
        """Test that region.show(crop=True, highlights=[...]) shows additional highlights."""
        page = sample_pdf.pages[0]
        region = page.region(left=50, right=200, top=50, bottom=200)

        # Create another region to highlight
        other_region = page.region(left=100, right=150, top=100, bottom=150)

        # Get render specs with crop=True and additional highlights
        additional_highlights = [
            {"elements": [other_region], "color": "green", "label": "Other Region"}
        ]
        specs = region._get_render_specs(mode="show", crop=True, highlights=additional_highlights)

        assert len(specs) == 1
        spec = specs[0]

        # Should NOT have the default region highlight
        # Should ONLY have the additional highlights
        assert len(spec.highlights) == 1
        assert spec.highlights[0].get("color") == "green"
        assert spec.highlights[0].get("label") == "Other Region"

        # Should be cropped to region bounds
        assert spec.crop_bbox == region.bbox

    def test_visual_output_crop_no_highlights(self, sample_pdf):
        """Test that the actual visual output has no highlights when crop=True."""
        page = sample_pdf.pages[0]
        region = page.region(left=50, right=200, top=50, bottom=200)

        # Generate actual image with crop=True
        img = region.show(crop=True)

        # Should get an image
        assert isinstance(img, Image.Image)
        assert img.width > 0
        assert img.height > 0

        # The image should be cropped (smaller than full page)
        full_page_img = page.show()
        assert img.width < full_page_img.width
        assert img.height < full_page_img.height

    def test_show_with_highlights_false_no_highlights(self, sample_pdf):
        """Test that region.show(highlights=False) disables all highlights."""
        page = sample_pdf.pages[0]
        region = page.region(left=50, right=200, top=50, bottom=200)

        # Get render specs with highlights=False and crop=False
        specs = region._get_render_specs(mode="show", highlights=False, crop=False)

        assert len(specs) == 1
        spec = specs[0]

        # Should NOT have any highlights
        assert len(spec.highlights) == 0

        # Should not be cropped
        assert spec.crop_bbox is None

    def test_show_with_highlights_false_and_crop_true(self, sample_pdf):
        """Test that region.show(highlights=False, crop=True) works correctly."""
        page = sample_pdf.pages[0]
        region = page.region(left=50, right=200, top=50, bottom=200)

        # Get render specs with highlights=False and crop=True
        specs = region._get_render_specs(mode="show", highlights=False, crop=True)

        assert len(specs) == 1
        spec = specs[0]

        # Should NOT have any highlights
        assert len(spec.highlights) == 0

        # Should be cropped
        assert spec.crop_bbox == region.bbox

    def test_show_with_highlights_false_explicit_color_still_no_highlights(self, sample_pdf):
        """Test that region.show(highlights=False, color='red') still has no highlights."""
        page = sample_pdf.pages[0]
        region = page.region(left=50, right=200, top=50, bottom=200)

        # Get render specs with highlights=False and explicit color
        specs = region._get_render_specs(mode="show", highlights=False, color="red")

        assert len(specs) == 1
        spec = specs[0]

        # Should NOT have highlights even with explicit color
        assert len(spec.highlights) == 0

    def test_visual_output_highlights_false(self, sample_pdf):
        """Test that the actual visual output has no highlights when highlights=False."""
        page = sample_pdf.pages[0]
        region = page.region(left=50, right=200, top=50, bottom=200)

        # Generate actual image with highlights=False and crop=False
        img = region.show(highlights=False, crop=False)

        # Should get an image
        assert isinstance(img, Image.Image)
        assert img.width > 0
        assert img.height > 0

        # The image should NOT be cropped (full page)
        full_page_img = page.show()
        assert img.width == full_page_img.width
        assert img.height == full_page_img.height

    def test_visual_output_highlights_false_crop_true(self, sample_pdf):
        """Test combined highlights=False and crop=True for clean cropped output."""
        page = sample_pdf.pages[0]
        region = page.region(left=50, right=200, top=50, bottom=200)

        # Generate actual image with highlights=False and crop=True
        img = region.show(highlights=False, crop=True)

        # Should get an image
        assert isinstance(img, Image.Image)
        assert img.width > 0
        assert img.height > 0

        # The image should be cropped (smaller than full page)
        full_page_img = page.show()
        assert img.width < full_page_img.width
        assert img.height < full_page_img.height
