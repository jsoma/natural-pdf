"""Test for element.show(crop=True) disabling highlights by default.

This test ensures that when crop=True is specified on an element's show() method,
highlights are automatically disabled (unless explicitly enabled).
"""

import pytest
from PIL import Image

from natural_pdf import PDF
from natural_pdf.core.render_spec import RenderSpec


class TestElementShowCropHighlights:
    """Test element.show() behavior with crop parameter."""

    @pytest.fixture
    def sample_pdf(self):
        """Load a sample PDF for testing."""
        pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/multicolumn.pdf")
        return pdf

    def test_element_show_default_no_crop_has_highlights(self, sample_pdf):
        """Test that element.show() default behavior includes highlights."""
        page = sample_pdf.pages[0]
        # Find a text element
        element = page.find('text:contains("This is some text")')

        # Get render specs with default behavior (crop=False by default for elements)
        specs = element._get_render_specs(mode="show")

        assert len(specs) == 1
        spec = specs[0]

        # Should have highlights
        assert len(spec.highlights) > 0
        assert spec.highlights[0].get("color") == "red"  # Default color for elements

        # Should not be cropped
        assert spec.crop_bbox is None

    def test_element_show_with_crop_false_has_highlights(self, sample_pdf):
        """Test that element.show(crop=False) includes highlights."""
        page = sample_pdf.pages[0]
        element = page.find('text:contains("This is some text")')

        # Get render specs with crop=False
        specs = element._get_render_specs(mode="show", crop=False)

        assert len(specs) == 1
        spec = specs[0]

        # Should have highlights
        assert len(spec.highlights) > 0
        assert spec.highlights[0].get("color") == "red"  # Default color for elements

        # Should not be cropped
        assert spec.crop_bbox is None

    def test_element_show_with_crop_true_no_highlights(self, sample_pdf):
        """Test that element.show(crop=True) disables highlights by default."""
        page = sample_pdf.pages[0]
        element = page.find('text:contains("This is some text")')

        # Get render specs with crop=True
        specs = element._get_render_specs(mode="show", crop=True)

        assert len(specs) == 1
        spec = specs[0]

        # Should NOT have highlights
        assert len(spec.highlights) == 0

        # Should be cropped to element bounds
        assert spec.crop_bbox == element.bbox

    def test_element_show_with_crop_true_explicit_color_has_highlights(self, sample_pdf):
        """Test that element.show(crop=True, color='blue') still shows highlights when explicitly requested."""
        page = sample_pdf.pages[0]
        element = page.find('text:contains("This is some text")')

        # Get render specs with crop=True but explicit color
        specs = element._get_render_specs(mode="show", crop=True, color="blue")

        assert len(specs) == 1
        spec = specs[0]

        # Should have highlights because color was explicitly specified
        assert len(spec.highlights) > 0
        assert spec.highlights[0].get("color") == "blue"

        # Should be cropped to element bounds
        assert spec.crop_bbox == element.bbox

    def test_element_show_with_highlights_false_no_highlights(self, sample_pdf):
        """Test that element.show(highlights=False) disables all highlights."""
        page = sample_pdf.pages[0]
        element = page.find('text:contains("This is some text")')

        # Get render specs with highlights=False
        specs = element._get_render_specs(mode="show", highlights=False)

        assert len(specs) == 1
        spec = specs[0]

        # Should NOT have any highlights
        assert len(spec.highlights) == 0

        # Should not be cropped by default
        assert spec.crop_bbox is None

    def test_image_element_show_crop_true_no_highlights(self):
        """Test that ImageElement.show(crop=True) disables highlights by default."""
        # Use classified.pdf which has images as shown in the user's screenshot
        pdf = PDF("pdfs/classified.pdf")
        page = pdf.pages[0]

        # Find an image
        image_element = page.find("image")
        assert image_element is not None, "No image found in classified.pdf"

        # Get render specs with crop=True
        specs = image_element._get_render_specs(mode="show", crop=True)

        assert len(specs) == 1
        spec = specs[0]

        # Should NOT have highlights
        assert len(spec.highlights) == 0

        # Should be cropped to image bounds
        assert spec.crop_bbox == image_element.bbox

    def test_visual_output_element_crop_no_highlights(self, sample_pdf):
        """Test that the actual visual output has no highlights when crop=True."""
        page = sample_pdf.pages[0]
        element = page.find('text:contains("This is some text")')

        # Generate actual image with crop=True
        img = element.show(crop=True)

        # Should get an image
        assert isinstance(img, Image.Image)
        assert img.width > 0
        assert img.height > 0

        # The image should be cropped (smaller than full page)
        full_page_img = page.show()
        assert img.width < full_page_img.width
        assert img.height < full_page_img.height

    def test_visual_output_element_highlights_false_crop_true(self, sample_pdf):
        """Test combined highlights=False and crop=True for clean cropped output."""
        page = sample_pdf.pages[0]
        element = page.find('text:contains("This is some text")')

        # Generate actual image with highlights=False and crop=True
        img = element.show(highlights=False, crop=True)

        # Should get an image
        assert isinstance(img, Image.Image)
        assert img.width > 0
        assert img.height > 0

        # The image should be cropped
        full_page_img = page.show()
        assert img.width < full_page_img.width
        assert img.height < full_page_img.height
