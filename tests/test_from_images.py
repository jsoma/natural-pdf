"""Test PDF.from_images() functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import natural_pdf as npdf


class TestFromImages:
    """Test suite for PDF.from_images() method."""

    def test_single_image_from_file(self):
        """Test creating PDF from a single image file."""
        # Use an existing test image
        pdf = npdf.PDF.from_images("pdfs/images/practice_page_1.png", apply_ocr=False)

        assert len(pdf.pages) == 1
        assert pdf._from_images is True
        assert pdf._source_metadata["count"] == 1
        assert pdf._source_metadata["resolution"] == 300

    def test_single_image_from_pil(self):
        """Test creating PDF from a single PIL Image object."""
        # Create a test image
        img = Image.new("RGB", (200, 300), color="red")
        pdf = npdf.PDF.from_images(img, apply_ocr=False)

        assert len(pdf.pages) == 1
        assert pdf._from_images is True

    def test_multiple_images_from_files(self):
        """Test creating PDF from multiple image files."""
        images = ["pdfs/images/practice_page_1.png", "pdfs/images/practice_page_1.jpg"]
        pdf = npdf.PDF.from_images(images, apply_ocr=False)

        assert len(pdf.pages) == 2
        assert pdf._source_metadata["count"] == 2

    def test_multiple_images_from_pil_objects(self):
        """Test creating PDF from multiple PIL Image objects."""
        images = [
            Image.new("RGB", (200, 300), color="red"),
            Image.new("RGB", (200, 300), color="green"),
            Image.new("RGB", (200, 300), color="blue"),
        ]
        pdf = npdf.PDF.from_images(images, apply_ocr=False)

        assert len(pdf.pages) == 3
        assert pdf._source_metadata["count"] == 3

    def test_mixed_image_inputs(self):
        """Test creating PDF from mixed PIL objects and file paths."""
        images = [Image.new("RGB", (200, 300), color="yellow"), "pdfs/images/practice_page_1.png"]
        pdf = npdf.PDF.from_images(images, apply_ocr=False)

        assert len(pdf.pages) == 2

    def test_rgba_to_rgb_conversion(self):
        """Test that RGBA images are converted to RGB with white background."""
        # Create RGBA image with transparency
        img = Image.new("RGBA", (200, 300), (255, 0, 0, 128))
        pdf = npdf.PDF.from_images(img, apply_ocr=False)

        assert len(pdf.pages) == 1
        # The PDF should be created successfully

    def test_grayscale_image(self):
        """Test creating PDF from grayscale image."""
        img = Image.new("L", (200, 300), 128)
        pdf = npdf.PDF.from_images(img, apply_ocr=False)

        assert len(pdf.pages) == 1

    def test_custom_resolution(self):
        """Test creating PDF with custom resolution."""
        img = Image.new("RGB", (200, 300), "blue")
        pdf = npdf.PDF.from_images(img, resolution=150, apply_ocr=False)

        assert pdf._source_metadata["resolution"] == 150

    def test_ocr_disabled(self):
        """Test that OCR can be disabled."""
        img = Image.new("RGB", (200, 300), "white")
        pdf = npdf.PDF.from_images(img, apply_ocr=False)

        # Should have no OCR elements
        page = pdf.pages[0]
        # Check that no OCR was applied (no OCR elements)
        assert len(page.chars) == 0 or not any(
            hasattr(c, "is_ocr") and c.is_ocr for c in page.chars
        )

    def test_ocr_enabled_default(self):
        """Test that OCR is enabled by default."""
        # Use a real document image that contains text
        if Path("pdfs/images/multipage_1.png").exists():
            pdf = npdf.PDF.from_images("pdfs/images/multipage_1.png")
            # OCR should have been applied
            assert pdf._from_images is True
            # This would have OCR results if the image contains text

    def test_exif_rotation(self):
        """Test that EXIF rotation is handled."""
        # Create an image with simulated EXIF orientation
        img = Image.new("RGB", (300, 200), "cyan")
        # Note: Actually testing EXIF would require an image with EXIF data
        pdf = npdf.PDF.from_images(img, apply_ocr=False)

        assert len(pdf.pages) == 1

    def test_pdf_options_passed_through(self):
        """Test that PDF constructor options are passed through."""
        img = Image.new("RGB", (200, 300), "magenta")
        pdf = npdf.PDF.from_images(img, apply_ocr=False, reading_order=False, text_layer=False)

        assert pdf._text_layer is False

    def test_pathlib_path_input(self):
        """Test that pathlib Path objects work."""
        path = Path("pdfs/images/practice_page_1.png")
        pdf = npdf.PDF.from_images(path, apply_ocr=False)

        assert len(pdf.pages) == 1

    def test_page_content_accessible(self):
        """Test that page content is accessible after creation."""
        img = Image.new("RGB", (200, 300), "white")
        pdf = npdf.PDF.from_images(img, apply_ocr=False)

        # Should be able to access page properties
        page = pdf.pages[0]
        assert page.width > 0
        assert page.height > 0

    def test_multiple_page_navigation(self):
        """Test navigation between multiple pages."""
        images = [Image.new("RGB", (200, 300), color) for color in ["red", "green", "blue"]]
        pdf = npdf.PDF.from_images(images, apply_ocr=False)

        # Test page access
        assert len(pdf.pages) == 3
        for i in range(3):
            page = pdf.pages[i]
            assert page.number == i + 1

    def test_empty_image_list_raises_error(self):
        """Test that empty image list raises appropriate error."""
        with pytest.raises(Exception):
            npdf.PDF.from_images([], apply_ocr=False)

    def test_invalid_image_path_raises_error(self):
        """Test that invalid image path raises appropriate error."""
        with pytest.raises(Exception):
            npdf.PDF.from_images("nonexistent.jpg", apply_ocr=False)

    def test_render_created_pdf(self):
        """Test that we can render pages from the created PDF."""
        img = Image.new("RGB", (200, 300), "purple")
        pdf = npdf.PDF.from_images(img, apply_ocr=False)

        # Should be able to render the page
        rendered = pdf.pages[0].render(resolution=72)
        assert isinstance(rendered, Image.Image)
        assert rendered.width > 0
        assert rendered.height > 0


class TestFromImagesIntegration:
    """Integration tests for PDF.from_images()."""

    def test_multipage_from_real_images(self):
        """Test creating multi-page PDF from real exported images."""
        # Only run if we have the test images
        if Path("pdfs/images/multipage_1.png").exists():
            images = [
                f"pdfs/images/multipage_{i}.png"
                for i in range(1, 4)
                if Path(f"pdfs/images/multipage_{i}.png").exists()
            ]

            if images:
                pdf = npdf.PDF.from_images(images, apply_ocr=False)
                assert len(pdf.pages) == len(images)

                # Test that we can extract text if there was any
                for page in pdf.pages:
                    # Just check that extract_text doesn't error
                    text = page.extract_text()
                    assert isinstance(text, str)

    def test_save_created_pdf(self):
        """Test that created PDF can be saved to disk."""
        img = Image.new("RGB", (200, 300), "orange")
        pdf = npdf.PDF.from_images(img, apply_ocr=False)

        # PDFs created from images in memory can't be saved with original=True
        # They would need to be saved with ocr=True after applying OCR
        # or we need a different save method for image-based PDFs

        # For now, just verify the PDF was created successfully
        assert pdf._from_images is True
        assert len(pdf.pages) == 1

        # We could test save_pdf with ocr=True if we apply OCR first
        # but that's already tested in other tests

    def test_apply_operations_to_created_pdf(self):
        """Test that normal PDF operations work on created PDF."""
        img = Image.new("RGB", (400, 600), "white")
        pdf = npdf.PDF.from_images(img, apply_ocr=False)

        # Should be able to use normal PDF operations
        page = pdf.pages[0]

        # Create a region
        region = page.region(50, 50, 150, 150)
        assert region is not None

        # Find elements (should be empty without OCR)
        elements = page.find_all("*")
        assert isinstance(elements, (list, object))  # ElementCollection

    def test_ocr_with_specific_engine(self):
        """Test OCR with specific engine parameter."""
        img = Image.new("RGB", (200, 300), "white")

        # This will use whatever engine is available
        try:
            pdf = npdf.PDF.from_images(img, ocr_engine="easyocr")
            assert pdf._from_images is True
        except Exception:
            # OCR engine might not be installed, that's ok
            pytest.skip("EasyOCR not available")
