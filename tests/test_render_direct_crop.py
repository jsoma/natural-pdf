from pathlib import Path

import pytest
from PIL import ImageChops

from natural_pdf import PDF
from natural_pdf.utils.visualization import render_cropped_page, render_plain_page

PDF_PATH = Path("pdfs/01-practice.pdf")


def test_direct_crop_matches_full_render_then_crop_pixels():
    if not PDF_PATH.exists():
        pytest.skip("Test requires pdfs/01-practice.pdf fixture")

    pdf = PDF(str(PDF_PATH))
    try:
        page = pdf.pages[0]
        resolution = 150
        scale = resolution / 72.0
        crop_bbox = (
            page.width * 0.2,
            page.height * 0.2,
            page.width * 0.6,
            page.height * 0.45,
        )

        full_image = render_plain_page(page, resolution=resolution)
        expected = full_image.crop(
            (
                int(crop_bbox[0] * scale),
                int(crop_bbox[1] * scale),
                int(crop_bbox[2] * scale),
                int(crop_bbox[3] * scale),
            )
        )
        direct = render_cropped_page(page, resolution=resolution, crop_bbox=crop_bbox)

        assert direct.size == expected.size
        assert ImageChops.difference(direct, expected).getbbox() is None
    finally:
        pdf.close()
