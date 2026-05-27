from pathlib import Path

import pytest
from PIL import Image, ImageChops

from natural_pdf import PDF
from natural_pdf.utils.visualization import (
    DirectCropRenderUnsupportedError,
    _validate_direct_crop_size,
    render_cropped_page,
    render_plain_page,
)

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


def test_direct_crop_size_validation_rejects_uncropped_bitmap():
    image = Image.new("RGB", (3400, 4403), "white")

    with pytest.raises(DirectCropRenderUnsupportedError, match="expected \\(3400, 2108\\)"):
        _validate_direct_crop_size(image, (3400, 2108))


def test_direct_crop_unsupported_falls_back(monkeypatch):
    if not PDF_PATH.exists():
        pytest.skip("Test requires pdfs/01-practice.pdf fixture")

    pdf = PDF(str(PDF_PATH))
    try:
        page = pdf.pages[0]
        crop_bbox = (
            page.width * 0.2,
            page.height * 0.2,
            page.width * 0.6,
            page.height * 0.45,
        )

        def unsupported(*args, **kwargs):
            raise DirectCropRenderUnsupportedError("unsupported page wrapper")

        monkeypatch.setattr(
            "natural_pdf.core.highlighting_service.render_cropped_page", unsupported
        )

        image = page.render(crop_bbox=crop_bbox)
        assert image is not None
    finally:
        pdf.close()


def test_direct_crop_unexpected_failure_propagates(monkeypatch):
    if not PDF_PATH.exists():
        pytest.skip("Test requires pdfs/01-practice.pdf fixture")

    pdf = PDF(str(PDF_PATH))
    try:
        page = pdf.pages[0]
        crop_bbox = (
            page.width * 0.2,
            page.height * 0.2,
            page.width * 0.6,
            page.height * 0.45,
        )

        def fail(*args, **kwargs):
            raise RuntimeError("direct crop bug")

        monkeypatch.setattr("natural_pdf.core.highlighting_service.render_cropped_page", fail)

        with pytest.raises(RuntimeError, match="direct crop bug"):
            page.render(crop_bbox=crop_bbox)
    finally:
        pdf.close()
