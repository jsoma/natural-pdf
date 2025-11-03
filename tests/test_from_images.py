"""Targeted coverage for PDF.from_images."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import natural_pdf as npdf

pytestmark = [pytest.mark.slow, pytest.mark.ocr]

SAMPLE_PATH = Path("pdfs/images/practice_page_1.png")
HOPPER_PATH = Path("pdfs/images/hopper.png")


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Required fixture missing: {path}")


@pytest.mark.parametrize(
    "source, expected_pages",
    [
        (lambda: str(SAMPLE_PATH), 1),
        (lambda: SAMPLE_PATH, 1),
        (lambda: Image.new("RGB", (120, 160), "red"), 1),
        (lambda: [Image.new("RGB", (50, 60), "green"), Image.new("RGB", (40, 80), "blue")], 2),
        (lambda: [Image.new("RGB", (64, 64), "yellow"), str(SAMPLE_PATH)], 2),
    ],
)
def test_from_images_accepts_various_inputs(source, expected_pages):
    _skip_if_missing(SAMPLE_PATH)
    images = source()
    pdf = npdf.PDF.from_images(images, apply_ocr=False)
    try:
        assert len(pdf.pages) == expected_pages
    finally:
        pdf.close()


@pytest.mark.parametrize("mode", ["RGBA", "L"])
def test_from_images_handles_color_modes(mode):
    color = (255, 0, 0, 128) if mode == "RGBA" else 128
    image = Image.new(mode, (90, 120), color)
    pdf = npdf.PDF.from_images(image, apply_ocr=False)
    try:
        assert len(pdf.pages) == 1
    finally:
        pdf.close()


def test_from_images_custom_resolution_and_options():
    image = Image.new("RGB", (200, 200), "purple")
    pdf = npdf.PDF.from_images(
        image,
        resolution=150,
        apply_ocr=False,
        reading_order=False,
        text_layer=False,
    )
    try:
        assert pdf._source_metadata["resolution"] == 150
        assert pdf._text_layer is False
    finally:
        pdf.close()


def test_from_images_calls_ocr_by_default(monkeypatch):
    _skip_if_missing(HOPPER_PATH)
    called = {}

    original = npdf.PDF.apply_ocr

    def fake_apply(self, **kwargs):
        called["kwargs"] = kwargs
        return original(self, **kwargs) if False else []  # avoid heavy OCR

    monkeypatch.setattr(npdf.PDF, "apply_ocr", fake_apply)

    pdf = npdf.PDF.from_images(str(HOPPER_PATH))
    try:
        assert called, "Expected apply_ocr to be invoked"
    finally:
        pdf.close()


def test_from_images_empty_and_invalid_inputs_raise():
    with pytest.raises(Exception):
        npdf.PDF.from_images([], apply_ocr=False)

    with pytest.raises(Exception):
        npdf.PDF.from_images("nonexistent.jpg", apply_ocr=False)


def test_from_images_render_page():
    image = Image.new("RGB", (100, 150), "white")
    pdf = npdf.PDF.from_images(image, apply_ocr=False)
    try:
        rendered = pdf.pages[0].render(resolution=72)
        assert isinstance(rendered, Image.Image)
    finally:
        pdf.close()


def test_from_images_multipage_real_images():
    paths = [Path(f"pdfs/images/multipage_{i}.png") for i in range(1, 4)]
    if not all(p.exists() for p in paths):
        pytest.skip("Multipage fixtures missing")

    pdf = npdf.PDF.from_images([str(p) for p in paths], apply_ocr=False)
    try:
        assert len(pdf.pages) == len(paths)
    finally:
        pdf.close()
