"""Tests that highlight behavior remains intact when cropping regions."""

from pathlib import Path

import pytest

import natural_pdf as npdf

PDF_PATH = Path("pdfs/01-practice.pdf")


def _load_pdf() -> npdf.PDF:
    if not PDF_PATH.exists():
        pytest.skip("Test requires pdfs/01-practice.pdf fixture")
    return npdf.PDF(str(PDF_PATH))


def test_crop_region_preserves_highlights():
    """Verify that cropping to a region retains highlight visibility."""
    pdf = _load_pdf()
    try:
        page = pdf.pages[0]
        text_elements = page.find_all("text")
        if not text_elements:
            pytest.skip("Sample PDF contains no text elements")

        element = text_elements[0]
        section = page.region(0, 0, page.width, page.height / 2)

        tight = element.show(crop=True)
        region_crop = element.show(crop=section)
        padded = element.show(crop=50)
        wide = element.show(crop="wide")

        assert tight is not None
        assert region_crop is not None
        assert padded is not None
        assert wide is not None

        merged = text_elements[:1].merge()
        merged_img = merged.show(crop=section)
        assert merged_img is not None
    finally:
        pdf.close()


def test_highlight_visual_examples(tmp_path):
    """Persist a few highlight variants for manual inspection when needed."""
    pdf = _load_pdf()
    try:
        page = pdf.pages[0]
        text = page.find("text")
        if text is None:
            pytest.skip("Sample PDF contains no text elements")

        crop_region = page.region(0, 0, page.width, page.height / 2)
        scenarios = [
            (True, "tight_crop"),
            (crop_region, "region_crop"),
            (50, "padding_crop"),
            ("wide", "wide_crop"),
        ]

        for crop_mode, suffix in scenarios:
            image = text.show(crop=crop_mode)
            assert image is not None, f"Expected image for crop={crop_mode!r}"
            image.save(tmp_path / f"highlight_{suffix}.png")
    finally:
        pdf.close()
