"""Tests for enhanced crop functionality on elements and regions."""

from pathlib import Path

import pytest

import natural_pdf as npdf

PDF_PATH = Path("pdfs/01-practice.pdf")


def _load_sample_pdf() -> npdf.PDF:
    if not PDF_PATH.exists():
        pytest.skip("Test requires pdfs/01-practice.pdf fixture")
    return npdf.PDF(str(PDF_PATH))


def test_crop_modes_across_elements():
    """Exercise a handful of crop modes on a representative element."""
    pdf = _load_sample_pdf()
    try:
        page = pdf.pages[0]
        elements = page.find_all("text")
        if not elements:
            pytest.skip("Sample PDF contains no text elements")

        element = elements[0]
        modes = [True, False, "tight", "loose", 20, 100]

        for mode in modes:
            image = element.render(crop=mode)
            assert image is not None, f"Element render returned None for crop={mode!r}"
    finally:
        pdf.close()


def test_crop_visual_comparison_saves_to_tmp(tmp_path):
    """Generate cropped variants and ensure they save successfully."""
    pdf = _load_sample_pdf()
    try:
        elements = pdf.find_all("text")
        if not elements:
            pytest.skip("Sample PDF contains no text elements")

        target = elements[len(elements) // 2]
        modes = [
            (True, "tight"),
            (50, "padding_50"),
            (100, "padding_100"),
            ("wide", "wide"),
        ]

        for crop_mode, suffix in modes:
            image = target.render(crop=crop_mode)
            assert image is not None, f"Expected image for crop={crop_mode!r}"
            image.save(tmp_path / f"crop_{suffix}.png")
    finally:
        pdf.close()
