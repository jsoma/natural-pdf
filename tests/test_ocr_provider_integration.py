"""End-to-end OCR provider integration tests."""

from __future__ import annotations

import pytest

import natural_pdf as npdf
import natural_pdf.engine_provider as provider_module
from natural_pdf.engine_provider import EngineProvider
from natural_pdf.ocr.engine import OCREngine, TextRegion


class _FakeOCREngine(OCREngine):
    """Minimal in-memory OCR engine used to test provider wiring."""

    def _initialize_model(
        self,
        languages,
        device,
        options,
    ):
        self._languages = languages
        self._device = device
        self._options = options

    def _preprocess_image(self, image):
        return image

    def _process_single_image(self, image, detect_only, options):
        width, height = image.size
        return [
            {
                "bbox": (0, 0, width * 0.25, height * 0.25),
                "text": None if detect_only else "fake-ocr",
                "confidence": 0.99,
            }
        ]

    def _standardize_results(self, raw_results, min_confidence, detect_only):
        regions = []
        for entry in raw_results:
            bbox = self._standardize_bbox(entry["bbox"])
            regions.append(
                TextRegion(
                    bbox=bbox,
                    text="" if detect_only else entry.get("text", ""),
                    confidence=float(entry.get("confidence", 1.0)),
                )
            )
        return regions

    def is_available(self) -> bool:
        return True


def test_page_apply_ocr_uses_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    for capability in ("ocr", "ocr.apply", "ocr.extract"):
        provider.register(capability, "fake", lambda **_: _FakeOCREngine(), replace=True)

    pdf = npdf.PDF("pdfs/tiny-ocr.pdf", text_layer=False)
    page = pdf.pages[0]

    before = len([w for w in page.words if getattr(w, "source", None) == "ocr"])
    page.apply_ocr(engine="fake", resolution=72)
    after = len([w for w in page.words if getattr(w, "source", None) == "ocr"])

    assert after > before, "OCR should add new elements via provider-backed engine"
    pdf.close()


def test_region_apply_ocr_respects_crop_offsets(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    for capability in ("ocr", "ocr.apply", "ocr.extract"):
        provider.register(capability, "fake", lambda **_: _FakeOCREngine(), replace=True)

    pdf = npdf.PDF("pdfs/tiny-ocr.pdf", text_layer=False)
    try:
        page = pdf.pages[0]
        region = page.create_region(100, 150, 220, 260)

        region.apply_ocr(engine="fake", resolution=72)

        ocr_words = [
            word
            for word in page.words
            if getattr(word, "source", None) == "ocr" and getattr(word, "text", None) == "fake-ocr"
        ]
        assert ocr_words, "Expected OCR words anchored to the crop region"
        word = ocr_words[0]

        assert word.x0 == pytest.approx(region.x0, abs=1e-3)
        assert word.top == pytest.approx(region.top, abs=1e-3)
        expected_width = region.width * 0.25
        assert word.width == pytest.approx(expected_width, rel=1e-3, abs=1e-3)
    finally:
        pdf.close()
