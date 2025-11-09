"""EngineProvider and OCR integration smoke tests."""

from __future__ import annotations

from PIL import Image

import natural_pdf.engine_provider as provider_module
from natural_pdf.engine_provider import EngineProvider
from natural_pdf.ocr.ocr_manager import run_ocr_engine


def test_engine_provider_caches_instances() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True  # Skip entry point discovery for the test

    call_count = {"value": 0}

    def factory(*, context=None, **opts):
        call_count["value"] += 1
        return object()

    provider.register("demo", "alpha", factory)

    ctx = object()
    first = provider.get("demo", context=ctx, name="alpha")
    second = provider.get("demo", context=ctx, name="alpha")

    assert first is second
    assert call_count["value"] == 1


def test_run_ocr_engine_with_custom_provider(monkeypatch) -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _FakeOCREngine:
        def __init__(self):
            self.calls = []

        def process_image(self, **kwargs):
            self.calls.append(kwargs)
            return [[{"bbox": [0, 0, 1, 1], "text": "hi", "confidence": 0.9}]]

    fake_engine = _FakeOCREngine()

    def factory(*, context=None, **opts):
        return fake_engine

    provider.register("ocr", "test-ocr", factory, replace=True)

    img = Image.new("RGB", (4, 4), color="white")
    result = run_ocr_engine(
        [img],
        context=object(),
        engine_name="test-ocr",
        languages=["en"],
        min_confidence=0.1,
        device="cpu",
        detect_only=False,
        options=None,
    )

    assert fake_engine.calls, "Engine should have been invoked"
    assert isinstance(result, list) and result[0][0]["text"] == "hi"
