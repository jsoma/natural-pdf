"""Regression tests for OCR option construction and backend forwarding."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from natural_pdf.ocr.engine_easyocr import EasyOCREngine
from natural_pdf.ocr.ocr_cache import OCRCache, set_default_cache
from natural_pdf.ocr.ocr_options import (
    BaseOCROptions,
    EasyOCROptions,
    PaddleOCROptions,
    PaddleOCRVLOptions,
    RapidOCROptions,
)
from natural_pdf.ocr.ocr_provider import normalize_ocr_options
from natural_pdf.ocr.unified_dispatch import (
    EngineCache,
    EngineEntry,
    OCRRunResult,
    _run_classic,
    run_ocr,
)
from natural_pdf.services.ocr_service import OCRService


def test_mapping_options_are_constructed_for_the_resolved_engine():
    options = normalize_ocr_options(
        {"decoder": "beamsearch", "beamWidth": 9, "batch_size": 3},
        engine_name="easyocr",
    )

    assert isinstance(options, EasyOCROptions)
    assert options.decoder == "beamsearch"
    assert options.beamWidth == 9
    assert options.batch_size == 3


def test_mapping_options_reject_unknown_fields_with_engine_context():
    with pytest.raises(TypeError, match=r"Unsupported option\(s\) for OCR engine 'easyocr': nope"):
        normalize_ocr_options({"nope": True}, engine_name="easyocr")


def test_deferred_mapping_rejects_custom_engine_without_an_options_class(monkeypatch):
    monkeypatch.setattr("natural_pdf.ocr.ocr_provider.get_ocr_options_class", lambda _: None)
    deferred = normalize_ocr_options({"custom_setting": True})

    with pytest.raises(TypeError, match="does not declare an options class"):
        normalize_ocr_options(deferred, engine_name="custom-engine")


def test_mapping_options_reject_an_incompatible_options_instance():
    with pytest.raises(TypeError, match="requires EasyOCROptions; got PaddleOCROptions"):
        normalize_ocr_options(PaddleOCROptions(), engine_name="easyocr")


def test_easyocr_forwards_all_declared_readtext_controls():
    engine = EasyOCREngine()
    reader = MagicMock()
    reader.readtext.return_value = []
    engine._model = reader
    options = EasyOCROptions(
        decoder="beamsearch",
        beamWidth=7,
        batch_size=4,
        workers=2,
        allowlist="ABC",
        blocklist="xyz",
    )

    engine._process_single_image(np.zeros((4, 4, 3), dtype=np.uint8), False, options)

    _, kwargs = reader.readtext.call_args
    assert {key: kwargs[key] for key in ("decoder", "beamWidth", "batch_size", "workers")} == {
        "decoder": "beamsearch",
        "beamWidth": 7,
        "batch_size": 4,
        "workers": 2,
    }
    assert kwargs["allowlist"] == "ABC"
    assert kwargs["blocklist"] == "xyz"


def test_paddle_constructor_identity_includes_lang_and_device():
    base = PaddleOCROptions()
    assert base._init_key() != PaddleOCROptions(lang="fr")._init_key()
    assert base._init_key() != PaddleOCROptions(device="cuda")._init_key()


def test_paddlevl_constructor_identity_includes_extra_args():
    assert (
        PaddleOCRVLOptions(extra_args={"foo": "one"})._init_key()
        != PaddleOCRVLOptions(extra_args={"foo": "two"})._init_key()
    )


class _MutableOptionValue:
    def __repr__(self):
        return "MutableOptionValue()"


def test_noncanonical_option_values_are_explicitly_uncacheable():
    mutable = _MutableOptionValue()

    assert BaseOCROptions(extra_args={"callback": mutable})._cache_key() is None
    assert BaseOCROptions(extra_args={"callback": mutable})._init_key() is None
    assert EasyOCROptions(recog_network=mutable)._init_key() is None
    assert PaddleOCRVLOptions(extra_args={"callback": mutable})._init_key() is None


def test_uncacheable_constructor_identity_bypasses_engine_reuse(monkeypatch):
    created = []
    cleaned = []

    class UncacheableOptions(BaseOCROptions):
        def _init_key(self):
            return None

    class FakeEngine:
        def __init__(self):
            created.append(self)

        def is_available(self):
            return True

        def process_image(self, *args, **kwargs):
            return []

        def cleanup(self):
            cleaned.append(self)

    options = UncacheableOptions(extra_args={"callback": _MutableOptionValue()})
    entry = EngineEntry(
        engine_type="classic",
        provider=FakeEngine,
        options_class=UncacheableOptions,
        needs_gpu_lock=False,
    )
    monkeypatch.setattr(
        "natural_pdf.ocr.ocr_provider.normalize_ocr_options", lambda value, **kwargs: value
    )
    monkeypatch.setattr("natural_pdf.ocr.unified_dispatch._engine_cache", EngineCache(maxsize=4))

    for _ in range(2):
        _run_classic(
            image=Image.new("RGB", (4, 4)),
            engine_name="uncacheable",
            entry=entry,
            languages=["en"],
            min_confidence=None,
            device="cpu",
            detect_only=False,
            options=options,
        )

    assert len(created) == 2
    assert cleaned == created


def test_uncacheable_options_skip_persistent_result_cache(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    previous_cache = set_default_cache(OCRCache(cache_dir=tmp_path / "ocr-cache"))
    try:
        page = SimpleNamespace(
            width=100,
            height=100,
            index=0,
            pdf=SimpleNamespace(_resolved_path=str(pdf_path)),
        )
        page._ocr_scope = lambda: "page"
        page._ocr_render_kwargs = lambda *, apply_exclusions: {"apply_exclusions": apply_exclusions}
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        calls = []
        monkeypatch.setattr(
            "natural_pdf.services.ocr_service.run_ocr",
            lambda **kwargs: calls.append(kwargs)
            or OCRRunResult(results=[], image_size=(100, 100)),
        )
        monkeypatch.setattr(
            service,
            "_process_ocr_payload",
            lambda *args, **kwargs: [],
        )
        options = RapidOCROptions(extra_args={"callback": _MutableOptionValue()})

        service.apply_ocr(page, engine="rapidocr", options=options, replace="none")
        service.apply_ocr(page, engine="rapidocr", options=options, replace="none")

        assert len(calls) == 2
        assert not list((tmp_path / "ocr-cache").rglob("*.json"))
    finally:
        set_default_cache(previous_cache)


def test_service_rejects_engine_invalid_options_before_render(monkeypatch):
    page = SimpleNamespace(width=100, height=100, index=0)
    page._ocr_scope = lambda: "page"
    page._ocr_render_kwargs = MagicMock(return_value={})
    service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
    run = MagicMock()
    monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", run)

    with pytest.raises(TypeError, match="Unsupported option.*rapidocr"):
        service.apply_ocr(
            page,
            engine="rapidocr",
            options={"definitely_invalid": True},
            replace="none",
        )

    page._ocr_render_kwargs.assert_not_called()
    run.assert_not_called()


def test_dispatch_reuses_only_matching_paddle_constructor_identity(monkeypatch):
    created_options = []

    class FakePaddleEngine:
        def is_available(self):
            return True

        def _initialize_model(self, languages, device, options):
            created_options.append(options)

        def process_image(self, *args, **kwargs):
            return []

    entry = EngineEntry(
        engine_type="classic",
        provider=FakePaddleEngine,
        options_class=PaddleOCROptions,
        needs_gpu_lock=False,
    )
    monkeypatch.setattr("natural_pdf.ocr.unified_dispatch.get_registry", lambda: {"paddle": entry})
    monkeypatch.setattr("natural_pdf.ocr.unified_dispatch._engine_cache", EngineCache(maxsize=4))
    target = MagicMock()
    target.render.return_value = Image.new("RGB", (4, 4))

    run_ocr(target=target, engine_name="paddle", resolution=72, options={"lang": "en"})
    run_ocr(target=target, engine_name="paddle", resolution=72, options={"lang": "en"})
    run_ocr(target=target, engine_name="paddle", resolution=72, options={"lang": "fr"})

    assert [options.lang for options in created_options] == ["en", "fr"]
