"""Tests for OCR result caching."""

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from natural_pdf.exceptions import OCRError
from natural_pdf.ocr.ocr_cache import (
    OCRCache,
    compute_cache_key,
    compute_render_kwargs_cache_key,
    resolve_ocr_cache_identity,
    set_default_cache,
)
from natural_pdf.ocr.ocr_options import RapidOCROptions
from natural_pdf.ocr.unified_dispatch import (
    EngineEntry,
    OCRRunResult,
    get_engine_cache,
    get_registry,
    register_engine,
)
from natural_pdf.services.ocr_service import OCRService

# ---------------------------------------------------------------------------
# Cache key tests
# ---------------------------------------------------------------------------


class TestCacheKey:
    """compute_cache_key determinism and sensitivity."""

    BASE = dict(
        pdf_path="test.pdf",
        file_mtime_ns=1000000000,
        file_size=5000,
        page_index=0,
        engine_name="rapidocr",
        languages=("en",),
        resolution=150,
        detect_only=False,
        device="cpu",
        options_init_key="",
        apply_exclusions=True,
        model=None,
        prompt=None,
        instructions=None,
        max_new_tokens=None,
    )

    def test_deterministic(self):
        """Same inputs produce the same key."""
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**self.BASE)
        assert key1 == key2

    def test_changes_with_mtime(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "file_mtime_ns": 2000000000})
        assert key1 != key2

    def test_changes_with_engine(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "engine_name": "easyocr"})
        assert key1 != key2

    def test_changes_with_page_index(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "page_index": 1})
        assert key1 != key2

    def test_changes_with_resolution(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "resolution": 300})
        assert key1 != key2

    def test_changes_with_languages(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "languages": ("en", "fr")})
        assert key1 != key2

    def test_changes_with_language_priority(self):
        preferred_french = compute_cache_key(**{**self.BASE, "languages": ("fr", "en")})
        preferred_english = compute_cache_key(**{**self.BASE, "languages": ("en", "fr")})
        assert preferred_french != preferred_english

    def test_changes_with_crop_bbox(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "crop_bbox": (10, 20, 110, 120)})
        assert key1 != key2

    def test_changes_with_min_confidence(self):
        key1 = compute_cache_key(**{**self.BASE, "min_confidence": 0.2})
        key2 = compute_cache_key(**{**self.BASE, "min_confidence": 0.5})
        assert key1 != key2

    def test_changes_with_options_cache_key(self):
        key1 = compute_cache_key(**{**self.BASE, "options_cache_key": "text_score=0.2"})
        key2 = compute_cache_key(**{**self.BASE, "options_cache_key": "text_score=0.5"})
        assert key1 != key2

    def test_changes_with_layout(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "layout": "rapidocr"})
        assert key1 != key2

    def test_changes_with_preserve_markup(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "preserve_markup": True})
        assert key1 != key2

    def test_glm_layout_rapidocr_differs_from_plain_rapidocr(self):
        glm_key = compute_cache_key(**{**self.BASE, "engine_name": "glm_ocr", "layout": "rapidocr"})
        rapidocr_key = compute_cache_key(**{**self.BASE, "engine_name": "rapidocr", "layout": None})
        assert glm_key != rapidocr_key

    def test_changes_with_resolved_execution_identity(self):
        local = compute_cache_key(
            **self.BASE,
            execution_identity={"engine": "builtin:vlm", "client": "local"},
        )
        remote = compute_cache_key(
            **self.BASE,
            execution_identity={"engine": "builtin:vlm", "client": "remote-a"},
        )
        assert local != remote

    def test_changes_with_render_kwargs_identity(self):
        grayscale = compute_cache_key(
            **self.BASE,
            render_kwargs_cache_key=compute_render_kwargs_cache_key({"grayscale": True}),
        )
        color = compute_cache_key(
            **self.BASE,
            render_kwargs_cache_key=compute_render_kwargs_cache_key({"grayscale": False}),
        )
        assert grayscale != color


def test_render_kwargs_cache_identity_is_canonical_and_strict():
    first = compute_render_kwargs_cache_key(
        {"quality": 90, "nested": {"flags": [True, None], "scale": 1.5}}
    )
    reordered = compute_render_kwargs_cache_key(
        {"nested": {"scale": 1.5, "flags": (True, None)}, "quality": 90}
    )
    changed = compute_render_kwargs_cache_key(
        {"quality": 80, "nested": {"flags": [True, None], "scale": 1.5}}
    )

    assert first == reordered
    assert first != changed
    assert compute_render_kwargs_cache_key({"custom": object()}) is None
    assert compute_render_kwargs_cache_key({"custom": {1: "non-string-key"}}) is None


def test_internal_exclusion_payload_is_fingerprinted_separately():
    first = compute_render_kwargs_cache_key(
        {"apply_exclusions": True, "_ocr_exclusion_bboxes": ((1, 2, 3, 4),)}
    )
    second = compute_render_kwargs_cache_key(
        {"apply_exclusions": True, "_ocr_exclusion_bboxes": ((5, 6, 7, 8),)}
    )
    assert first == second


def test_remote_client_requires_explicit_cache_namespace():
    anonymous_client = SimpleNamespace()
    assert (
        resolve_ocr_cache_identity(
            engine_name="vlm",
            device="cpu",
            model="example-model",
            client=anonymous_client,
        )
        is None
    )

    namespaced_client = SimpleNamespace(natural_pdf_cache_namespace="account-a/deployment-1")
    identity = resolve_ocr_cache_identity(
        engine_name="vlm",
        device="cpu",
        model="example-model",
        client=namespaced_client,
    )
    assert identity is not None
    assert identity["client"] == "account-a/deployment-1"


def test_custom_engine_requires_an_explicit_cache_namespace():
    engine_name = "cache-identity-test-engine"
    registry = get_registry()
    previous = registry.get(engine_name)
    try:
        register_engine(engine_name, EngineEntry(engine_type="classic", provider=object()))
        assert (
            resolve_ocr_cache_identity(
                engine_name=engine_name,
                device="cpu",
                model=None,
                client=None,
            )
            is None
        )

        register_engine(
            engine_name,
            EngineEntry(
                engine_type="classic",
                provider=object(),
                cache_namespace="example-plugin/v2",
            ),
        )
        identity = resolve_ocr_cache_identity(
            engine_name=engine_name,
            device="cpu",
            model=None,
            client=None,
        )
        assert identity is not None
        assert identity["engine"] == "example-plugin/v2"
    finally:
        get_engine_cache().invalidate(engine_name)
        if previous is None:
            registry.pop(engine_name, None)
        else:
            register_engine(engine_name, previous)


def test_overriding_a_builtin_name_requires_a_new_cache_namespace():
    engine_name = "rapidocr"
    registry = get_registry()
    builtin = registry[engine_name]
    try:
        assert (
            resolve_ocr_cache_identity(
                engine_name=engine_name,
                device="cpu",
                model=None,
                client=None,
            )
            is not None
        )
        register_engine(engine_name, EngineEntry(engine_type="classic", provider=object()))
        assert (
            resolve_ocr_cache_identity(
                engine_name=engine_name,
                device="cpu",
                model=None,
                client=None,
            )
            is None
        )
    finally:
        register_engine(engine_name, builtin)


def test_apply_ocr_does_not_persistently_cache_anonymous_remote_client(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    previous_cache = set_default_cache(OCRCache(cache_dir=tmp_path / "ocr-cache"))
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _FakePage(pdf_path)
        calls = []

        def fake_run_ocr(**kwargs):
            calls.append(kwargs)
            return OCRRunResult(
                results=[{"bbox": [0, 0, 10, 10], "text": "fresh", "confidence": 0.99}],
                image_size=(100, 100),
                engine_type="vlm",
            )

        monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)
        client = SimpleNamespace()
        service.apply_ocr(page, engine="vlm", model="example-model", client=client)
        service.apply_ocr(page, engine="vlm", model="example-model", client=client)

        assert len(calls) == 2
    finally:
        set_default_cache(previous_cache)


def test_malformed_detection_payload_is_not_persistently_cached(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    cache = OCRCache(cache_dir=tmp_path / "ocr-cache")
    cache.put = MagicMock(wraps=cache.put)
    previous_cache = set_default_cache(cache)
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _FakePage(pdf_path)
        monkeypatch.setattr(
            "natural_pdf.services.ocr_service.run_ocr",
            lambda **kwargs: OCRRunResult(
                results=[{"text": "missing bbox", "confidence": 0.99}],
                image_size=(100, 100),
                engine_type="classic",
            ),
        )

        with pytest.raises(OCRError, match=r"result 0.*bbox"):
            service.apply_ocr(page, engine="rapidocr", detect_only=True)

        cache.put.assert_not_called()
        assert page.manager.created == []
    finally:
        set_default_cache(previous_cache)


def test_mixed_classic_payload_is_rejected_before_cache_or_replacement(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    cache = OCRCache(cache_dir=tmp_path / "ocr-cache")
    cache.put = MagicMock(wraps=cache.put)
    previous_cache = set_default_cache(cache)
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _FakePage(pdf_path)
        page.manager.remove_text_elements_in_bbox = MagicMock(return_value=(0, 0))
        monkeypatch.setattr(
            "natural_pdf.services.ocr_service.run_ocr",
            lambda **kwargs: OCRRunResult(
                results=[
                    {"bbox": [0, 0, 10, 10], "text": "valid", "confidence": 0.99},
                    {"text": "missing bbox", "confidence": 0.75},
                ],
                image_size=(100, 100),
                engine_type="classic",
            ),
        )

        with pytest.raises(OCRError, match=r"result 1.*bbox"):
            service.apply_ocr(page, engine="rapidocr", replace="ocr")

        cache.put.assert_not_called()
        page.manager.remove_text_elements_in_bbox.assert_not_called()
        assert page.manager.created == []
    finally:
        set_default_cache(previous_cache)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({"bbox": [0, 0, float("nan"), 10], "text": "bad"}, "finite"),
        ({"bbox": [10, 0, 0, 10], "text": "bad"}, "ordered"),
        ({"bbox": [0, 0, 10, 10], "text": 123}, "recognition 'text'"),
        (
            {"bbox": [0, 0, 10, 10], "text": "bad", "confidence": float("inf")},
            "confidence",
        ),
        ({"bbox": [0, 0, 10, 10], "text": "   "}, "non-empty"),
        (
            {"bbox": [0, 0, 10, 10], "text": None, "_ocr_detection_only": "yes"},
            "_ocr_detection_only",
        ),
    ],
)
def test_direct_classic_conversion_rejects_invalid_entry_atomically(tmp_path, result, message):
    service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
    page = _FakePage(tmp_path / "unused.pdf")

    with pytest.raises(OCRError, match=message):
        service.create_text_elements_from_ocr(page, [result])

    assert page.manager.created == []


# ---------------------------------------------------------------------------
# Cache store / retrieve tests
# ---------------------------------------------------------------------------


class TestOCRCache:
    """OCRCache put/get/clear operations."""

    def _make_result(self, text="Hello"):
        return OCRRunResult(
            results=[{"bbox": (100, 200, 300, 250), "text": text, "confidence": 0.95}],
            image_size=(1000, 1000),
            engine_type="classic",
        )

    def test_put_and_get(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        result = self._make_result()
        cache.put("test_key", result, "rapidocr", 0)

        retrieved = cache.get("test_key")
        assert retrieved is not None
        assert len(retrieved.results) == 1
        assert retrieved.results[0]["text"] == "Hello"
        assert retrieved.image_size == (1000, 1000)
        assert retrieved.engine_type == "classic"

    def test_miss_returns_none(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        assert cache.get("nonexistent") is None

    def test_clear(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        result = self._make_result()
        cache.put("key1", result, "rapidocr", 0)
        cache.put("key2", result, "rapidocr", 1)

        removed = cache.clear()
        assert removed == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_clear_empty_cache(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        assert cache.clear() == 0

    def test_multiple_entries(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        r1 = self._make_result("First")
        r2 = self._make_result("Second")
        cache.put("k1", r1, "rapidocr", 0)
        cache.put("k2", r2, "rapidocr", 1)

        assert cache.get("k1").results[0]["text"] == "First"
        assert cache.get("k2").results[0]["text"] == "Second"

    def test_overwrite(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        r1 = self._make_result("Old")
        r2 = self._make_result("New")
        cache.put("key", r1, "rapidocr", 0)
        cache.put("key", r2, "rapidocr", 0)

        assert cache.get("key").results[0]["text"] == "New"

    def test_bbox_tuple_round_trip(self, tmp_path):
        """Bounding boxes stored as lists should be retrievable."""
        cache = OCRCache(cache_dir=tmp_path)
        result = OCRRunResult(
            results=[{"bbox": (10, 20, 30, 40), "text": "test", "confidence": 0.9}],
            image_size=(500, 500),
        )
        cache.put("bbox_test", result, "rapidocr", 0)

        retrieved = cache.get("bbox_test")
        # JSON round-trip converts tuples to lists
        assert retrieved.results[0]["bbox"] == [10, 20, 30, 40]

    def test_delete_removes_single_entry(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        result = self._make_result()
        cache.put("key", result, "rapidocr", 0)

        assert cache.delete("key") is True
        assert cache.get("key") is None
        assert cache.delete("key") is False

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are unavailable")
    def test_cache_directories_and_files_are_private_under_permissive_umask(
        self, monkeypatch, tmp_path
    ):
        cache_dir = tmp_path / "ocr-cache"
        cache_dir.mkdir(mode=0o777)
        cache_dir.chmod(0o777)
        cache = OCRCache(cache_dir=cache_dir)
        observed = {}
        real_replace = Path.replace

        def recording_replace(source, target):
            observed["temp_mode"] = stat.S_IMODE(source.stat().st_mode)
            return real_replace(source, target)

        monkeypatch.setattr(Path, "replace", recording_replace)
        previous_umask = os.umask(0)
        try:
            cache.put("permission-key", self._make_result(), "rapidocr", 0)
        finally:
            os.umask(previous_umask)

        final_path = cache._key_path("permission-key")
        assert observed["temp_mode"] == 0o600
        assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(final_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(final_path.stat().st_mode) == 0o600

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are unavailable")
    def test_read_hardens_entries_written_by_older_versions(self, tmp_path):
        cache_dir = tmp_path / "ocr-cache"
        cache = OCRCache(cache_dir=cache_dir)
        cache.put("legacy-key", self._make_result(), "rapidocr", 0)
        path = cache._key_path("legacy-key")

        cache_dir.chmod(0o777)
        path.parent.chmod(0o777)
        path.chmod(0o666)

        assert cache.get("legacy-key") is not None
        assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


# ---------------------------------------------------------------------------
# Service integration regression tests
# ---------------------------------------------------------------------------


class _RecordingOCRManager:
    def __init__(self):
        self.created = []

    def remove_ocr_elements(self):
        return 0

    def clear_text_layer(self):
        return (0, 0)

    def remove_text_elements_in_bbox(self, bbox, *, sources=None, predicate=None):
        return (0, 0)

    def create_text_elements_from_ocr(
        self,
        ocr_results,
        scale_x=None,
        scale_y=None,
        offset_x=0.0,
        offset_y=0.0,
        engine_name=None,
    ):
        call = {
            "ocr_results": ocr_results,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "engine_name": engine_name,
        }
        self.created.append(call)
        return [SimpleNamespace(text=result.get("text")) for result in ocr_results]


class _FakePage:
    width = 100
    height = 100
    index = 0

    def __init__(self, pdf_path):
        self.pdf = SimpleNamespace(_resolved_path=str(pdf_path))
        self.manager = _RecordingOCRManager()

    def _ocr_element_manager(self):
        return self.manager

    def _ocr_scope(self):
        return "page"

    def _ocr_render_kwargs(self, *, apply_exclusions=True):
        return {"apply_exclusions": apply_exclusions}


class _FakeRegion:
    width = 50
    height = 50
    bbox = (25, 25, 75, 75)

    def __init__(self, page):
        self.page = page
        self.manager = _RecordingOCRManager()

    def _ocr_element_manager(self):
        return self.manager

    def _ocr_scope(self):
        return "region"

    def _ocr_render_kwargs(self, *, apply_exclusions=True):
        return {"crop": True}


class _CustomRenderPage(_FakePage):
    def __init__(self, pdf_path, render_value):
        super().__init__(pdf_path)
        self.render_value = render_value

    def _ocr_render_kwargs(self, *, apply_exclusions=True):
        return {
            "apply_exclusions": apply_exclusions,
            "custom_render_value": self.render_value,
        }


def test_render_hook_kwargs_partition_persistent_results(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    previous_cache = set_default_cache(OCRCache(cache_dir=tmp_path / "ocr-cache"))
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _CustomRenderPage(pdf_path, "first")
        calls = []

        def fake_run_ocr(**kwargs):
            calls.append(kwargs)
            return OCRRunResult(
                results=[{"bbox": [0, 0, 10, 10], "text": "result", "confidence": 0.99}],
                image_size=(100, 100),
            )

        monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)
        service.apply_ocr(page, engine="rapidocr", replace="none")
        page.render_value = "second"
        service.apply_ocr(page, engine="rapidocr", replace="none")
        service.apply_ocr(page, engine="rapidocr", replace="none")

        assert len(calls) == 2
        assert calls[0]["render_kwargs"]["custom_render_value"] == "first"
        assert calls[1]["render_kwargs"]["custom_render_value"] == "second"
    finally:
        set_default_cache(previous_cache)


def test_unstable_render_hook_value_disables_persistent_cache(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    previous_cache = set_default_cache(OCRCache(cache_dir=tmp_path / "ocr-cache"))
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _CustomRenderPage(pdf_path, object())
        calls = []

        def fake_run_ocr(**kwargs):
            calls.append(kwargs)
            return OCRRunResult(
                results=[{"bbox": [0, 0, 10, 10], "text": "result", "confidence": 0.99}],
                image_size=(100, 100),
            )

        monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)
        service.apply_ocr(page, engine="rapidocr", replace="none")
        service.apply_ocr(page, engine="rapidocr", replace="none")

        assert len(calls) == 2
    finally:
        set_default_cache(previous_cache)


def test_region_ocr_cache_isolated_from_full_page_payload(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    previous_cache = set_default_cache(OCRCache(cache_dir=tmp_path / "ocr-cache"))
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _FakePage(pdf_path)
        region = _FakeRegion(page)
        calls = []

        def fake_run_ocr(**kwargs):
            calls.append(kwargs)
            target = kwargs["target"]
            if target is page:
                return OCRRunResult(
                    results=[{"bbox": [80, 80, 100, 100], "text": "outside", "confidence": 0.99}],
                    image_size=(100, 100),
                )
            if target is region:
                return OCRRunResult(
                    results=[{"bbox": [0, 0, 10, 10], "text": "inside", "confidence": 0.99}],
                    image_size=(50, 50),
                )
            raise AssertionError(f"Unexpected OCR target: {target!r}")

        monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)

        service.apply_ocr(page, engine="rapidocr", languages=["en"], device="cpu")
        service.apply_ocr(region, engine="rapidocr", languages=["en"], device="cpu")

        assert [call["target"] for call in calls] == [page, region]
        assert calls[1]["render_kwargs"] == {
            "crop": True,
            "crop_bbox": (25.0, 25.0, 75.0, 75.0),
            "apply_exclusions": True,
            "_ocr_exclusion_bboxes": (),
        }

        region_create_call = region.manager.created[-1]
        assert region_create_call["ocr_results"][0]["text"] == "inside"
        assert region_create_call["scale_x"] == 1.0
        assert region_create_call["scale_y"] == 1.0
        assert region_create_call["offset_x"] == 25
        assert region_create_call["offset_y"] == 25
    finally:
        set_default_cache(previous_cache)


def test_apply_ocr_cache_key_uses_runtime_option_fields(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    previous_cache = set_default_cache(OCRCache(cache_dir=tmp_path / "ocr-cache"))
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _FakePage(pdf_path)
        captured = {}

        def fake_run_ocr(**kwargs):
            return OCRRunResult(
                results=[{"bbox": [0, 0, 10, 10], "text": "inside", "confidence": 0.99}],
                image_size=(100, 100),
            )

        real_compute_cache_key = compute_cache_key

        def recording_compute_cache_key(**kwargs):
            captured.update(kwargs)
            return real_compute_cache_key(**kwargs)

        monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)
        monkeypatch.setattr(
            "natural_pdf.ocr.ocr_cache.compute_cache_key",
            recording_compute_cache_key,
        )

        service.apply_ocr(
            page,
            engine="rapidocr",
            options=RapidOCROptions(text_score=0.2),
            languages=["en"],
            min_confidence=0.4,
            device="cpu",
        )

        options_payload = json.loads(captured["options_cache_key"])
        assert options_payload["class"] == "RapidOCROptions"
        assert options_payload["options"]["text_score"] == 0.2
        assert captured["min_confidence"] == 0.4
    finally:
        set_default_cache(previous_cache)


def test_invalid_cached_vlm_payload_is_evicted_and_retried(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    cache = OCRCache(cache_dir=tmp_path / "ocr-cache")
    previous_cache = set_default_cache(cache)
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _FakePage(pdf_path)
        calls = []

        def fake_run_ocr(**kwargs):
            calls.append(kwargs)
            return OCRRunResult(
                results=[{"bbox": [0, 0, 10, 10], "text": "fresh", "confidence": 0.99}],
                image_size=(100, 100),
                engine_type="vlm",
            )

        monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)
        key = "shared-vlm-cache-key"
        monkeypatch.setattr("natural_pdf.ocr.ocr_cache.compute_cache_key", lambda **kwargs: key)
        cache.put(
            key,
            OCRRunResult(results=[], image_size=(100, 100), engine_type="vlm"),
            "vlm",
            0,
        )

        service.apply_ocr(
            page,
            engine="vlm",
            model="gemini-3.1-flash-lite",
            languages=["en"],
            device="cpu",
        )

        assert len(calls) == 1
        assert page.manager.created[-1]["ocr_results"][0]["text"] == "fresh"
        cached = cache.get(key)
        assert cached is not None
        assert cached.results[0]["text"] == "fresh"
    finally:
        set_default_cache(previous_cache)


@pytest.mark.parametrize(("confidence", "expected"), [(None, None), ("0.9", 0.9)])
def test_cached_vlm_accepts_supported_confidence_forms(
    monkeypatch,
    tmp_path,
    confidence,
    expected,
):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    cache = OCRCache(cache_dir=tmp_path / "ocr-cache")
    previous_cache = set_default_cache(cache)
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _FakePage(pdf_path)
        key = f"vlm-confidence-{confidence}"
        monkeypatch.setattr("natural_pdf.ocr.ocr_cache.compute_cache_key", lambda **_: key)
        cache.put(
            key,
            OCRRunResult(
                results=[
                    {
                        "bbox": [0, 0, 10, 10],
                        "text": "cached",
                        "confidence": confidence,
                    }
                ],
                image_size=(100, 100),
                engine_type="vlm",
            ),
            "vlm",
            0,
        )
        monkeypatch.setattr(
            "natural_pdf.services.ocr_service.run_ocr",
            lambda **_: (_ for _ in ()).throw(AssertionError("valid cache entry was ignored")),
        )

        service.apply_ocr(
            page,
            engine="vlm",
            model="gemini-3.1-flash-lite",
            languages=["en"],
            device="cpu",
            min_confidence=0.5,
            replace="none",
        )

        result = page.manager.created[-1]["ocr_results"][0]
        assert result["text"] == "cached"
        assert result["confidence"] == expected
    finally:
        set_default_cache(previous_cache)


def test_mixed_invalid_cached_classic_detection_is_evicted_before_conversion(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    cache = OCRCache(cache_dir=tmp_path / "ocr-cache")
    previous_cache = set_default_cache(cache)
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _FakePage(pdf_path)
        page.manager._ocr_converter = SimpleNamespace(
            convert=lambda *args, **kwargs: ([SimpleNamespace()], [])
        )
        calls = []

        def fake_run_ocr(**kwargs):
            calls.append(kwargs)
            return OCRRunResult(
                results=[{"bbox": [20, 20, 40, 40], "text": None, "confidence": 0.8}],
                image_size=(100, 100),
                engine_type="classic",
            )

        monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)
        key = "shared-classic-detection-cache-key"
        monkeypatch.setattr("natural_pdf.ocr.ocr_cache.compute_cache_key", lambda **kwargs: key)
        cache.put(
            key,
            OCRRunResult(
                results=[
                    {"bbox": [0, 0, 10, 10], "text": None, "confidence": 0.9},
                    "not a result mapping",
                ],
                image_size=(100, 100),
                engine_type="classic",
            ),
            "rapidocr",
            0,
        )

        service.apply_ocr(
            page,
            engine="rapidocr",
            languages=["en"],
            device="cpu",
            detect_only=True,
        )

        assert len(calls) == 1
        assert len(page.manager.created) == 1
        assert page.manager.created[0]["ocr_results"][0]["bbox"] == (
            20.0,
            20.0,
            40.0,
            40.0,
        )
        cached = cache.get(key)
        assert cached is not None
        assert len(cached.results) == 1
    finally:
        set_default_cache(previous_cache)


def test_malformed_multi_table_cache_entry_is_evicted_before_replacement(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    cache = OCRCache(cache_dir=tmp_path / "ocr-cache")
    previous_cache = set_default_cache(cache)
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _FakePage(pdf_path)
        calls = []

        def fake_run_ocr(**kwargs):
            calls.append(kwargs)
            return OCRRunResult(
                results=[{"bbox": [0, 0, 10, 10], "text": "fresh", "confidence": 0.99}],
                image_size=(100, 100),
                engine_type="vlm",
            )

        monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)
        key = "malformed-multi-table-key"
        monkeypatch.setattr("natural_pdf.ocr.ocr_cache.compute_cache_key", lambda **kwargs: key)
        cache.put(
            key,
            OCRRunResult(
                results=[
                    {
                        "bbox": [0, 0, 10, 10],
                        "text": "valid\ttable",
                        "source_category": "table",
                    },
                    {"text": "missing bbox", "source_category": "table"},
                ],
                image_size=(100, 100),
                engine_type="vlm",
            ),
            "vlm",
            0,
        )

        service.apply_ocr(
            page,
            engine="vlm",
            model="gemini-3.1-flash-lite",
            languages=["en"],
            device="cpu",
        )

        assert len(calls) == 1
        assert page.manager.created[-1]["ocr_results"][0]["text"] == "fresh"
        cached = cache.get(key)
        assert cached is not None
        assert cached.results[0]["text"] == "fresh"
    finally:
        set_default_cache(previous_cache)


def test_empty_vlm_payload_is_not_cached(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    cache = OCRCache(cache_dir=tmp_path / "ocr-cache")
    previous_cache = set_default_cache(cache)
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _FakePage(pdf_path)

        def fake_run_ocr(**kwargs):
            return OCRRunResult(results=[], image_size=(100, 100), engine_type="vlm")

        monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)
        key = "empty-vlm-cache-key"
        monkeypatch.setattr("natural_pdf.ocr.ocr_cache.compute_cache_key", lambda **kwargs: key)

        service.apply_ocr(
            page,
            engine="vlm",
            model="gemini-3.1-flash-lite",
            languages=["en"],
            device="cpu",
        )

        assert cache.get(key) is None
    finally:
        set_default_cache(previous_cache)
