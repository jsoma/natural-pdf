"""Tests for the unified OCR dispatch module."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from natural_pdf.ocr.ocr_options import (
    BaseOCROptions,
    ChandraOCROptions,
    DoctrOCROptions,
    EasyOCROptions,
    PaddleOCROptions,
    PaddleOCRVLOptions,
    RapidOCROptions,
    SuryaOCROptions,
)
from natural_pdf.ocr.unified_dispatch import (
    EngineCache,
    EngineEntry,
    OCRRunResult,
    get_registry,
    list_engines,
    register_engine,
    run_ocr,
)

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_contains_all_classic_engines(self):
        registry = get_registry()
        for name in ("easyocr", "rapidocr", "surya", "paddle", "doctr", "chandra2"):
            assert name in registry, f"Missing classic engine: {name}"
            assert registry[name].engine_type == "classic"

    def test_paddlevl_auto_platform(self):
        registry = get_registry()
        assert "paddlevl" in registry
        assert registry["paddlevl"].engine_type == "auto_platform"
        assert registry["paddlevl"].model_resolver is not None

    def test_contains_vlm_engines(self):
        registry = get_registry()
        for name in ("dots", "glm_ocr", "chandra"):
            assert name in registry, f"Missing VLM engine: {name}"
            assert registry[name].engine_type == "vlm_shorthand"

    def test_contains_generic_vlm(self):
        registry = get_registry()
        assert "vlm" in registry
        assert registry["vlm"].engine_type == "vlm_generic"

    def test_vlm_shorthands_have_model_resolvers(self):
        registry = get_registry()
        for name in ("dots", "glm_ocr", "chandra"):
            assert registry[name].model_resolver is not None

    def test_list_engines_returns_copy(self):
        engines = list_engines()
        assert isinstance(engines, dict)
        assert "easyocr" in engines

    def test_register_engine(self):
        register_engine("test_engine", EngineEntry(engine_type="classic"))
        registry = get_registry()
        assert "test_engine" in registry
        # Cleanup
        del registry["test_engine"]


# ---------------------------------------------------------------------------
# EngineCache tests
# ---------------------------------------------------------------------------


class TestEngineCache:
    def test_cache_hit(self):
        cache = EngineCache(maxsize=2)
        engine = MagicMock()
        result = cache.get_or_create("test", ("en",), "cpu", "", lambda: engine)
        assert result is engine
        # Second call should return same instance
        result2 = cache.get_or_create("test", ("en",), "cpu", "", lambda: MagicMock())
        assert result2 is engine

    def test_cache_miss_on_different_languages(self):
        cache = EngineCache(maxsize=4)
        engine_en = MagicMock()
        engine_ja = MagicMock()
        cache.get_or_create("test", ("en",), "cpu", "", lambda: engine_en)
        result = cache.get_or_create("test", ("ja",), "cpu", "", lambda: engine_ja)
        assert result is engine_ja

    def test_cache_miss_on_different_init_key(self):
        cache = EngineCache(maxsize=4)
        engine1 = MagicMock()
        engine2 = MagicMock()
        cache.get_or_create("test", ("en",), "cpu", "key1", lambda: engine1)
        result = cache.get_or_create("test", ("en",), "cpu", "key2", lambda: engine2)
        assert result is engine2

    def test_lru_eviction(self):
        cache = EngineCache(maxsize=2)
        e1 = MagicMock()
        e2 = MagicMock()
        e3 = MagicMock()
        cache.get_or_create("a", ("en",), "cpu", "", lambda: e1)
        cache.get_or_create("b", ("en",), "cpu", "", lambda: e2)
        # This should evict e1
        cache.get_or_create("c", ("en",), "cpu", "", lambda: e3)
        # Verify e1 is evicted by checking it creates a new one
        e1_new = MagicMock()
        result = cache.get_or_create("a", ("en",), "cpu", "", lambda: e1_new)
        assert result is e1_new

    def test_eviction_calls_cleanup(self):
        cache = EngineCache(maxsize=1)
        e1 = MagicMock()
        e1.cleanup = MagicMock()
        cache.get_or_create("a", ("en",), "cpu", "", lambda: e1)
        # Evict e1 by adding another
        cache.get_or_create("b", ("en",), "cpu", "", lambda: MagicMock())
        e1.cleanup.assert_called_once()

    def test_clear(self):
        cache = EngineCache(maxsize=4)
        e1 = MagicMock()
        e1.cleanup = MagicMock()
        cache.get_or_create("a", ("en",), "cpu", "", lambda: e1)
        count = cache.clear()
        assert count == 1
        e1.cleanup.assert_called_once()

    def test_thread_safety(self):
        cache = EngineCache(maxsize=4)
        results = {}
        errors = []

        def worker(name):
            try:
                engine = cache.get_or_create(name, ("en",), "cpu", "", lambda: MagicMock())
                results[name] = engine
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"e{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 4

    def test_maxsize_setter(self):
        cache = EngineCache(maxsize=4)
        for i in range(4):
            cache.get_or_create(f"e{i}", ("en",), "cpu", "", lambda: MagicMock())
        cache.maxsize = 2
        assert cache.maxsize == 2
        # Should have evicted 2 entries
        count = cache.clear()
        assert count == 2


# ---------------------------------------------------------------------------
# _init_key tests
# ---------------------------------------------------------------------------


class TestInitKeys:
    def test_base_options_default(self):
        assert BaseOCROptions()._init_key() == '{"extra_args":{}}'

    def test_surya_options_default(self):
        assert SuryaOCROptions()._init_key() == '{"extra_args":{}}'

    def test_easyocr_options(self):
        opts = EasyOCROptions()
        key = opts._init_key()
        assert "english_g2" in key  # default recog_network
        assert "craft" in key  # default detect_network

    def test_easyocr_different_network(self):
        opts1 = EasyOCROptions(recog_network="english_g2")
        opts2 = EasyOCROptions(recog_network="latin_g2")
        assert opts1._init_key() != opts2._init_key()

    def test_rapidocr_options(self):
        opts = RapidOCROptions()
        key = opts._init_key()
        assert "mobile" in key

    def test_rapidocr_different_model_type(self):
        opts1 = RapidOCROptions(det_model_type="mobile")
        opts2 = RapidOCROptions(det_model_type="server")
        assert opts1._init_key() != opts2._init_key()

    def test_rapidocr_runtime_options_change_result_key_not_init_key(self):
        opts1 = RapidOCROptions(text_score=0.2)
        opts2 = RapidOCROptions(text_score=0.5)
        assert opts1._init_key() == opts2._init_key()
        assert opts1._cache_key() != opts2._cache_key()

    def test_doctr_options(self):
        opts = DoctrOCROptions()
        key = opts._init_key()
        assert "db_resnet50" in key

    def test_paddle_options(self):
        opts = PaddleOCROptions()
        key = opts._init_key()
        assert isinstance(key, str)
        assert len(key) > 0

    def test_chandra_options(self):
        opts = ChandraOCROptions()
        key = opts._init_key()
        assert "hf" in key

    def test_paddlevl_options(self):
        opts = PaddleOCRVLOptions()
        key = opts._init_key()
        assert isinstance(key, str)

    def test_paddlevl_generation_options_change_result_key_not_init_key(self):
        opts1 = PaddleOCRVLOptions(max_new_tokens=1024)
        opts2 = PaddleOCRVLOptions(max_new_tokens=2048)
        assert opts1._init_key() == opts2._init_key()
        assert opts1._cache_key() != opts2._cache_key()


# ---------------------------------------------------------------------------
# run_ocr dispatch tests
# ---------------------------------------------------------------------------


class TestRunOcr:
    def test_unknown_engine_raises(self):
        target = MagicMock()
        with pytest.raises(LookupError, match="Unknown OCR engine"):
            run_ocr(target=target, engine_name="nonexistent", resolution=72)

    def test_vlm_generic_requires_model_or_client(self):
        target = MagicMock()
        target.render.return_value = Image.new("RGB", (100, 100))
        with patch(
            "natural_pdf.ocr.unified_dispatch.get_registry",
            return_value=get_registry(),
        ):
            with patch(
                "natural_pdf.core.vlm_client.get_default_client",
                return_value=(None, None),
            ):
                with pytest.raises(ValueError, match="requires a model"):
                    run_ocr(target=target, engine_name="vlm", resolution=72)

    def test_vlm_explicit_model_never_uses_default_client(self, monkeypatch):
        """apply_ocr(engine='vlm', model=...) must run locally even when a
        default client is configured (default applies only when neither
        model= nor client= is passed)."""
        import natural_pdf.core.vlm_client as vlm_client

        default_client = MagicMock()
        monkeypatch.setattr(vlm_client, "_default_client", default_client)
        monkeypatch.setattr(vlm_client, "_default_model", "default-remote-model")

        local_calls = []

        def fake_local(image, prompt, *, model, max_new_tokens):
            local_calls.append(model)
            return ""

        def boom_remote(*args, **kwargs):
            raise AssertionError("default client must not receive the image")

        monkeypatch.setattr(vlm_client, "_generate_local", fake_local)
        monkeypatch.setattr(vlm_client, "_generate_remote", boom_remote)

        target = MagicMock()
        target.render.return_value = Image.new("RGB", (100, 100))
        result = run_ocr(
            target=target,
            engine_name="vlm",
            resolution=72,
            model="Qwen/Qwen3-VL-2B-Instruct",
            layout=False,
        )

        assert local_calls == ["Qwen/Qwen3-VL-2B-Instruct"]
        assert result.results == []
        default_client.chat.completions.create.assert_not_called()

    def test_vlm_default_client_used_when_neither_model_nor_client(self, monkeypatch):
        """apply_ocr(engine='vlm') with neither model= nor client= uses the
        default client and its default model."""
        import natural_pdf.core.vlm_client as vlm_client

        default_client = MagicMock()
        monkeypatch.setattr(vlm_client, "_default_client", default_client)
        monkeypatch.setattr(vlm_client, "_default_model", "default-remote-model")

        remote_calls = []

        def fake_remote(image, prompt, *, client, model, max_new_tokens, response_format=None):
            remote_calls.append((client, model))
            return ""

        monkeypatch.setattr(vlm_client, "_generate_remote", fake_remote)

        target = MagicMock()
        target.render.return_value = Image.new("RGB", (100, 100))
        result = run_ocr(target=target, engine_name="vlm", resolution=72, layout=False)

        assert remote_calls == [(default_client, "default-remote-model")]
        assert result.results == []


# ---------------------------------------------------------------------------
# Classic engine payload normalization (fail-closed)
# ---------------------------------------------------------------------------


class TestNormalizeEngineOutput:
    def test_none_is_empty(self):
        from natural_pdf.ocr.unified_dispatch import _normalize_engine_output

        assert _normalize_engine_output(None, engine_name="fake") == []

    def test_empty_list_is_empty(self):
        from natural_pdf.ocr.unified_dispatch import _normalize_engine_output

        assert _normalize_engine_output([], engine_name="fake") == []

    def test_list_of_dicts_passes_through(self):
        from natural_pdf.ocr.unified_dispatch import _normalize_engine_output

        payload = [{"text": "hi", "bbox": [0, 0, 1, 1], "confidence": 0.9}]
        assert _normalize_engine_output(payload, engine_name="fake") == payload

    def test_batch_list_unwraps_first(self):
        from natural_pdf.ocr.unified_dispatch import _normalize_engine_output

        inner = [{"text": "hi", "bbox": [0, 0, 1, 1], "confidence": 0.9}]
        assert _normalize_engine_output([inner], engine_name="fake") == inner

    def test_empty_batch_inner_is_empty(self):
        from natural_pdf.ocr.unified_dispatch import _normalize_engine_output

        assert _normalize_engine_output([[]], engine_name="fake") == []

    def test_multiple_batches_raise_ocr_error(self):
        # A single-image call must never silently discard extra batches.
        from natural_pdf.exceptions import OCRError
        from natural_pdf.ocr.unified_dispatch import _normalize_engine_output

        batch_a = [{"text": "a", "bbox": [0, 0, 1, 1], "confidence": 0.9}]
        batch_b = [{"text": "b", "bbox": [0, 0, 1, 1], "confidence": 0.9}]
        with pytest.raises(OCRError, match=r"'fake'.*2 result batches"):
            _normalize_engine_output([batch_a, batch_b], engine_name="fake")

    def test_batch_inner_non_dict_beyond_first_raises(self):
        # Every inner element is validated, not just the first.
        from natural_pdf.exceptions import OCRError
        from natural_pdf.ocr.unified_dispatch import _normalize_engine_output

        inner = [{"text": "a", "bbox": [0, 0, 1, 1], "confidence": 0.9}, "junk"]
        with pytest.raises(OCRError, match=r"'fake'.*str"):
            _normalize_engine_output([inner], engine_name="fake")

    def test_flat_list_non_dict_beyond_first_raises(self):
        # Every element of a flat result list is validated too.
        from natural_pdf.exceptions import OCRError
        from natural_pdf.ocr.unified_dispatch import _normalize_engine_output

        payload = [{"text": "a", "bbox": [0, 0, 1, 1], "confidence": 0.9}, 42]
        with pytest.raises(OCRError, match=r"'fake'.*int"):
            _normalize_engine_output(payload, engine_name="fake")

    def test_dict_payload_raises_ocr_error(self):
        from natural_pdf.exceptions import OCRError
        from natural_pdf.ocr.unified_dispatch import _normalize_engine_output

        with pytest.raises(OCRError, match=r"'myengine'.*dict"):
            _normalize_engine_output({"text": "hi"}, engine_name="myengine")

    def test_string_payload_raises_ocr_error(self):
        from natural_pdf.exceptions import OCRError
        from natural_pdf.ocr.unified_dispatch import _normalize_engine_output

        with pytest.raises(OCRError, match=r"'myengine'.*str"):
            _normalize_engine_output("some text", engine_name="myengine")

    def test_list_of_unsupported_items_raises_ocr_error(self):
        from natural_pdf.exceptions import OCRError
        from natural_pdf.ocr.unified_dispatch import _normalize_engine_output

        with pytest.raises(OCRError, match=r"'myengine'.*tuple"):
            _normalize_engine_output([("hi", 0.9)], engine_name="myengine")

    def test_batch_of_unsupported_items_raises_ocr_error(self):
        from natural_pdf.exceptions import OCRError
        from natural_pdf.ocr.unified_dispatch import _normalize_engine_output

        with pytest.raises(OCRError, match=r"'myengine'.*str"):
            _normalize_engine_output([["not-a-dict"]], engine_name="myengine")


# ---------------------------------------------------------------------------
# Comparison service spec normalization
# ---------------------------------------------------------------------------


class TestComparisonSpecs:
    def test_string_spec(self):
        from natural_pdf.services.ocr_comparison_service import _normalize_spec

        result = _normalize_spec(
            "easyocr",
            default_resolution=150,
            default_languages=None,
            default_device=None,
            default_min_confidence=None,
        )
        assert result["engine"] == "easyocr"
        assert result["label"] == "easyocr"
        assert result["resolution"] == 150

    def test_dict_spec_with_override(self):
        from natural_pdf.services.ocr_comparison_service import _normalize_spec

        result = _normalize_spec(
            {"engine": "rapidocr", "resolution": 72},
            default_resolution=150,
            default_languages=None,
            default_device=None,
            default_min_confidence=None,
        )
        assert result["engine"] == "rapidocr"
        assert result["resolution"] == 72
        assert "resolution=72" in result["label"]

    def test_dict_spec_with_explicit_label(self):
        from natural_pdf.services.ocr_comparison_service import _normalize_spec

        result = _normalize_spec(
            {"engine": "rapidocr", "resolution": 72, "label": "rapid-lo"},
            default_resolution=150,
            default_languages=None,
            default_device=None,
            default_min_confidence=None,
        )
        assert result["label"] == "rapid-lo"

    def test_dict_spec_missing_engine_raises(self):
        from natural_pdf.services.ocr_comparison_service import _normalize_spec

        with pytest.raises(ValueError, match="must have an 'engine' key"):
            _normalize_spec(
                {"resolution": 72},
                default_resolution=150,
                default_languages=None,
                default_device=None,
                default_min_confidence=None,
            )

    def test_dict_spec_vlm_params(self):
        from natural_pdf.services.ocr_comparison_service import _normalize_spec

        result = _normalize_spec(
            {"engine": "vlm", "model": "some-model"},
            default_resolution=150,
            default_languages=None,
            default_device=None,
            default_min_confidence=None,
        )
        assert result["model"] == "some-model"
