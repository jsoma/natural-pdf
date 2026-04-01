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
        assert BaseOCROptions()._init_key() == ""

    def test_surya_options_default(self):
        assert SuryaOCROptions()._init_key() == ""

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
