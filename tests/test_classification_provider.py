from __future__ import annotations

import logging

import natural_pdf as npdf
import natural_pdf.engine_provider as provider_module
from natural_pdf.classification.pipelines import (
    _CACHE_LOCK,
    _PIPELINE_CACHE,
    _parse_raw_scores,
    cleanup_models,
)
from natural_pdf.classification.results import CategoryScore, ClassificationResult
from natural_pdf.engine_provider import EngineProvider

# ---------- Existing test ----------


def test_page_classify_uses_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubClassificationEngine:
        def infer_using(self, model_id, using):
            return using or "text"

        def default_model(self, using):
            return "stub-model"

        def classify_item(self, **kwargs):
            return ClassificationResult(
                scores=[CategoryScore("stub", 0.9)],
                model_id=kwargs.get("model_id", "stub-model"),
                using=kwargs.get("using", "text"),
            )

        def classify_batch(self, **kwargs):
            return [
                self.classify_item(model_id=kwargs.get("model_id"), using=kwargs.get("using"))
                for _ in kwargs["item_contents"]
            ]

    provider.register(
        "classification", "default", lambda **_: _StubClassificationEngine(), replace=True
    )

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    page.classify(labels=["stub"])
    result = page.analyses["classification"]
    assert result.category == "stub"
    pdf.close()


# ---------- _parse_raw_scores ----------


class TestParseRawScores:
    def test_text_format(self):
        raw = {"labels": ["cat", "dog", "bird"], "scores": [0.8, 0.15, 0.05]}
        scores = _parse_raw_scores(raw, 0.0, "test-model")
        assert len(scores) == 3
        assert scores[0].label == "cat"
        assert scores[0].score == 0.8

    def test_vision_format(self):
        raw = [
            {"label": "cat", "score": 0.9},
            {"label": "dog", "score": 0.1},
        ]
        scores = _parse_raw_scores(raw, 0.0, "test-model")
        assert len(scores) == 2
        assert scores[0].label == "cat"

    def test_min_confidence_filtering(self):
        raw = {"labels": ["a", "b", "c"], "scores": [0.8, 0.3, 0.05]}
        scores = _parse_raw_scores(raw, 0.2, "test-model")
        assert len(scores) == 2
        labels = {s.label for s in scores}
        assert "c" not in labels

    def test_unexpected_format_returns_empty(self, caplog):
        with caplog.at_level(logging.WARNING):
            scores = _parse_raw_scores("unexpected", 0.0, "test-model")
        assert scores == []
        assert "Unexpected raw result format" in caplog.text

    def test_vision_format_skips_incomplete_items(self):
        raw = [
            {"label": "cat", "score": 0.9},
            {"label": None, "score": 0.5},  # missing label
            {"score": 0.3},  # no label key
            {"label": "dog"},  # no score key
        ]
        scores = _parse_raw_scores(raw, 0.0, "test-model")
        assert len(scores) == 1
        assert scores[0].label == "cat"


# ---------- cleanup_models ----------


class TestCleanupModels:
    def test_empty_cache_returns_zero(self):
        # Ensure cache is empty
        with _CACHE_LOCK:
            _PIPELINE_CACHE.clear()
        assert cleanup_models() == 0

    def test_cleanup_removes_from_cache(self):
        # Insert a fake pipeline
        class _FakePipeline:
            model = None

        with _CACHE_LOCK:
            _PIPELINE_CACHE["fake_key_text_None"] = _FakePipeline()

        cleaned = cleanup_models()
        assert cleaned == 1
        assert "fake_key_text_None" not in _PIPELINE_CACHE

    def test_cleanup_by_model_id_uses_prefix_match(self):
        """cleanup_models(model_id=...) uses prefix match, not substring."""

        class _FakePipeline:
            model = None

        with _CACHE_LOCK:
            _PIPELINE_CACHE.clear()
            # Key format: "{model_id}_{using}_{device}"
            _PIPELINE_CACHE["openai/clip-vit_vision_None"] = _FakePipeline()
            _PIPELINE_CACHE["my-clip-model_text_None"] = _FakePipeline()

        # Should only match prefix "openai/clip-vit_", not substring "clip"
        cleaned = cleanup_models(model_id="openai/clip-vit")
        assert cleaned == 1
        assert "openai/clip-vit_vision_None" not in _PIPELINE_CACHE
        assert "my-clip-model_text_None" in _PIPELINE_CACHE

        # Clean up
        with _CACHE_LOCK:
            _PIPELINE_CACHE.clear()


# ---------- Batch mismatch ----------


class TestBatchMismatch:
    def test_element_collection_batch_mismatch_logs_error(self, monkeypatch, caplog):
        """ClassificationBatchMixin logs error on mismatch, returns self."""
        provider = EngineProvider()
        provider._entry_points_loaded = True
        monkeypatch.setattr(provider_module, "_PROVIDER", provider)

        class _MismatchEngine:
            def infer_using(self, model_id, using):
                return using or "text"

            def default_model(self, using):
                return "stub-model"

            def classify_item(self, **kwargs):
                return ClassificationResult(
                    scores=[CategoryScore("stub", 0.9)],
                    model_id="stub-model",
                    using="text",
                )

            def classify_batch(self, **kwargs):
                # Return wrong number of results (empty)
                return []

        provider.register("classification", "default", lambda **_: _MismatchEngine(), replace=True)

        pdf = npdf.PDF("pdfs/01-practice.pdf")
        try:
            elements = pdf.pages[0].find_all("text")[:3]
            assert len(elements) > 0
            with caplog.at_level(logging.ERROR):
                result = elements.classify_all(labels=["a", "b"])
            assert result is elements  # Returns self
            assert "mismatch" in caplog.text.lower()
        finally:
            pdf.close()

    def test_empty_collection_returns_self(self):
        """Empty ElementCollection returns self without error."""
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        try:
            empty = pdf.pages[0].find_all('text:contains("ZZZNONEXISTENT")')
            assert len(empty) == 0
            result = empty.classify_all(labels=["a"])
            assert result is empty
        finally:
            pdf.close()
