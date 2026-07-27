from __future__ import annotations

import logging

import pytest

import natural_pdf as npdf
import natural_pdf.engine_provider as provider_module
from natural_pdf.classification.pipelines import (
    _CACHE_LOCK,
    _PIPELINE_CACHE,
    _parse_raw_scores,
    cleanup_models,
)
from natural_pdf.classification.results import CategoryScore, ClassificationResult
from natural_pdf.core.context import PDFContext
from natural_pdf.engine_provider import EngineProvider
from natural_pdf.exceptions import ClassificationError
from natural_pdf.services import classification_service
from natural_pdf.services.classification_service import ClassificationService

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


def test_pdf_text_classification_rejects_removed_use_exclusions():
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        with pytest.raises(TypeError, match="use_exclusions was removed.*apply_exclusions"):
            pdf.classify(labels=["stub"], using="text", use_exclusions=False)

        with pytest.raises(TypeError, match="use_exclusions was removed.*apply_exclusions"):
            pdf._get_classification_content(
                model_type="text",
                use_exclusions=False,
            )
    finally:
        pdf.close()


def test_text_classification_propagates_unexpected_text_extraction_error(monkeypatch):
    """A broken text extractor must not be treated as a scanned document."""

    class StubEngine:
        def infer_using(self, model_id, using):
            return using or "text"

        def default_model(self, using):
            return "stub-model"

    class BrokenTextHost:
        analyses = {}

        def _get_classification_content(self, model_type, **kwargs):
            if model_type == "text":
                raise OSError("damaged text layer")
            raise AssertionError("vision fallback must not run after a text extraction failure")

    monkeypatch.setattr(
        classification_service, "get_classification_engine", lambda *_: StubEngine()
    )
    monkeypatch.setattr(
        classification_service,
        "run_classification_item",
        lambda **_: (_ for _ in ()).throw(AssertionError("model must not run")),
    )
    with pytest.raises(RuntimeError, match="Failed to extract text content") as exc_info:
        ClassificationService(PDFContext.with_defaults()).classify(
            BrokenTextHost(), labels=["stub"], using="text"
        )

    assert isinstance(exc_info.value.__cause__, OSError)


def test_text_classification_does_not_guess_empty_state_from_error_substrings(monkeypatch):
    class StubEngine:
        def infer_using(self, model_id, using):
            return using or "text"

        def default_model(self, using):
            return "stub-model"

    class BrokenTextHost:
        analyses = {}

        def _get_classification_content(self, model_type, **kwargs):
            if model_type == "text":
                raise ValueError("no extractable text because parser configuration is invalid")
            raise AssertionError("vision fallback must not run after a text extraction failure")

    monkeypatch.setattr(
        classification_service, "get_classification_engine", lambda *_: StubEngine()
    )

    with pytest.raises(RuntimeError, match="Failed to extract text content") as exc_info:
        ClassificationService(PDFContext.with_defaults()).classify(
            BrokenTextHost(), labels=["stub"], using="text"
        )

    assert isinstance(exc_info.value.__cause__, ValueError)


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

    def test_unexpected_format_raises(self):
        """Fail closed: an unknown payload shape must raise, not silently
        become an empty (category=None) result."""
        with pytest.raises(ClassificationError, match="Unexpected result format"):
            _parse_raw_scores("unexpected", 0.0, "test-model")

    def test_vision_format_rejects_incomplete_items(self):
        raw = [
            {"label": "cat", "score": 0.9},
            {"score": 0.3},  # no label key
        ]
        with pytest.raises(ClassificationError, match="Malformed entry"):
            _parse_raw_scores(raw, 0.0, "test-model")

    def test_non_numeric_score_raises(self):
        raw = {"labels": ["cat"], "scores": ["high"]}
        with pytest.raises(ClassificationError, match="Non-numeric score"):
            _parse_raw_scores(raw, 0.0, "test-model")

    def test_mismatched_labels_scores_raises(self):
        """A labels/scores length mismatch must raise instead of being
        silently truncated by zip()."""
        raw = {"labels": ["cat", "dog", "bird"], "scores": [0.8, 0.2]}
        with pytest.raises(ClassificationError, match="Mismatched"):
            _parse_raw_scores(raw, 0.0, "test-model")
        raw = {"labels": ["cat"], "scores": [0.8, 0.2]}
        with pytest.raises(ClassificationError, match="Mismatched"):
            _parse_raw_scores(raw, 0.0, "test-model")


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


# ---------- Batch kwarg routing ----------


class _RecordingBatchEngine:
    """Stub engine that records classify_batch kwargs."""

    def __init__(self, calls):
        self._calls = calls

    def infer_using(self, model_id, using):
        return using or "text"

    def default_model(self, using):
        return "stub-model"

    def classify_item(self, **kwargs):
        raise AssertionError("batch entry points must not call classify_item")

    def classify_batch(self, **kwargs):
        self._calls.append(kwargs)
        return [
            ClassificationResult(
                scores=[CategoryScore("stub", 0.9)],
                model_id="stub-model",
                using="text",
            )
            for _ in kwargs["contents"]
        ]


class TestBatchKwargRouting:
    def _install_provider(self, monkeypatch):
        provider = EngineProvider()
        provider._entry_points_loaded = True
        monkeypatch.setattr(provider_module, "_PROVIDER", provider)
        engine_calls = []
        factory_calls = []

        def factory(**_):
            factory_calls.append(1)
            return _RecordingBatchEngine(engine_calls)

        provider.register("classification", "default", factory, replace=True)
        return engine_calls, factory_calls

    def test_pdf_classify_pages_routes_device_to_engine(self, monkeypatch):
        """device= must reach the engine batch call; resolution= must reach
        the content getter only."""
        engine_calls, factory_calls = self._install_provider(monkeypatch)

        pdf = npdf.PDF("pdfs/01-practice.pdf")
        try:
            content_calls = []

            def fake_content(model_type, **kwargs):
                content_calls.append(kwargs)
                return "page text"

            for page in pdf.pages:
                monkeypatch.setattr(page, "_get_classification_content", fake_content)

            pdf.classify_pages(labels=["a"], device="cpu", resolution=99, progress_bar=False)

            assert engine_calls, "engine classify_batch was never called"
            assert engine_calls[-1]["device"] == "cpu"
            assert "resolution" not in engine_calls[-1]
            assert content_calls and all(c == {"resolution": 99} for c in content_calls)
            assert len(factory_calls) == 1  # engine resolved once for this context
        finally:
            pdf.close()

    def test_collection_classify_all_routes_device_to_engine(self, monkeypatch):
        """Same split for PDFCollection.classify_all — and the engine must be
        resolved once (single context), not once per resolution site."""
        from natural_pdf.core.pdf_collection import PDFCollection

        engine_calls, factory_calls = self._install_provider(monkeypatch)

        collection = PDFCollection(["pdfs/01-practice.pdf"])
        try:
            content_calls = []

            def fake_content(model_type, **kwargs):
                content_calls.append(kwargs)
                return "pdf text"

            for pdf in collection.pdfs:
                monkeypatch.setattr(pdf, "_get_classification_content", fake_content)

            collection.classify_all(labels=["a"], device="cpu", resolution=99, progress_bar=False)

            assert engine_calls, "engine classify_batch was never called"
            assert engine_calls[-1]["device"] == "cpu"
            assert "resolution" not in engine_calls[-1]
            assert content_calls == [{"resolution": 99}]
            assert len(factory_calls) == 1  # single engine instance, context=collection
        finally:
            for pdf in collection.pdfs:
                pdf.close()


# ---------- Batch mismatch ----------


class TestBatchMismatch:
    def test_element_collection_batch_mismatch_raises(self, monkeypatch, caplog):
        """ClassificationBatchMixin raises on a result-count mismatch instead of
        silently returning the collection unchanged."""
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
            with pytest.raises(ClassificationError, match="returned 0 results"):
                elements.classify_all(labels=["a", "b"])
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
