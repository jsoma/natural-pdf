from __future__ import annotations

import logging

import pytest

import natural_pdf as npdf
import natural_pdf.classification.pipelines as pipelines_module
import natural_pdf.engine_provider as provider_module
from natural_pdf.classification.pipelines import (
    _CACHE_LOCK,
    _PIPELINE_CACHE,
    _parse_raw_scores,
    classify_batch_contents,
    classify_single,
    cleanup_models,
    validate_classification_labels,
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


def _stub_engine_checkout(engine):
    from contextlib import contextmanager

    @contextmanager
    def fake_checkout(context, engine_name=None):
        yield engine

    return fake_checkout


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
        classification_service,
        "checkout_classification_engine",
        _stub_engine_checkout(StubEngine()),
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
        classification_service,
        "checkout_classification_engine",
        _stub_engine_checkout(StubEngine()),
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

    def test_string_labels_container_raises(self):
        # A plain string would zip character-by-character; must be rejected.
        raw = {"labels": "cat", "scores": [0.5, 0.3, 0.2]}
        with pytest.raises(ClassificationError, match="labels"):
            _parse_raw_scores(raw, 0.0, "test-model")

    def test_non_sequence_scores_raises(self):
        # A scalar would leak a raw TypeError from len(); must be a
        # ClassificationError instead.
        raw = {"labels": ["cat"], "scores": 0.5}
        with pytest.raises(ClassificationError, match="scores"):
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

    def test_mapping_labels_container_raises(self):
        raw = {"labels": {"cat": 1}, "scores": [0.5]}
        with pytest.raises(ClassificationError, match="labels"):
            _parse_raw_scores(raw, 0.0, "test-model")

    def test_mapping_scores_container_raises(self):
        raw = {"labels": ["cat"], "scores": {"cat": 0.5}}
        with pytest.raises(ClassificationError, match="scores"):
            _parse_raw_scores(raw, 0.0, "test-model")

    def test_empty_labels_container_raises(self):
        with pytest.raises(ClassificationError, match="Empty"):
            _parse_raw_scores({"labels": [], "scores": []}, 0.0, "test-model")

    def test_empty_list_payload_raises(self):
        with pytest.raises(ClassificationError, match="Empty"):
            _parse_raw_scores([], 0.0, "test-model")

    def test_non_string_label_raises(self):
        raw = {"labels": [5], "scores": [0.5]}
        with pytest.raises(ClassificationError, match="Non-string label"):
            _parse_raw_scores(raw, 0.0, "test-model")

    def test_non_string_label_in_list_format_raises(self):
        raw = [{"label": {"x": 1}, "score": 0.5}]
        with pytest.raises(ClassificationError, match="Non-string label"):
            _parse_raw_scores(raw, 0.0, "test-model")

    def test_bool_score_raises(self):
        raw = {"labels": ["cat"], "scores": [True]}
        with pytest.raises(ClassificationError, match="Non-numeric score"):
            _parse_raw_scores(raw, 0.0, "test-model")

    def test_infinite_score_raises(self):
        raw = {"labels": ["cat"], "scores": [float("inf")]}
        with pytest.raises(ClassificationError, match="Non-finite score"):
            _parse_raw_scores(raw, 0.0, "test-model")

    def test_nan_score_raises(self):
        raw = {"labels": ["cat"], "scores": [float("nan")]}
        with pytest.raises(ClassificationError, match="Non-finite score"):
            _parse_raw_scores(raw, 0.0, "test-model")

    def test_tuple_containers_accepted(self):
        raw = {"labels": ("cat", "dog"), "scores": (0.7, 0.3)}
        scores = _parse_raw_scores(raw, 0.0, "test-model")
        assert [s.label for s in scores] == ["cat", "dog"]

    @pytest.mark.parametrize(
        "raw",
        [
            {"labels": ["   "], "scores": [0.5]},
            [{"label": "\t", "score": 0.5}],
        ],
    )
    def test_blank_label_raises(self, raw):
        with pytest.raises(ClassificationError, match="Blank label"):
            _parse_raw_scores(raw, 0.0, "test-model")

    def test_labels_are_whitespace_normalized(self):
        raw = {"labels": ["  cat \n"], "scores": [0.7]}
        scores = _parse_raw_scores(raw, 0.0, "test-model")
        assert [score.label for score in scores] == ["cat"]


@pytest.mark.parametrize(
    ("labels", "error_type"),
    [
        ([], ValueError),
        (["   "], ValueError),
        (["valid", 3], TypeError),
    ],
)
def test_single_item_invalid_labels_fail_before_engine_checkout(monkeypatch, labels, error_type):
    class Host:
        analyses = None

    def checkout_must_not_run(*_args, **_kwargs):
        raise AssertionError("empty labels must fail before engine checkout")

    monkeypatch.setattr(
        classification_service,
        "checkout_classification_engine",
        checkout_must_not_run,
    )

    host = Host()
    with pytest.raises(error_type):
        ClassificationService(PDFContext.with_defaults()).classify(host, labels=labels)
    assert host.analyses is None


@pytest.mark.parametrize(
    ("labels", "error_type"),
    [
        ([], ValueError),
        (["\t"], ValueError),
        (["valid", object()], TypeError),
    ],
)
def test_label_validator_rejects_invalid_entries(labels, error_type):
    with pytest.raises(error_type):
        validate_classification_labels(labels)


@pytest.mark.parametrize("entrypoint", ["single", "batch"])
@pytest.mark.parametrize(
    ("labels", "error_type"),
    [
        ([], ValueError),
        (["  "], ValueError),
        (["valid", 1], TypeError),
    ],
)
def test_pipeline_entries_validate_labels_before_dependency_probe(
    monkeypatch, entrypoint, labels, error_type
):
    dependency_checks = []

    def dependency_probe():
        dependency_checks.append(True)
        return True

    monkeypatch.setattr(pipelines_module, "_check_classification_dependencies", dependency_probe)

    with pytest.raises(error_type):
        if entrypoint == "single":
            classify_single(item_content="text", labels=labels)
        else:
            classify_batch_contents(contents=["text"], labels=labels)

    assert dependency_checks == []


def test_public_batch_entries_reject_blank_labels_before_checkout(monkeypatch):
    from natural_pdf.core.pdf_collection import PDFCollection

    def checkout_must_not_run(*_args, **_kwargs):
        raise AssertionError("invalid labels must fail before engine checkout")

    monkeypatch.setattr(
        classification_service,
        "checkout_classification_engine",
        checkout_must_not_run,
    )

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        with pytest.raises(ValueError, match="cannot be blank"):
            pdf.classify_pages(labels=[" "])

        collection = PDFCollection([pdf])
        with pytest.raises(ValueError, match="cannot be blank"):
            collection.classify_all(labels=[" "])

        elements = pdf.pages[0].find_all("text")[:1]
        with pytest.raises(ValueError, match="cannot be blank"):
            elements.classify_all(labels=[" "])
    finally:
        pdf.close()


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


def test_classify_custom_engine_invokes_custom_classify_item(monkeypatch):
    """classification_engine='custom' must route the actual classification to
    the custom engine, not just model/mode inference."""
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    custom_calls = []

    class _CustomEngine:
        def infer_using(self, model_id, using):
            return using or "text"

        def default_model(self, using):
            return "custom-model"

        def classify_item(self, **kwargs):
            custom_calls.append(kwargs)
            return ClassificationResult(
                scores=[CategoryScore("custom", 0.99)],
                model_id=kwargs.get("model_id", "custom-model"),
                using=kwargs.get("using", "text"),
            )

        def classify_batch(self, **kwargs):
            raise AssertionError("single-item classify must not call classify_batch")

    class _DefaultEngine(_CustomEngine):
        def classify_item(self, **kwargs):
            raise AssertionError(
                "classification_engine='custom' must not fall back to the default engine"
            )

    provider.register("classification", "default", lambda **_: _DefaultEngine(), replace=True)
    provider.register("classification", "custom", lambda **_: _CustomEngine(), replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        page.classify(labels=["custom"], classification_engine="custom")
        assert custom_calls, "custom engine classify_item was never invoked"
        assert page.analyses["classification"].category == "custom"
    finally:
        pdf.close()


def test_classify_all_transient_engine_resolved_once(monkeypatch):
    """classify_all must create exactly one engine instance even for
    transient-lifetime registrations (no second resolution inside
    run_classification_batch)."""
    from natural_pdf.core.pdf_collection import PDFCollection

    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    engine_calls = []
    factory_calls = []

    def factory(**_):
        factory_calls.append(1)
        return _RecordingBatchEngine(engine_calls)

    provider.register("classification", "default", factory, replace=True, lifetime="transient")

    collection = PDFCollection(["pdfs/01-practice.pdf"])
    try:
        for pdf in collection.pdfs:
            monkeypatch.setattr(
                pdf, "_get_classification_content", lambda model_type, **kw: "pdf text"
            )
        collection.classify_all(labels=["a"], progress_bar=False)
        assert engine_calls, "engine classify_batch was never called"
        assert len(factory_calls) == 1, "transient engine factory ran more than once"
    finally:
        for pdf in collection.pdfs:
            pdf.close()


# ---------- Transient engine lifecycle (checkout cleanup) ----------


class _ClosableEngine:
    """Stub engine recording classify calls and close() invocations."""

    def __init__(self, closed):
        self._closed = closed

    def infer_using(self, model_id, using):
        return using or "text"

    def default_model(self, using):
        return "stub-model"

    def _result(self):
        return ClassificationResult(
            scores=[CategoryScore("stub", 0.9)],
            model_id="stub-model",
            using="text",
        )

    def classify_item(self, **kwargs):
        return self._result()

    def classify_batch(self, **kwargs):
        return [self._result() for _ in kwargs["contents"]]

    def close(self):
        self._closed.append(self)


class TestTransientEngineCleanup:
    def _install(self, monkeypatch, lifetime):
        provider = EngineProvider()
        provider._entry_points_loaded = True
        monkeypatch.setattr(provider_module, "_PROVIDER", provider)
        closed = []
        factory_calls = []

        def factory(**_):
            factory_calls.append(1)
            return _ClosableEngine(closed)

        provider.register("classification", "default", factory, replace=True, lifetime=lifetime)
        return closed, factory_calls

    def test_page_classify_closes_transient_engine_once_per_call(self, monkeypatch):
        closed, factory_calls = self._install(monkeypatch, "transient")
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        try:
            page = pdf.pages[0]
            page.classify(labels=["stub"])
            assert len(factory_calls) == 1
            assert len(closed) == 1
            page.classify(labels=["stub"])
            assert len(factory_calls) == 2
            assert len(closed) == 2
        finally:
            pdf.close()

    def test_classify_pages_closes_transient_engine_once(self, monkeypatch):
        closed, factory_calls = self._install(monkeypatch, "transient")
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        try:
            for page in pdf.pages:
                monkeypatch.setattr(
                    page, "_get_classification_content", lambda model_type, **kw: "page text"
                )
            pdf.classify_pages(labels=["a"], progress_bar=False)
            assert len(factory_calls) == 1
            assert len(closed) == 1
        finally:
            pdf.close()

    def test_collection_classify_all_closes_transient_engine_once(self, monkeypatch):
        from natural_pdf.core.pdf_collection import PDFCollection

        closed, factory_calls = self._install(monkeypatch, "transient")
        collection = PDFCollection(["pdfs/01-practice.pdf"])
        try:
            for pdf in collection.pdfs:
                monkeypatch.setattr(
                    pdf, "_get_classification_content", lambda model_type, **kw: "pdf text"
                )
            collection.classify_all(labels=["a"], progress_bar=False)
            assert len(factory_calls) == 1
            assert len(closed) == 1
        finally:
            for pdf in collection.pdfs:
                pdf.close()

    def test_element_collection_classify_all_closes_transient_engine_once(self, monkeypatch):
        closed, factory_calls = self._install(monkeypatch, "transient")
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        try:
            elements = pdf.pages[0].find_all("text")[:3]
            assert len(elements) > 0
            elements.classify_all(labels=["a"], progress_bar=False)
            assert len(factory_calls) == 1, "transient engine factory ran more than once"
            assert len(closed) == 1
        finally:
            pdf.close()

    def test_direct_run_item_closes_transient_engine_once(self, monkeypatch):
        from natural_pdf.classification import run_classification_item

        closed, factory_calls = self._install(monkeypatch, "transient")
        result = run_classification_item(
            context=PDFContext.with_defaults(),
            content="text",
            labels=["stub"],
            model_id="stub-model",
            using="text",
            min_confidence=0.0,
            multi_label=False,
        )

        assert result.category == "stub"
        assert len(factory_calls) == 1
        assert len(closed) == 1

    @pytest.mark.parametrize("entrypoint", ["item", "batch"])
    @pytest.mark.parametrize(
        ("labels", "error_type"),
        [
            ([], ValueError),
            (["  "], ValueError),
            (["valid", 1], TypeError),
        ],
    )
    def test_direct_run_helpers_reject_invalid_labels_before_checkout(
        self, monkeypatch, entrypoint, labels, error_type
    ):
        from natural_pdf.classification import run_classification_batch, run_classification_item

        closed, factory_calls = self._install(monkeypatch, "transient")
        with pytest.raises(error_type):
            common = {
                "context": PDFContext.with_defaults(),
                "labels": labels,
                "model_id": "stub-model",
                "using": "text",
                "min_confidence": 0.0,
                "multi_label": False,
            }
            if entrypoint == "item":
                run_classification_item(content="text", **common)
            else:
                run_classification_batch(
                    contents=["text"], batch_size=1, progress_bar=False, **common
                )

        assert factory_calls == []
        assert closed == []

    def test_direct_run_batch_closes_transient_engine_once(self, monkeypatch):
        from natural_pdf.classification import run_classification_batch

        closed, factory_calls = self._install(monkeypatch, "transient")
        results = run_classification_batch(
            context=PDFContext.with_defaults(),
            contents=["one", "two"],
            labels=["stub"],
            model_id="stub-model",
            using="text",
            min_confidence=0.0,
            multi_label=False,
            batch_size=2,
            progress_bar=False,
        )

        assert [result.category for result in results] == ["stub", "stub"]
        assert len(factory_calls) == 1
        assert len(closed) == 1

    def test_pre_resolved_engine_remains_caller_owned(self):
        from natural_pdf.classification import run_classification_item

        closed = []
        engine = _ClosableEngine(closed)
        result = run_classification_item(
            context=PDFContext.with_defaults(),
            content="text",
            labels=["stub"],
            model_id="stub-model",
            using="text",
            min_confidence=0.0,
            multi_label=False,
            engine=engine,
        )

        assert result.category == "stub"
        assert closed == []

    @pytest.mark.parametrize("lifetime", ["context", "singleton"])
    def test_cached_lifetimes_are_never_cleaned(self, monkeypatch, lifetime):
        closed, factory_calls = self._install(monkeypatch, lifetime)
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        try:
            page = pdf.pages[0]
            page.classify(labels=["stub"])
            page.classify(labels=["stub"])
            assert len(factory_calls) == 1
            assert closed == []
        finally:
            pdf.close()


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

    def test_element_collection_routes_options_to_one_boundary(self, monkeypatch):
        """Element batches use the same content/engine split as PDF batches."""
        engine_calls, _ = self._install_provider(monkeypatch)
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        try:
            elements = pdf.pages[0].find_all("text")[:2]
            content_calls = []

            def fake_content(model_type, **kwargs):
                content_calls.append(kwargs)
                return "element text"

            for element in elements:
                monkeypatch.setattr(element, "_get_classification_content", fake_content)

            elements.classify_all(labels=["a"], device="cpu", resolution=99, progress_bar=False)

            assert engine_calls[-1]["device"] == "cpu"
            assert "resolution" not in engine_calls[-1]
            assert content_calls == [{"resolution": 99}, {"resolution": 99}]
        finally:
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

    def test_empty_collection_still_validates_labels(self):
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        try:
            empty = pdf.pages[0].find_all('text:contains("ZZZNONEXISTENT")')
            with pytest.raises(ValueError, match="Labels list cannot be empty"):
                empty.classify_all(labels=[])
        finally:
            pdf.close()
