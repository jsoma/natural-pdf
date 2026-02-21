"""Tests for ClassificationResult and accessor mixin."""

from __future__ import annotations

from datetime import datetime

import pytest

from natural_pdf.classification.accessors import ClassificationResultAccessorMixin
from natural_pdf.classification.results import CategoryScore, ClassificationResult

# ---------- ClassificationResult ----------


class TestClassificationResultConstruction:
    def test_basic_construction(self):
        scores = [CategoryScore("cat", 0.8), CategoryScore("dog", 0.2)]
        result = ClassificationResult(
            scores=scores,
            model_id="test-model",
            using="text",
        )
        assert result.category == "cat"
        assert result.score == 0.8
        assert result.model_id == "test-model"
        assert result.using == "text"
        assert len(result.scores) == 2

    def test_empty_scores(self):
        result = ClassificationResult(
            scores=[],
            model_id="test-model",
            using="text",
        )
        assert result.category is None
        assert result.score == 0.0
        assert result.scores == []

    def test_scores_sorted_descending(self):
        scores = [
            CategoryScore("low", 0.1),
            CategoryScore("high", 0.9),
            CategoryScore("mid", 0.5),
        ]
        result = ClassificationResult(
            scores=scores,
            model_id="test-model",
            using="text",
        )
        assert result.scores[0].label == "high"
        assert result.scores[1].label == "mid"
        assert result.scores[2].label == "low"
        # Top category is the highest score
        assert result.category == "high"
        assert result.score == 0.9

    def test_custom_timestamp(self):
        ts = datetime(2025, 1, 1, 12, 0, 0)
        result = ClassificationResult(
            scores=[CategoryScore("a", 1.0)],
            model_id="m",
            using="text",
            timestamp=ts,
        )
        assert result.timestamp == ts


class TestClassificationResultToDict:
    def test_correctness(self):
        result = ClassificationResult(
            scores=[CategoryScore("cat", 0.8), CategoryScore("dog", 0.2)],
            model_id="test-model",
            using="text",
        )
        d = result.to_dict()
        assert d["category"] == "cat"
        assert d["score"] == 0.8
        assert len(d["scores"]) == 2
        assert d["model_id"] == "test-model"
        assert d["using"] == "text"
        assert "timestamp" in d

    def test_caching(self):
        result = ClassificationResult(
            scores=[CategoryScore("a", 1.0)],
            model_id="m",
            using="text",
        )
        d1 = result.to_dict()
        d2 = result.to_dict()
        assert d1 is d2  # Same dict object returned


class TestClassificationResultMapping:
    def test_getitem(self):
        result = ClassificationResult(
            scores=[CategoryScore("cat", 0.8)],
            model_id="test-model",
            using="text",
        )
        assert result["category"] == "cat"
        assert result["score"] == 0.8

    def test_getitem_missing_key(self):
        result = ClassificationResult(
            scores=[CategoryScore("cat", 0.8)],
            model_id="test-model",
            using="text",
        )
        with pytest.raises(KeyError):
            result["nonexistent"]

    def test_iter(self):
        result = ClassificationResult(
            scores=[CategoryScore("cat", 0.8)],
            model_id="test-model",
            using="text",
        )
        keys = list(result)
        assert "category" in keys
        assert "score" in keys
        assert "model_id" in keys

    def test_len(self):
        result = ClassificationResult(
            scores=[CategoryScore("cat", 0.8)],
            model_id="test-model",
            using="text",
        )
        assert len(result) == 7  # category, score, scores, model_id, using, parameters, timestamp


class TestClassificationResultRepr:
    def test_repr_format(self):
        result = ClassificationResult(
            scores=[CategoryScore("cat", 0.8)],
            model_id="test-model",
            using="text",
        )
        r = repr(result)
        assert "ClassificationResult" in r
        assert "cat" in r
        assert "0.800" in r
        assert "test-model" in r


# ---------- ClassificationResultAccessorMixin ----------


class _FakeHost(ClassificationResultAccessorMixin):
    """Minimal host for accessor testing."""

    def __init__(self):
        self.analyses: dict = {}


class TestAccessorMixin:
    def test_category_with_classification_result(self):
        host = _FakeHost()
        host.analyses["classification"] = ClassificationResult(
            scores=[CategoryScore("invoice", 0.95)],
            model_id="m",
            using="text",
        )
        assert host.category == "invoice"
        assert host.category_confidence == 0.95

    def test_category_when_no_classification(self):
        host = _FakeHost()
        assert host.category is None
        assert host.category_confidence is None

    def test_classification_results_dict(self):
        host = _FakeHost()
        result = ClassificationResult(
            scores=[CategoryScore("report", 0.7)],
            model_id="m",
            using="text",
        )
        host.analyses["classification"] = result
        d = host.classification_results
        assert isinstance(d, dict)
        assert d["category"] == "report"

    def test_classification_results_when_none(self):
        host = _FakeHost()
        assert host.classification_results is None

    def test_category_with_dict_result(self):
        host = _FakeHost()
        host.analyses["classification"] = {"category": "memo", "score": 0.6}
        assert host.category == "memo"
        assert host.category_confidence == 0.6

    def test_category_confidence_zero_is_not_treated_as_missing(self):
        """A score of 0.0 should be returned, not treated as falsy/missing."""
        host = _FakeHost()
        host.analyses["classification"] = {"category": "unlikely", "score": 0.0}
        assert host.category_confidence == 0.0

    def test_category_confidence_zero_does_not_fallthrough_to_confidence_key(self):
        """score=0.0 should NOT fall through to the 'confidence' key."""
        host = _FakeHost()
        host.analyses["classification"] = {
            "category": "x",
            "score": 0.0,
            "confidence": 0.99,
        }
        assert host.category_confidence == 0.0
