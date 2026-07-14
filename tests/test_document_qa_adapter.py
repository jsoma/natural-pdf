"""Regression tests for the local document-QA adapter.

These tests deliberately use a fake pipeline.  They exercise the coordinate
and result-shaping contract without loading a Hugging Face checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from natural_pdf.qa.document_qa import DocumentQA


class _FakePipeline:
    """Capture pipeline calls and return configurable QA results."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        if callable(self.result):
            return self.result(payload)
        if self.result is not None:
            return self.result
        if isinstance(payload, list):
            return [{"answer": "", "score": 0.0, "start": -1, "end": -1} for _ in payload]
        return {"answer": "", "score": 0.0, "start": -1, "end": -1}


def _engine(pipe: _FakePipeline) -> DocumentQA:
    """Build an initialized adapter without invoking its model constructor."""
    engine = DocumentQA.__new__(DocumentQA)
    engine.pipe = pipe
    engine.device = "cpu"
    engine.model_name = "test"
    engine._is_initialized = True
    return engine


def test_normalize_word_boxes_clamps_and_supports_float_origins():
    boxes = [
        ["outside", [0.5, -12.5, 120.5, 102.5]],
        ["inside", [35.5, 22.5, 85.5, 47.5]],
    ]

    normalized = DocumentQA._normalize_word_boxes(
        boxes,
        bounds=(10.5, -2.5, 110.5, 97.5),
    )

    assert normalized == [
        ["outside", [0, 0, 1000, 1000]],
        ["inside", [250, 250, 750, 500]],
    ]


@pytest.mark.parametrize(
    "bounds",
    [
        (0, 0, 0, 100),
        (0, 0, 100, 0),
        (5, 5, 4, 6),
        (5, 5, 6, 4),
    ],
)
def test_normalize_word_boxes_rejects_degenerate_dimensions(bounds):
    with pytest.raises(ValueError, match="positive width and height"):
        DocumentQA._normalize_word_boxes([["word", [0, 0, 1, 1]]], bounds)


def test_ask_forwards_defaults_and_explicit_pipeline_overrides():
    pipe = _FakePipeline({"answer": "", "score": 0.0, "start": -1, "end": -1})
    engine = _engine(pipe)
    image = Image.new("RGB", (20, 30), "white")

    engine.ask(image, "first", min_confidence=0)
    assert pipe.calls[-1][1] == {
        "handle_impossible_answer": True,
        "max_answer_len": 30,
    }

    engine.ask(
        image,
        "second",
        min_confidence=0,
        handle_impossible_answer=False,
        max_answer_len=12,
    )
    assert pipe.calls[-1][1] == {
        "handle_impossible_answer": False,
        "max_answer_len": 12,
    }


def test_ask_passes_already_normalized_boxes_unchanged():
    pipe = _FakePipeline({"answer": "", "score": 0.0, "start": -1, "end": -1})
    engine = _engine(pipe)
    boxes = [["word", [125, 250, 875, 750]]]

    engine.ask(Image.new("RGB", (20, 30)), "where?", word_boxes=boxes, min_confidence=0)

    query = pipe.calls[-1][0]
    assert query["word_boxes"] == boxes


def test_ask_high_score_empty_answer_is_not_found():
    pipe = _FakePipeline({"answer": "", "score": 0.99, "start": 0, "end": 0})
    result = _engine(pipe).ask(Image.new("RGB", (20, 30)), "missing?", min_confidence=0)

    assert result["answer"] == ""
    assert result["found"] is False
    assert result["confidence"] == pytest.approx(0.99)


def test_ask_preserves_single_and_batch_result_shapes():
    pipe = _FakePipeline(
        lambda payload: (
            [
                {"answer": "alpha", "score": 0.9, "start": 0, "end": 0},
                {"answer": "beta", "score": 0.8, "start": 1, "end": 1},
            ]
            if isinstance(payload, list)
            else {"answer": "alpha", "score": 0.9, "start": 0, "end": 0}
        )
    )
    engine = _engine(pipe)
    image = Image.new("RGB", (20, 30))

    single = engine.ask(image, "one", min_confidence=0)
    batch = engine.ask(image, ["one", "two"], min_confidence=0)

    assert isinstance(single, dict)
    assert single["answer"] == "alpha"
    assert isinstance(batch, list)
    assert [item["answer"] for item in batch] == ["alpha", "beta"]


def test_ask_empty_batch_returns_without_calling_pipeline():
    pipe = _FakePipeline()
    result = _engine(pipe).ask(Image.new("RGB", (20, 30)), [])

    assert result == []
    assert pipe.calls == []


@pytest.mark.parametrize("pipeline_result", [[], {}, {"score": 0.5}])
def test_ask_handles_empty_or_malformed_pipeline_result(pipeline_result):
    pipe = _FakePipeline(pipeline_result)
    result = _engine(pipe).ask(Image.new("RGB", (20, 30)), "missing?", min_confidence=0)

    assert result["answer"] == ""
    assert result["found"] is False


class _FakePage:
    index = 4
    width = 612.0
    height = 792.0

    def __init__(self, elements):
        self.elements = elements
        self.render_calls = []

    def find_all(self, selector):
        assert selector == "text"
        return self.elements

    def render(self, **kwargs):
        self.render_calls.append(kwargs)
        # Render is four pixels per PDF point; wrappers must not treat these
        # pixels as though they were source PDF coordinates.
        return Image.new("RGB", (2448, 3168), "white")


class _FakeRegion:
    x0, top, x1, bottom = 100.0, 200.0, 300.0, 500.0

    def __init__(self, page, elements):
        self.page = page
        self.elements = elements
        self.render_calls = []

    def find_all(self, selector):
        assert selector == "text"
        return self.elements

    def render(self, **kwargs):
        self.render_calls.append(kwargs)
        # This image is already cropped to the region's source bounds.
        return Image.new("RGB", (800, 1200), "white")


def _element(text, x0, top, x1, bottom):
    return SimpleNamespace(text=text, x0=x0, top=top, x1=x1, bottom=bottom)


def test_page_wrapper_renders_pil_directly_and_normalizes_page_coordinates():
    element = _element("center", 153.0, 198.0, 306.0, 396.0)
    page = _FakePage([element])
    pipe = _FakePipeline({"answer": "", "score": 0.0, "start": -1, "end": -1})

    _engine(pipe).ask_pdf_page(page, "where?", min_confidence=0)

    assert page.render_calls
    assert page.render_calls[0].get("resolution") == 300
    payload, _ = pipe.calls[-1]
    assert isinstance(payload["image"], Image.Image)
    assert payload["image"].size == (2448, 3168)
    assert payload["word_boxes"] == [["center", [250, 250, 500, 500]]]


def test_region_wrapper_uses_region_render_and_region_coordinates():
    element = _element("region", 150.0, 250.0, 250.0, 450.0)
    page = _FakePage([])
    region = _FakeRegion(page, [element])
    pipe = _FakePipeline({"answer": "", "score": 0.0, "start": -1, "end": -1})

    _engine(pipe).ask_pdf_region(region, "where?", min_confidence=0)

    assert region.render_calls == [{"resolution": 300, "crop": True, "highlights": False}]
    payload, _ = pipe.calls[-1]
    assert isinstance(payload["image"], Image.Image)
    assert payload["image"].size == (800, 1200)
    assert payload["word_boxes"] == [["region", [250, 167, 750, 833]]]


def test_duplicate_text_mapping_uses_answer_span_index_not_text_matching():
    first = _element("same", 10, 10, 20, 20)
    second = _element("same", 30, 30, 40, 40)
    page = _FakePage([first, second])
    pipe = _FakePipeline({"answer": "same", "score": 0.9, "start": 1, "end": 1})

    result = _engine(pipe).ask_pdf_page(page, "which?", min_confidence=0)

    assert list(result["source_elements"]) == [second]
