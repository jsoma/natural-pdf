"""Tests for the VLM OCR parser and pipeline."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from natural_pdf.ocr.vlm_ocr import parse_grounding_response, scale_ocr_results

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_defaults():
    """Reset module-level VLM defaults."""
    import natural_pdf.core.vlm_client as mod

    orig_client, orig_model = mod._default_client, mod._default_model
    mod._default_client = None
    mod._default_model = None
    yield
    mod._default_client = orig_client
    mod._default_model = orig_model


# ---------------------------------------------------------------------------
# parse_grounding_response
# ---------------------------------------------------------------------------


class TestParseGroundingResponse:
    def test_basic_json_array(self):
        raw = json.dumps(
            [
                {"bbox": [10, 20, 200, 45], "text": "Hello World", "confidence": 0.95},
                {"bbox": [10, 50, 300, 75], "text": "Second line", "confidence": 0.8},
            ]
        )
        results = parse_grounding_response(raw)
        assert len(results) == 2
        assert results[0]["text"] == "Hello World"
        assert results[0]["bbox"] == [10.0, 20.0, 200.0, 45.0]
        assert results[0]["confidence"] == 0.95

    def test_markdown_fenced_json(self):
        raw = '```json\n[{"bbox": [1, 2, 3, 4], "text": "Test", "confidence": 0.9}]\n```'
        results = parse_grounding_response(raw)
        assert len(results) == 1
        assert results[0]["text"] == "Test"

    def test_json_with_extra_text(self):
        raw = 'Here are the results:\n[{"bbox": [1, 2, 3, 4], "text": "Found", "confidence": 0.7}]\nDone.'
        results = parse_grounding_response(raw)
        assert len(results) == 1
        assert results[0]["text"] == "Found"

    def test_missing_confidence_defaults(self):
        raw = json.dumps([{"bbox": [0, 0, 10, 10], "text": "No conf"}])
        results = parse_grounding_response(raw)
        assert len(results) == 1
        assert results[0]["confidence"] == 0.5

    def test_invalid_bbox_skipped(self):
        raw = json.dumps(
            [
                {"bbox": [1, 2], "text": "Short bbox"},  # only 2 values
                {"bbox": [1, 2, 3, 4], "text": "Valid"},
            ]
        )
        results = parse_grounding_response(raw)
        assert len(results) == 1
        assert results[0]["text"] == "Valid"

    def test_empty_text_skipped(self):
        raw = json.dumps(
            [
                {"bbox": [1, 2, 3, 4], "text": ""},
                {"bbox": [5, 6, 7, 8], "text": "OK"},
            ]
        )
        results = parse_grounding_response(raw)
        assert len(results) == 1
        assert results[0]["text"] == "OK"

    def test_invalid_json_returns_empty(self):
        results = parse_grounding_response("not json at all")
        assert results == []

    def test_non_array_returns_empty(self):
        results = parse_grounding_response('{"key": "value"}')
        assert results == []

    def test_confidence_clamped(self):
        raw = json.dumps([{"bbox": [0, 0, 1, 1], "text": "X", "confidence": 1.5}])
        results = parse_grounding_response(raw)
        assert results[0]["confidence"] == 1.0

        raw = json.dumps([{"bbox": [0, 0, 1, 1], "text": "X", "confidence": -0.5}])
        results = parse_grounding_response(raw)
        assert results[0]["confidence"] == 0.0


# ---------------------------------------------------------------------------
# scale_ocr_results
# ---------------------------------------------------------------------------


class TestScaleOCRResults:
    def test_basic_scaling(self):
        results = [{"bbox": [0, 0, 100, 50], "text": "Hi", "confidence": 0.9}]
        scaled = scale_ocr_results(
            results,
            image_width=200,
            image_height=100,
            page_width=400,
            page_height=200,
        )
        assert len(scaled) == 1
        assert scaled[0]["bbox"] == [0.0, 0.0, 200.0, 100.0]
        assert scaled[0]["text"] == "Hi"

    def test_with_offset(self):
        results = [{"bbox": [10, 20, 30, 40], "text": "T", "confidence": 1.0}]
        scaled = scale_ocr_results(
            results,
            image_width=100,
            image_height=100,
            page_width=200,
            page_height=200,
            offset_x=50,
            offset_y=100,
        )
        # scale is 2x in both dimensions, plus offset
        assert scaled[0]["bbox"] == [70.0, 140.0, 110.0, 180.0]

    def test_zero_image_size_returns_unscaled(self):
        results = [{"bbox": [10, 20, 30, 40], "text": "T", "confidence": 1.0}]
        scaled = scale_ocr_results(
            results,
            image_width=0,
            image_height=0,
            page_width=100,
            page_height=100,
        )
        assert scaled is results  # returned as-is

    def test_inverted_bbox_normalized(self):
        """Inverted coordinates (x1 < x0) should be swapped."""
        results = [{"bbox": [100, 50, 10, 20], "text": "Inv", "confidence": 0.9}]
        scaled = scale_ocr_results(
            results,
            image_width=200,
            image_height=100,
            page_width=200,
            page_height=100,
        )
        assert len(scaled) == 1
        bbox = scaled[0]["bbox"]
        assert bbox[0] < bbox[2]  # x0 < x1
        assert bbox[1] < bbox[3]  # y0 < y1

    def test_bbox_clamped_to_page(self):
        """Bboxes extending beyond page bounds should be clamped."""
        results = [{"bbox": [-10, -5, 250, 150], "text": "Big", "confidence": 0.9}]
        scaled = scale_ocr_results(
            results,
            image_width=200,
            image_height=100,
            page_width=200,
            page_height=100,
        )
        assert len(scaled) == 1
        bbox = scaled[0]["bbox"]
        assert bbox[0] >= 0
        assert bbox[1] >= 0
        assert bbox[2] <= 200
        assert bbox[3] <= 100

    def test_degenerate_bbox_skipped(self):
        """Zero-area boxes should be filtered out."""
        results = [
            {"bbox": [10, 10, 10, 10], "text": "Point", "confidence": 0.9},
            {"bbox": [10, 20, 50, 40], "text": "OK", "confidence": 0.9},
        ]
        scaled = scale_ocr_results(
            results,
            image_width=100,
            image_height=100,
            page_width=100,
            page_height=100,
        )
        assert len(scaled) == 1
        assert scaled[0]["text"] == "OK"


# ---------------------------------------------------------------------------
# apply_ocr with VLM path (integration-style test with mocks)
# ---------------------------------------------------------------------------


class TestApplyOCRVLMPath:
    def test_vlm_ocr_routes_correctly(self):
        """apply_ocr(model=...) should route to VLM OCR instead of traditional."""
        import natural_pdf

        pdf = natural_pdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = MagicMock()
        mock_response = json.dumps(
            [{"bbox": [10, 20, 200, 45], "text": "VLM Text", "confidence": 0.95}]
        )
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=mock_response))]
        )

        page.apply_ocr(model="test-model", client=mock_client)

        # The VLM client should have been called
        mock_client.chat.completions.create.assert_called_once()
        pdf.close()
