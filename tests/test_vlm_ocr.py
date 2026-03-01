"""Tests for the VLM OCR parser and pipeline."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from natural_pdf.core.vlm_prompts import build_ocr_prompt, detect_model_family
from natural_pdf.ocr.vlm_ocr import (
    normalize_gemini_coordinates,
    normalize_qwen_coordinates,
    parse_grounding_response,
    scale_ocr_results,
)

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

    def test_engine_vlm_routes_to_vlm_ocr(self):
        """apply_ocr(engine='vlm', ...) should route to _apply_vlm_ocr."""
        import natural_pdf

        pdf = natural_pdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = MagicMock()
        mock_response = json.dumps(
            [{"bbox": [10, 20, 200, 45], "text": "Engine VLM", "confidence": 0.9}]
        )
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=mock_response))]
        )

        page.apply_ocr(engine="vlm", model="test-model", client=mock_client)

        mock_client.chat.completions.create.assert_called_once()
        pdf.close()

    def test_engine_vlm_without_client_raises(self):
        """apply_ocr(engine='vlm') without model/client should raise ValueError."""
        import natural_pdf

        pdf = natural_pdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        with pytest.raises(ValueError, match='engine="vlm"'):
            page.apply_ocr(engine="vlm")

        pdf.close()

    def test_engine_vlm_uses_default_client(self):
        """apply_ocr(engine='vlm') should work if a default client is set."""
        import natural_pdf
        from natural_pdf.core.vlm_client import set_default_client

        pdf = natural_pdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = MagicMock()
        mock_response = json.dumps(
            [{"bbox": [10, 20, 200, 45], "text": "Default Client", "confidence": 0.9}]
        )
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=mock_response))]
        )

        set_default_client(mock_client, model="test-default-model")

        page.apply_ocr(engine="vlm")
        mock_client.chat.completions.create.assert_called_once()
        pdf.close()

    def test_region_engine_vlm_routes_correctly(self):
        """Region.apply_ocr(engine='vlm', ...) should route to _apply_vlm_ocr."""
        import natural_pdf

        pdf = natural_pdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = MagicMock()
        mock_response = json.dumps(
            [{"bbox": [5, 10, 100, 30], "text": "Region VLM", "confidence": 0.85}]
        )
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=mock_response))]
        )

        region = page.create_region(50, 50, 300, 200)
        region.apply_ocr(engine="vlm", model="test-model", client=mock_client)

        mock_client.chat.completions.create.assert_called_once()
        pdf.close()

    def test_region_engine_vlm_without_client_raises(self):
        """Region.apply_ocr(engine='vlm') without model/client should raise."""
        import natural_pdf

        pdf = natural_pdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]
        region = page.create_region(50, 50, 300, 200)

        with pytest.raises(ValueError, match='engine="vlm"'):
            region.apply_ocr(engine="vlm")

        pdf.close()

    def test_region_model_kwarg_routes_to_vlm(self):
        """Region.apply_ocr(model=...) should route to VLM OCR."""
        import natural_pdf

        pdf = natural_pdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = MagicMock()
        mock_response = json.dumps(
            [{"bbox": [5, 10, 100, 30], "text": "Region Model", "confidence": 0.85}]
        )
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=mock_response))]
        )

        region = page.create_region(50, 50, 300, 200)
        region.apply_ocr(model="test-model", client=mock_client)

        mock_client.chat.completions.create.assert_called_once()
        pdf.close()


# ---------------------------------------------------------------------------
# detect_model_family
# ---------------------------------------------------------------------------


class TestDetectModelFamily:
    def test_qwen3_vl_full_path(self):
        assert detect_model_family("Qwen/Qwen3-VL-2B-Instruct") == "qwen_vl"

    def test_qwen3_vl_short(self):
        assert detect_model_family("qwen3-vl-2b") == "qwen_vl"

    def test_qwen25_vl(self):
        assert detect_model_family("Qwen/Qwen2.5-VL-7B-Instruct") == "qwen_vl"

    def test_gutenocr_priority_over_qwen(self):
        """GutenOCR is checked before Qwen since it's built on Qwen2.5-VL."""
        assert detect_model_family("rootsautomation/GutenOCR-3B") == "gutenocr"

    def test_generic_model(self):
        assert detect_model_family("some/random-model") == "generic"

    def test_none_model(self):
        assert detect_model_family(None) == "generic"

    def test_case_insensitive(self):
        assert detect_model_family("QWEN3-VL-2B") == "qwen_vl"
        assert detect_model_family("GUTENOCR-3b") == "gutenocr"

    def test_gemini_flash(self):
        assert detect_model_family("gemini-2.5-flash") == "gemini"

    def test_gemini_pro(self):
        assert detect_model_family("gemini-2.5-pro") == "gemini"

    def test_gemini_case_insensitive(self):
        assert detect_model_family("GEMINI-2.0-FLASH") == "gemini"


# ---------------------------------------------------------------------------
# build_ocr_prompt with family
# ---------------------------------------------------------------------------


class TestBuildOCRPromptFamily:
    def test_generic_prompt_unchanged_no_family(self):
        prompt = build_ocr_prompt(grounding=True)
        assert "bbox" in prompt
        assert "text" in prompt

    def test_qwen_prompt_contains_bbox_2d(self):
        prompt = build_ocr_prompt(grounding=True, family="qwen_vl")
        assert "bbox_2d" in prompt
        assert "1000" in prompt

    def test_gutenocr_prompt_contains_text2d(self):
        prompt = build_ocr_prompt(grounding=True, family="gutenocr")
        assert "TEXT2D" in prompt

    def test_gemini_prompt_contains_box_2d(self):
        prompt = build_ocr_prompt(grounding=True, family="gemini")
        assert "box_2d" in prompt
        assert "y_min, x_min, y_max, x_max" in prompt
        assert "1000" in prompt

    def test_grounding_false_ignores_family(self):
        prompt_generic = build_ocr_prompt(grounding=False, family="generic")
        prompt_qwen = build_ocr_prompt(grounding=False, family="qwen_vl")
        assert prompt_generic == prompt_qwen
        assert "reading order" in prompt_generic


# ---------------------------------------------------------------------------
# parse_grounding_response with family
# ---------------------------------------------------------------------------


class TestParseGroundingResponseFamily:
    def test_gemini_box_2d_parsed(self):
        raw = json.dumps(
            [
                {"box_2d": [20, 100, 45, 500], "label": "Gemini Text"},
            ]
        )
        results = parse_grounding_response(raw, family="gemini")
        assert len(results) == 1
        assert results[0]["bbox"] == [20.0, 100.0, 45.0, 500.0]
        assert results[0]["text"] == "Gemini Text"
        assert results[0]["confidence"] == 0.75  # gemini default

    def test_qwen_keys_parsed(self):
        raw = json.dumps(
            [
                {"bbox_2d": [100, 200, 500, 250], "label": "Hello"},
            ]
        )
        results = parse_grounding_response(raw, family="qwen_vl")
        assert len(results) == 1
        assert results[0]["bbox"] == [100.0, 200.0, 500.0, 250.0]
        assert results[0]["text"] == "Hello"
        assert results[0]["confidence"] == 0.75  # qwen_vl default

    def test_gutenocr_keys_parsed(self):
        raw = json.dumps(
            [
                {"bbox": [10, 20, 300, 50], "text": "World"},
            ]
        )
        results = parse_grounding_response(raw, family="gutenocr")
        assert len(results) == 1
        assert results[0]["text"] == "World"
        assert results[0]["confidence"] == 0.8  # gutenocr default

    def test_generic_family_falls_back_to_bbox_2d(self):
        """Generic family can still parse bbox_2d if model outputs it."""
        raw = json.dumps(
            [
                {"bbox_2d": [10, 20, 30, 40], "label": "Adaptive"},
            ]
        )
        results = parse_grounding_response(raw, family="generic")
        assert len(results) == 1
        assert results[0]["text"] == "Adaptive"

    def test_qwen_family_falls_back_to_bbox(self):
        """Qwen family can still parse bbox/text if model ignores prompt."""
        raw = json.dumps(
            [
                {"bbox": [10, 20, 30, 40], "text": "Fallback"},
            ]
        )
        results = parse_grounding_response(raw, family="qwen_vl")
        assert len(results) == 1
        assert results[0]["text"] == "Fallback"

    def test_backward_compat_no_family(self):
        """Calling without family works as before."""
        raw = json.dumps(
            [
                {"bbox": [1, 2, 3, 4], "text": "Old", "confidence": 0.9},
            ]
        )
        results = parse_grounding_response(raw)
        assert len(results) == 1
        assert results[0]["text"] == "Old"
        assert results[0]["confidence"] == 0.9

    def test_explicit_confidence_overrides_default(self):
        """Explicit confidence in data takes precedence over family default."""
        raw = json.dumps(
            [
                {"bbox_2d": [10, 20, 30, 40], "label": "X", "confidence": 0.99},
            ]
        )
        results = parse_grounding_response(raw, family="qwen_vl")
        assert results[0]["confidence"] == 0.99


# ---------------------------------------------------------------------------
# normalize_qwen_coordinates
# ---------------------------------------------------------------------------


class TestNormalizeQwenCoordinates:
    def test_full_page_coords(self):
        results = [{"bbox": [0, 0, 1000, 1000], "text": "Full", "confidence": 0.9}]
        scaled = normalize_qwen_coordinates(results, 800, 600)
        assert scaled[0]["bbox"] == [0.0, 0.0, 800.0, 600.0]

    def test_partial_coords_scale(self):
        results = [{"bbox": [500, 250, 750, 500], "text": "Partial", "confidence": 0.8}]
        scaled = normalize_qwen_coordinates(results, 1000, 800)
        assert scaled[0]["bbox"] == [500.0, 200.0, 750.0, 400.0]

    def test_out_of_range_clamped_high(self):
        results = [{"bbox": [0, 0, 1200, 1100], "text": "Overflow", "confidence": 0.7}]
        scaled = normalize_qwen_coordinates(results, 800, 600)
        # Values >1000 clamped to 1000 → full image size
        assert scaled[0]["bbox"] == [0.0, 0.0, 800.0, 600.0]

    def test_out_of_range_clamped_low(self):
        results = [{"bbox": [-50, -100, 500, 500], "text": "Negative", "confidence": 0.7}]
        scaled = normalize_qwen_coordinates(results, 800, 600)
        assert scaled[0]["bbox"][0] == 0.0
        assert scaled[0]["bbox"][1] == 0.0

    def test_other_fields_preserved(self):
        results = [
            {"bbox": [0, 0, 1000, 1000], "text": "Keep", "confidence": 0.95, "extra": "data"}
        ]
        scaled = normalize_qwen_coordinates(results, 100, 100)
        assert scaled[0]["text"] == "Keep"
        assert scaled[0]["confidence"] == 0.95
        assert scaled[0]["extra"] == "data"


# ---------------------------------------------------------------------------
# normalize_gemini_coordinates (y,x → x,y swap + 0-1000 scaling)
# ---------------------------------------------------------------------------


class TestNormalizeGeminiCoordinates:
    def test_full_page_coords(self):
        # Gemini: [y_min, x_min, y_max, x_max] → should become [x0, y0, x1, y1]
        results = [{"bbox": [0, 0, 1000, 1000], "text": "Full", "confidence": 0.9}]
        scaled = normalize_gemini_coordinates(results, 800, 600)
        assert scaled[0]["bbox"] == [0.0, 0.0, 800.0, 600.0]

    def test_yx_swap(self):
        # Gemini input: [y_min=200, x_min=100, y_max=400, x_max=500]
        # Expected output: [x0=100/1000*800, y0=200/1000*600, x1=500/1000*800, y1=400/1000*600]
        #                = [80.0, 120.0, 400.0, 240.0]
        results = [{"bbox": [200, 100, 400, 500], "text": "Swapped", "confidence": 0.8}]
        scaled = normalize_gemini_coordinates(results, 800, 600)
        assert scaled[0]["bbox"] == [80.0, 120.0, 400.0, 240.0]

    def test_clamped_values(self):
        results = [{"bbox": [-50, -10, 1200, 1100], "text": "Clamped", "confidence": 0.7}]
        scaled = normalize_gemini_coordinates(results, 800, 600)
        assert scaled[0]["bbox"] == [0.0, 0.0, 800.0, 600.0]

    def test_other_fields_preserved(self):
        results = [
            {"bbox": [0, 0, 1000, 1000], "text": "Keep", "confidence": 0.95, "extra": "data"}
        ]
        scaled = normalize_gemini_coordinates(results, 100, 100)
        assert scaled[0]["text"] == "Keep"
        assert scaled[0]["confidence"] == 0.95
        assert scaled[0]["extra"] == "data"


# ---------------------------------------------------------------------------
# run_vlm_ocr with family detection (integration with mocks)
# ---------------------------------------------------------------------------


class TestRunVLMOCRFamily:
    def _make_mock_host(self):
        host = MagicMock()
        img = Image.new("RGB", (800, 600), "white")
        host.render.return_value = img
        return host, img

    def _make_mock_client(self, response_json):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_json))]
        )
        return client

    def test_qwen_model_uses_qwen_prompt_and_normalizes(self):
        from natural_pdf.ocr.vlm_ocr import run_vlm_ocr

        host, img = self._make_mock_host()
        qwen_response = json.dumps(
            [
                {"bbox_2d": [0, 0, 500, 500], "label": "Top-left"},
            ]
        )
        client = self._make_mock_client(qwen_response)

        results, size = run_vlm_ocr(host, model="Qwen/Qwen3-VL-2B-Instruct", client=client)

        # Should have called with Qwen prompt
        call_args = client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][1]["text"]
        assert "bbox_2d" in prompt_text

        # Coordinates should be normalized from 0-1000 to pixel coords
        assert len(results) == 1
        assert results[0]["bbox"] == [0.0, 0.0, 400.0, 300.0]  # 500/1000 * 800, 500/1000 * 600

    def test_custom_prompt_overrides_family_prompt(self):
        from natural_pdf.ocr.vlm_ocr import run_vlm_ocr

        host, img = self._make_mock_host()
        # Return generic format despite being called with qwen model
        generic_response = json.dumps(
            [
                {"bbox_2d": [0, 0, 500, 500], "label": "Custom"},
            ]
        )
        client = self._make_mock_client(generic_response)

        results, size = run_vlm_ocr(
            host,
            model="Qwen/Qwen3-VL-2B-Instruct",
            client=client,
            prompt="My custom prompt",
        )

        # Should have used custom prompt
        call_args = client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][1]["text"]
        assert prompt_text == "My custom prompt"

        # But family parsing and normalization still apply
        assert len(results) == 1
        assert results[0]["bbox"] == [0.0, 0.0, 400.0, 300.0]

    def test_default_model_triggers_family_detection(self):
        from natural_pdf.core.vlm_client import set_default_client
        from natural_pdf.ocr.vlm_ocr import run_vlm_ocr

        host, img = self._make_mock_host()
        qwen_response = json.dumps(
            [
                {"bbox_2d": [0, 0, 1000, 1000], "label": "Full"},
            ]
        )
        client = self._make_mock_client(qwen_response)

        set_default_client(client, model="Qwen/Qwen2.5-VL-7B-Instruct")

        results, size = run_vlm_ocr(host)

        # Should detect Qwen family from default model
        assert len(results) == 1
        # Full-page coords normalized: 1000/1000 * 800, 1000/1000 * 600
        assert results[0]["bbox"] == [0.0, 0.0, 800.0, 600.0]

    def test_gemini_model_uses_gemini_prompt_and_swaps_yx(self):
        from natural_pdf.ocr.vlm_ocr import run_vlm_ocr

        host, img = self._make_mock_host()
        # Gemini returns box_2d with [y_min, x_min, y_max, x_max]
        gemini_response = json.dumps(
            [
                {"box_2d": [0, 0, 500, 500], "label": "Top-left"},
            ]
        )
        client = self._make_mock_client(gemini_response)

        results, size = run_vlm_ocr(host, model="gemini-2.5-flash", client=client)

        # Should have called with Gemini prompt containing box_2d
        call_args = client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][1]["text"]
        assert "box_2d" in prompt_text
        assert "y_min, x_min" in prompt_text

        # Coordinates: [y_min=0, x_min=0, y_max=500, x_max=500]
        # After swap+scale: x0=0, y0=0, x1=500/1000*800=400, y1=500/1000*600=300
        assert len(results) == 1
        assert results[0]["bbox"] == [0.0, 0.0, 400.0, 300.0]

    def test_generic_model_no_normalization(self):
        from natural_pdf.ocr.vlm_ocr import run_vlm_ocr

        host, img = self._make_mock_host()
        generic_response = json.dumps(
            [
                {"bbox": [10, 20, 200, 45], "text": "Generic", "confidence": 0.9},
            ]
        )
        client = self._make_mock_client(generic_response)

        results, size = run_vlm_ocr(host, model="some-model", client=client)

        # No normalization for generic models — coords pass through as-is
        assert len(results) == 1
        assert results[0]["bbox"] == [10.0, 20.0, 200.0, 45.0]


# ---------------------------------------------------------------------------
# run_vlm_ocr with languages parameter
# ---------------------------------------------------------------------------


class TestRunVLMOCRLanguages:
    def _make_mock_host(self):
        host = MagicMock()
        img = Image.new("RGB", (800, 600), "white")
        host.render.return_value = img
        return host, img

    def _make_mock_client(self, response_json):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_json))]
        )
        return client

    def test_languages_appear_in_prompt(self):
        from natural_pdf.ocr.vlm_ocr import run_vlm_ocr

        host, _ = self._make_mock_host()
        response = json.dumps(
            [
                {"bbox": [10, 20, 200, 45], "text": "Test", "confidence": 0.9},
            ]
        )
        client = self._make_mock_client(response)

        run_vlm_ocr(host, model="some-model", client=client, languages=["ja"])

        call_args = client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][1]["text"]
        assert "Japanese" in prompt_text

    def test_no_hint_when_languages_omitted(self):
        from natural_pdf.ocr.vlm_ocr import run_vlm_ocr

        host, _ = self._make_mock_host()
        response = json.dumps(
            [
                {"bbox": [10, 20, 200, 45], "text": "Test", "confidence": 0.9},
            ]
        )
        client = self._make_mock_client(response)

        run_vlm_ocr(host, model="some-model", client=client)

        call_args = client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][1]["text"]
        assert "The document is in" not in prompt_text

    def test_custom_prompt_ignores_languages(self):
        from natural_pdf.ocr.vlm_ocr import run_vlm_ocr

        host, _ = self._make_mock_host()
        response = json.dumps(
            [
                {"bbox": [10, 20, 200, 45], "text": "Test", "confidence": 0.9},
            ]
        )
        client = self._make_mock_client(response)

        run_vlm_ocr(
            host,
            model="some-model",
            client=client,
            prompt="My custom prompt",
            languages=["ja"],
        )

        call_args = client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][1]["text"]
        assert prompt_text == "My custom prompt"
        assert "Japanese" not in prompt_text

    def test_english_only_no_hint(self):
        from natural_pdf.ocr.vlm_ocr import run_vlm_ocr

        host, _ = self._make_mock_host()
        response = json.dumps(
            [
                {"bbox": [10, 20, 200, 45], "text": "Test", "confidence": 0.9},
            ]
        )
        client = self._make_mock_client(response)

        run_vlm_ocr(host, model="some-model", client=client, languages=["en"])

        call_args = client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][1]["text"]
        assert "The document is in" not in prompt_text


# ---------------------------------------------------------------------------
# instructions parameter
# ---------------------------------------------------------------------------


class TestRunVLMOCRInstructions:
    def _make_mock_host(self):
        host = MagicMock()
        img = Image.new("RGB", (800, 600), "white")
        host.render.return_value = img
        return host, img

    def _make_mock_client(self, response_json):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_json))]
        )
        return client

    def test_instructions_appended_to_prompt(self):
        from natural_pdf.ocr.vlm_ocr import run_vlm_ocr

        host, _ = self._make_mock_host()
        response = json.dumps([{"bbox": [10, 20, 200, 45], "text": "Test", "confidence": 0.9}])
        client = self._make_mock_client(response)

        run_vlm_ocr(
            host,
            model="some-model",
            client=client,
            instructions="Focus on handwritten text only.",
        )

        call_args = client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][1]["text"]
        assert "Focus on handwritten text only." in prompt_text
        # The base prompt should still be there
        assert "bbox" in prompt_text or "text" in prompt_text

    def test_instructions_ignored_when_prompt_set(self):
        from natural_pdf.ocr.vlm_ocr import run_vlm_ocr

        host, _ = self._make_mock_host()
        response = json.dumps([{"bbox": [10, 20, 200, 45], "text": "Test", "confidence": 0.9}])
        client = self._make_mock_client(response)

        run_vlm_ocr(
            host,
            model="some-model",
            client=client,
            prompt="My full custom prompt",
            instructions="This should be ignored",
        )

        call_args = client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][1]["text"]
        assert prompt_text == "My full custom prompt"
        assert "This should be ignored" not in prompt_text


# ---------------------------------------------------------------------------
# ElementCollection VLM correction mode
# ---------------------------------------------------------------------------


class TestElementCollectionVLMCorrection:
    """When apply_ocr(engine='vlm') is called on a collection of existing OCR
    elements, it should correct text in-place rather than deleting elements
    and running grounded VLM OCR on tiny regions."""

    def _make_mock_client(self, response_text):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
        )
        return client

    def test_vlm_correction_updates_text_in_place(self):
        """OCR elements should have their text updated, not be deleted."""
        import natural_pdf

        pdf = natural_pdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        page.apply_ocr(engine="easyocr")
        ocr_elements = page.find_all("text[source=ocr]")
        if not ocr_elements:
            pdf.close()
            pytest.skip("No OCR elements created — cannot test correction")

        original_count = len(ocr_elements)

        mock_client = self._make_mock_client("Corrected Text")

        ocr_elements.apply_ocr(
            engine="vlm",
            model="gemini-2.5-flash",
            client=mock_client,
            instructions="Return corrected text.",
        )

        # Elements should still exist (not deleted)
        remaining = page.find_all("text[source=ocr]")
        assert len(remaining) >= original_count

        # One API call per OCR element
        assert mock_client.chat.completions.create.call_count == original_count

        # First element's text should be updated
        assert ocr_elements[0].text == "Corrected Text"
        assert ocr_elements[0].source == "ocr"

        pdf.close()

    def test_vlm_correction_uses_instructions_as_prompt(self):
        """User instructions should be used as the prompt, not the grounding prompt."""
        import natural_pdf

        pdf = natural_pdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        page.apply_ocr(engine="easyocr")
        ocr_elements = page.find_all("text[source=ocr]")
        if not ocr_elements:
            pdf.close()
            pytest.skip("No OCR elements created")

        mock_client = self._make_mock_client("Fixed")

        ocr_elements.apply_ocr(
            engine="vlm",
            model="gemini-2.5-flash",
            client=mock_client,
            instructions="Return only the exact text.",
        )

        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][1]["text"]
        assert prompt_text == "Return only the exact text."
        assert "bbox" not in prompt_text
        assert "JSON" not in prompt_text

        pdf.close()

    def test_non_ocr_elements_use_grounded_path(self):
        """Non-OCR elements should still use the standard grounded VLM OCR path."""
        import natural_pdf

        pdf = natural_pdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        # Create a region (not an OCR element)
        region = page.create_region(50, 50, 300, 200)

        mock_client = MagicMock()
        grounded_response = json.dumps(
            [{"bbox": [10, 20, 200, 45], "text": "Grounded", "confidence": 0.9}]
        )
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=grounded_response))]
        )

        # Calling apply_ocr on a region should use the grounded path
        region.apply_ocr(engine="vlm", model="gemini-2.5-flash", client=mock_client)

        # The prompt should contain grounding instructions
        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][1]["text"]
        assert "box_2d" in prompt_text  # Gemini grounding prompt

        pdf.close()


# ---------------------------------------------------------------------------
# Region._apply_vlm_ocr transactional replace
# ---------------------------------------------------------------------------


class TestRegionVLMOCRTransactional:
    """Region._apply_vlm_ocr should not delete existing OCR elements when
    the VLM returns no parseable results."""

    def test_no_deletion_on_empty_results(self):
        """If VLM returns unparseable text, original OCR elements survive."""
        import natural_pdf

        pdf = natural_pdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        page.apply_ocr(engine="easyocr")
        before_count = len(page.find_all("text[source=ocr]"))
        if before_count == 0:
            pdf.close()
            pytest.skip("No OCR elements created")

        # Mock client that returns plain text (not JSON) — simulates the bug
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Just plain text, no JSON"))]
        )

        region = page.create_region(0, 0, float(page.width), float(page.height))
        region._apply_vlm_ocr(model="some-model", client=mock_client, replace=True)

        # Original OCR elements should still be there
        after_count = len(page.find_all("text[source=ocr]"))
        assert after_count == before_count

        pdf.close()
