"""Tests for the PaddleOCR-VL engine adapter."""

import types

import numpy as np
import pytest

from natural_pdf.ocr.engine_paddleocr_vl import (
    _DEFAULT_BLOCK_CONFIDENCE,
    PaddleOCRVLEngine,
    _strip_html_tags,
)
from natural_pdf.ocr.ocr_options import PaddleOCRVLOptions

# ---------------------------------------------------------------------------
# Unit tests (no optional deps required)
# ---------------------------------------------------------------------------


class TestStripHtmlTags:
    def test_basic_html(self):
        assert _strip_html_tags("<p>Hello</p>") == "Hello"

    def test_table_html(self):
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        result = _strip_html_tags(html)
        assert "A" in result
        assert "B" in result
        assert "<" not in result

    def test_empty_string(self):
        assert _strip_html_tags("") == ""

    def test_no_tags(self):
        assert _strip_html_tags("plain text") == "plain text"


class TestPaddleOCRVLOptions:
    def test_defaults(self):
        opts = PaddleOCRVLOptions()
        assert opts.pipeline_version is None
        assert opts.use_layout_detection is None
        assert opts.use_chart_recognition is None
        assert opts.use_seal_recognition is None
        assert opts.use_doc_orientation_classify is None
        assert opts.use_doc_unwarping is None
        assert opts.format_block_content is None
        assert opts.extra_args == {}

    def test_custom_values(self):
        opts = PaddleOCRVLOptions(
            pipeline_version="v1.5",
            use_layout_detection=True,
            use_chart_recognition=False,
        )
        assert opts.pipeline_version == "v1.5"
        assert opts.use_layout_detection is True
        assert opts.use_chart_recognition is False


class TestPaddleOCRVLEngineUnit:
    """Unit tests using mocked PaddleOCR-VL internals."""

    def _make_block(self, label, bbox, content):
        """Create a mock PaddleOCRVLBlock-like object."""
        block = types.SimpleNamespace(label=label, bbox=bbox, content=content)
        return block

    def _make_result(self, blocks):
        """Create a mock PaddleOCRVLResult-like object."""
        return types.SimpleNamespace(blocks=blocks)

    def test_standardize_text_blocks(self):
        engine = PaddleOCRVLEngine()
        blocks = [
            self._make_block("text", [10, 20, 200, 50], "Hello world"),
            self._make_block("paragraph_title", [10, 60, 200, 80], "Title"),
            self._make_block("header", [10, 0, 200, 15], "Page Header"),
            self._make_block("footer", [10, 500, 200, 520], "Page Footer"),
        ]
        raw_results = [self._make_result(blocks)]
        regions = engine._standardize_results(raw_results, min_confidence=0.0, detect_only=False)

        assert len(regions) == 4
        assert regions[0].text == "Hello world"
        assert regions[1].text == "Title"
        assert regions[2].text == "Page Header"
        assert regions[3].text == "Page Footer"
        for r in regions:
            assert r.confidence == _DEFAULT_BLOCK_CONFIDENCE

    def test_skip_image_blocks(self):
        engine = PaddleOCRVLEngine()
        blocks = [
            self._make_block("text", [10, 20, 200, 50], "Visible"),
            self._make_block("image", [10, 60, 200, 200], ""),
            self._make_block("chart", [10, 210, 200, 400], "chart data"),
            self._make_block("seal", [300, 300, 400, 400], "seal"),
        ]
        raw_results = [self._make_result(blocks)]
        regions = engine._standardize_results(raw_results, min_confidence=0.0, detect_only=False)

        assert len(regions) == 1
        assert regions[0].text == "Visible"

    def test_table_block_strips_html(self):
        engine = PaddleOCRVLEngine()
        blocks = [
            self._make_block(
                "table",
                [10, 20, 400, 200],
                "<table><tr><td>Name</td><td>Value</td></tr></table>",
            ),
        ]
        raw_results = [self._make_result(blocks)]
        regions = engine._standardize_results(raw_results, min_confidence=0.0, detect_only=False)

        assert len(regions) == 1
        assert "<" not in regions[0].text
        assert "Name" in regions[0].text
        assert "Value" in regions[0].text

    def test_detect_only_mode(self):
        engine = PaddleOCRVLEngine()
        blocks = [
            self._make_block("text", [10, 20, 200, 50], "Hello"),
            self._make_block("table", [10, 60, 400, 200], "<table><tr><td>X</td></tr></table>"),
        ]
        raw_results = [self._make_result(blocks)]
        regions = engine._standardize_results(raw_results, min_confidence=0.0, detect_only=True)

        assert len(regions) == 2
        for r in regions:
            assert r.text == ""
            assert r.confidence == 0.0

    def test_empty_results(self):
        engine = PaddleOCRVLEngine()
        assert engine._standardize_results([], 0.0, False) == []
        assert engine._standardize_results(None, 0.0, False) == []
        # Non-list, non-dict, non-block object returns empty
        assert engine._standardize_results("unexpected", 0.0, False) == []

    def test_single_result_object_normalized(self):
        """A single result object (not wrapped in list) should be handled."""
        engine = PaddleOCRVLEngine()
        single_result = self._make_result(
            [
                self._make_block("text", [10, 20, 200, 50], "Single"),
            ]
        )
        regions = engine._standardize_results(single_result, min_confidence=0.0, detect_only=False)
        assert len(regions) == 1
        assert regions[0].text == "Single"

    def test_dict_style_blocks(self):
        """Test handling when blocks come as dicts instead of objects."""
        engine = PaddleOCRVLEngine()
        raw_results = [
            {
                "blocks": [
                    {"label": "text", "bbox": [10, 20, 200, 50], "content": "Dict text"},
                    {"label": "image", "bbox": [10, 60, 200, 200], "content": ""},
                ]
            }
        ]
        regions = engine._standardize_results(raw_results, min_confidence=0.0, detect_only=False)

        assert len(regions) == 1
        assert regions[0].text == "Dict text"

    def test_dict_parsing_res_list_key(self):
        """Test handling PaddleOCR-VL 3.4+ format with 'parsing_res_list' key."""
        engine = PaddleOCRVLEngine()
        raw_results = [
            {
                "parsing_res_list": [
                    self._make_block("text", [207, 359, 1025, 416], "Site: Durham's"),
                    self._make_block("header", [1665, 149, 2359, 238], "Page Header"),
                    self._make_block("image", [10, 60, 200, 200], ""),
                ],
            }
        ]
        regions = engine._standardize_results(raw_results, min_confidence=0.0, detect_only=False)

        assert len(regions) == 2
        assert regions[0].text == "Site: Durham's"
        assert regions[1].text == "Page Header"

    def test_bbox_as_numpy_array(self):
        engine = PaddleOCRVLEngine()
        blocks = [
            self._make_block("text", np.array([10, 20, 200, 50]), "Numpy bbox"),
        ]
        raw_results = [self._make_result(blocks)]
        regions = engine._standardize_results(raw_results, min_confidence=0.0, detect_only=False)

        assert len(regions) == 1
        assert regions[0].bbox == (10.0, 20.0, 200.0, 50.0)

    def test_skip_empty_content(self):
        engine = PaddleOCRVLEngine()
        blocks = [
            self._make_block("text", [10, 20, 200, 50], ""),
            self._make_block("text", [10, 60, 200, 80], "   "),
        ]
        raw_results = [self._make_result(blocks)]
        regions = engine._standardize_results(raw_results, min_confidence=0.0, detect_only=False)

        assert len(regions) == 0

    def test_skip_unknown_labels(self):
        """Unknown block labels should be ignored (allowlist-based filtering)."""
        engine = PaddleOCRVLEngine()
        blocks = [
            self._make_block("formula", [10, 20, 200, 50], "E=mc^2"),
            self._make_block("caption", [10, 60, 200, 80], "Figure 1"),
            self._make_block("footnote_mark", [10, 90, 200, 100], "1"),
            self._make_block("text", [10, 110, 200, 130], "Kept"),
        ]
        raw_results = [self._make_result(blocks)]
        regions = engine._standardize_results(raw_results, min_confidence=0.0, detect_only=False)

        assert len(regions) == 1
        assert regions[0].text == "Kept"

    def test_min_confidence_filter(self):
        engine = PaddleOCRVLEngine()
        blocks = [
            self._make_block("text", [10, 20, 200, 50], "High conf"),
        ]
        raw_results = [self._make_result(blocks)]

        # Default confidence is 0.99, so filtering at 1.0 should exclude
        regions = engine._standardize_results(raw_results, min_confidence=1.0, detect_only=False)
        assert len(regions) == 0

        # Filtering at 0.5 should include
        regions = engine._standardize_results(raw_results, min_confidence=0.5, detect_only=False)
        assert len(regions) == 1


class TestEngineRegistration:
    """Test that paddlevl is registered in the engine registry."""

    def test_registry_entry(self):
        from natural_pdf.ocr.ocr_provider import ENGINE_REGISTRY

        assert "paddlevl" in ENGINE_REGISTRY
        assert ENGINE_REGISTRY["paddlevl"]["provider"] is PaddleOCRVLEngine
        assert ENGINE_REGISTRY["paddlevl"]["options_class"] is PaddleOCRVLOptions


# ---------------------------------------------------------------------------
# Integration tests (require paddleocr + paddlepaddle)
# ---------------------------------------------------------------------------


@pytest.mark.optional_deps
class TestPaddleOCRVLIntegration:
    """Integration tests requiring PaddleOCR-VL to be installed."""

    @pytest.fixture(autouse=True)
    def check_available(self):
        engine = PaddleOCRVLEngine()
        if not engine.is_available():
            pytest.skip("PaddleOCR-VL dependencies not installed")

    def test_ocr_on_needs_ocr_pdf(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/needs-ocr.pdf")
        try:
            page = pdf.pages[0]
            page.apply_ocr(engine="paddlevl")
            text = page.extract_text()
            assert len(text) > 0, "Expected non-empty text from OCR"

            elements = page.find_all("text[source=ocr]")
            assert len(elements) > 0, "Expected OCR text elements"

            # Check bboxes are reasonable
            for el in elements:
                assert el.x0 >= 0
                assert el.x1 <= page.width + 1  # small tolerance
                assert el.top >= 0
                assert el.bottom <= page.height + 1
        finally:
            pdf.close()
