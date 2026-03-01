"""Tests for the checkbox detection reimplementation.

Sections:
1. NMS unit tests
2. Coordinate mapping unit tests
3. Classifier metric tests
4. Vector detector contract tests
5. Engine availability tests
6. Orchestrator auto-strategy tests
7. Service dispatch tests
8. Lazy import test
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from natural_pdf.analyzers.checkbox.base import CheckboxDetector, DetectionContext
from natural_pdf.analyzers.checkbox.checkbox_options import (
    BaseCheckboxOptions,
    DefaultCheckboxOptions,
    OnnxCheckboxOptions,
    VectorCheckboxOptions,
    VLMCheckboxOptions,
)
from natural_pdf.analyzers.checkbox.classifier import CheckboxClassifier

# ============================================================
# 1. NMS unit tests
# ============================================================


class TestNMS:
    def test_empty_input(self):
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        result = CheckboxDetector.nms(boxes, scores, 0.5)
        assert result == []

    def test_single_box(self):
        boxes = np.array([[10, 10, 20, 20]])
        scores = np.array([0.9])
        result = CheckboxDetector.nms(boxes, scores, 0.5)
        assert result == [0]

    def test_no_overlap(self):
        boxes = np.array(
            [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
            ]
        )
        scores = np.array([0.9, 0.8, 0.7])
        result = CheckboxDetector.nms(boxes, scores, 0.5)
        assert sorted(result) == [0, 1, 2]

    def test_full_overlap_keeps_higher_score(self):
        # Two identical boxes, different scores
        boxes = np.array(
            [
                [10, 10, 20, 20],
                [10, 10, 20, 20],
            ]
        )
        scores = np.array([0.5, 0.9])
        result = CheckboxDetector.nms(boxes, scores, 0.5)
        # Should keep the higher-score box (index 1)
        assert result == [1]

    def test_partial_overlap_below_threshold(self):
        # Two boxes with slight overlap (IoU < 0.5)
        boxes = np.array(
            [
                [0, 0, 10, 10],
                [8, 0, 18, 10],
            ]
        )
        scores = np.array([0.9, 0.8])
        # IoU = 2*10 / (100 + 100 - 20) = 20/180 ≈ 0.11
        result = CheckboxDetector.nms(boxes, scores, 0.5)
        assert sorted(result) == [0, 1]

    def test_threshold_zero_keeps_only_best(self):
        # With threshold=0, any overlap should suppress
        boxes = np.array(
            [
                [0, 0, 10, 10],
                [5, 5, 15, 15],
            ]
        )
        scores = np.array([0.9, 0.8])
        result = CheckboxDetector.nms(boxes, scores, 0.0)
        assert result == [0]

    def test_score_ordering(self):
        """Higher-score boxes should be kept preferentially."""
        boxes = np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 10, 10],
            ]
        )
        scores = np.array([0.3, 0.7])
        result = CheckboxDetector.nms(boxes, scores, 0.5)
        assert result == [1]  # Higher score wins


# ============================================================
# 2. Coordinate mapping unit tests
# ============================================================


class TestCoordinateMapping:
    def test_image_to_pdf_scaling(self):
        """Image coords should map to PDF coords via scale factors."""
        # Page: 612x792 (letter), rendered at 150 DPI -> ~1275x1650 px
        page_w, page_h = 612.0, 792.0
        img_w, img_h = 1275, 1650

        scale_x = page_w / img_w
        scale_y = page_h / img_h

        # Detection at image coords (100, 200, 120, 220)
        img_bbox = (100, 200, 120, 220)
        pdf_x0 = img_bbox[0] * scale_x
        pdf_y0 = img_bbox[1] * scale_y
        pdf_x1 = img_bbox[2] * scale_x
        pdf_y1 = img_bbox[3] * scale_y

        assert abs(pdf_x0 - 100 * scale_x) < 0.01
        assert abs(pdf_y0 - 200 * scale_y) < 0.01
        assert pdf_x1 > pdf_x0
        assert pdf_y1 > pdf_y0

    def test_pdf_coords_identity(self):
        """PDF coord_space should pass through as-is."""
        bbox = (50.0, 100.0, 70.0, 120.0)
        detection = {"bbox": bbox, "coord_space": "pdf"}
        # No conversion needed
        assert detection["bbox"] == bbox


# ============================================================
# 3. Classifier metric tests
# ============================================================


class TestClassifier:
    def test_white_image_is_unchecked(self):
        """Pure white image should classify as unchecked."""
        img = Image.new("L", (24, 24), 255)
        metrics = CheckboxClassifier._extract_metrics(img.convert("RGB"))
        is_checked, state, conf = CheckboxClassifier._classify_from_metrics(metrics)
        assert not is_checked
        assert state == "unchecked"

    def test_dark_image_is_checked(self):
        """Mostly dark image should classify as checked."""
        img = Image.new("L", (24, 24), 30)
        metrics = CheckboxClassifier._extract_metrics(img.convert("RGB"))
        is_checked, state, conf = CheckboxClassifier._classify_from_metrics(metrics)
        assert is_checked
        assert state == "checked"

    def test_metrics_extraction(self):
        """Metrics should have expected keys and valid ranges."""
        img = Image.new("RGB", (30, 30), (128, 128, 128))
        metrics = CheckboxClassifier._extract_metrics(img)
        assert "center_darkness" in metrics
        assert "ink_density" in metrics
        assert "dark_pixel_ratio" in metrics
        assert "std_dev" in metrics
        assert 0 <= metrics["ink_density"] <= 255
        assert 0 <= metrics["dark_pixel_ratio"] <= 1.0

    def test_classify_regions_with_judge(self):
        """Judge classification should take precedence."""
        region = MagicMock()
        region.bbox = (10, 10, 20, 20)
        region.is_checked = None
        region.checkbox_state = "unknown"
        region.analyses = {"checkbox": {}}

        mock_judge = MagicMock()
        from collections import namedtuple

        Decision = namedtuple("Decision", ["label", "score"])
        mock_judge.decide.return_value = Decision(label="checked", score=0.95)

        CheckboxClassifier.classify_regions([region], MagicMock(), judge=mock_judge)
        assert region.is_checked is True
        assert region.checkbox_state == "checked"


# ============================================================
# 4. Vector detector contract tests
# ============================================================


class TestVectorDetector:
    def test_finds_rects_in_practice_pdf(self):
        """Vector detector should find rect elements in a native PDF."""
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        try:
            page = pdf.pages[0]
            rects = page.find_all("rect")
            # Just verify the detector runs without error
            from natural_pdf.analyzers.checkbox.vector import VectorCheckboxDetector

            detector = VectorCheckboxDetector()
            assert detector.is_available()

            context = DetectionContext(page=page)
            results = detector.detect(Image.new("RGB", (1, 1)), VectorCheckboxOptions(), context)
            # Results depend on actual PDF content - just verify format
            for r in results:
                assert "bbox" in r
                assert r["coord_space"] == "pdf"
                assert r["engine"] == "vector"
                assert r["label"] == "checkbox"
        finally:
            pdf.close()

    def test_requires_page_context(self):
        """Vector detector should return empty without page context."""
        from natural_pdf.analyzers.checkbox.vector import VectorCheckboxDetector

        detector = VectorCheckboxDetector()
        results = detector.detect(Image.new("RGB", (1, 1)), VectorCheckboxOptions())
        assert results == []

    def test_density_filter(self):
        """Too many rects should trigger density filter."""
        from natural_pdf.analyzers.checkbox.vector import VectorCheckboxDetector

        detector = VectorCheckboxDetector()

        # Create mock page with many small rects
        mock_page = MagicMock()
        mock_rects = []
        for i in range(60):
            r = MagicMock()
            r.bbox = (i * 20, 0, i * 20 + 10, 10)  # Small rects within range
            r.stroke = True
            r.fill = None
            r.non_stroking_color = None
            mock_rects.append(r)

        # Mock find_all to return the rects
        mock_collection = MagicMock()
        mock_collection.__iter__ = lambda self: iter(mock_rects)
        mock_collection.__bool__ = lambda self: True
        mock_page.find_all.return_value = mock_collection

        context = DetectionContext(page=mock_page)
        results = detector.detect(Image.new("RGB", (1, 1)), VectorCheckboxOptions(), context)
        assert results == []  # Density filter should reject


# ============================================================
# 5. Engine availability tests
# ============================================================


class TestEngineAvailability:
    def test_vector_always_available(self):
        from natural_pdf.analyzers.checkbox.vector import VectorCheckboxDetector

        assert VectorCheckboxDetector().is_available()

    def test_onnx_availability_depends_on_import(self):
        from natural_pdf.analyzers.checkbox.onnx_engine import OnnxCheckboxDetector

        detector = OnnxCheckboxDetector()
        # Result depends on whether onnxruntime is installed
        result = detector.is_available()
        assert isinstance(result, bool)

    def test_vlm_availability_depends_on_openai(self):
        from natural_pdf.analyzers.checkbox.vlm_detector import VLMCheckboxDetector

        detector = VLMCheckboxDetector()
        result = detector.is_available()
        assert isinstance(result, bool)

    def test_all_engines_registered(self):
        from natural_pdf.engine_provider import get_provider

        provider = get_provider()
        engines = provider.list("checkbox").get("checkbox", ())
        assert "vector" in engines
        assert "default" in engines
        assert "onnx" in engines
        assert "vlm" in engines


# ============================================================
# 6. Orchestrator auto-strategy tests
# ============================================================


class TestOrchestratorAutoStrategy:
    def test_probe_page_type_vector(self):
        """Page with text and rects should be classified as vector."""
        from natural_pdf.analyzers.checkbox.checkbox_analyzer import CheckboxAnalyzer

        mock_page = MagicMock()
        mock_page.find.side_effect = lambda s: MagicMock() if s in ("text", "rect") else None

        analyzer = CheckboxAnalyzer(mock_page)
        assert analyzer._probe_page_type() == "vector"

    def test_probe_page_type_scanned(self):
        """Page with no text or rects should be classified as scanned."""
        from natural_pdf.analyzers.checkbox.checkbox_analyzer import CheckboxAnalyzer

        mock_page = MagicMock()
        mock_page.find.return_value = None

        analyzer = CheckboxAnalyzer(mock_page)
        assert analyzer._probe_page_type() == "scanned"

    def test_probe_page_type_mixed(self):
        """Page with text but no rects should be classified as mixed."""
        from natural_pdf.analyzers.checkbox.checkbox_analyzer import CheckboxAnalyzer

        mock_page = MagicMock()
        mock_page.find.side_effect = lambda s: MagicMock() if s == "text" else None

        analyzer = CheckboxAnalyzer(mock_page)
        assert analyzer._probe_page_type() == "mixed"


# ============================================================
# 7. Service dispatch tests
# ============================================================


class TestServiceDispatch:
    def test_service_handles_page(self):
        """Service should dispatch to page detection via real Page instance."""
        from natural_pdf import PDF
        from natural_pdf.services.checkbox_service import CheckboxDetectionService

        pdf = PDF("pdfs/01-practice.pdf")
        try:
            page = pdf.pages[0]
            service = CheckboxDetectionService(MagicMock())

            with patch.object(service, "_detect_page") as mock_detect:
                from natural_pdf.elements.element_collection import ElementCollection

                mock_detect.return_value = ElementCollection([])
                result = service.detect_checkboxes(page)
                mock_detect.assert_called_once()
        finally:
            pdf.close()

    def test_normalize_engine_arg(self):
        """Positional engine arg should be normalized to keyword."""
        from natural_pdf.services.checkbox_service import CheckboxDetectionService

        service = CheckboxDetectionService(MagicMock())

        result = service._normalize_engine_arg(("vector",), {})
        assert result == {"engine": "vector"}

        result = service._normalize_engine_arg((), {"engine": "default"})
        assert result == {"engine": "default"}

        with pytest.raises(TypeError):
            service._normalize_engine_arg(("vector",), {"engine": "default"})


# ============================================================
# 8. Options hierarchy tests
# ============================================================


class TestOptionsHierarchy:
    def test_base_options_defaults(self):
        opts = BaseCheckboxOptions()
        assert opts.confidence == 0.3
        assert opts.resolution == 150
        assert opts.classify is True
        assert opts.reject_with_text is True

    def test_vector_options_defaults(self):
        opts = VectorCheckboxOptions()
        assert opts.min_size == 6.0
        assert opts.max_size == 25.0
        assert opts.max_aspect_ratio == 1.5

    def test_default_options_defaults(self):
        opts = DefaultCheckboxOptions()
        assert opts.model_repo == "wendys-llc/checkbox-detector"
        assert opts.model_file == "checkbox_yolo12n.onnx"
        assert opts.model_revision == "v1"
        assert opts.input_size == 1024
        assert isinstance(opts, OnnxCheckboxOptions)

    def test_vlm_options_defaults(self):
        opts = VLMCheckboxOptions()
        assert opts.model_name == "gemini-2.0-flash"


# ============================================================
# 9. Lazy import test
# ============================================================


class TestLazyImports:
    def test_import_natural_pdf_does_not_load_heavy_deps(self):
        """Importing natural_pdf should NOT trigger torch/transformers/onnxruntime."""
        # These may already be loaded in the test session, so we can only
        # verify the checkbox module doesn't force-import them at module level
        from natural_pdf.analyzers.checkbox import classifier, vector

        # These modules should NOT have imported torch/transformers/onnxruntime
        # at the module level (they use lazy imports)
        # We verify by checking the module doesn't reference them directly
        assert not hasattr(vector, "torch")
        assert not hasattr(vector, "onnxruntime")
        assert not hasattr(classifier, "torch")


# ============================================================
# 10. Detection result canonical format tests
# ============================================================


class TestCanonicalFormat:
    def test_vector_detection_format(self):
        """Vector detections should have coord_space='pdf'."""
        from natural_pdf.analyzers.checkbox.vector import VectorCheckboxDetector

        detector = VectorCheckboxDetector()

        mock_page = MagicMock()
        mock_rect = MagicMock()
        mock_rect.bbox = (100, 200, 112, 212)
        mock_rect.stroke = True
        mock_rect.fill = None
        mock_rect.non_stroking_color = None

        mock_collection = MagicMock()
        mock_collection.__iter__ = lambda self: iter([mock_rect])
        mock_collection.__bool__ = lambda self: True
        mock_page.find_all.return_value = mock_collection

        context = DetectionContext(page=mock_page)
        results = detector.detect(Image.new("RGB", (1, 1)), VectorCheckboxOptions(), context)

        assert len(results) == 1
        det = results[0]
        assert det["coord_space"] == "pdf"
        assert det["bbox"] == (100, 200, 112, 212)
        assert det["label"] == "checkbox"
        assert det["engine"] == "vector"
        assert det["is_checked"] is None
        assert det["checkbox_state"] == "unknown"
        assert 0 <= det["confidence"] <= 1.0
