"""Unit tests for RapidOCR engine implementation."""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from natural_pdf.ocr.engine import TextRegion
from natural_pdf.ocr.engine_rapidocr import RapidOCREngine
from natural_pdf.ocr.ocr_options import RapidOCROptions


def rapidocr_available() -> bool:
    """Check if RapidOCR is installed."""
    return importlib.util.find_spec("rapidocr") is not None


class TestRapidOCREngineAvailability:
    """Tests for is_available method."""

    def test_is_available_returns_bool(self):
        engine = RapidOCREngine()
        result = engine.is_available()
        assert isinstance(result, bool)

    @patch("natural_pdf.ocr.engine_rapidocr.importlib.util.find_spec")
    def test_is_available_when_installed(self, mock_find_spec):
        mock_find_spec.return_value = MagicMock()
        engine = RapidOCREngine()
        assert engine.is_available() is True

    @patch("natural_pdf.ocr.engine_rapidocr.importlib.util.find_spec")
    def test_is_available_when_not_installed(self, mock_find_spec):
        mock_find_spec.return_value = None
        engine = RapidOCREngine()
        assert engine.is_available() is False


class TestRapidOCROptions:
    """Tests for RapidOCROptions dataclass."""

    def test_default_options(self):
        opts = RapidOCROptions()
        assert opts.det_model_type == "mobile"
        assert opts.rec_model_type == "mobile"
        assert opts.det_thresh == 0.3
        assert opts.use_det is True
        assert opts.use_cls is True
        assert opts.use_rec is True
        assert opts.config_path is None

    def test_custom_options(self):
        opts = RapidOCROptions(
            det_model_type="server",
            rec_model_type="server",
            det_thresh=0.5,
            use_cls=False,
        )
        assert opts.det_model_type == "server"
        assert opts.rec_model_type == "server"
        assert opts.det_thresh == 0.5
        assert opts.use_cls is False

    def test_det_thresh_validation(self):
        # Valid threshold
        opts = RapidOCROptions(det_thresh=0.5)
        assert opts.det_thresh == 0.5

        # Threshold at boundaries
        opts_zero = RapidOCROptions(det_thresh=0.0)
        assert opts_zero.det_thresh == 0.0

        opts_one = RapidOCROptions(det_thresh=1.0)
        assert opts_one.det_thresh == 1.0


class TestRapidOCREngineStandardization:
    """Tests for result standardization."""

    def test_standardize_empty_results(self):
        engine = RapidOCREngine()
        result = engine._standardize_results(None, 0.5, False)
        assert result == []

    def test_standardize_results_with_no_boxes(self):
        engine = RapidOCREngine()

        # Mock result with None boxes
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_result.txts = None
        mock_result.scores = None

        result = engine._standardize_results(mock_result, 0.5, False)
        assert result == []

    def test_standardize_results_with_data(self):
        engine = RapidOCREngine()

        # Create mock RapidOCR output
        mock_result = MagicMock()
        # Polygon format: 4 corners (top-left, top-right, bottom-right, bottom-left)
        mock_result.boxes = np.array(
            [
                [[10, 20], [100, 20], [100, 50], [10, 50]],
                [[10, 60], [150, 60], [150, 90], [10, 90]],
            ]
        )
        mock_result.txts = ("Hello", "World")
        mock_result.scores = (0.95, 0.88)

        result = engine._standardize_results(mock_result, 0.5, False)

        assert len(result) == 2
        assert all(isinstance(r, TextRegion) for r in result)

        # Check first region
        assert result[0].text == "Hello"
        assert result[0].confidence == 0.95
        assert result[0].bbox == (10.0, 20.0, 100.0, 50.0)

        # Check second region
        assert result[1].text == "World"
        assert result[1].confidence == 0.88
        assert result[1].bbox == (10.0, 60.0, 150.0, 90.0)

    def test_standardize_results_filters_low_confidence(self):
        engine = RapidOCREngine()

        mock_result = MagicMock()
        mock_result.boxes = np.array(
            [
                [[10, 20], [100, 20], [100, 50], [10, 50]],
                [[10, 60], [150, 60], [150, 90], [10, 90]],
            ]
        )
        mock_result.txts = ("High", "Low")
        mock_result.scores = (0.95, 0.3)

        # With min_confidence=0.5, should filter out the low confidence result
        result = engine._standardize_results(mock_result, 0.5, False)

        assert len(result) == 1
        assert result[0].text == "High"

    def test_standardize_results_detect_only(self):
        engine = RapidOCREngine()

        mock_result = MagicMock()
        mock_result.boxes = np.array(
            [
                [[10, 20], [100, 20], [100, 50], [10, 50]],
            ]
        )
        mock_result.txts = None
        mock_result.scores = None

        # In detect_only mode, should include all boxes regardless of score
        result = engine._standardize_results(mock_result, 0.5, detect_only=True)

        assert len(result) == 1
        assert result[0].text == ""
        assert result[0].confidence == 0.0


class TestRapidOCREnginePreprocess:
    """Tests for image preprocessing."""

    def test_preprocess_returns_same_image(self):
        engine = RapidOCREngine()
        img = Image.new("RGB", (100, 100), color="white")

        result = engine._preprocess_image(img)

        # RapidOCR accepts PIL images directly
        assert result is img


class TestRapidOCREngineInitialization:
    """Tests for engine initialization."""

    @patch("natural_pdf.ocr.engine_rapidocr.importlib.util.find_spec")
    def test_initialize_raises_when_not_available(self, mock_find_spec):
        mock_find_spec.return_value = None
        engine = RapidOCREngine()

        with pytest.raises(ImportError, match="RapidOCR library is not installed"):
            engine._initialize_model(["en"], "cpu", None)


class TestRapidOCREngineRegistry:
    """Tests for engine registration."""

    def test_engine_in_registry(self):
        from natural_pdf.ocr.ocr_provider import ENGINE_REGISTRY

        assert "rapidocr" in ENGINE_REGISTRY
        assert ENGINE_REGISTRY["rapidocr"]["options_class"] is RapidOCROptions

    def test_engine_in_factory_preference(self):
        from natural_pdf.ocr.ocr_factory import _ENGINE_PREFERENCE

        assert "rapidocr" in _ENGINE_PREFERENCE
        # Should be second in preference (after easyocr)
        assert _ENGINE_PREFERENCE.index("rapidocr") == 1


@pytest.mark.skipif(not rapidocr_available(), reason="RapidOCR not installed")
class TestRapidOCREngineIntegration:
    """Integration tests that require RapidOCR to be installed."""

    def test_initialize_with_defaults(self):
        engine = RapidOCREngine()
        engine._initialize_model(["en"], "cpu", None)
        assert engine._engine is not None

    def test_initialize_with_options(self):
        engine = RapidOCREngine()
        opts = RapidOCROptions(det_model_type="mobile")
        engine._initialize_model(["en"], "cpu", opts)
        assert engine._engine is not None

    def test_process_simple_image(self):
        engine = RapidOCREngine()
        engine._initialize_model(["en"], "cpu", None)

        # Create a simple test image with text
        img = Image.new("RGB", (200, 100), color="white")

        # Process should not raise
        result = engine._process_single_image(img, detect_only=False, options=None)
        # Result may be empty for a blank image, but should not error
        assert result is not None

    def test_full_pipeline(self):
        engine = RapidOCREngine()

        # Create a test image
        img = Image.new("RGB", (200, 100), color="white")

        # Run through full process_image pipeline
        results = engine.process_image(
            img,
            languages=["en"],
            min_confidence=0.1,
            device="cpu",
        )

        # Should return a list (possibly empty for blank image)
        assert isinstance(results, list)
