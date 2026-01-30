# tests/test_option_validation.py
"""
Tests for option validation helpers and config validation.

Tests cover:
- Individual validation functions
- Warning emission
- Strict mode behavior
- Options class validation
"""

import logging
import os

import pytest

from natural_pdf.exceptions import InvalidOptionError
from natural_pdf.utils.option_validation import (
    is_strict_mode,
    validate_confidence,
    validate_device,
    validate_non_empty_string,
    validate_option_type,
    validate_path_exists,
    validate_positive_int,
)


class TestValidateConfidence:
    """Tests for validate_confidence function."""

    def test_valid_confidence_values(self):
        """Valid confidence values should pass through unchanged."""
        assert validate_confidence(0.0) == 0.0
        assert validate_confidence(0.5) == 0.5
        assert validate_confidence(1.0) == 1.0
        assert validate_confidence(0.75) == 0.75

    def test_none_passes_through(self):
        """None should pass through unchanged."""
        assert validate_confidence(None) is None

    def test_confidence_below_zero_clamped(self, caplog):
        """Confidence below 0.0 should be clamped to 0.0 with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_confidence(-0.5, "confidence", "TestClass")
        assert result == 0.0
        assert "[TestClass] confidence=-0.5" in caplog.text
        assert "using 0.0" in caplog.text

    def test_confidence_above_one_clamped(self, caplog):
        """Confidence above 1.0 should be clamped to 1.0 with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_confidence(1.5, "confidence", "TestClass")
        assert result == 1.0
        assert "[TestClass] confidence=1.5" in caplog.text
        assert "using 1.0" in caplog.text

    def test_integer_accepted(self):
        """Integer values should be accepted and converted to float."""
        assert validate_confidence(1) == 1.0
        assert validate_confidence(0) == 0.0


class TestValidateDevice:
    """Tests for validate_device function."""

    def test_valid_devices(self):
        """Valid device strings should pass through."""
        assert validate_device("cpu") == "cpu"
        assert validate_device("cuda") == "cuda"
        assert validate_device("mps") == "mps"
        assert validate_device("auto") == "auto"

    def test_cuda_with_device_number(self):
        """cuda:N format should be accepted."""
        assert validate_device("cuda:0") == "cuda:0"
        assert validate_device("cuda:1") == "cuda:1"

    def test_none_passes_through(self):
        """None should pass through unchanged."""
        assert validate_device(None) is None

    def test_invalid_device_falls_back(self, caplog):
        """Invalid device should fall back to 'cpu' with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_device("invalid_device", "device", "TestClass")
        assert result == "cpu"
        assert "[TestClass]" in caplog.text
        assert "using 'cpu'" in caplog.text

    def test_non_string_falls_back(self, caplog):
        """Non-string device should fall back to 'cpu' with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_device(123, "device", "TestClass")
        assert result == "cpu"
        assert "must be string" in caplog.text


class TestValidatePositiveInt:
    """Tests for validate_positive_int function."""

    def test_valid_positive_int(self):
        """Valid positive integers should pass through."""
        assert validate_positive_int(1, "batch_size") == 1
        assert validate_positive_int(10, "batch_size") == 10
        assert validate_positive_int(100, "batch_size") == 100

    def test_none_passes_through(self):
        """None should pass through unchanged."""
        assert validate_positive_int(None, "batch_size") is None

    def test_zero_falls_back(self, caplog):
        """Zero should fall back to default with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_positive_int(0, "batch_size", "TestClass", default=1)
        assert result == 1
        assert "[TestClass] batch_size=0" in caplog.text

    def test_negative_falls_back(self, caplog):
        """Negative value should fall back to default with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_positive_int(-5, "batch_size", "TestClass", default=1)
        assert result == 1


class TestValidateNonEmptyString:
    """Tests for validate_non_empty_string function."""

    def test_valid_string(self):
        """Valid non-empty strings should pass through."""
        assert validate_non_empty_string("hello", "field") == "hello"
        assert validate_non_empty_string("a", "field") == "a"

    def test_none_passes_through(self):
        """None should pass through unchanged."""
        assert validate_non_empty_string(None, "field") is None

    def test_empty_string_falls_back(self, caplog):
        """Empty string should fall back to default with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_non_empty_string("", "model_name", "TestClass", default="default")
        assert result == "default"
        assert "is empty" in caplog.text

    def test_whitespace_only_falls_back(self, caplog):
        """Whitespace-only string should fall back to default with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_non_empty_string("   ", "model_name", "TestClass", default="default")
        assert result == "default"


class TestValidatePathExists:
    """Tests for validate_path_exists function."""

    def test_existing_path_passes(self, tmp_path):
        """Existing path should pass through unchanged."""
        test_file = tmp_path / "test.txt"
        test_file.touch()
        result = validate_path_exists(str(test_file), "model_path", "TestClass")
        assert result == str(test_file)

    def test_none_passes_through(self):
        """None should pass through unchanged."""
        assert validate_path_exists(None, "model_path") is None

    def test_nonexistent_path_raises(self):
        """Non-existent path should raise InvalidOptionError."""
        with pytest.raises(InvalidOptionError) as exc_info:
            validate_path_exists("/nonexistent/path/to/file.txt", "model_path", "TestClass")
        assert "does not exist" in str(exc_info.value)


class TestValidateOptionType:
    """Tests for validate_option_type function."""

    def test_correct_type_passes(self):
        """Correct type should pass through unchanged."""
        from natural_pdf.ocr.ocr_options import EasyOCROptions

        options = EasyOCROptions()
        result, was_default = validate_option_type(options, EasyOCROptions, "TestEngine")
        assert result is options
        assert was_default is False

    def test_wrong_type_creates_default(self, caplog):
        """Wrong type should create default with warning."""
        from natural_pdf.ocr.ocr_options import EasyOCROptions, PaddleOCROptions

        options = PaddleOCROptions()
        with caplog.at_level(logging.WARNING):
            result, was_default = validate_option_type(options, EasyOCROptions, "TestEngine")
        assert isinstance(result, EasyOCROptions)
        assert was_default is True
        assert "[TestEngine]" in caplog.text
        assert "Expected EasyOCROptions" in caplog.text


class TestStrictMode:
    """Tests for strict mode behavior."""

    def test_strict_mode_disabled_by_default(self):
        """Strict mode should be disabled by default."""
        # Ensure env var is not set
        os.environ.pop("NATURAL_PDF_STRICT", None)
        assert is_strict_mode() is False

    def test_strict_mode_enabled(self):
        """Strict mode should be enabled when env var is set."""
        os.environ["NATURAL_PDF_STRICT"] = "1"
        try:
            assert is_strict_mode() is True
        finally:
            os.environ.pop("NATURAL_PDF_STRICT", None)

    def test_strict_mode_raises_on_invalid_confidence(self):
        """In strict mode, invalid confidence should raise error."""
        os.environ["NATURAL_PDF_STRICT"] = "1"
        try:
            with pytest.raises(InvalidOptionError):
                validate_confidence(1.5, "confidence", "TestClass")
        finally:
            os.environ.pop("NATURAL_PDF_STRICT", None)

    def test_strict_mode_raises_on_invalid_device(self):
        """In strict mode, invalid device should raise error."""
        os.environ["NATURAL_PDF_STRICT"] = "1"
        try:
            with pytest.raises(InvalidOptionError):
                validate_device("invalid", "device", "TestClass")
        finally:
            os.environ.pop("NATURAL_PDF_STRICT", None)


class TestOCROptionsValidation:
    """Tests for OCR options __post_init__ validation."""

    def test_easyocr_options_valid(self):
        """Valid EasyOCR options should be created without issues."""
        from natural_pdf.ocr.ocr_options import EasyOCROptions

        opts = EasyOCROptions(batch_size=4, min_size=20)
        assert opts.batch_size == 4
        assert opts.min_size == 20

    def test_easyocr_options_invalid_batch_size(self, caplog):
        """Invalid batch_size should be corrected with warning."""
        from natural_pdf.ocr.ocr_options import EasyOCROptions

        with caplog.at_level(logging.WARNING):
            opts = EasyOCROptions(batch_size=-1)
        assert opts.batch_size == 1
        assert "batch_size" in caplog.text

    def test_paddle_options_invalid_confidence(self, caplog):
        """Invalid confidence threshold should be corrected with warning."""
        from natural_pdf.ocr.ocr_options import PaddleOCROptions

        with caplog.at_level(logging.WARNING):
            opts = PaddleOCROptions(text_det_thresh=1.5)
        assert opts.text_det_thresh == 1.0

    def test_doctr_options_valid(self):
        """Valid DocTR options should be created without issues."""
        from natural_pdf.ocr.ocr_options import DoctrOCROptions

        opts = DoctrOCROptions(batch_size=2, bin_thresh=0.3)
        assert opts.batch_size == 2
        assert opts.bin_thresh == 0.3


class TestLayoutOptionsValidation:
    """Tests for layout options __post_init__ validation."""

    def test_base_layout_options_valid(self):
        """Valid base layout options should be created without issues."""
        from natural_pdf.analyzers.layout.layout_options import BaseLayoutOptions

        opts = BaseLayoutOptions(confidence=0.7, device="cuda")
        assert opts.confidence == 0.7
        assert opts.device == "cuda"

    def test_base_layout_options_invalid_confidence(self, caplog):
        """Invalid confidence should be corrected with warning."""
        from natural_pdf.analyzers.layout.layout_options import BaseLayoutOptions

        with caplog.at_level(logging.WARNING):
            opts = BaseLayoutOptions(confidence=2.0)
        assert opts.confidence == 1.0

    def test_yolo_options_valid(self):
        """Valid YOLO options should be created without issues."""
        from natural_pdf.analyzers.layout.layout_options import YOLOLayoutOptions

        opts = YOLOLayoutOptions(image_size=512, confidence=0.5)
        assert opts.image_size == 512
        assert opts.confidence == 0.5

    def test_yolo_options_invalid_image_size(self, caplog):
        """Invalid image_size should be corrected with warning."""
        from natural_pdf.analyzers.layout.layout_options import YOLOLayoutOptions

        with caplog.at_level(logging.WARNING):
            opts = YOLOLayoutOptions(image_size=-100)
        assert opts.image_size == 1024  # default

    def test_tatr_options_valid(self):
        """Valid TATR options should be created without issues."""
        from natural_pdf.analyzers.layout.layout_options import TATRLayoutOptions

        opts = TATRLayoutOptions(max_detection_size=1000, max_structure_size=1200)
        assert opts.max_detection_size == 1000
        assert opts.max_structure_size == 1200


class TestSetOptionValidation:
    """Tests for set_option validation."""

    def test_set_option_valid_confidence(self):
        """Valid confidence should be set without issue."""
        import natural_pdf as npdf

        npdf.set_option("ocr.min_confidence", 0.8)
        assert npdf.options.ocr.min_confidence == 0.8

    def test_set_option_invalid_confidence_corrected(self, caplog):
        """Invalid confidence should be corrected with warning."""
        import natural_pdf as npdf

        with caplog.at_level(logging.WARNING):
            npdf.set_option("ocr.min_confidence", 1.5)
        assert npdf.options.ocr.min_confidence == 1.0
        assert "[set_option]" in caplog.text

    def test_set_option_valid_resolution(self):
        """Valid resolution should be set without issue."""
        import natural_pdf as npdf

        npdf.set_option("image.resolution", 300)
        assert npdf.options.image.resolution == 300

    def test_set_option_invalid_resolution_corrected(self, caplog):
        """Invalid resolution should be corrected with warning."""
        import natural_pdf as npdf

        with caplog.at_level(logging.WARNING):
            npdf.set_option("image.resolution", -50)
        assert npdf.options.image.resolution >= 1

    def test_set_option_unknown_option_raises(self):
        """Unknown option should raise KeyError."""
        import natural_pdf as npdf

        with pytest.raises(KeyError):
            npdf.set_option("unknown.option", "value")
