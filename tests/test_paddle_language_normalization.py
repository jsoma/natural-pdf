"""Tests for PaddleOCR language code normalization and device translation."""

from __future__ import annotations

import logging

import pytest

from natural_pdf.ocr.engine_paddle import _normalize_paddle_language, _translate_device_for_paddle


class TestNormalizePaddleLanguage:
    def test_ja_to_japan(self):
        assert _normalize_paddle_language("ja") == "japan"

    def test_ko_to_korean(self):
        assert _normalize_paddle_language("ko") == "korean"

    def test_zh_to_ch(self):
        assert _normalize_paddle_language("zh") == "ch"

    def test_zh_cn_to_ch(self):
        assert _normalize_paddle_language("zh-cn") == "ch"

    def test_zh_hans_to_ch(self):
        assert _normalize_paddle_language("zh-hans") == "ch"

    def test_zh_tw_to_chinese_cht(self):
        assert _normalize_paddle_language("zh-tw") == "chinese_cht"

    def test_zh_hant_to_chinese_cht(self):
        assert _normalize_paddle_language("zh-hant") == "chinese_cht"

    def test_de_to_german(self):
        assert _normalize_paddle_language("de") == "german"

    def test_fr_to_french(self):
        assert _normalize_paddle_language("fr") == "french"

    def test_unknown_pass_through(self):
        assert _normalize_paddle_language("en") == "en"
        assert _normalize_paddle_language("ar") == "ar"

    def test_case_insensitive(self):
        assert _normalize_paddle_language("JA") == "japan"
        assert _normalize_paddle_language("Zh-TW") == "chinese_cht"
        assert _normalize_paddle_language("DE") == "german"

    def test_whitespace_stripped(self):
        assert _normalize_paddle_language("  ja  ") == "japan"
        assert _normalize_paddle_language(" en ") == "en"

    def test_already_paddle_codes_unchanged(self):
        assert _normalize_paddle_language("japan") == "japan"
        assert _normalize_paddle_language("korean") == "korean"
        assert _normalize_paddle_language("ch") == "ch"
        assert _normalize_paddle_language("chinese_cht") == "chinese_cht"
        assert _normalize_paddle_language("german") == "german"
        assert _normalize_paddle_language("french") == "french"


class TestTranslateDeviceForPaddle:
    """Tests for _translate_device_for_paddle helper."""

    _logger = logging.getLogger("test_paddle_device")

    def test_cuda_to_gpu0(self):
        assert _translate_device_for_paddle("cuda", self._logger) == "gpu:0"

    def test_cuda_uppercase(self):
        assert _translate_device_for_paddle("CUDA", self._logger) == "gpu:0"

    def test_cuda_with_index(self):
        assert _translate_device_for_paddle("cuda:1", self._logger) == "gpu:1"
        assert _translate_device_for_paddle("cuda:0", self._logger) == "gpu:0"

    def test_bare_gpu_gets_index(self):
        assert _translate_device_for_paddle("gpu", self._logger) == "gpu:0"

    def test_mps_falls_back_to_cpu(self):
        assert _translate_device_for_paddle("mps", self._logger) == "cpu"

    def test_cpu_passthrough(self):
        assert _translate_device_for_paddle("cpu", self._logger) == "cpu"

    def test_gpu0_passthrough(self):
        assert _translate_device_for_paddle("gpu:0", self._logger) == "gpu:0"

    def test_npu_passthrough(self):
        assert _translate_device_for_paddle("npu:0", self._logger) == "npu:0"

    def test_none_returns_none(self):
        assert _translate_device_for_paddle(None, self._logger) is None

    def test_idempotent(self):
        """Applying translation twice should give the same result."""
        result1 = _translate_device_for_paddle("cuda", self._logger)
        result2 = _translate_device_for_paddle(result1, self._logger)
        assert result1 == result2 == "gpu:0"
