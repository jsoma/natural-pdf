"""Tests for PaddleOCR language code normalization."""

from __future__ import annotations

import pytest

from natural_pdf.ocr.engine_paddle import _normalize_paddle_language


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
