"""Tests for VLM language hint utilities and prompt integration."""

from __future__ import annotations

import pytest

from natural_pdf.core.vlm_prompts import _LANGUAGE_NAMES, build_ocr_prompt, languages_to_hint

# ---------------------------------------------------------------------------
# languages_to_hint
# ---------------------------------------------------------------------------


class TestLanguagesToHint:
    def test_none_returns_empty(self):
        assert languages_to_hint(None) == ""

    def test_empty_list_returns_empty(self):
        assert languages_to_hint([]) == ""

    def test_english_only_returns_empty(self):
        assert languages_to_hint(["en"]) == ""

    def test_single_language(self):
        assert languages_to_hint(["ja"]) == "The document is in Japanese."

    def test_multiple_languages(self):
        result = languages_to_hint(["ja", "en"])
        assert result == "The document is in Japanese and English."

    def test_three_languages(self):
        result = languages_to_hint(["ja", "zh", "en"])
        assert result == "The document is in Japanese, Chinese and English."

    def test_paddle_codes(self):
        assert languages_to_hint(["japan"]) == "The document is in Japanese."
        assert languages_to_hint(["korean"]) == "The document is in Korean."
        assert languages_to_hint(["ch"]) == "The document is in Chinese."

    def test_unknown_code_capitalized(self):
        result = languages_to_hint(["xyz"])
        assert result == "The document is in Xyz."

    def test_case_insensitive(self):
        assert languages_to_hint(["JA"]) == "The document is in Japanese."
        assert languages_to_hint(["JAPAN"]) == "The document is in Japanese."

    def test_no_duplicates(self):
        result = languages_to_hint(["ja", "JA", "ja"])
        assert result == "The document is in Japanese."

    def test_whitespace_stripped(self):
        assert languages_to_hint(["  ja  "]) == "The document is in Japanese."

    def test_en_with_other_languages_not_suppressed(self):
        result = languages_to_hint(["en", "ja"])
        assert "English" in result
        assert "Japanese" in result

    def test_non_string_entries_skipped(self):
        assert languages_to_hint([None, "ja"]) == "The document is in Japanese."
        assert languages_to_hint([None]) == ""
        assert languages_to_hint([42, "ko"]) == "The document is in Korean."


# ---------------------------------------------------------------------------
# build_ocr_prompt with languages
# ---------------------------------------------------------------------------


class TestBuildOCRPromptWithLanguages:
    def test_no_languages_unchanged(self):
        base = build_ocr_prompt(grounding=True, family="generic")
        with_none = build_ocr_prompt(grounding=True, family="generic", languages=None)
        assert base == with_none

    def test_english_only_unchanged(self):
        base = build_ocr_prompt(grounding=True, family="generic")
        with_en = build_ocr_prompt(grounding=True, family="generic", languages=["en"])
        assert base == with_en

    def test_japanese_prepended(self):
        prompt = build_ocr_prompt(grounding=True, family="generic", languages=["ja"])
        assert prompt.startswith("The document is in Japanese.")
        # Original prompt content still present
        assert "bbox" in prompt

    def test_works_with_qwen_family(self):
        prompt = build_ocr_prompt(grounding=True, family="qwen_vl", languages=["ko"])
        assert prompt.startswith("The document is in Korean.")
        assert "bbox_2d" in prompt

    def test_works_with_gutenocr_family(self):
        prompt = build_ocr_prompt(grounding=True, family="gutenocr", languages=["zh"])
        assert prompt.startswith("The document is in Chinese.")
        assert "TEXT2D" in prompt

    def test_works_with_non_grounding(self):
        prompt = build_ocr_prompt(grounding=False, languages=["fr"])
        assert prompt.startswith("The document is in French.")
        assert "reading order" in prompt

    def test_non_grounding_no_languages_unchanged(self):
        base = build_ocr_prompt(grounding=False)
        with_none = build_ocr_prompt(grounding=False, languages=None)
        assert base == with_none
