"""Tests for to_llm() features: element boundary separators and OCR garble rate."""

import pytest

from natural_pdf import PDF

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def practice_page():
    pdf = PDF("pdfs/01-practice.pdf")
    yield pdf.pages[0]
    pdf.close()


# ---------------------------------------------------------------------------
# Feature 1: Element boundary separators
# ---------------------------------------------------------------------------


class TestLayoutBoundarySeparators:
    """Layout preview ┃ separators between element groups."""

    SEPARATOR = "\u2503"  # ┃

    def test_boundaries_present_by_default(self, practice_page):
        """to_llm() should include ┃ separators by default."""
        output = practice_page.to_llm(detail="standard")
        assert self.SEPARATOR in output

    def test_boundaries_absent_when_disabled(self, practice_page):
        """show_boundaries=False should use classic layout (no ┃)."""
        output = practice_page.to_llm(detail="standard", show_boundaries=False)
        assert self.SEPARATOR not in output

    def test_layout_preview_section_present(self, practice_page):
        """LAYOUT PREVIEW section should appear at standard/full detail."""
        output = practice_page.to_llm(detail="standard")
        assert "LAYOUT PREVIEW" in output

    def test_brief_detail_no_preview(self, practice_page):
        """Brief detail should skip layout preview entirely."""
        output = practice_page.to_llm(detail="brief")
        assert "LAYOUT PREVIEW" not in output

    def test_separators_not_between_every_word(self, practice_page):
        """Separators should appear between groups, not every word."""
        output = practice_page.to_llm(detail="standard")
        # Extract layout preview lines
        lines = output.split("\n")
        preview_lines = []
        in_preview = False
        for line in lines:
            if "LAYOUT PREVIEW" in line:
                in_preview = True
                continue
            elif in_preview and line.strip() and not line.startswith("  "):
                break
            elif in_preview:
                preview_lines.append(line)

        separator_count = sum(line.count(self.SEPARATOR) for line in preview_lines)
        word_count = sum(len(line.split()) for line in preview_lines)

        # Should have some separators but far fewer than words
        assert separator_count > 0
        assert separator_count < word_count / 2

    def test_table_rows_have_separators(self, practice_page):
        """Table header and data rows should have column separators."""
        output = practice_page.to_llm(detail="standard")
        lines = output.split("\n")
        # Find the "Statute" line (table header)
        header_lines = [l for l in lines if "Statute" in l and self.SEPARATOR in l]
        assert len(header_lines) > 0, "Table header should have ┃ separators"

    def test_label_value_separated(self, practice_page):
        """Bold labels like 'Site:' should be separated from their values."""
        output = practice_page.to_llm(detail="standard")
        lines = output.split("\n")
        site_lines = [l for l in lines if "Site:" in l and self.SEPARATOR in l]
        assert len(site_lines) > 0, "'Site:' label should be separated from value"


# ---------------------------------------------------------------------------
# Feature 2: Garble rate
# ---------------------------------------------------------------------------


class TestGarbleRate:
    """OCR garble rate via pyspellchecker."""

    def test_no_garble_rate_for_native_text(self, practice_page):
        """Native PDF text should not show garble rate."""
        output = practice_page.to_llm(detail="standard")
        assert "garble rate" not in output.lower()

    def test_garble_rate_helper_returns_none_without_spellchecker(self, monkeypatch):
        """When pyspellchecker is not installed, helper returns None."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "spellchecker":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from natural_pdf.describe.to_llm_sections import _compute_garble_rate

        result = _compute_garble_rate([])
        assert result is None

    def test_garble_rate_clean_words(self):
        """Clean English words should have low garble rate."""
        pytest.importorskip("spellchecker")
        from natural_pdf.describe.to_llm_sections import _compute_garble_rate

        # Create mock elements with clean text
        class FakeEl:
            def __init__(self, text):
                self.text = text
                self.source = "ocr"

        elements = [
            FakeEl("The"),
            FakeEl("quick"),
            FakeEl("brown"),
            FakeEl("fox"),
            FakeEl("jumps"),
            FakeEl("over"),
            FakeEl("the"),
            FakeEl("lazy"),
            FakeEl("dog"),
        ]
        result = _compute_garble_rate(elements)
        assert result is not None
        assert result["supported"] is True
        assert result["garble_rate"] < 0.2

    def test_garble_rate_garbage_words(self):
        """Garbage OCR output should have high garble rate."""
        pytest.importorskip("spellchecker")
        from natural_pdf.describe.to_llm_sections import _compute_garble_rate

        class FakeEl:
            def __init__(self, text):
                self.text = text
                self.source = "ocr"

        elements = [
            FakeEl("xkjf"),
            FakeEl("qwrp"),
            FakeEl("bnmz"),
            FakeEl("tlkq"),
            FakeEl("zxvn"),
        ]
        result = _compute_garble_rate(elements)
        assert result is not None
        assert result["garble_rate"] > 0.5

    def test_garble_rate_skips_short_words(self):
        """Words shorter than 3 characters should be excluded."""
        pytest.importorskip("spellchecker")
        from natural_pdf.describe.to_llm_sections import _compute_garble_rate

        class FakeEl:
            def __init__(self, text):
                self.text = text
                self.source = "ocr"

        # Only short words — should result in 0 alpha words
        elements = [FakeEl("A"), FakeEl("is"), FakeEl("to"), FakeEl("an")]
        result = _compute_garble_rate(elements)
        assert result is not None
        assert result["alpha_word_count"] == 0

    def test_garble_rate_skips_allcaps(self):
        """ALL_CAPS words (likely acronyms) should be excluded."""
        pytest.importorskip("spellchecker")
        from natural_pdf.describe.to_llm_sections import _compute_garble_rate

        class FakeEl:
            def __init__(self, text):
                self.text = text
                self.source = "ocr"

        elements = [FakeEl("FBI"), FakeEl("CIA"), FakeEl("NASA"), FakeEl("hello")]
        result = _compute_garble_rate(elements)
        assert result is not None
        # Only "hello" should be counted
        assert result["alpha_word_count"] == 1

    def test_garble_rate_handles_hyphenated_words(self):
        """Hyphenated words should be split and parts checked."""
        pytest.importorskip("spellchecker")
        from natural_pdf.describe.to_llm_sections import _compute_garble_rate

        class FakeEl:
            def __init__(self, text):
                self.text = text
                self.source = "ocr"

        elements = [FakeEl("state-of-the-art"), FakeEl("well-known")]
        result = _compute_garble_rate(elements)
        assert result is not None
        # "state", "the", "art", "well", "known" are real words (>= 3 chars)
        assert result["alpha_word_count"] > 0
        assert result["garble_rate"] < 0.3

    def test_garble_rate_handles_possessives(self):
        """Possessives like Durham's should be handled."""
        pytest.importorskip("spellchecker")
        from natural_pdf.describe.to_llm_sections import _compute_garble_rate

        class FakeEl:
            def __init__(self, text):
                self.text = text
                self.source = "ocr"

        elements = [FakeEl("Durham's"), FakeEl("company's"), FakeEl("children")]
        result = _compute_garble_rate(elements)
        assert result is not None
        assert result["alpha_word_count"] > 0
