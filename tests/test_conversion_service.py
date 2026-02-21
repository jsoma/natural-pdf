"""Tests for the ConversionService and to_markdown() wiring."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from natural_pdf.core.vlm_client import set_default_client

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


@pytest.fixture()
def pdf():
    import natural_pdf

    return natural_pdf.PDF("pdfs/01-practice.pdf")


@pytest.fixture()
def page(pdf):
    return pdf.pages[0]


# ---------------------------------------------------------------------------
# to_markdown fallback (no model configured)
# ---------------------------------------------------------------------------


class TestToMarkdownFallback:
    def test_falls_back_to_extract_text(self, page, caplog):
        """When no model is configured, to_markdown() should warn and fallback."""
        with caplog.at_level(logging.WARNING, logger="natural_pdf"):
            result = page.to_markdown()

        assert isinstance(result, str)
        assert len(result) > 0  # extract_text should return something
        assert "Falling back to extract_text" in caplog.text


# ---------------------------------------------------------------------------
# to_markdown with mock VLM
# ---------------------------------------------------------------------------


class TestToMarkdownWithVLM:
    def test_returns_vlm_output(self, page):
        """When a model is configured, to_markdown() should call VLM generate."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="# Title\n\nParagraph"))]
        )

        result = page.to_markdown(model="test-model", client=mock_client)
        assert result == "# Title\n\nParagraph"

    def test_custom_prompt(self, page):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="custom output"))]
        )

        result = page.to_markdown(
            model="test-model", client=mock_client, prompt="Custom instruction"
        )
        assert result == "custom output"

        # Verify the custom prompt was sent
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        text_parts = [
            c["text"] for m in messages for c in m["content"] if isinstance(c, dict) and "text" in c
        ]
        assert "Custom instruction" in text_parts


# ---------------------------------------------------------------------------
# PDF-level to_markdown
# ---------------------------------------------------------------------------


class TestPDFToMarkdown:
    @pytest.fixture()
    def multi_page_pdf(self):
        import natural_pdf

        return natural_pdf.PDF("pdfs/Atlanta_Public_Schools_GA_sample.pdf")

    def test_joins_pages(self, multi_page_pdf):
        """PDF.to_markdown() should join page results with separator."""
        mock_client = MagicMock()
        call_count = 0

        def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=f"Page {call_count}"))]
            )

        mock_client.chat.completions.create.side_effect = fake_create

        result = multi_page_pdf.to_markdown(
            pages=[0, 1], model="test-model", client=mock_client, separator="\n---\n"
        )

        assert "Page 1" in result
        assert "Page 2" in result
        assert "\n---\n" in result

    def test_pdf_to_markdown_fallback(self, pdf, caplog):
        """PDF.to_markdown() with no model should fallback on each page."""
        with caplog.at_level(logging.WARNING, logger="natural_pdf"):
            result = pdf.to_markdown(pages=[0])

        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# PageCollection to_markdown
# ---------------------------------------------------------------------------


class TestPageCollectionToMarkdown:
    def test_page_collection_to_markdown(self, pdf, caplog):
        collection = pdf.pages[:2]
        with caplog.at_level(logging.WARNING, logger="natural_pdf"):
            result = collection.to_markdown()

        assert isinstance(result, str)
        assert len(result) > 0
