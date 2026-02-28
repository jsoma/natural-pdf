#!/usr/bin/env python3
"""Test to reproduce extraction error when content appears empty."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from natural_pdf import PDF
from natural_pdf.extraction.result import StructuredDataResult

pytestmark = [pytest.mark.qa]


def test_extraction_with_apparently_empty_content():
    """Test that extraction returns a failed result when content is empty."""
    source = Path("pdfs/needs-ocr.pdf")
    if not source.exists():
        pytest.skip("Test requires pdfs/needs-ocr.pdf fixture")

    pdf = PDF(str(source))
    try:
        page = pdf.pages[0]

        text = page.extract_text()
        assert not text or not text.strip(), f"Expected empty text but got: {repr(text[:100])}"

        mock_client = Mock()
        mock_client.beta = Mock()
        mock_client.beta.chat = Mock()
        mock_client.beta.chat.completions = Mock()
        mock_client.beta.chat.completions.parse = Mock(
            side_effect=Exception("Should not reach API call")
        )

        fields = [
            "site",
            "date",
            "violation count",
            "inspection service",
            "summary",
            "city",
            "state",
        ]
        result = page.extract(fields, client=mock_client, model="gpt-4.1-nano", using="text")

        assert (
            not mock_client.beta.chat.completions.parse.called
        ), "API should not be called when content is empty"

        # Result should be a failed StructuredDataResult
        assert isinstance(result, StructuredDataResult)
        assert not result.success

        # .extracted() should return the failed StructuredDataResult
        stored = page.extracted()
        assert isinstance(stored, StructuredDataResult)
        assert not stored.success
    finally:
        pdf.close()


def test_extraction_content_method():
    """Test the _get_extraction_content method directly."""
    pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/practice.pdf")
    page = pdf.pages[0]

    content = page._get_extraction_content(using="text", layout=True)

    assert content is not None, "_get_extraction_content returned None"
    assert content, "_get_extraction_content returned empty/falsy value"
    if isinstance(content, str):
        assert content.strip(), "_get_extraction_content returned only whitespace"


if __name__ == "__main__":
    print("=== Running extraction content test ===")
    try:
        test_extraction_content_method()
        print("Content test passed")
    except AssertionError as e:
        print(f"Content test failed: {e}")
    except Exception as e:
        print(f"Content test error: {type(e).__name__}: {e}")

    print("\n=== Running extraction error test ===")
    try:
        test_extraction_with_apparently_empty_content()
        print("Extraction error test passed")
    except AssertionError as e:
        print(f"Extraction error test failed: {e}")
    except Exception as e:
        print(f"Extraction error test error: {type(e).__name__}: {e}")
