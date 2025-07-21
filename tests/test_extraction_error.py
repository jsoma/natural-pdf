#!/usr/bin/env python3
"""Test to reproduce extraction error when content appears empty."""

from unittest.mock import MagicMock, Mock

import pytest

from natural_pdf import PDF
from natural_pdf.extraction.result import StructuredDataResult


def test_extraction_with_apparently_empty_content():
    """Test that extraction returns None when content is empty."""
    # Load a PDF that needs OCR (has no text layer)
    pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/needs-ocr.pdf")
    page = pdf.pages[0]

    # Verify that extract_text() returns empty content
    text = page.extract_text()
    assert not text or not text.strip(), f"Expected empty text but got: {repr(text[:100])}"

    # Create a mock client that would fail if actually called
    mock_client = Mock()
    mock_client.beta = Mock()
    mock_client.beta.chat = Mock()
    mock_client.beta.chat.completions = Mock()
    mock_client.beta.chat.completions.parse = Mock(
        side_effect=Exception("Should not reach API call")
    )

    # Try the extraction - it should fail before reaching the API
    fields = ["site", "date", "violation count", "inspection service", "summary", "city", "state"]
    page.extract(fields, client=mock_client, model="gpt-4.1-nano", using="text")

    # The mock client should not have been called since there's no content
    assert (
        not mock_client.beta.chat.completions.parse.called
    ), "API should not be called when content is empty"

    # Check that we get None instead of raising an error
    result = page.extracted()
    assert result is None, "Should return None for failed extraction instead of raising"


def test_extraction_content_method():
    """Test the _get_extraction_content method directly."""
    pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/practice.pdf")
    page = pdf.pages[0]

    # Test getting content directly
    content = page._get_extraction_content(using="text", layout=True)

    print(f"Content type: {type(content)}")
    print(f"Content length: {len(content) if content else 'None'}")
    print(f"Content preview: {repr(content[:200]) if content else 'None'}")
    print(f"Is content None? {content is None}")
    print(f"Is content empty string? {content == ''}")
    print(f"Is content whitespace only? {isinstance(content, str) and not content.strip()}")

    # This should not be None or empty
    assert content is not None, "_get_extraction_content returned None"
    assert content, "_get_extraction_content returned empty/falsy value"
    if isinstance(content, str):
        assert content.strip(), "_get_extraction_content returned only whitespace"


if __name__ == "__main__":
    # Run the tests
    print("=== Running extraction content test ===")
    try:
        test_extraction_content_method()
        print("✓ Content test passed")
    except AssertionError as e:
        print(f"✗ Content test failed: {e}")
    except Exception as e:
        print(f"✗ Content test error: {type(e).__name__}: {e}")

    print("\n=== Running extraction error test ===")
    try:
        test_extraction_with_apparently_empty_content()
        print("✓ Extraction error test passed")
    except AssertionError as e:
        print(f"✗ Extraction error test failed: {e}")
    except Exception as e:
        print(f"✗ Extraction error test error: {type(e).__name__}: {e}")
