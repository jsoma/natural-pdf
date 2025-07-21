#!/usr/bin/env python3
"""Test to ensure extraction works with the given code pattern."""

from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from natural_pdf import PDF
from natural_pdf.extraction.result import StructuredDataResult


class ExtractedData(BaseModel):
    """Schema matching the fields requested."""

    site: Optional[str] = None
    date: Optional[str] = None
    violation_count: Optional[str] = None
    inspection_service: Optional[str] = None
    summary: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None


def test_extraction_with_mock_client():
    """Test the exact code pattern from the user with a mocked client."""
    # Load the PDF
    pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/practice.pdf")
    page = pdf.pages[0]

    # Verify that extract_text returns content
    text_content = page.extract_text()
    assert text_content, "Page should have text content"
    print(f"Text content length: {len(text_content)}")
    print(f"First 200 chars: {text_content[:200]}")

    # Create a mock client that returns structured data
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()

    # Create mock parsed data
    mock_parsed_data = ExtractedData(
        site="123 Main St",
        date="2024-01-15",
        violation_count="3",
        inspection_service="Health Department",
        summary="Minor violations found",
        city="New York",
        state="NY",
    )

    # Set up the mock chain
    mock_message.parsed = mock_parsed_data
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.beta.chat.completions.parse.return_value = mock_response

    # Perform extraction
    fields = ["site", "date", "violation count", "inspection service", "summary", "city", "state"]
    page.extract(fields, client=mock_client, model="gpt-4.1-nano", using="text")

    # Verify the mock was called
    assert mock_client.beta.chat.completions.parse.called, "Client should be called"

    # Get the extracted data
    result = page.extracted()

    # Verify the result
    assert result is not None, "Should return extracted data"
    assert hasattr(result, "site"), "Should have site field"
    assert result.site == "123 Main St", "Site should match mock data"
    assert result.date == "2024-01-15", "Date should match mock data"
    assert result.violation_count == "3", "Violation count should match"

    # Test accessing specific fields
    assert page.extracted("site") == "123 Main St"
    assert page.extracted("date") == "2024-01-15"
    assert page.extracted("violation_count") == "3"


def test_extraction_handles_api_errors():
    """Test that API errors are raised immediately (fail fast)."""
    pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/practice.pdf")
    page = pdf.pages[0]

    # Create a mock client that raises an error
    mock_client = Mock()
    mock_client.beta.chat.completions.parse.side_effect = Exception("API Error: Invalid API key")

    # API errors should be raised immediately during extract()
    fields = ["site", "date"]
    with pytest.raises(Exception) as exc_info:
        page.extract(fields, client=mock_client, model="test-model", using="text")

    assert "API Error: Invalid API key" in str(exc_info.value)


def test_extraction_with_empty_content():
    """Test extraction behavior when content is empty."""
    pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/practice.pdf")
    page = pdf.pages[0]

    # Mock extract_text to return empty
    with patch.object(page, "extract_text", return_value=""):
        mock_client = Mock()
        fields = ["site", "date"]
        page.extract(fields, client=mock_client, model="test", using="text")

        # Should return None instead of raising
        result = page.extracted()
        assert result is None, "Should return None for empty content"


if __name__ == "__main__":
    # Run tests
    print("=== Testing extraction with mock client ===")
    try:
        test_extraction_with_mock_client()
        print("✓ Mock client test passed")
    except AssertionError as e:
        print(f"✗ Mock client test failed: {e}")
    except Exception as e:
        print(f"✗ Mock client test error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    print("\n=== Testing API error handling ===")
    try:
        test_extraction_handles_api_errors()
        print("✓ API error test passed")
    except AssertionError as e:
        print(f"✗ API error test failed: {e}")
    except Exception as e:
        print(f"✗ API error test error: {type(e).__name__}: {e}")

    print("\n=== Testing empty content handling ===")
    try:
        test_extraction_with_empty_content()
        print("✓ Empty content test passed")
    except AssertionError as e:
        print(f"✗ Empty content test failed: {e}")
    except Exception as e:
        print(f"✗ Empty content test error: {type(e).__name__}: {e}")
