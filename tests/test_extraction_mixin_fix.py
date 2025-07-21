#!/usr/bin/env python3
"""Test for extraction mixin fix - ensure it works without to_image method."""

from typing import Optional
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from natural_pdf.extraction.mixin import ExtractionMixin
from natural_pdf.extraction.result import StructuredDataResult


class MockPage(ExtractionMixin):
    """Mock page that has extract_text but not to_image."""

    def __init__(self):
        self.analyses = {}
        self._text_content = "Sample text content for testing"

    def extract_text(self, layout=True, **kwargs):
        """Return mock text content."""
        if layout:
            return f"   {self._text_content}   "  # With layout spacing
        return self._text_content

    def render(self, resolution=72, **kwargs):
        """Return mock image."""
        mock_image = Mock()
        mock_image.size = (612, 792)
        return mock_image

    # Removed get_manager method - will be set on pdf mock instead

    @property
    def pdf(self):
        """Return mock PDF with manager."""
        if not hasattr(self, "_pdf"):
            self._pdf = Mock()
            # get_manager will be set in the test
        return self._pdf


class ExtractionTestData(BaseModel):
    field1: Optional[str] = None
    field2: Optional[str] = None


def test_extraction_without_to_image():
    """Test that text extraction works without to_image method."""
    page = MockPage()

    # Verify page doesn't have to_image (migrated to render)
    assert not hasattr(page, "to_image"), "Mock page should not have to_image (migrated to render)"

    # Test _get_extraction_content for text
    content = page._get_extraction_content(using="text")
    assert content is not None, "Should return content for text extraction"
    assert "Sample text content" in content

    # Test _get_extraction_content for vision
    content = page._get_extraction_content(using="vision")
    assert content is not None, "Should return content for vision extraction"


def test_extraction_with_mock_client():
    """Test full extraction flow without to_image."""
    page = MockPage()

    # Create mock client
    mock_client = Mock()
    mock_data = ExtractionTestData(field1="value1", field2="value2")
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(parsed=mock_data))]
    mock_client.beta.chat.completions.parse.return_value = mock_response

    # Set up mock manager
    mock_manager = Mock()
    mock_manager.is_available.return_value = True
    mock_manager.extract.return_value = StructuredDataResult(
        data=mock_data, success=True, error_message=None, model_used="test-model"
    )
    # Create a proper mock for get_manager
    page.pdf.get_manager = Mock(return_value=mock_manager)

    # Perform extraction
    fields = ["field1", "field2"]
    page.extract(fields, client=mock_client, model="test-model", using="text")

    # Verify results
    result = page.extracted()
    assert result.field1 == "value1"
    assert result.field2 == "value2"

    # Verify manager was called with text content
    assert mock_manager.extract.called
    call_args = mock_manager.extract.call_args[1]
    assert "Sample text content" in call_args["content"]


def test_extraction_requires_correct_method():
    """Test that extraction checks for the right method based on 'using' parameter."""
    # Page with only extract_text
    page_text = MockPage()
    # Remove render method by setting it to None instead of delattr
    page_text.render = None

    # Should work for text
    content = page_text._get_extraction_content(using="text")
    assert content is not None

    # Should fail for vision (render is None, so not callable)
    content = page_text._get_extraction_content(using="vision")
    assert content is None

    # Page with only render
    page_vision = MockPage()
    # Remove extract_text method by setting it to None
    page_vision.extract_text = None

    # Should fail for text (extract_text is None, so not callable)
    content = page_vision._get_extraction_content(using="text")
    assert content is None

    # Should work for vision
    content = page_vision._get_extraction_content(using="vision")
    assert content is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
