#!/usr/bin/env python3
"""Test extraction with both text and vision modes."""

import base64
import io
from typing import Optional
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from pydantic import BaseModel

from natural_pdf import PDF
from natural_pdf.extraction.result import StructuredDataResult


class InspectionData(BaseModel):
    """Schema for inspection data extraction."""

    site: Optional[str] = None
    date: Optional[str] = None
    violation_count: Optional[str] = None
    inspection_service: Optional[str] = None
    summary: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None


def create_mock_client(parsed_data):
    """Create a mock OpenAI client with given parsed data."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(parsed=parsed_data))]
    mock_client.beta.chat.completions.parse.return_value = mock_response
    return mock_client


def test_text_extraction():
    """Test extraction using text mode."""
    pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/practice.pdf")
    page = pdf.pages[0]

    # Verify text extraction works
    text = page.extract_text()
    assert text and len(text) > 0, "Page should have text content"

    # Create mock data
    mock_data = InspectionData(
        site="Durham's Meatpacking Chicago, Ill.", date="February 3, 1905", violation_count="7"
    )
    mock_client = create_mock_client(mock_data)

    # Perform extraction
    fields = ["site", "date", "violation count"]
    page.extract(fields, client=mock_client, model="gpt-4o-mini", using="text")

    # Verify API was called with text content
    assert mock_client.beta.chat.completions.parse.called
    call_args = mock_client.beta.chat.completions.parse.call_args[1]
    messages = call_args["messages"]

    # Check that text was sent
    user_message = messages[1]
    assert isinstance(user_message["content"], str)
    assert len(user_message["content"]) > 1000  # Should have substantial text

    # Get results
    result = page.extracted()
    assert result.site == "Durham's Meatpacking Chicago, Ill."
    assert result.date == "February 3, 1905"
    assert result.violation_count == "7"


def test_vision_extraction():
    """Test extraction using vision mode."""
    pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/practice.pdf")
    page = pdf.pages[0]

    # Verify render works
    image = page.render()
    assert isinstance(image, Image.Image), "Page should render to PIL Image"

    # Create mock data for vision
    mock_data = InspectionData(
        site="Vision: Durham's Meatpacking",
        date="Vision: February 3, 1905",
        violation_count="Vision: 7",
    )
    mock_client = create_mock_client(mock_data)

    # Perform extraction with vision
    fields = ["site", "date", "violation count"]
    page.extract(
        fields, client=mock_client, model="gpt-4o", using="vision", analysis_key="vision-test"
    )

    # Verify API was called with image
    assert mock_client.beta.chat.completions.parse.called
    call_args = mock_client.beta.chat.completions.parse.call_args[1]
    messages = call_args["messages"]

    # Check that image was sent
    user_message = messages[1]
    assert isinstance(user_message["content"], list), "Vision content should be a list"

    # Find the image part
    image_found = False
    for part in user_message["content"]:
        if part.get("type") == "image_url":
            image_found = True
            url = part["image_url"]["url"]
            assert url.startswith("data:image/png;base64,"), "Image should be base64 PNG"

    assert image_found, "Image should be included in vision request"

    # Get results
    result = page.extracted(analysis_key="vision-test")
    assert result.site == "Vision: Durham's Meatpacking"
    assert result.violation_count == "Vision: 7"


def test_vision_extraction_with_custom_resolution():
    """Test vision extraction with custom resolution."""
    pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/practice.pdf")
    page = pdf.pages[0]

    mock_data = InspectionData(site="Test")
    mock_client = create_mock_client(mock_data)

    # Test with high resolution
    page.extract(
        ["site"],
        client=mock_client,
        model="gpt-4o",
        using="vision",
        analysis_key="high-res",
        resolution=216,
    )

    # Check the image size in the API call
    call_args = mock_client.beta.chat.completions.parse.call_args[1]
    messages = call_args["messages"]

    # The resolution should affect the image size
    # We can't directly check the image size from base64, but we can verify
    # the base64 string is longer for higher resolution
    image_url = None
    for part in messages[1]["content"]:
        if part.get("type") == "image_url":
            image_url = part["image_url"]["url"]
            break

    assert image_url is not None
    base64_data = image_url.split(",")[1]

    # Higher resolution should produce larger base64 string
    # 216 DPI should be roughly 3x the data of 72 DPI
    assert len(base64_data) > 200000, "High resolution image should be large"


def test_extraction_without_render_method_fails_for_vision():
    """Test that vision extraction fails gracefully without render method."""

    class PageWithoutRender:
        """Mock page without render method."""

        def __init__(self):
            self.analyses = {}

        def extract_text(self, **kwargs):
            return "Some text"

        def _get_extraction_content(self, using="text", **kwargs):
            from natural_pdf.extraction.mixin import ExtractionMixin

            return ExtractionMixin._get_extraction_content(self, using, **kwargs)

    page = PageWithoutRender()

    # Text extraction should work
    content = page._get_extraction_content(using="text")
    assert content == "Some text"

    # Vision extraction should return None
    content = page._get_extraction_content(using="vision")
    assert content is None


def test_api_error_propagates():
    """Test that API errors are properly propagated."""
    pdf = PDF("https://github.com/jsoma/abraji25-pdfs/raw/refs/heads/main/practice.pdf")
    page = pdf.pages[0]

    # Create client that raises API error
    mock_client = Mock()
    mock_client.beta.chat.completions.parse.side_effect = Exception(
        "Error code: 401 - Invalid API key"
    )

    # Since we removed error swallowing, extract should now raise immediately
    with pytest.raises(Exception) as exc_info:
        page.extract(
            ["site"], client=mock_client, model="test", using="text", analysis_key="error-test"
        )

    assert "Invalid API key" in str(exc_info.value)


if __name__ == "__main__":
    # Run tests manually
    print("=== Testing text extraction ===")
    try:
        test_text_extraction()
        print("✓ Text extraction test passed")
    except Exception as e:
        print(f"✗ Text extraction test failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n=== Testing vision extraction ===")
    try:
        test_vision_extraction()
        print("✓ Vision extraction test passed")
    except Exception as e:
        print(f"✗ Vision extraction test failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n=== Testing custom resolution ===")
    try:
        test_vision_extraction_with_custom_resolution()
        print("✓ Custom resolution test passed")
    except Exception as e:
        print(f"✗ Custom resolution test failed: {e}")

    print("\n=== Testing without render method ===")
    try:
        test_extraction_without_render_method_fails_for_vision()
        print("✓ No render method test passed")
    except Exception as e:
        print(f"✗ No render method test failed: {e}")

    print("\n=== Testing API error propagation ===")
    try:
        test_api_error_propagates()
        print("✓ API error test passed")
    except Exception as e:
        print(f"✗ API error test failed: {e}")
