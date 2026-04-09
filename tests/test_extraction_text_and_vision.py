#!/usr/bin/env python3
"""Test extraction with both text and vision modes."""

import builtins
from typing import Optional
from unittest.mock import Mock

import pytest
from PIL import Image
from pydantic import BaseModel

from natural_pdf.core.context import PDFContext
from natural_pdf.extraction.result import StructuredDataResult
from natural_pdf.services import extraction_service
from natural_pdf.services.extraction_service import ExtractionService, ResolvedExtractionMode


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


pytestmark = [pytest.mark.qa]


def test_text_extraction(practice_pdf):
    """Test extraction using text mode."""
    page = practice_pdf.pages[0]

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
    result = page.extract(fields, client=mock_client, model="gpt-4o-mini", using="text")

    # Verify API was called with text content
    assert mock_client.beta.chat.completions.parse.called
    call_args = mock_client.beta.chat.completions.parse.call_args[1]
    messages = call_args["messages"]

    # Check that text was sent
    user_message = messages[1]
    assert isinstance(user_message["content"], str)
    assert len(user_message["content"]) > 1000  # Should have substantial text

    # Get results via return value
    assert isinstance(result, StructuredDataResult)
    assert result.site == "Durham's Meatpacking Chicago, Ill."
    assert result.date == "February 3, 1905"
    assert result.violation_count == "7"


def test_vision_extraction(practice_pdf):
    """Test extraction using vision mode."""
    page = practice_pdf.pages[0]

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
    result = page.extract(
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
    assert isinstance(result, StructuredDataResult)
    assert result.site == "Vision: Durham's Meatpacking"
    assert result.violation_count == "Vision: 7"


def test_vision_requires_client_or_vlm_engine(practice_pdf):
    page = practice_pdf.pages[0]

    with pytest.raises(ValueError, match="using='vision' requires either a client"):
        page.extract(["site"], using="vision")


def test_engine_vlm_forces_vision_mode():
    mode = ExtractionService._resolve_mode(using="text", engine="vlm", client=None)

    assert mode == ResolvedExtractionMode(using="vision", engine="vlm")


def test_default_mode_prefers_llm_when_client_is_provided():
    mode = ExtractionService._resolve_mode(using="text", engine=None, client=Mock())

    assert mode == ResolvedExtractionMode(using="text", engine="llm")


def test_default_mode_prefers_doc_qa_without_client():
    mode = ExtractionService._resolve_mode(using="text", engine=None, client=None)

    assert mode == ResolvedExtractionMode(using="text", engine="doc_qa")


def test_docqa_dependency_error_mentions_core_complete_install(practice_pdf, monkeypatch):
    service = ExtractionService(PDFContext.with_defaults())
    page = practice_pdf.pages[0]
    original_import = builtins.__import__

    def fail_docqa_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "natural_pdf.qa.document_qa":
            raise ImportError("doc_qa missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_docqa_import)

    with pytest.raises(RuntimeError, match=r"natural-pdf\[all\]"):
        service._perform_docqa_extraction(
            host=page,
            schema=InspectionData,
            analysis_key="structured",
            model=None,
            overwrite=True,
        )


def test_vlm_dependency_error_mentions_core_complete_install(monkeypatch):
    service = ExtractionService(PDFContext.with_defaults())
    host = _MockExtractionHost()
    original_import = builtins.__import__

    def fail_vlm_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "natural_pdf.extraction.vlm_adapter":
            raise ImportError("vlm missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_vlm_import)

    with pytest.raises(RuntimeError, match=r"natural-pdf\[all\]"):
        service._perform_vlm_extraction(
            host=host,
            schema=InspectionData,
            analysis_key="structured",
            prompt=None,
            model=None,
        )


def test_pdf_vision_extraction_rejects_multi_page_documents(pdf_factory):
    pdf = pdf_factory("pdfs/pdf_sample_land_registry_japan.PDF")

    with pytest.raises(NotImplementedError, match="does not support multi-page PDFs yet"):
        pdf.extract(["site"], using="vision", client=Mock())


def test_vision_extraction_with_custom_resolution(practice_pdf):
    """Test vision extraction with custom resolution."""
    page = practice_pdf.pages[0]

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

    image_url = None
    for part in messages[1]["content"]:
        if part.get("type") == "image_url":
            image_url = part["image_url"]["url"]
            break

    assert image_url is not None
    base64_data = image_url.split(",")[1]
    assert len(base64_data) > 50000, "High resolution image should be large"


def test_extraction_without_render_method_fails_for_vision():
    """Test that vision extraction fails gracefully without render method."""

    service = ExtractionService(PDFContext.with_defaults())

    class PageWithoutRender:
        def __init__(self):
            self.analyses = {}

        def extract_text(self, **kwargs):
            return "Some text"

    page = PageWithoutRender()

    content = service._default_extraction_content(page, using="text")
    assert content == "Some text"

    content = service._default_extraction_content(page, using="vision")
    assert content is None


def test_api_error_propagates(practice_pdf):
    """Test that API errors are properly propagated."""
    page = practice_pdf.pages[0]

    mock_client = Mock()
    mock_client.beta.chat.completions.parse.side_effect = Exception(
        "Error code: 401 - Invalid API key"
    )

    with pytest.raises(Exception) as exc_info:
        page.extract(
            ["site"], client=mock_client, model="test", using="text", analysis_key="error-test"
        )

    assert "Invalid API key" in str(exc_info.value)


class _MockExtractionHost:
    """Lightweight host that mimics Page/Region for extraction service tests."""

    def __init__(self):
        self.analyses = {}
        self._text_content = "Sample text content for testing"

    def extract_text(self, layout=True, **kwargs):
        return self._text_content if not layout else f"   {self._text_content}   "

    def render(self, resolution=72, **kwargs):
        mock_image = Mock()
        mock_image.size = (612, 792)
        return mock_image

    @property
    def pdf(self):
        if not hasattr(self, "_pdf"):
            self._pdf = Mock()
        return self._pdf


def test_extraction_service_default_content_helpers():
    """Ensure the extraction service content helpers handle text and vision safely."""

    service = ExtractionService(PDFContext.with_defaults())
    host = _MockExtractionHost()

    text_content = service._default_extraction_content(host, using="text")
    assert "Sample text content" in text_content

    vision_content = service._default_extraction_content(host, using="vision")
    assert vision_content is not None

    # Remove extract_text to force failure
    host.extract_text = None
    assert service._default_extraction_content(host, using="text") is None

    # Remove render to force failure
    host.render = None
    assert service._default_extraction_content(host, using="vision") is None


def test_extraction_service_with_mock_client(monkeypatch):
    """Ensure ExtractionService integrates with structured data managers."""

    service = ExtractionService(PDFContext.with_defaults())
    host = _MockExtractionHost()

    mock_client = create_mock_client(
        InspectionData(site="value1", date="value2", violation_count="value3")
    )
    mock_result = StructuredDataResult(
        data=InspectionData(site="value1", date="value2", violation_count="value3"),
        success=True,
        error_message=None,
        model_used="test-model",
    )
    monkeypatch.setattr(extraction_service, "structured_data_is_available", lambda: True)
    mock_extract = Mock(return_value=mock_result)
    monkeypatch.setattr(extraction_service, "extract_structured_data", mock_extract)

    result = service.extract(
        host,
        schema=["site", "violation_count"],
        client=mock_client,
        model="test-model",
        using="text",
    )

    # The monkeypatched mock stores our mock_result, but the service
    # returns what's in host.analyses — which is the mock_result
    assert result.data.site == "value1"
    assert result.data.violation_count == "value3"

    mock_extract.assert_called_once()
