"""Contract tests for extract(engine="vlm") routing through the shared VLM client.

The old ``natural_pdf.extraction.vlm_adapter`` (HFVLMAdapter) was removed;
``ExtractionService._perform_vlm_extraction`` now goes through
``natural_pdf.core.vlm_client.generate``. These tests mock that entry point,
so no GPU or model downloads are needed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional
from unittest.mock import Mock

import pytest
from PIL import Image
from pydantic import BaseModel

import natural_pdf.core.vlm_client as vlm_client
from natural_pdf.core.context import PDFContext
from natural_pdf.extraction.result import StructuredDataResult
from natural_pdf.services.extraction_service import DEFAULT_VLM_MODEL, ExtractionService


class InvoiceSchema(BaseModel):
    total: Optional[str] = None
    date: Optional[str] = None


class _Host:
    """Minimal Page/Region stand-in with render() and analyses."""

    def __init__(self):
        self.analyses = {}
        self.render_calls = []

    def render(self, resolution=72, **kwargs):
        self.render_calls.append(resolution)
        return Image.new("RGB", (100, 100))


@pytest.fixture
def service():
    return ExtractionService(PDFContext.with_defaults())


@pytest.fixture
def generate_spy(monkeypatch):
    """Patch vlm_client.generate and record its calls."""
    calls = []

    def fake_generate(image, prompt, *, model=None, client=None, max_new_tokens=None, **kw):
        calls.append(
            {
                "image": image,
                "prompt": prompt,
                "model": model,
                "client": client,
                "max_new_tokens": max_new_tokens,
            }
        )
        return '{"total": "$100.00", "date": "2024-01-15"}'

    monkeypatch.setattr(vlm_client, "generate", fake_generate)
    # Make sure module-level defaults do not leak between tests
    monkeypatch.setattr(vlm_client, "get_default_client", lambda: (None, None))
    return calls


def test_vlm_extraction_returns_validated_model(service, generate_spy):
    host = _Host()
    service._perform_vlm_extraction(
        host=host,
        schema=InvoiceSchema,
        analysis_key="structured",
        prompt=None,
        model="test-model",
    )

    result = host.analyses["structured"]
    assert isinstance(result, StructuredDataResult)
    assert result.success is True
    assert isinstance(result.data, InvoiceSchema)
    assert result.data.total == "$100.00"
    assert result.data.date == "2024-01-15"
    assert result.model_used == "test-model"
    assert result.raw_output == '{"total": "$100.00", "date": "2024-01-15"}'


def test_prompt_contains_schema_json(service, generate_spy):
    host = _Host()
    service._perform_vlm_extraction(
        host=host,
        schema=InvoiceSchema,
        analysis_key="structured",
        prompt="Extract invoice data",
        model="test-model",
    )

    prompt = generate_spy[0]["prompt"]
    assert "Extract invoice data" in prompt
    assert "JSON" in prompt
    match = re.search(r"```json\n(.*?)```", prompt, re.DOTALL)
    assert match, "No ```json block found in prompt"
    parsed = json.loads(match.group(1))
    assert "properties" in parsed
    assert "total" in parsed["properties"]
    assert "date" in parsed["properties"]


def test_explicit_client_is_forwarded_through_extract(service, generate_spy):
    """extract(engine='vlm', client=...) must pass the client to vlm_client.generate."""
    host = _Host()
    fake_client = Mock()

    service.extract(
        host,
        schema=InvoiceSchema,
        engine="vlm",
        client=fake_client,
        model="remote-model",
    )

    assert len(generate_spy) == 1
    assert generate_spy[0]["client"] is fake_client
    assert generate_spy[0]["model"] == "remote-model"
    assert host.analyses["structured"].success is True


def test_default_client_used_when_model_and_client_are_none(service, generate_spy, monkeypatch):
    """With both model= and client= unset, the set_default_client() model is used."""
    default_client = Mock()
    monkeypatch.setattr(vlm_client, "get_default_client", lambda: (default_client, "default-model"))

    host = _Host()
    service._perform_vlm_extraction(
        host=host,
        schema=InvoiceSchema,
        analysis_key="structured",
        prompt=None,
        model=None,
        client=None,
    )

    # client stays None here: vlm_client.generate() resolves the default
    # client itself, exactly as OCR dispatch does.
    assert generate_spy[0]["client"] is None
    assert generate_spy[0]["model"] == "default-model"


def test_local_default_model_when_no_client_configured(service, generate_spy):
    """No model, no client, no default client: fall back to the local default model."""
    host = _Host()
    service._perform_vlm_extraction(
        host=host,
        schema=InvoiceSchema,
        analysis_key="structured",
        prompt=None,
        model=None,
        client=None,
    )

    assert generate_spy[0]["model"] == DEFAULT_VLM_MODEL
    assert generate_spy[0]["client"] is None


def test_generation_failure_produces_unsuccessful_result(service, monkeypatch):
    def boom(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("model exploded")

    monkeypatch.setattr(vlm_client, "generate", boom)
    monkeypatch.setattr(vlm_client, "get_default_client", lambda: (None, None))

    host = _Host()
    service._perform_vlm_extraction(
        host=host,
        schema=InvoiceSchema,
        analysis_key="structured",
        prompt=None,
        model="test-model",
    )

    result = host.analyses["structured"]
    assert result.success is False
    assert "model exploded" in result.error_message
    assert result.data is None


def test_unparseable_response_produces_unsuccessful_result(service, monkeypatch):
    monkeypatch.setattr(vlm_client, "generate", lambda *a, **kw: "not json at all")
    monkeypatch.setattr(vlm_client, "get_default_client", lambda: (None, None))

    host = _Host()
    service._perform_vlm_extraction(
        host=host,
        schema=InvoiceSchema,
        analysis_key="structured",
        prompt=None,
        model="test-model",
    )

    result = host.analyses["structured"]
    assert result.success is False
    assert result.data is None
    assert result.raw_output == "not json at all"


def test_resolution_kwarg_reaches_render(service, generate_spy):
    host = _Host()
    service._perform_vlm_extraction(
        host=host,
        schema=InvoiceSchema,
        analysis_key="structured",
        prompt=None,
        model="test-model",
        resolution=216,
    )

    assert host.render_calls == [216]


def test_host_without_render_raises(service, generate_spy):
    class NoRender:
        analyses: dict = {}

    with pytest.raises(RuntimeError, match="render"):
        service._perform_vlm_extraction(
            host=NoRender(),
            schema=InvoiceSchema,
            analysis_key="structured",
            prompt=None,
            model="test-model",
        )
