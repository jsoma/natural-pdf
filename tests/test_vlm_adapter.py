"""Tests for natural_pdf.extraction.vlm_adapter (monkeypatched, no GPU needed)."""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from natural_pdf.extraction.vlm_adapter import HFVLMAdapter, get_vlm_adapter


class InvoiceSchema(BaseModel):
    total: Optional[str] = None
    date: Optional[str] = None


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        return "fake_template_text"

    def __call__(self, text, images, return_tensors):
        mock_inputs = MagicMock()
        mock_inputs.input_ids = MagicMock()
        mock_inputs.input_ids.shape = (1, 10)
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        return mock_inputs

    def decode(self, tokens, skip_special_tokens=True):
        return '{"total": "$100.00", "date": "2024-01-15"}'


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        import torch

        return torch.tensor([[0] * 10 + [1, 2, 3]])

    def to(self, device):
        return self


@pytest.fixture
def adapter():
    adapter = HFVLMAdapter(model_name="test-model")
    adapter._processor = FakeProcessor()
    adapter._model = FakeModel()
    return adapter


def test_generate_returns_validated_model(adapter):
    from PIL import Image

    image = Image.new("RGB", (100, 100))
    result = adapter.generate(
        image=image,
        prompt="Extract invoice data",
        schema=InvoiceSchema,
    )
    assert isinstance(result, InvoiceSchema)
    assert result.total == "$100.00"
    assert result.date == "2024-01-15"


def test_schema_prompt_contains_schema(adapter):
    prompt = adapter._schema_prompt(InvoiceSchema)
    assert "total" in prompt
    assert "date" in prompt
    assert "JSON" in prompt


def test_schema_prompt_is_valid_json(adapter):
    """_schema_prompt output should contain valid JSON inside the fence."""
    import json
    import re

    prompt = adapter._schema_prompt(InvoiceSchema)
    match = re.search(r"```json\n(.*?)```", prompt, re.DOTALL)
    assert match, "No ```json block found in prompt"
    parsed = json.loads(match.group(1))
    assert "properties" in parsed
    assert "total" in parsed["properties"]


def test_get_vlm_adapter_caches():
    """get_vlm_adapter should return the same instance for the same model name."""
    from natural_pdf.extraction import vlm_adapter

    # Clear cache
    vlm_adapter._adapter_cache.clear()

    a1 = get_vlm_adapter("test-model-a")
    a2 = get_vlm_adapter("test-model-a")
    a3 = get_vlm_adapter("test-model-b")

    assert a1 is a2
    assert a1 is not a3

    vlm_adapter._adapter_cache.clear()


def test_get_vlm_adapter_thread_safe():
    """get_vlm_adapter should create only one instance even from concurrent threads."""
    import threading

    from natural_pdf.extraction import vlm_adapter

    vlm_adapter._adapter_cache.clear()

    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        results.append(get_vlm_adapter("thread-test-model"))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 2
    assert results[0] is results[1], "Both threads should get the same adapter instance"

    vlm_adapter._adapter_cache.clear()


def test_mps_device_detection(monkeypatch):
    """_ensure_loaded should use MPS when CUDA is unavailable but MPS is."""
    import sys
    import types

    import torch

    adapter = HFVLMAdapter(model_name="test-mps")

    # Mock CUDA unavailable
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Mock MPS available
    mock_mps = MagicMock()
    mock_mps.is_available.return_value = True
    monkeypatch.setattr(torch.backends, "mps", mock_mps)

    # Track what device .to() is called with
    to_calls = []
    fake_model = MagicMock()
    fake_model.to.side_effect = lambda d: (to_calls.append(d), fake_model)[1]

    fake_processor = FakeProcessor()

    # Mock the transformers module to avoid needing it installed
    mock_transformers = types.ModuleType("transformers")
    mock_transformers.AutoProcessor = MagicMock()
    mock_transformers.AutoProcessor.from_pretrained.return_value = fake_processor
    mock_transformers.AutoModelForVision2Seq = MagicMock()
    mock_transformers.AutoModelForVision2Seq.from_pretrained.return_value = fake_model
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    adapter._ensure_loaded()

    assert "mps" in to_calls, f"Expected .to('mps') call, got: {to_calls}"
    assert adapter._model is not None
