"""Tests for natural_pdf.core.vlm_client."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from natural_pdf.core.vlm_client import (
    _encode_image_base64,
    generate,
    get_default_client,
    set_default_client,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_defaults():
    """Reset module-level defaults before and after each test."""
    import natural_pdf.core.vlm_client as mod

    orig_client, orig_model = mod._default_client, mod._default_model
    mod._default_client = None
    mod._default_model = None
    yield
    mod._default_client = orig_client
    mod._default_model = orig_model


@pytest.fixture()
def tiny_image():
    return Image.new("RGB", (100, 50), color="white")


# ---------------------------------------------------------------------------
# set_default_client / get_default_client
# ---------------------------------------------------------------------------


class TestDefaultClient:
    def test_roundtrip(self):
        sentinel = object()
        set_default_client(sentinel, model="test-model")
        client, model = get_default_client()
        assert client is sentinel
        assert model == "test-model"

    def test_defaults_are_none(self):
        client, model = get_default_client()
        assert client is None
        assert model is None

    def test_set_without_model(self):
        sentinel = object()
        set_default_client(sentinel)
        client, model = get_default_client()
        assert client is sentinel
        assert model is None


# ---------------------------------------------------------------------------
# _encode_image_base64
# ---------------------------------------------------------------------------


class TestEncodeImage:
    def test_returns_data_uri(self, tiny_image):
        result = _encode_image_base64(tiny_image)
        assert result.startswith("data:image/jpeg;base64,")
        # base64 portion should be non-empty
        b64_part = result.split(",", 1)[1]
        assert len(b64_part) > 10


# ---------------------------------------------------------------------------
# generate — remote path
# ---------------------------------------------------------------------------


class TestGenerateRemote:
    def test_remote_with_explicit_client(self, tiny_image):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="# Hello"))]
        )

        result = generate(tiny_image, "convert", model="test-model", client=mock_client)

        assert result == "# Hello"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"

    def test_remote_uses_defaults(self, tiny_image):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="result"))]
        )
        set_default_client(mock_client, model="default-model")

        result = generate(tiny_image, "prompt")

        assert result == "result"
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "default-model"

    def test_remote_requires_model(self, tiny_image):
        mock_client = MagicMock()
        set_default_client(mock_client)  # no model

        with pytest.raises(ValueError, match="model= is required"):
            generate(tiny_image, "prompt")


# ---------------------------------------------------------------------------
# generate — local path
# ---------------------------------------------------------------------------


class TestGenerateLocal:
    def test_local_calls_adapter(self, tiny_image):
        mock_adapter = MagicMock()
        mock_adapter.generate.return_value = "local result"

        with patch("natural_pdf.core.vlm_client._get_local_adapter", return_value=mock_adapter):
            result = generate(tiny_image, "prompt", model="some/model")

        assert result == "local result"
        mock_adapter.generate.assert_called_once_with(tiny_image, "prompt", max_new_tokens=4096)


# ---------------------------------------------------------------------------
# generate — no model or client
# ---------------------------------------------------------------------------


class TestGenerateRemoteValidation:
    def test_empty_choices_raises(self, tiny_image):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[])

        with pytest.raises(RuntimeError, match="returned no choices"):
            generate(tiny_image, "prompt", model="test-model", client=mock_client)

    def test_none_content_raises(self, tiny_image):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )

        with pytest.raises(RuntimeError, match="returned None content"):
            generate(tiny_image, "prompt", model="test-model", client=mock_client)


class TestGenerateNoConfig:
    def test_raises_when_nothing_configured(self, tiny_image):
        with pytest.raises(ValueError, match="No VLM model or client configured"):
            generate(tiny_image, "prompt")
