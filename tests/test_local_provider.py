"""Tests for the local model provider in the benchmark framework."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from benchmark.providers import (
    LocalProvider,
    get_provider,
    parse_local_model_spec,
    render_pdf_pages,
    validate_models,
)

# --- parse_local_model_spec ---


class TestParseLocalModelSpec:
    def test_simple_model(self):
        name, url = parse_local_model_spec("local/llama3.2-vision")
        assert name == "llama3.2-vision"
        assert url is None

    def test_model_with_tag(self):
        name, url = parse_local_model_spec("local/qwen2.5-vl:7b")
        assert name == "qwen2.5-vl:7b"
        assert url is None

    def test_model_with_inline_url(self):
        name, url = parse_local_model_spec("local/mymodel@http://localhost:8000/v1")
        assert name == "mymodel"
        assert url == "http://localhost:8000/v1"

    def test_model_with_tag_and_url(self):
        name, url = parse_local_model_spec("local/qwen2.5-vl:7b@http://gpu-box:1234/v1")
        assert name == "qwen2.5-vl:7b"
        assert url == "http://gpu-box:1234/v1"


# --- get_provider routing ---


class TestGetProviderRouting:
    def test_local_prefix_returns_local_provider(self):
        provider = get_provider("local/llama3.2-vision")
        assert isinstance(provider, LocalProvider)
        assert provider.provider_name == "local"

    def test_local_model_name_parsed(self):
        provider = get_provider("local/mymodel")
        assert isinstance(provider, LocalProvider)
        assert provider.model_name == "mymodel"

    def test_local_inline_url_parsed(self):
        provider = get_provider("local/mymodel@http://localhost:9999/v1")
        assert isinstance(provider, LocalProvider)
        assert provider.model_name == "mymodel"
        assert provider.base_url == "http://localhost:9999/v1"

    def test_local_default_base_url(self):
        """Without inline URL or env var, defaults to Ollama URL."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove LOCAL_MODEL_BASE_URL if set
            import os

            os.environ.pop("LOCAL_MODEL_BASE_URL", None)
            provider = get_provider("local/test-model")
            assert isinstance(provider, LocalProvider)
            assert provider.base_url == "http://localhost:11434/v1"

    def test_local_env_var_base_url(self):
        """LOCAL_MODEL_BASE_URL env var overrides default."""
        with patch.dict("os.environ", {"LOCAL_MODEL_BASE_URL": "http://custom:5000/v1"}):
            provider = get_provider("local/test-model")
            assert isinstance(provider, LocalProvider)
            assert provider.base_url == "http://custom:5000/v1"

    def test_local_inline_url_overrides_env(self):
        """Inline @url takes priority over env var."""
        with patch.dict("os.environ", {"LOCAL_MODEL_BASE_URL": "http://env:5000/v1"}):
            provider = get_provider("local/test-model@http://inline:9999/v1")
            assert isinstance(provider, LocalProvider)
            assert provider.base_url == "http://inline:9999/v1"

    def test_openai_model_not_local(self):
        """Non-local models should not be routed to LocalProvider."""
        provider = get_provider("gpt-5.2")
        assert not isinstance(provider, LocalProvider)


# --- validate_models ---


class TestValidateModels:
    def test_local_models_skip_key_check(self):
        """validate_models should not raise for local models (no API key needed)."""
        # This should not raise even without any API keys set
        validate_models(["local/llama3.2-vision"])

    def test_local_mixed_with_cloud_missing_key(self):
        """Local models should pass; cloud models with missing keys should fail."""
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="Missing API keys"):
                validate_models(["local/llama3.2-vision", "gpt-5.2"])


# --- render_pdf_pages ---


class TestRenderPdfPages:
    def test_renders_non_empty_images(self):
        """render_pdf_pages should return base64 images from a real PDF."""
        images = render_pdf_pages("pdfs/01-practice.pdf", max_pages=1)
        assert len(images) == 1
        # Should be valid base64
        decoded = base64.b64decode(images[0])
        # PNG magic bytes
        assert decoded[:4] == b"\x89PNG"

    def test_max_pages_limits_output(self):
        images = render_pdf_pages("pdfs/01-practice.pdf", max_pages=1)
        assert len(images) == 1


# --- LocalProvider.call with mock ---


@pytest.mark.optional_deps
class TestLocalProviderCall:
    def test_call_sends_image_content_blocks(self):
        """Verify that call() sends image_url content blocks to the API."""
        provider = LocalProvider(
            model_name="test-model",
            base_url="http://localhost:11434/v1",
        )

        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"field": "value"}'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            result = provider.call(
                prompt="Extract data",
                pdf_path="pdfs/01-practice.pdf",
                model="local/test-model",
                max_pages=1,
            )

            # Verify the API was called
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args

            # Check messages structure
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            assert len(messages) == 1
            content = messages[0]["content"]

            # First block should be text
            assert content[0]["type"] == "text"
            assert content[0]["text"] == "Extract data"

            # Second block should be image_url (rendered page)
            assert content[1]["type"] == "image_url"
            assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

            # Verify response
            assert result.raw_text == '{"field": "value"}'
            assert result.tokens_input == 100
            assert result.tokens_output == 50
            assert result.provider == "local"


# --- Cache / schema normalization ---


class TestModelNameNormalization:
    def test_cache_normalizes_at_symbol(self):
        from benchmark.cache import ResponseCache

        cache = ResponseCache("/tmp/test-cache")
        normalized = cache._normalize_model_name("local/model@http://host:1234/v1")
        assert "@" not in normalized
        assert "/" not in normalized

    def test_schema_normalizes_at_symbol(self):
        from benchmark.schemas import BenchmarkOutput

        output = BenchmarkOutput("/tmp/test-output")
        path = output.llm_result_path("test_pdf", "local/model@http://host:1234/v1")
        assert "@" not in path.name
        assert "/" not in path.stem
