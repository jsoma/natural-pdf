"""
LLM Provider Abstraction Layer

Unified interface for calling PDF-native APIs from:
- OpenAI (GPT-4o, GPT-4V, GPT-5.2)
- Google (Gemini Pro, Gemini 2.5 Pro, Gemini 3 Pro)
- OpenRouter (Claude, Llama, etc.)

Handles:
- Native PDF input (no image rendering)
- Rate limiting with exponential backoff
- Retries on transient errors
- Cost calculation
- Response parsing
"""

import base64
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProviderResponse:
    """Response from an LLM provider."""

    raw_text: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    model: str
    provider: str


def pdf_to_base64(pdf_path: str) -> str:
    """Convert PDF file to base64."""
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_pdf_bytes(pdf_path: str) -> bytes:
    """Load PDF file as bytes."""
    with open(pdf_path, "rb") as f:
        return f.read()


def truncate_pdf_bytes(pdf_path: str, max_pages: int) -> bytes:
    """
    Load PDF and return bytes with only the first max_pages.

    Uses pikepdf to create a truncated copy in memory.
    """
    import io

    try:
        import pikepdf
    except ImportError:
        raise ImportError("pikepdf package required: pip install pikepdf")

    with pikepdf.open(pdf_path) as pdf:
        if len(pdf.pages) <= max_pages:
            # No truncation needed
            return load_pdf_bytes(pdf_path)

        # Create new PDF with only first max_pages
        with pikepdf.Pdf.new() as output:
            for i in range(min(max_pages, len(pdf.pages))):
                output.pages.append(pdf.pages[i])

            # Write to bytes buffer
            buffer = io.BytesIO()
            output.save(buffer)
            return buffer.getvalue()


def parse_csv_response(text: str) -> list[dict[str, Any]]:
    """Parse CSV from LLM response, handling multiple tables."""
    import csv
    import io

    # Remove markdown code blocks if present
    text = re.sub(r"```csv\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    if not text:
        return []

    results = []

    # Split by blank lines to handle multiple tables
    tables = re.split(r"\n\s*\n", text)

    for table_text in tables:
        table_text = table_text.strip()
        if not table_text:
            continue

        try:
            lines = table_text.split("\n")

            # Skip label lines like "TABLE 1 - Form Fields:" that don't have commas
            # Find the first line that looks like a CSV header (has commas)
            start_idx = 0
            for i, line in enumerate(lines):
                if "," in line:
                    start_idx = i
                    break

            csv_lines = lines[start_idx:]
            if len(csv_lines) < 2:
                continue

            csv_text = "\n".join(csv_lines)
            reader = csv.DictReader(io.StringIO(csv_text))

            for row in reader:
                # Clean up the row - remove None keys and empty values
                clean_row = {}
                for k, v in row.items():
                    if k is not None and v is not None:
                        clean_row[k.strip()] = v.strip() if isinstance(v, str) else v
                if clean_row:
                    results.append(clean_row)
        except Exception:
            continue

    return results


def parse_json_response(text: str) -> list[dict[str, Any]]:
    """Parse JSON from LLM response."""
    # Remove markdown code blocks if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        return []
    except json.JSONDecodeError:
        return []


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 120.0,
    ):
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name."""
        pass

    @abstractmethod
    def _call_api(
        self,
        prompt: str,
        pdf_bytes: bytes,
        model: str,
    ) -> ProviderResponse:
        """Make the actual API call with native PDF input. Implement in subclass."""
        pass

    def call(
        self,
        prompt: str,
        pdf_path: str,
        model: str,
        max_pages: int | None = None,
    ) -> ProviderResponse:
        """
        Call the LLM with a PDF and prompt.

        Args:
            prompt: The prompt to send
            pdf_path: Path to the PDF file
            model: Model name to use
            max_pages: If set, truncate PDF to first N pages before sending

        Handles retries with exponential backoff.
        """
        if max_pages:
            pdf_bytes = truncate_pdf_bytes(pdf_path, max_pages)
        else:
            pdf_bytes = load_pdf_bytes(pdf_path)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self._call_api(prompt, pdf_bytes, model)
                response.latency_ms = int((time.time() - start) * 1000)
                return response
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)

        raise RuntimeError(f"API call failed after {self.max_retries} retries: {last_error}")


class OpenAIProvider(LLMProvider):
    """OpenAI API provider with native PDF support."""

    @property
    def provider_name(self) -> str:
        return "openai"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key or os.environ.get("OPENAI_API_KEY"), **kwargs)

    def _call_api(
        self,
        prompt: str,
        pdf_bytes: bytes,
        model: str,
    ) -> ProviderResponse:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = OpenAI(api_key=self.api_key)

        # Use max_completion_tokens for newer models (gpt-5.x, o1, etc.)
        # Fall back to max_tokens for older models
        extra_params = {}
        if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
            extra_params["max_completion_tokens"] = 16384
        else:
            extra_params["max_tokens"] = 16384

        # Native PDF support via file content block
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "file",
                            "file": {
                                "filename": "document.pdf",
                                "file_data": f"data:application/pdf;base64,{pdf_base64}",
                            },
                        },
                    ],
                }
            ],
            timeout=self.timeout,
            **extra_params,
        )

        return ProviderResponse(
            raw_text=response.choices[0].message.content or "",
            tokens_input=response.usage.prompt_tokens if response.usage else 0,
            tokens_output=response.usage.completion_tokens if response.usage else 0,
            latency_ms=0,
            model=model,
            provider=self.provider_name,
        )


class GoogleProvider(LLMProvider):
    """Google Gemini API provider with native PDF support."""

    @property
    def provider_name(self) -> str:
        return "google"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            **kwargs,
        )

    def _call_api(
        self,
        prompt: str,
        pdf_bytes: bytes,
        model: str,
    ) -> ProviderResponse:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai package required: pip install google-genai")

        client = genai.Client(api_key=self.api_key)

        # Native PDF support via Part.from_bytes with PDF mime type
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=16384,
            ),
        )

        # Extract token counts
        tokens_input = 0
        tokens_output = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_input = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            tokens_output = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return ProviderResponse(
            raw_text=response.text if response.text else "",
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=0,
            model=model,
            provider=self.provider_name,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic API provider with native PDF support."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key or os.environ.get("ANTHROPIC_API_KEY"), **kwargs)

    def _call_api(
        self,
        prompt: str,
        pdf_bytes: bytes,
        model: str,
    ) -> ProviderResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)

        # Normalize model name (e.g., "anthropic/claude-sonnet-4.5" -> "claude-sonnet-4-5-20250514")
        api_model = normalize_anthropic_model(model)

        # Native PDF support via document block
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        response = client.messages.create(
            model=api_model,
            max_tokens=16384,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_base64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        # Extract text from response content blocks
        raw_text = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    raw_text = block.text
                    break

        return ProviderResponse(
            raw_text=raw_text,
            tokens_input=response.usage.input_tokens if response.usage else 0,
            tokens_output=response.usage.output_tokens if response.usage else 0,
            latency_ms=0,
            model=model,  # Keep original model name for display
            provider=self.provider_name,
        )


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider for models not available directly."""

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key or os.environ.get("OPENROUTER_API_KEY"), **kwargs)

    def _call_api(
        self,
        prompt: str,
        pdf_bytes: bytes,
        model: str,
    ) -> ProviderResponse:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        # Native PDF support via document block
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_base64,
                            },
                        },
                    ],
                }
            ],
            max_tokens=16384,
        )

        return ProviderResponse(
            raw_text=response.choices[0].message.content or "",
            tokens_input=response.usage.prompt_tokens if response.usage else 0,
            tokens_output=response.usage.completion_tokens if response.usage else 0,
            latency_ms=0,
            model=model,
            provider=self.provider_name,
        )


def normalize_anthropic_model(model: str) -> str:
    """
    Normalize Anthropic model names to API aliases.

    Uses aliases (e.g., claude-sonnet-4-5) which auto-point to latest snapshots.
    """
    # Remove provider prefix if present
    if "/" in model:
        model = model.split("/")[-1]

    # Map user-friendly names to API aliases
    # Aliases auto-update to latest snapshots
    model_map = {
        # 4.5 models
        "claude-opus-4.5": "claude-opus-4-5",
        "claude-sonnet-4.5": "claude-sonnet-4-5",
        "claude-haiku-4.5": "claude-haiku-4-5",
        # 4.0 models
        "claude-opus-4": "claude-opus-4-0",
        "claude-sonnet-4": "claude-sonnet-4-0",
    }

    return model_map.get(model, model)


def get_provider(model: str, api_key: Optional[str] = None) -> LLMProvider:
    """Get the appropriate provider for a model."""
    model_lower = model.lower()

    if (
        model_lower.startswith("gpt")
        or model_lower.startswith("o1")
        or model_lower.startswith("o3")
    ):
        return OpenAIProvider(api_key=api_key)
    elif model_lower.startswith("gemini"):
        return GoogleProvider(api_key=api_key)
    elif model_lower.startswith("claude") or "claude" in model_lower or "anthropic" in model_lower:
        # Direct Anthropic API for Claude models
        return AnthropicProvider(api_key=api_key)
    elif "/" in model:
        return OpenRouterProvider(api_key=api_key)
    else:
        # Default to OpenAI
        return OpenAIProvider(api_key=api_key)


def validate_models(models: list[str]) -> None:
    """
    Validate that all required API keys are available for the given models.

    Raises:
        ValueError: If any required API key is missing.
    """
    missing = []

    for model in models:
        provider = get_provider(model)
        if not provider.api_key:
            provider_name = provider.provider_name.upper()
            if provider_name == "GOOGLE":
                env_var = "GOOGLE_API_KEY or GEMINI_API_KEY"
            elif provider_name == "OPENROUTER":
                env_var = "OPENROUTER_API_KEY"
            else:
                env_var = f"{provider_name}_API_KEY"
            missing.append(f"  - {model} requires {env_var}")

    if missing:
        raise ValueError(
            "Missing API keys for the following models:\n"
            + "\n".join(missing)
            + "\n\nSet environment variables or configure in benchmark.json"
        )
