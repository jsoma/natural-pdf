"""ConversionService — VLM-powered document-to-markdown conversion."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from natural_pdf.services.registry import register_delegate
from natural_pdf.utils.locks import pdf_render_lock

logger = logging.getLogger(__name__)


class ConversionService:
    """Converts page images to Markdown (or plain text) via VLM inference."""

    def __init__(self, context: Any) -> None:
        self._context = context

    @staticmethod
    def _render_page_image(
        host: Any, resolution: int, render_kwargs: Optional[Dict[str, Any]]
    ) -> Any:
        """Render the host (Page/Region) to a PIL Image."""
        render_fn = getattr(host, "render", None)
        if not callable(render_fn):
            raise AttributeError("Host object does not support render() for conversion.")
        kwargs = dict(render_kwargs or {})
        with pdf_render_lock:
            image = render_fn(resolution=resolution, **kwargs)
        return image

    @register_delegate("conversion", "to_markdown")
    def to_markdown(
        self,
        host: Any,
        *,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        resolution: int = 144,
        render_kwargs: Optional[Dict[str, Any]] = None,
        max_new_tokens: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Convert a page/region to Markdown using a VLM.

        If no *model* and no *client* (and no defaults), falls back to
        ``host.extract_text()`` with a warning.

        Recommended models (olmOCR-bench scores):

        - **Local:** ``"rednote-hilab/dots.mocr"`` (83.9, 3B),
          ``"lightonai/LightOnOCR-2-1B"`` (83.2, 1B — ``pip install transformers>=5.0.0``),
          ``"Qwen/Qwen2.5-VL-7B-Instruct"`` (65.5, 7B).
        - **Remote:** ``"gpt-4o"`` (69.9), ``"gemini-2.0-flash"`` (63.8).

        Args:
            host: Page or Region object.
            model: HuggingFace model ID or remote model name.
            client: OpenAI-compatible client for remote inference.
            resolution: DPI for rendering the page image.
            render_kwargs: Extra keyword arguments for ``host.render()``.
            max_new_tokens: Maximum tokens for the VLM to generate.
            prompt: Custom prompt override (default uses the built-in conversion prompt).

        Returns:
            Markdown string.
        """
        from natural_pdf.core.vlm_client import generate, get_default_client

        # Check if any VLM is available (explicit args or defaults)
        default_client, default_model = get_default_client()
        effective_client = client or default_client
        effective_model = model or default_model

        if effective_client is None and effective_model is None:
            logger.warning(
                "to_markdown() called without a VLM model or client configured. "
                "Falling back to extract_text(). Set a model via "
                "model= parameter or natural_pdf.set_default_client()."
            )
            extract_fn = getattr(host, "extract_text", None)
            if callable(extract_fn):
                return extract_fn()
            return ""

        from natural_pdf.core.vlm_prompts import build_conversion_prompt

        effective_prompt = prompt or build_conversion_prompt()

        image = self._render_page_image(host, resolution, render_kwargs)

        gen_kwargs: Dict[str, Any] = dict(model=model, client=client)
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens

        return generate(image, effective_prompt, **gen_kwargs)
