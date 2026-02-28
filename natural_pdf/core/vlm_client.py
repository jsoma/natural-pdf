"""Shared VLM inference client for document conversion and OCR.

Provides ``generate(image, prompt, *, model, client, max_new_tokens) -> str``
that handles both remote (OpenAI-compatible) and local (HF transformers) inference.

Also provides module-level default client/model management via
``set_default_client()`` / ``get_default_client()``.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from typing import Any, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

_default_client: Optional[Any] = None
_default_model: Optional[str] = None
_defaults_lock = threading.Lock()


def set_default_client(client: Any, *, model: Optional[str] = None) -> None:
    """Set a default OpenAI-compatible client (and optionally model) for VLM calls.

    Args:
        client: An OpenAI-compatible client object.
        model: Optional model name to use with the client.
    """
    global _default_client, _default_model
    with _defaults_lock:
        _default_client = client
        if model is not None:
            _default_model = model


def get_default_client() -> Tuple[Optional[Any], Optional[str]]:
    """Return the current ``(client, model)`` defaults."""
    with _defaults_lock:
        return _default_client, _default_model


# ---------------------------------------------------------------------------
# Local HF adapter cache (reuses pattern from vlm_adapter.py)
# ---------------------------------------------------------------------------

_local_cache: dict[str, "_LocalVLMAdapter"] = {}
_cache_lock = threading.Lock()


def _get_local_adapter(model_name: str) -> "_LocalVLMAdapter":
    with _cache_lock:
        if model_name not in _local_cache:
            _local_cache[model_name] = _LocalVLMAdapter(model_name)
        return _local_cache[model_name]


class _LocalVLMAdapter:
    """Thin wrapper around HF transformers for local VLM inference."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._processor: Any = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._do_load()

    def _do_load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor

            try:
                from transformers import AutoModelForImageTextToText as AutoVLM
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoVLM
        except ImportError as exc:
            raise RuntimeError(
                "Local VLM inference requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            ) from exc

        logger.info("Loading VLM model %r (this may take a moment)…", self.model_name)

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
            device_map = "auto"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float32
            device_map = None
        else:
            device = "cpu"
            dtype = torch.float32
            device_map = None

        processor_kwargs: dict[str, Any] = {}
        from natural_pdf.core.vlm_prompts import detect_model_family

        if detect_model_family(self.model_name) in ("qwen_vl", "gutenocr"):
            # Qwen-VL defaults have no real pixel cap (longest_edge=16M),
            # which causes OOM on even modest document images.  Use the
            # values recommended in the Qwen-VL documentation.
            processor_kwargs["min_pixels"] = 256 * 28 * 28  # ~200k
            processor_kwargs["max_pixels"] = 1280 * 28 * 28  # ~1M

        self._processor = AutoProcessor.from_pretrained(self.model_name, **processor_kwargs)
        self._model = AutoVLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if device_map is None:
            self._model = self._model.to(device)
        self._model.eval()
        logger.info("VLM model %r loaded on %s.", self.model_name, device)

    def generate(self, image: Image.Image, prompt: str, *, max_new_tokens: int = 4096) -> str:
        import torch

        self._ensure_loaded()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(
            self._model.device
        )

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = output_ids[0][inputs.input_ids.shape[1] :]
        return self._processor.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Image encoding helper for remote API calls
# ---------------------------------------------------------------------------


def _encode_image_base64(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 JPEG data URI for OpenAI-compatible APIs.

    Uses JPEG (quality 90) by default — much smaller than PNG and perfectly
    adequate for VLM inference on document images.
    """
    buf = io.BytesIO()
    # Convert RGBA→RGB if needed (JPEG doesn't support alpha)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
    image.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate(
    image: Image.Image,
    prompt: str,
    *,
    model: Optional[str] = None,
    client: Optional[Any] = None,
    max_new_tokens: int = 4096,
) -> str:
    """Run VLM inference and return the raw text response.

    Args:
        image: A PIL Image of the page/region.
        prompt: The text prompt to send to the model.
        model: HuggingFace model ID (local) or model name (remote).
        client: An OpenAI-compatible client for remote inference.
            If ``None``, falls back to module-level defaults set via
            :func:`set_default_client`, then to local HF inference.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The model's text response as a plain string.
    """
    # Resolve client/model from defaults if not explicitly provided
    default_client, default_model = get_default_client()
    effective_client = client or default_client
    effective_model = model or default_model

    if effective_client is not None:
        return _generate_remote(
            image,
            prompt,
            client=effective_client,
            model=effective_model,
            max_new_tokens=max_new_tokens,
        )

    if effective_model is not None:
        return _generate_local(image, prompt, model=effective_model, max_new_tokens=max_new_tokens)

    raise ValueError(
        "No VLM model or client configured. Either pass model= and/or client= "
        "explicitly, or call set_default_client() first."
    )


def _generate_remote(
    image: Image.Image,
    prompt: str,
    *,
    client: Any,
    model: Optional[str],
    max_new_tokens: int,
) -> str:
    """Use an OpenAI-compatible client for inference."""
    if model is None:
        raise ValueError(
            "model= is required when using a remote client. "
            "Pass it directly or via set_default_client(client, model=...)."
        )

    data_uri = _encode_image_base64(image)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=max_new_tokens,
    )

    # Validate response shape — some clients/models return None or empty choices
    if not getattr(response, "choices", None):
        raise RuntimeError(f"VLM remote call returned no choices (model={model!r}).")
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError(
            f"VLM remote call returned None content (model={model!r}). "
            "The model may have refused the request or hit a filter."
        )
    return content


def _generate_local(
    image: Image.Image,
    prompt: str,
    *,
    model: str,
    max_new_tokens: int,
) -> str:
    """Use a local HF transformers model for inference."""
    adapter = _get_local_adapter(model)
    return adapter.generate(image, prompt, max_new_tokens=max_new_tokens)
