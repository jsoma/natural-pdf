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
from typing import Any, Dict, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

#: Default max tokens for VLM generation.  OCR grounding and layout models
#: produce verbose structured output, so this needs to be generous.
DEFAULT_VLM_MAX_TOKENS = 32768

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

_local_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()


def _is_mlx_model(model_name: str) -> bool:
    """Check if a model name refers to an MLX model."""
    name = model_name.lower()
    return name.startswith("mlx-community/") or "/mlx-" in name or name.endswith("-mlx")


def _get_local_adapter(model_name: str) -> "_LocalVLMAdapter | _MLXVLMAdapter":
    with _cache_lock:
        if model_name not in _local_cache:
            if _is_mlx_model(model_name):
                _local_cache[model_name] = _MLXVLMAdapter(model_name)
            else:
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
        import time

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

        logger.info("Loading VLM model %r on %s...", self.model_name, device)
        t0 = time.perf_counter()

        processor_kwargs: dict[str, Any] = {}
        from natural_pdf.core.vlm_prompts import detect_model_family

        family = detect_model_family(self.model_name)
        if family in ("qwen_vl", "gutenocr", "dots_mocr", "chandra"):
            # Qwen-VL defaults have no real pixel cap (longest_edge=16M),
            # which causes OOM on even modest document images.  Use the
            # values recommended in the Qwen-VL documentation.
            processor_kwargs["min_pixels"] = 256 * 28 * 28  # ~200k
            processor_kwargs["max_pixels"] = 1280 * 28 * 28  # ~1M

        trust_remote = family in ("qwen_vl", "gutenocr", "dots_mocr", "glm_ocr", "chandra")

        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=trust_remote, **processor_kwargs
        )
        self._model = AutoVLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote,
        )
        if device_map is None:
            self._model = self._model.to(device)
        self._model.eval()

        elapsed = time.perf_counter() - t0
        logger.info("VLM model %r loaded on %s in %.1fs.", self.model_name, device, elapsed)

    def generate(
        self, image: Image.Image, prompt: str, *, max_new_tokens: int = DEFAULT_VLM_MAX_TOKENS
    ) -> str:
        import time

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

        # GLM models include token_type_ids which must be removed before generate()
        inputs.pop("token_type_ids", None)

        # Log image dimension info for coordinate debugging
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            tensor_shape = pixel_values.shape
            logger.info(
                "VLM input: original image %dx%d, tensor shape %s.",
                image.size[0],
                image.size[1],
                list(tensor_shape),
            )
            # Check for image_grid_thw (Qwen-VL specific)
            grid_thw = inputs.get("image_grid_thw")
            if grid_thw is not None:
                logger.info("VLM input: image_grid_thw=%s.", grid_thw.tolist())

        logger.info(
            "VLM generation started (model=%r, max_new_tokens=%d).", self.model_name, max_new_tokens
        )
        t0 = time.perf_counter()

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated = output_ids[0][inputs.input_ids.shape[1] :]
        n_tokens = len(generated)
        elapsed = time.perf_counter() - t0
        tps = n_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            "VLM generation finished: %d tokens in %.1fs (%.1f tok/s).", n_tokens, elapsed, tps
        )

        if n_tokens >= max_new_tokens:
            logger.warning(
                "VLM output was truncated (generated %d tokens = max_new_tokens). "
                "Output may be incomplete — consider increasing max_new_tokens.",
                n_tokens,
            )

        return self._processor.decode(generated, skip_special_tokens=True)


class _MLXVLMAdapter:
    """Thin wrapper around mlx-vlm for local VLM inference on Apple Silicon."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._processor: Any = None
        self._config: Any = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._do_load()

    def _do_load(self) -> None:
        import time

        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise RuntimeError(
                "MLX VLM inference requires 'mlx-vlm'. " "Install with: pip install mlx-vlm"
            ) from exc

        logger.info("Loading MLX VLM model %r...", self.model_name)
        t0 = time.perf_counter()

        self._model, self._processor = load(self.model_name)
        self._config = load_config(self.model_name)

        elapsed = time.perf_counter() - t0
        logger.info("MLX VLM model %r loaded in %.1fs.", self.model_name, elapsed)

    def generate(
        self, image: Image.Image, prompt: str, *, max_new_tokens: int = DEFAULT_VLM_MAX_TOKENS
    ) -> str:
        import tempfile
        import time

        from mlx_vlm import generate as mlx_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        self._ensure_loaded()

        # mlx-vlm expects file paths, not PIL images — write to a temp file
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                if image.mode in ("RGBA", "LA", "P"):
                    image = image.convert("RGB")
                image.save(tmp, format="JPEG", quality=95)

            formatted_prompt = apply_chat_template(
                self._processor, self._config, prompt, num_images=1
            )

            logger.info(
                "MLX VLM generation started (model=%r, max_new_tokens=%d).",
                self.model_name,
                max_new_tokens,
            )
            t0 = time.perf_counter()

            result = mlx_generate(
                self._model,
                self._processor,
                formatted_prompt,
                [tmp_path],
                max_tokens=max_new_tokens,
                verbose=False,
            )

            # mlx-vlm returns a GenerationResult dataclass; extract the text
            output = result.text if hasattr(result, "text") else str(result)

            elapsed = time.perf_counter() - t0
            tps = getattr(result, "generation_tps", 0.0)
            n_tokens = getattr(result, "generation_tokens", 0)
            logger.info(
                "MLX VLM generation finished: %d tokens in %.1fs (%.1f tok/s).",
                n_tokens,
                elapsed,
                tps,
            )

            if n_tokens >= max_new_tokens:
                logger.warning(
                    "MLX VLM output was truncated (generated %d tokens = max_new_tokens). "
                    "Output may be incomplete — consider increasing max_new_tokens.",
                    n_tokens,
                )

            return output
        finally:
            if tmp_path:
                import os

                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


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
    max_new_tokens: int = DEFAULT_VLM_MAX_TOKENS,
    response_format: Optional[Dict[str, Any]] = None,
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
        response_format: Optional response format constraint for remote
            clients, e.g. ``{"type": "json_object"}``.

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
            response_format=response_format,
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
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """Use an OpenAI-compatible client for inference."""
    if model is None:
        raise ValueError(
            "model= is required when using a remote client. "
            "Pass it directly or via set_default_client(client, model=...)."
        )

    data_uri = _encode_image_base64(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # OpenAI-compatible clients raise errors with status_code for bad
    # request parameters (HTTP 400). We catch these to fall back on
    # parameter incompatibilities (max_tokens vs max_completion_tokens,
    # response_format support).
    def _is_bad_request(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        return status == 400

    def _is_param_error(exc: Exception, param: str) -> bool:
        """Check if a 400 error mentions a specific parameter."""
        return _is_bad_request(exc) and param in str(exc)

    def _call(*, use_response_format: bool = True, **overrides: Any) -> Any:
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        kwargs.update(overrides)
        if use_response_format and response_format is not None:
            kwargs["response_format"] = response_format
        return client.chat.completions.create(**kwargs)

    # Try max_completion_tokens first (required by newer OpenAI models),
    # then max_tokens, then max_tokens without response_format.
    try:
        response = _call(max_completion_tokens=max_new_tokens)
    except Exception as exc:
        if not _is_param_error(exc, "max_completion_tokens"):
            raise
        try:
            response = _call(max_tokens=max_new_tokens)
        except Exception as exc2:
            if not _is_param_error(exc2, "response_format"):
                raise
            response = _call(use_response_format=False, max_tokens=max_new_tokens)

    # Validate response shape — some clients/models return None or empty choices
    if not getattr(response, "choices", None):
        raise RuntimeError(f"VLM remote call returned no choices (model={model!r}).")
    choice = response.choices[0]
    content = choice.message.content
    finish_reason = getattr(choice, "finish_reason", "unknown")
    refusal = getattr(choice.message, "refusal", None)

    if finish_reason == "length":
        logger.warning(
            "VLM output was truncated (finish_reason='length', model=%s, "
            "max_new_tokens=%d). Output may be incomplete — consider increasing "
            "max_new_tokens.",
            model,
            max_new_tokens,
        )
    if refusal:
        logger.warning("VLM remote call refused (model=%s): %s", model, refusal)
    if not content:
        logger.warning(
            "VLM remote call returned empty content (model=%s, finish_reason=%s). "
            "The model may not support vision or JSON mode for this request.",
            model,
            finish_reason,
        )
    if content is None:
        raise RuntimeError(
            f"VLM remote call returned None content (model={model!r}, "
            f"finish_reason={finish_reason!r}). "
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
