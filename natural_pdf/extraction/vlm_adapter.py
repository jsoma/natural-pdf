"""Local VLM adapter using HuggingFace transformers.

Provides ``engine='vlm'`` support for ``.extract()`` and ``.ask()``
without requiring an external server.  Uses ``AutoProcessor.apply_chat_template()``
to handle model-specific image tokens automatically.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Optional, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

# Module-level adapter cache keyed by model_name
_adapter_cache: dict[str, "HFVLMAdapter"] = {}
_cache_lock = threading.Lock()


def get_vlm_adapter(model_name: Optional[str] = None) -> "HFVLMAdapter":
    """Return a cached :class:`HFVLMAdapter` for the given model name."""
    name = model_name or DEFAULT_VLM_MODEL
    with _cache_lock:
        if name not in _adapter_cache:
            _adapter_cache[name] = HFVLMAdapter(model_name=name)
        return _adapter_cache[name]


class HFVLMAdapter:
    """Vision-language model adapter backed by HuggingFace transformers.

    The model and processor are loaded lazily on first call to
    :meth:`generate`.
    """

    def __init__(self, model_name: str = DEFAULT_VLM_MODEL):
        self.model_name = model_name
        self._model: Any = None
        self._processor: Any = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Lazy-load model + processor on first use."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._do_load()

    def _do_load(self) -> None:
        """Actual model loading (called under lock)."""

        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "VLM engine requires 'transformers' and 'torch'. "
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

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if device_map is None:
            self._model = self._model.to(device)
        logger.info("VLM model %r loaded on %s.", self.model_name, device)

    def _schema_prompt(self, schema: Type[BaseModel]) -> str:
        """Build a system prompt instructing the model to return JSON matching *schema*."""
        schema_json = schema.model_json_schema()
        return (
            "You are a document extraction assistant. "
            "Return ONLY valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema_json, indent=2)}\n```\n"
            "Do not include any text before or after the JSON."
        )

    def generate(
        self,
        image: Any,
        prompt: str,
        schema: Type[BaseModel],
        *,
        max_new_tokens: int = 512,
    ) -> BaseModel:
        """Run inference and parse the output into *schema*.

        Args:
            image: A PIL Image.
            prompt: The user prompt text.
            schema: Pydantic model class to validate the response against.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            A validated Pydantic model instance.
        """
        from natural_pdf.extraction.json_parser import parse_json_response

        self._ensure_loaded()

        messages = [
            {"role": "system", "content": self._schema_prompt(schema)},
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

        output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Slice off the input tokens to get only the generated part
        generated = output_ids[0][inputs.input_ids.shape[1] :]
        response = self._processor.decode(generated, skip_special_tokens=True)

        logger.debug("VLM raw response: %s", response[:500])
        return parse_json_response(response, schema)
