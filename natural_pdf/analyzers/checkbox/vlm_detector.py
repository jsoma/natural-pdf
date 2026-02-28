"""VLM-based checkbox detector using any OpenAI-compatible Vision API.

Reuses the VLM layout infrastructure pattern from analyzers/layout/vlm.py.
Asks the VLM to find all checkboxes and report both location and state.
"""

import base64
import importlib.util
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

from PIL import Image
from pydantic import BaseModel, Field

from .base import CheckboxDetector, DetectionContext
from .checkbox_options import BaseCheckboxOptions, VLMCheckboxOptions

logger = logging.getLogger(__name__)


class _CheckboxResult(BaseModel):
    bbox: List[float] = Field(description="Bounding box [xmin, ymin, xmax, ymax] in pixels.")
    state: str = Field(description="'checked' or 'unchecked'")
    confidence: float = Field(description="Confidence score 0.0-1.0", default=0.9)


class VLMCheckboxDetector(CheckboxDetector):
    """Checkbox detection using any OpenAI-compatible Vision API."""

    def is_available(self) -> bool:
        return importlib.util.find_spec("openai") is not None

    def detect(
        self,
        image: Image.Image,
        options: BaseCheckboxOptions,
        context: Optional[DetectionContext] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(options, VLMCheckboxOptions):
            opts = VLMCheckboxOptions(
                confidence=options.confidence,
                resolution=options.resolution,
                device=options.device,
            )
        else:
            opts = options

        client, use_structured = self._resolve_client(opts)
        if client is None:
            self.logger.error(
                "No VLM client available. Set a default client or pass client= in options."
            )
            return []

        prompt = self._build_prompt(image.size, opts)
        messages = self._build_messages(image, prompt)

        model_name = opts.model_name

        if use_structured:
            return self._detect_structured(messages, client, model_name, opts)
        else:
            return self._detect_json_fallback(messages, client, model_name, opts)

    def _resolve_client(self, options: VLMCheckboxOptions):
        """Resolve client: explicit -> default -> None."""
        client = options.client

        if client is None:
            try:
                from natural_pdf.core.vlm_client import get_default_client

                client, _ = get_default_client()
            except Exception:
                pass

        if client is None:
            return None, False

        has_structured = (
            hasattr(client, "beta")
            and hasattr(getattr(client.beta, "chat", None), "completions")
            and hasattr(getattr(client.beta.chat.completions, "parse", None), "__call__")
        )
        return client, has_structured

    def _build_prompt(self, image_size, options: VLMCheckboxOptions) -> str:
        w, h = image_size

        lang_hint = ""
        if options.languages:
            from natural_pdf.core.vlm_prompts import languages_to_hint

            lang_hint = languages_to_hint(options.languages)

        return (
            f"{lang_hint} "
            f"Analyze this document image ({w}x{h} pixels). "
            f"Find ALL checkboxes, check marks, tick boxes, or selection boxes. "
            f"For each one, report its bounding box [xmin, ymin, xmax, ymax] in pixels "
            f"and whether it is 'checked' or 'unchecked'. "
            f"Return ONLY a JSON object with a 'checkboxes' key containing a list of objects, "
            f"each with 'bbox' (list of 4 floats), 'state' ('checked' or 'unchecked'), "
            f"and 'confidence' (float 0.0-1.0)."
        ).strip()

    def _build_messages(self, image: Image.Image, prompt: str) -> List[Dict]:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ],
            }
        ]

    def _detect_structured(self, messages, client, model_name, options):
        """Use structured output if client supports it."""

        class _Checkboxes(BaseModel):
            checkboxes: List[_CheckboxResult]

        try:
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=_Checkboxes,
                temperature=options.extra_args.get("temperature", 0.0),
                max_tokens=options.extra_args.get("max_tokens", 4096),
            )

            if not completion.choices:
                return []

            parsed = completion.choices[0].message.parsed.checkboxes
            return self._convert_results(parsed, options)

        except Exception as e:
            self.logger.warning("Structured output failed, falling back to JSON: %s", e)
            return self._detect_json_fallback(messages, client, model_name, options)

    def _detect_json_fallback(self, messages, client, model_name, options):
        """Parse JSON from free-text response."""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=options.extra_args.get("temperature", 0.0),
                max_tokens=options.extra_args.get("max_tokens", 4096),
            )

            if not response.choices:
                return []

            text = response.choices[0].message.content or ""
            return self._parse_json_response(text, options)

        except Exception as e:
            self.logger.error("VLM checkbox detection failed: %s", e)
            return []

    def _parse_json_response(self, text: str, options) -> List[Dict[str, Any]]:
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            self.logger.error("No JSON found in VLM response")
            return []

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse JSON: %s", e)
            return []

        items = data.get("checkboxes", [])
        if not isinstance(items, list):
            return []

        parsed = []
        for item in items:
            try:
                parsed.append(
                    _CheckboxResult(
                        bbox=item["bbox"],
                        state=item.get("state", "unknown"),
                        confidence=item.get("confidence", 0.9),
                    )
                )
            except (KeyError, TypeError, ValueError) as e:
                self.logger.warning("Skipping malformed VLM result: %s", e)
                continue

        return self._convert_results(parsed, options)

    def _convert_results(self, results: List[_CheckboxResult], options) -> List[Dict[str, Any]]:
        """Convert parsed results to canonical detection format."""
        detections = []

        for item in results:
            if len(item.bbox) != 4:
                continue

            if item.confidence < options.confidence:
                continue

            state = item.state.lower().strip()
            is_checked = state == "checked"

            detections.append(
                {
                    "bbox": tuple(item.bbox),
                    "coord_space": "image",
                    "confidence": item.confidence,
                    "label": "checkbox",
                    "engine": "vlm",
                    "is_checked": is_checked,
                    "checkbox_state": state if state in ("checked", "unchecked") else "unknown",
                }
            )

        return detections
