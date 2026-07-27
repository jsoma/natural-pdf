# layout_detector_vlm.py
import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

from PIL import Image
from pydantic import BaseModel, Field

from natural_pdf.core.vlm_prompts import languages_to_hint
from natural_pdf.utils.option_validation import validate_option_type

from .base import LayoutDetector
from .layout_options import BaseLayoutOptions, VLMLayoutOptions

logger = logging.getLogger(__name__)


# Define Pydantic model for the expected output structure
# This is used by the openai library's `response_format`
class DetectedRegion(BaseModel):
    label: str = Field(description="The identified class name.")
    bbox: List[float] = Field(description="Bounding box coordinates [xmin, ymin, xmax, ymax].")
    confidence: float = Field(description="Confidence score [0.0, 1.0].")


class VLMLayoutDetector(LayoutDetector):
    """
    VLMLayoutDetector: Layout analysis using any OpenAI-compatible Vision API.

    Supports remote clients (OpenAI, Gemini, etc.) via structured output,
    local HF models via vlm_client fallback, and module-level default clients.
    """

    # Dynamic classes — pass through as-is
    TYPE_MAP: Dict[str, str] = {}

    def __init__(self):
        super().__init__()
        self.supported_classes = set()  # Indicate dynamic nature

    def is_available(self) -> bool:
        """
        Check if the VLM detector is available.

        Returns True if the openai library is installed (for remote clients)
        OR if transformers is installed (for local models).
        """
        import importlib.util

        has_openai = importlib.util.find_spec("openai") is not None
        has_transformers = importlib.util.find_spec("transformers") is not None
        return has_openai or has_transformers

    def _get_cache_key(self, options: BaseLayoutOptions) -> str:
        """Generate cache key based on model name."""
        options, _ = validate_option_type(options, VLMLayoutOptions, "VLMLayoutDetector")

        model_key = options.model_name
        return f"{self.__class__.__name__}_{model_key}"

    def _load_model_from_options(self, options: BaseLayoutOptions) -> str:
        """Validate options and return the model name."""
        if not isinstance(options, VLMLayoutOptions):
            raise TypeError("Incorrect options type provided for VLM model loading.")
        return options.model_name

    def _resolve_client(self, options: "VLMLayoutOptions"):
        """Resolve client using fallback chain: explicit -> default -> None.

        Returns:
            (client, use_structured) tuple. use_structured is True when the
            client supports beta.chat.completions.parse (structured output).
        """
        client = getattr(options, "client", None)

        if client is None:
            from natural_pdf.core.vlm_client import get_default_client

            default_client, _ = get_default_client()
            client = default_client

        if client is None:
            return None, False

        # Check if client supports structured output
        has_structured = (
            hasattr(client, "beta")
            and hasattr(getattr(client.beta, "chat", None), "completions")
            and hasattr(getattr(client.beta.chat.completions, "parse", None), "__call__")
        )
        return client, has_structured

    def detect(
        self, image: Image.Image, options: BaseLayoutOptions, context=None
    ) -> List[Dict[str, Any]]:
        """Detect layout elements in an image using a VLM."""
        # Ensure options are the correct type
        final_options: VLMLayoutOptions
        if isinstance(options, VLMLayoutOptions):
            final_options = options
        else:
            self.logger.warning(
                "Received BaseLayoutOptions, expected VLMLayoutOptions. Converting and using defaults."
            )
            final_options = VLMLayoutOptions(
                confidence=options.confidence,
                classes=options.classes,
                exclude_classes=options.exclude_classes,
                device=options.device,
                extra_args=options.extra_args,
            )

        model_name = self._get_model(final_options)

        if not final_options.classes:
            logger.error("VLM layout detection requires a list of classes to find.")
            return []

        client, use_structured = self._resolve_client(final_options)

        if client is not None:
            if use_structured:
                return self._detect_structured(image, final_options, client, model_name)
            else:
                return self._detect_json_fallback(image, final_options, client, model_name)
        else:
            # Fall back to local model via vlm_client
            return self._detect_local(image, final_options, model_name)

    def _build_prompt(
        self,
        width: int,
        height: int,
        classes: List[str],
        languages: Optional[List[str]] = None,
    ) -> str:
        hint = languages_to_hint(languages)
        class_list_str = ", ".join(f"`{c}`" for c in classes)
        base = (
            f"Analyze the provided image of a document page ({width}x{height}). "
            f"Identify all regions corresponding to the following types: {class_list_str}. "
            f"Return ONLY a JSON object with a 'regions' key containing a list of objects, "
            f"each with 'label' (str), 'bbox' ([xmin, ymin, xmax, ymax] floats), "
            f"and 'confidence' (float 0.0-1.0)."
        )
        return f"{hint} {base}" if hint else base

    def _build_messages(self, image: Image.Image, prompt: str) -> List[Dict]:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_url = f"data:image/png;base64,{img_base64}"

        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ]

    def _detect_structured(
        self, image: Image.Image, options: VLMLayoutOptions, client: Any, model_name: str
    ) -> List[Dict[str, Any]]:
        """Use structured output (beta.chat.completions.parse) for remote clients."""
        width, height = image.size
        prompt = self._build_prompt(
            width, height, options.classes, getattr(options, "languages", None)
        )
        messages = self._build_messages(image, prompt)

        logger.debug(
            "Running VLM detection via structured output (Model: %s). Classes: %s",
            model_name,
            options.classes,
        )

        completion_kwargs = {
            "temperature": options.extra_args.get("temperature", 0.0),
            "max_tokens": options.extra_args.get("max_tokens", 4096),
        }
        completion_kwargs = {k: v for k, v in completion_kwargs.items() if v is not None}

        class ImageContents(BaseModel):
            regions: List[DetectedRegion]

        completion: Any = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=ImageContents,
            **completion_kwargs,
        )

        if not completion.choices:
            logger.error("VLM response contained no choices.")
            return []

        parsed_results = completion.choices[0].message.parsed.regions
        if not parsed_results or not isinstance(parsed_results, list):
            logger.error("VLM response did not contain a valid list of parsed regions.")
            return []

        return self._filter_detections(parsed_results, options)

    def _detect_json_fallback(
        self, image: Image.Image, options: VLMLayoutOptions, client: Any, model_name: str
    ) -> List[Dict[str, Any]]:
        """Use plain chat.completions.create and parse JSON from response text."""
        width, height = image.size
        prompt = self._build_prompt(
            width, height, options.classes, getattr(options, "languages", None)
        )
        messages = self._build_messages(image, prompt)

        logger.debug(
            "Running VLM detection via JSON fallback (Model: %s). Classes: %s",
            model_name,
            options.classes,
        )

        completion_kwargs = {
            "temperature": options.extra_args.get("temperature", 0.0),
            "max_tokens": options.extra_args.get("max_tokens", 4096),
        }
        completion_kwargs = {k: v for k, v in completion_kwargs.items() if v is not None}

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **completion_kwargs,
        )

        if not response.choices:
            logger.error("VLM response contained no choices.")
            return []

        text = response.choices[0].message.content or ""
        return self._parse_json_response(text, options)

    def _detect_local(
        self, image: Image.Image, options: VLMLayoutOptions, model_name: str
    ) -> List[Dict[str, Any]]:
        """Use local HF model via vlm_client.generate()."""
        from natural_pdf.core.vlm_client import generate

        width, height = image.size
        prompt = self._build_prompt(
            width, height, options.classes, getattr(options, "languages", None)
        )

        logger.debug(
            "Running VLM detection via local model (Model: %s). Classes: %s",
            model_name,
            options.classes,
        )

        text = generate(image, prompt, model=model_name)
        return self._parse_json_response(text, options)

    def _parse_json_response(self, text: str, options: VLMLayoutOptions) -> List[Dict[str, Any]]:
        """Extract JSON from a free-text response and convert to detections."""
        # Try to find JSON in the response
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            logger.error("Could not find JSON in VLM response: %s", text[:200])
            return []

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from VLM response: %s", e)
            return []

        regions = data.get("regions", [])
        if not isinstance(regions, list):
            logger.error("VLM response 'regions' is not a list.")
            return []

        parsed_results = []
        for item in regions:
            try:
                parsed_results.append(
                    DetectedRegion(
                        label=item["label"],
                        bbox=item["bbox"],
                        confidence=item.get("confidence", 0.9),
                    )
                )
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("Skipping malformed region in VLM response: %s", e)
                continue

        return self._filter_detections(parsed_results, options)

    def _filter_detections(
        self, parsed_results: List[DetectedRegion], options: VLMLayoutOptions
    ) -> List[Dict[str, Any]]:
        """Apply class and confidence filtering to parsed results."""
        detections = []

        normalized_classes_req = {self._normalize_class_name(c) for c in options.classes}
        normalized_classes_excl = (
            {self._normalize_class_name(c) for c in options.exclude_classes}
            if options.exclude_classes
            else set()
        )

        for item in parsed_results:
            label = item.label
            bbox_raw = item.bbox
            confidence_score = item.confidence

            if len(bbox_raw) != 4:
                logger.warning(
                    "Skipping VLM region '%s' with invalid bbox length %d (expected 4).",
                    label,
                    len(bbox_raw),
                )
                continue

            xmin, ymin, xmax, ymax = tuple(bbox_raw)
            normalized_class = self._normalize_class_name(label)

            if normalized_class not in normalized_classes_req:
                logger.warning(
                    "VLM returned unexpected class '%s' despite prompt. Skipping.", label
                )
                continue

            if normalized_class in normalized_classes_excl:
                continue

            if confidence_score < options.confidence:
                continue

            detections.append(
                {
                    "bbox": (xmin, ymin, xmax, ymax),
                    "class": label,
                    "confidence": confidence_score,
                    "normalized_class": normalized_class,
                    "source": "layout",
                    "model": "vlm",
                }
            )

        self.logger.info(
            "VLM processed response. Detected %d layout elements matching criteria.",
            len(detections),
        )
        return detections

    def validate_classes(self, classes: List[str]):
        # Override: VLM supports dynamic classes, skip validation
        pass
