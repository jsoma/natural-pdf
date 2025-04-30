# layout_detector_gemini.py
import base64
import importlib.util
import io
import logging
import os
from typing import Any, Dict, List, Optional

from PIL import Image
from pydantic import BaseModel, Field

# Use OpenAI library for interaction
try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion

    # Import OpenAIError for exception handling if needed
except ImportError:
    OpenAI = None
    ChatCompletion = None

try:
    from .base import LayoutDetector
    from .layout_options import BaseLayoutOptions, GeminiLayoutOptions
except ImportError:
    # Placeholders if run standalone or imports fail
    class BaseLayoutOptions:
        pass

    class GeminiLayoutOptions(BaseLayoutOptions):
        pass

    class LayoutDetector:
        def __init__(self):
            self.logger = logging.getLogger()
            self.supported_classes = set()  # Will be dynamic based on user request

        def _get_model(self, options):
            raise NotImplementedError

        def _normalize_class_name(self, n):
            return n.lower().replace("_", "-").replace(" ", "-")

        def validate_classes(self, c):
            pass  # Less strict validation needed for LLM

    logging.basicConfig()

logger = logging.getLogger(__name__)


# Define Pydantic model for the expected output structure
# This is used by the openai library's `response_format`
class DetectedRegion(BaseModel):
    label: str = Field(description="The identified class name.")
    bbox: List[float] = Field(description="Bounding box coordinates [xmin, ymin, xmax, ymax].")
    confidence: float = Field(description="Confidence score [0.0, 1.0].")


class GeminiLayoutDetector(LayoutDetector):
    """Document layout detector using Google's Gemini models via OpenAI compatibility layer."""

    # Base URL for the Gemini OpenAI-compatible endpoint
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self):
        super().__init__()
        self.supported_classes = set()  # Indicate dynamic nature

    def is_available(self) -> bool:
        """Check if openai library is installed."""
        if OpenAI is None:
            logger.warning(
                "openai package not found. Gemini detector (via OpenAI lib) will not be available. Run: pip install openai"
            )
            return False
        return True

    def _get_cache_key(self, options: GeminiLayoutOptions) -> str:
        """Generate cache key based on model name."""
        if not isinstance(options, GeminiLayoutOptions):
            options = GeminiLayoutOptions()  # Use defaults

        model_key = options.model_name
        # Prompt is built dynamically, so not part of cache key based on options
        return f"{self.__class__.__name__}_{model_key}"

    def _load_model_from_options(self, options: GeminiLayoutOptions) -> Any:
        """Validate options and return the model name."""
        if not self.is_available():
            raise RuntimeError("OpenAI library not installed. Please run: pip install openai")

        if not isinstance(options, GeminiLayoutOptions):
            raise TypeError("Incorrect options type provided for Gemini model loading.")

        # Model loading is deferred to detect() based on whether a client is provided
        return options.model_name

    def detect(self, image: Image.Image, options: BaseLayoutOptions) -> List[Dict[str, Any]]:
        """Detect layout elements in an image using Gemini via OpenAI library."""
        if not self.is_available():
            # The is_available check now only confirms library presence
            raise RuntimeError("OpenAI library not installed. Please run: pip install openai")

        # Ensure options are the correct type
        final_options: GeminiLayoutOptions
        if isinstance(options, GeminiLayoutOptions):
            final_options = options
        else:
            # If base options are passed, try to convert, keeping extra_args
            # Note: This won't transfer a 'client' if it was somehow attached to BaseLayoutOptions
            self.logger.warning(
                "Received BaseLayoutOptions, expected GeminiLayoutOptions. Converting and using defaults."
            )
            final_options = GeminiLayoutOptions(
                confidence=options.confidence,
                classes=options.classes,
                exclude_classes=options.exclude_classes,
                device=options.device,  # device is not used by Gemini detector currently
                extra_args=options.extra_args,
                # client will be None here, forcing default client creation below
            )

        model_name = self._get_model(final_options)
        detections = []

        try:
            # --- 1. Initialize OpenAI Client ---
            client: Optional[OpenAI] = None
            # Use the provided client instance
            if hasattr(final_options.client, "beta") and hasattr(
                final_options.client.beta.chat.completions, "parse"
            ):
                client = final_options.client
                logger.debug("Using provided client instance.")
            else:
                logger.error(
                    "Provided client does not seem compatible (missing beta.chat.completions.parse)."
                )
                raise TypeError(
                    "Provided client is not compatible with the expected OpenAI interface."
                )

            if not client:
                # This should not happen if logic above is correct, but as a safeguard
                raise RuntimeError("Failed to obtain a valid client for Gemini detection.")

            # --- 2. Prepare Input for OpenAI API ---
            if not final_options.classes:
                logger.error("Gemini layout detection requires a list of classes to find.")
                return []

            width, height = image.size

            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_url = f"data:image/png;base64,{img_base64}"

            class_list_str = ", ".join(f"`{c}`" for c in final_options.classes)
            prompt_text = (
                f"Analyze the provided image of a document page ({width}x{height}). "
                f"Identify all regions corresponding to the following types: {class_list_str}. "
                f"Return ONLY the structured data requested as formatted JSON."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ]

            logger.debug(
                f"Running Gemini detection via OpenAI lib (Model: {model_name}). Asking for classes: {final_options.classes}"
            )

            completion_kwargs = {
                "temperature": final_options.extra_args.get(
                    "temperature", 0.0
                ),  # Default to low temp
                "max_tokens": final_options.extra_args.get("max_tokens", 4096),
            }

            completion_kwargs = {k: v for k, v in completion_kwargs.items() if v is not None}

            class ImageContents(BaseModel):
                regions: List[DetectedRegion]

            completion: ChatCompletion = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=ImageContents,
                **completion_kwargs,
            )

            logger.debug(f"Gemini response received via OpenAI lib.")

            # --- 4. Process Parsed Response ---
            if not completion.choices:
                logger.error("Gemini response (via OpenAI lib) contained no choices.")
                return []

            # Get the parsed Pydantic objects
            parsed_results = completion.choices[0].message.parsed.regions
            if not parsed_results or not isinstance(parsed_results, list):
                logger.error(
                    f"Gemini response (via OpenAI lib) did not contain a valid list of parsed regions. Found: {type(parsed_results)}"
                )
                return []

            # --- 5. Convert to Detections & Filter ---
            normalized_classes_req = {self._normalize_class_name(c) for c in final_options.classes}
            normalized_classes_excl = (
                {self._normalize_class_name(c) for c in final_options.exclude_classes}
                if final_options.exclude_classes
                else set()
            )

            for item in parsed_results:
                # The item is already a validated DetectedRegion Pydantic object
                # Access fields directly
                label = item.label
                bbox_raw = item.bbox
                confidence_score = item.confidence

                # Coordinates should already be floats, but ensure tuple format
                xmin, ymin, xmax, ymax = tuple(bbox_raw)

                # --- Apply Filtering ---
                normalized_class = self._normalize_class_name(label)

                # Check against requested classes (Should be guaranteed by schema, but doesn't hurt)
                if normalized_class not in normalized_classes_req:
                    logger.warning(
                        f"Gemini (via OpenAI) returned unexpected class '{label}' despite schema. Skipping."
                    )
                    continue

                # Check against excluded classes
                if normalized_class in normalized_classes_excl:
                    logger.debug(
                        f"Skipping excluded class '{label}' (normalized: {normalized_class})."
                    )
                    continue

                # Check against base confidence threshold from options
                if confidence_score < final_options.confidence:
                    logger.debug(
                        f"Skipping item with confidence {confidence_score:.3f} below threshold {final_options.confidence}."
                    )
                    continue

                # Add detection
                detections.append(
                    {
                        "bbox": (xmin, ymin, xmax, ymax),
                        "class": label,  # Use original label from LLM
                        "confidence": confidence_score,
                        "normalized_class": normalized_class,
                        "source": "layout",
                        "model": "gemini",  # Keep model name generic as gemini
                    }
                )

            self.logger.info(
                f"Gemini (via OpenAI lib) processed response. Detected {len(detections)} layout elements matching criteria."
            )

        except Exception as e:
            # Catch potential OpenAI API errors or other issues
            self.logger.error(f"Error during Gemini detection (via OpenAI lib): {e}", exc_info=True)
            return []

        return detections

    def _normalize_class_name(self, name: str) -> str:
        """Normalizes class names for filtering (lowercase, hyphenated)."""
        return super()._normalize_class_name(name)

    def validate_classes(self, classes: List[str]):
        """Validation is less critical as we pass requested classes to the LLM."""
        pass  # Override base validation if needed, but likely not necessary
