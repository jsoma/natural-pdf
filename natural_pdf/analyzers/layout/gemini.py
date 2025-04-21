# layout_detector_gemini.py
import importlib.util
import logging
import os
from typing import Any, Dict, List, Optional
import base64
import io

from pydantic import BaseModel, Field
from PIL import Image

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
            self.supported_classes = set() # Will be dynamic based on user request

        def _get_model(self, options):
            raise NotImplementedError

        def _normalize_class_name(self, n):
            return n.lower().replace("_", "-").replace(" ", "-")

        def validate_classes(self, c):
            pass # Less strict validation needed for LLM

    logging.basicConfig()

logger = logging.getLogger(__name__)

# Define Pydantic model for the expected output structure
# This is used by the openai library's `response_format`
class DetectedRegion(BaseModel):
    label: str = Field(description="The identified class name.")
    bbox: List[float] = Field(description="Bounding box coordinates [xmin, ymin, xmax, ymax].", min_items=4, max_items=4)
    confidence: float = Field(description="Confidence score [0.0, 1.0].", ge=0.0, le=1.0)


class GeminiLayoutDetector(LayoutDetector):
    """Document layout detector using Google's Gemini models via OpenAI compatibility layer."""

    # Base URL for the Gemini OpenAI-compatible endpoint
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self):
        super().__init__()
        self.supported_classes = set() # Indicate dynamic nature

    def is_available(self) -> bool:
        """Check if openai library is installed and GOOGLE_API_KEY is available."""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY environment variable not set. Gemini detector (via OpenAI lib) will not be available.")
            return False
        if OpenAI is None:
             logger.warning("openai package not found. Gemini detector (via OpenAI lib) will not be available.")
             return False
        return True

    def _get_cache_key(self, options: GeminiLayoutOptions) -> str:
        """Generate cache key based on model name."""
        if not isinstance(options, GeminiLayoutOptions):
            options = GeminiLayoutOptions() # Use defaults

        model_key = options.model_name
        # Prompt is built dynamically, so not part of cache key based on options
        return f"{self.__class__.__name__}_{model_key}"

    def _load_model_from_options(self, options: GeminiLayoutOptions) -> Any:
        """Validate options and return the model name."""
        if not self.is_available():
            raise RuntimeError(
                "OpenAI library not installed or GOOGLE_API_KEY not set. Please run: pip install openai"
            )

        if not isinstance(options, GeminiLayoutOptions):
            raise TypeError("Incorrect options type provided for Gemini model loading.")

        # Simply return the model name, client is created in detect()
        return options.model_name

    def detect(self, image: Image.Image, options: BaseLayoutOptions) -> List[Dict[str, Any]]:
        """Detect layout elements in an image using Gemini via OpenAI library."""
        if not self.is_available():
            raise RuntimeError(
                "OpenAI library not installed or GOOGLE_API_KEY not set."
            )

        # Ensure options are the correct type
        if not isinstance(options, GeminiLayoutOptions):
            self.logger.warning(
                "Received BaseLayoutOptions, expected GeminiLayoutOptions. Using defaults."
            )
            options = GeminiLayoutOptions(
                confidence=options.confidence,
                classes=options.classes,
                exclude_classes=options.exclude_classes,
                device=options.device,
                extra_args=options.extra_args,
            )

        model_name = self._get_model(options)
        api_key = os.environ.get("GOOGLE_API_KEY")

        detections = []
        try:
            # --- 1. Initialize OpenAI Client for Gemini ---
            client = OpenAI(
                api_key=api_key,
                base_url=self.GEMINI_BASE_URL
            )

            # --- 2. Prepare Input for OpenAI API ---
            if not options.classes:
                logger.error("Gemini layout detection requires a list of classes to find.")
                return []

            width, height = image.size

            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_url = f"data:image/png;base64,{img_base64}"

            # Construct the prompt text
            class_list_str = ", ".join(f'`{c}`' for c in options.classes)
            prompt_text = (
                f"Analyze the provided image of a document page ({width}x{height}). "
                f"Identify all regions corresponding to the following types: {class_list_str}. "
                f"Return ONLY the structured data requested."
            )

            # Prepare messages for chat completions endpoint
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

            # --- 3. Call OpenAI API using .parse for structured output ---
            logger.debug(f"Running Gemini detection via OpenAI lib (Model: {model_name}). Asking for classes: {options.classes}")

            # Extract relevant generation parameters from extra_args if provided
            # Mapping common names: temperature, top_p, max_tokens
            completion_kwargs = {
                "temperature": options.extra_args.get("temperature", 0.2), # Default to low temp
                "top_p": options.extra_args.get("top_p"),
                "max_tokens": options.extra_args.get("max_tokens", 4096), # Map from max_output_tokens
            }
            # Filter out None values
            completion_kwargs = {k: v for k, v in completion_kwargs.items() if v is not None}

            completion: ChatCompletion = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=List[DetectedRegion], # Pass the Pydantic model list
                **completion_kwargs
            )

            logger.debug(f"Gemini response received via OpenAI lib.")

            # --- 4. Process Parsed Response --- 
            if not completion.choices:
                logger.error("Gemini response (via OpenAI lib) contained no choices.")
                return []

            # Get the parsed Pydantic objects
            parsed_results = completion.choices[0].message.parsed
            if not parsed_results or not isinstance(parsed_results, list):
                 logger.error(f"Gemini response (via OpenAI lib) did not contain a valid list of parsed regions. Found: {type(parsed_results)}")
                 return []

            # --- 5. Convert to Detections & Filter --- 
            normalized_classes_req = {
                self._normalize_class_name(c) for c in options.classes
            }
            normalized_classes_excl = {
                self._normalize_class_name(c) for c in options.exclude_classes
            } if options.exclude_classes else set()

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
                    logger.warning(f"Gemini (via OpenAI) returned unexpected class '{label}' despite schema. Skipping.")
                    continue

                # Check against excluded classes
                if normalized_class in normalized_classes_excl:
                    logger.debug(f"Skipping excluded class '{label}' (normalized: {normalized_class}).")
                    continue
                
                # Check against base confidence threshold from options
                if confidence_score < options.confidence:
                    logger.debug(f"Skipping item with confidence {confidence_score:.3f} below threshold {options.confidence}.")
                    continue

                # Add detection
                detections.append({
                    "bbox": (xmin, ymin, xmax, ymax),
                    "class": label, # Use original label from LLM
                    "confidence": confidence_score,
                    "normalized_class": normalized_class,
                    "source": "layout",
                    "model": "gemini", # Keep model name generic as gemini
                })

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
        pass # Override base validation if needed, but likely not necessary

