# ocr_engine_paddleocr.py
import importlib.util
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from natural_pdf.utils.option_validation import validate_option_type

from .engine import OCREngine, TextRegion
from .ocr_options import BaseOCROptions, PaddleOCROptions

logger = logging.getLogger(__name__)

# Common ISO 639-1 codes mapped to PaddleOCR's internal language codes.
_PADDLE_LANG_ALIASES = {
    "ja": "japan",
    "ko": "korean",
    "zh": "ch",
    "zh-cn": "ch",
    "zh-hans": "ch",
    "zh-tw": "chinese_cht",
    "zh-hant": "chinese_cht",
    "de": "german",
    "fr": "french",
}


def _normalize_paddle_language(code: str) -> str:
    """Normalize a language code to PaddleOCR's expected format.

    Maps common ISO 639-1 codes (e.g. ``"ja"``, ``"zh"``) to PaddleOCR's
    internal codes (``"japan"``, ``"ch"``). Unknown codes pass through as-is.
    """
    lower = code.strip().lower()
    mapped = _PADDLE_LANG_ALIASES.get(lower)
    if mapped:
        logger.debug("PaddleOCR: normalized language code '%s' -> '%s'", code, mapped)
        return mapped
    return lower


def _translate_device_for_paddle(device: str, logger: logging.Logger) -> str:
    """Translate standard device strings to PaddleOCR-compatible format.

    PaddleOCR uses ``"gpu:0"`` instead of ``"cuda"`` and does not support
    ``"mps"``.  This helper bridges the gap between the device strings
    returned by ``resolve_auto_device()`` (``"cuda"``, ``"mps"``, ``"cpu"``)
    and what PaddleOCR actually accepts.
    """
    if device is None:
        return None

    lower = device.strip().lower()
    if lower == "cuda" or lower == "gpu":
        logger.debug("Translating device '%s' -> 'gpu:0' for PaddleOCR", device)
        return "gpu:0"
    if lower.startswith("cuda:"):
        # e.g. "cuda:1" -> "gpu:1"
        idx = lower.split(":", 1)[1]
        translated = f"gpu:{idx}"
        logger.debug("Translating device '%s' -> '%s' for PaddleOCR", device, translated)
        return translated
    if lower == "mps":
        logger.warning("PaddleOCR does not support MPS (Apple Silicon GPU). Falling back to CPU.")
        return "cpu"
    # Already a PaddleOCR-native string (e.g. "cpu", "gpu:0", "npu:0") — pass through.
    return device


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine implementation."""

    SUPPORT_MATRIX = {
        "PP-OCRv5": {"ch", "chinese_cht", "en", "japan"},
        "PP-OCRv4": {"ch", "en"},
        "PP-OCRv3": {
            "abq",
            "af",
            "ady",
            "ang",
            "ar",
            "ava",
            "az",
            "be",
            "bg",
            "bgc",
            "bh",
            "bho",
            "bs",
            "ch",
            "che",
            "chinese_cht",
            "cs",
            "cy",
            "da",
            "dar",
            "de",
            "german",
            "en",
            "es",
            "et",
            "fa",
            "fr",
            "french",
            "ga",
            "gom",
            "hi",
            "hr",
            "hu",
            "id",
            "inh",
            "is",
            "it",
            "japan",
            "ka",
            "kbd",
            "korean",
            "ku",
            "la",
            "lbe",
            "lez",
            "lt",
            "lv",
            "mah",
            "mai",
            "mi",
            "mn",
            "mr",
            "ms",
            "mt",
            "ne",
            "new",
            "nl",
            "no",
            "oc",
            "pi",
            "pl",
            "pt",
            "ro",
            "rs_cyrillic",
            "rs_latin",
            "ru",
            "sa",
            "sck",
            "sk",
            "sl",
            "sq",
            "sv",
            "sw",
            "ta",
            "tab",
            "te",
            "tl",
            "tr",
            "ug",
            "uk",
            "ur",
            "uz",
            "vi",
        },
    }

    def __init__(self):
        super().__init__()

    def is_available(self) -> bool:
        """Check if PaddleOCR and paddlepaddle are installed."""
        paddle_installed = (
            importlib.util.find_spec("paddle") is not None
            or importlib.util.find_spec("paddlepaddle") is not None
        )
        paddleocr_installed = importlib.util.find_spec("paddleocr") is not None
        return paddle_installed and paddleocr_installed

    def _initialize_model(
        self, languages: List[str], device: str, options: Optional[BaseOCROptions]
    ):
        """Initialize the PaddleOCR model using the >=3.0.0 pipeline API."""
        try:
            import paddleocr  # type: ignore[import-untyped]

            self.logger.info("PaddleOCR module imported successfully.")
        except ImportError as e:
            self.logger.error(f"Failed to import PaddleOCR/PaddlePaddle: {e}")
            raise RuntimeError(
                'paddleocr is not available. Install via: pip install "natural-pdf[paddle]"'
            ) from e

        paddle_options, _ = validate_option_type(options, PaddleOCROptions, "PaddleOCREngine")

        # Normalize language codes (e.g. "ja" -> "japan", "zh" -> "ch")
        normalized = [_normalize_paddle_language(l) for l in languages] if languages else ["en"]

        if len(normalized) > 1:
            self.logger.warning(
                "PaddleOCR >= 3.0.0 only supports one language at a time. "
                "Using the first language provided: '%s'",
                normalized[0],
            )
        primary_lang = normalized[0]

        # Determine the appropriate ocr_version based on language support
        user_ocr_version = paddle_options.ocr_version
        final_ocr_version = user_ocr_version
        version_preference = ["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"]

        # --- RESTORE: Language/version support check logic ---
        user_specified_model = (
            getattr(paddle_options, "text_recognition_model_name", None) is not None
            or getattr(paddle_options, "text_detection_model_name", None) is not None
        )
        if user_specified_model and user_ocr_version:
            if primary_lang not in self.SUPPORT_MATRIX.get(user_ocr_version, set()):
                self.logger.warning(
                    f"Model '{user_ocr_version}' was explicitly specified, but language '{primary_lang}' is not officially supported. Proceeding anyway as requested."
                )

        if user_ocr_version:
            if primary_lang not in self.SUPPORT_MATRIX.get(user_ocr_version, set()):
                self.logger.warning(
                    f"Language '{primary_lang}' is not supported by the requested ocr_version '{user_ocr_version}'. "
                    f"Attempting to find a compatible version."
                )
                self.logger.warning(
                    "Language '%s' is not supported by the requested ocr_version '%s'. "
                    "Attempting to find a compatible version.",
                    primary_lang,
                    user_ocr_version,
                )
                final_ocr_version = None  # Reset to find a compatible version

        # If no version was specified or the specified one was incompatible, find the best fit.
        if not final_ocr_version:
            found_compatible = False
            for version in version_preference:
                if primary_lang in self.SUPPORT_MATRIX[version]:
                    final_ocr_version = version
                    found_compatible = True
                    break

            if not found_compatible:
                if not languages or not primary_lang:
                    final_ocr_version = "PP-OCRv5"
                    self.logger.info(
                        "No language specified and no match found. Defaulting to ocr_version 'PP-OCRv5'. Note: 'PP-OCRv3' has the widest language support among PaddleOCR versions."
                    )
                else:
                    # Collect all supported codes and find similar ones
                    all_supported = set()
                    for codes in self.SUPPORT_MATRIX.values():
                        all_supported.update(codes)
                    similar = sorted(c for c in all_supported if c.startswith(primary_lang[:2]))
                    hint = ""
                    if similar:
                        hint = f" Similar supported codes: {', '.join(similar)}."
                    self.logger.error(
                        "Language '%s' is not supported by any available PaddleOCR version (v3, v4, v5). "
                        "Proceeding without a specific version, but this is likely to fail.%s",
                        primary_lang,
                        hint,
                    )
                    final_ocr_version = None  # Let paddleocr handle the error
            elif final_ocr_version != "PP-OCRv5":
                self.logger.warning(
                    f"Automatically selected ocr_version '{final_ocr_version}' for language '{primary_lang}'. This is not the default (PP-OCRv5)."
                )
                self.logger.warning(
                    "Automatically selected ocr_version '%s' for language '%s'. This is not the default (PP-OCRv5).",
                    final_ocr_version,
                    primary_lang,
                )
            # else: if PP-OCRv5, no need to log
        elif final_ocr_version != "PP-OCRv5":
            self.logger.warning(
                f"Using user-specified ocr_version '{final_ocr_version}' for language '{primary_lang}'. This is not the default (PP-OCRv5)."
            )
            self.logger.warning(
                "Using user-specified ocr_version '%s' for language '%s'. This is not the default (PP-OCRv5).",
                final_ocr_version,
                primary_lang,
            )
        # --- END RESTORE ---

        # Build PaddleOCR config dict from valid constructor arguments.
        # See: https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/OCR.html
        valid_init_args = {
            "doc_orientation_classify_model_name",
            "doc_orientation_classify_model_dir",
            "doc_unwarping_model_name",
            "doc_unwarping_model_dir",
            "text_detection_model_name",
            "text_detection_model_dir",
            "textline_orientation_model_name",
            "textline_orientation_model_dir",
            "text_recognition_model_name",
            "text_recognition_model_dir",
            "textline_orientation_batch_size",
            "text_recognition_batch_size",
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_textline_orientation",
            "text_det_limit_side_len",
            "text_det_limit_type",
            "text_det_thresh",
            "text_det_box_thresh",
            "text_det_unclip_ratio",
            "text_det_input_shape",
            "text_rec_score_thresh",
            "text_rec_input_shape",
            "lang",
            "ocr_version",
            "device",
            "enable_hpi",
            "use_tensorrt",
            "precision",
            "enable_mkldnn",
            # "mkldnn_cache_capacity",
            "cpu_threads",
            "paddlex_config",
        }

        # Translate device string for PaddleOCR compatibility
        # (e.g. "cuda" -> "gpu:0", "mps" -> "cpu")
        device = _translate_device_for_paddle(device, self.logger)

        # Start with defaults passed from the main apply_ocr call.
        ocr_config = {
            "lang": primary_lang,
            "device": device,
        }

        # Add the determined ocr_version to the config if available
        if final_ocr_version:
            ocr_config["ocr_version"] = final_ocr_version

        # Populate ocr_config from paddle_options with non-None values
        # that are valid for the constructor. This allows overriding defaults.
        for arg in valid_init_args:
            if hasattr(paddle_options, arg):
                value = getattr(paddle_options, arg)
                if value is not None:
                    ocr_config[arg] = value

        # If the user set device in PaddleOCROptions, it may also need translation
        if "device" in ocr_config and ocr_config["device"]:
            ocr_config["device"] = _translate_device_for_paddle(ocr_config["device"], self.logger)

        try:
            # The new API uses PaddleOCR as a pipeline object.
            self._model = paddleocr.PaddleOCR(**ocr_config)
            self.logger.info("PaddleOCR model created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create PaddleOCR model: {e}")
            raise

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to BGR numpy array for PaddleOCR."""
        if image.mode == "BGR":
            return np.array(image)
        img_rgb = image.convert("RGB")
        img_array_rgb = np.array(img_rgb)
        img_array_bgr = img_array_rgb[:, :, ::-1]  # Convert RGB to BGR
        return img_array_bgr

    def _process_single_image(
        self, image: Any, detect_only: bool, options: Optional[BaseOCROptions]
    ) -> Any:
        """Process a single image with PaddleOCR using the .predict() method."""
        if self._model is None:
            raise RuntimeError("PaddleOCR model not initialized")

        if not isinstance(image, np.ndarray):
            raise TypeError("PaddleOCREngine expects preprocessed numpy arrays")

        # Prepare arguments for the .predict() method from PaddleOCROptions.
        # See: https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/OCR.html
        predict_args: Dict[str, Any] = {}
        paddle_options = options if isinstance(options, PaddleOCROptions) else None
        if paddle_options is not None:
            valid_predict_args = {
                "use_doc_orientation_classify",
                "use_doc_unwarping",
                "use_textline_orientation",
                "text_det_limit_side_len",
                "text_det_limit_type",
                "text_det_thresh",
                "text_det_box_thresh",
                "text_det_unclip_ratio",
                "text_rec_score_thresh",
            }
            for arg in valid_predict_args:
                if hasattr(paddle_options, arg):
                    value = getattr(paddle_options, arg)
                    if value is not None:
                        predict_args[arg] = value

        # The `detect_only` flag is handled in `_standardize_results` by ignoring
        # the recognized text and confidence, as the new .predict() API does not
        # have a direct flag to disable only the recognition step.

        # Run OCR using the new .predict() method.
        raw_results = self._model.predict(image, **predict_args)
        return raw_results

    def _standardize_results(
        self, raw_results: Any, min_confidence: float, detect_only: bool, **kwargs
    ) -> List[TextRegion]:
        """Convert PaddleOCR results to standardized TextRegion objects."""
        standardized_regions: List[TextRegion] = []

        if not raw_results or not isinstance(raw_results, list) or len(raw_results) == 0:
            return standardized_regions

        # New PaddleOCR 3.x format: list of dicts with keys like 'rec_texts', 'rec_scores', 'rec_boxes'
        if isinstance(raw_results[0], dict):
            for page in raw_results:
                rec_texts = page.get("rec_texts", [])
                rec_scores = page.get("rec_scores", [])
                rec_boxes = page.get("rec_boxes", [])
                # Fallback to dt_polys if rec_boxes is not present or empty
                if rec_boxes is None or len(rec_boxes) == 0:
                    rec_boxes = page.get("dt_polys", [])
                for i in range(len(rec_texts)):
                    text_value = str(rec_texts[i]) if i < len(rec_texts) else ""
                    confidence_value = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                    # --- Bounding box format note ---
                    # PaddleOCR 3.x may return bounding boxes in several formats:
                    # - Rectangle: [x1, y1, x2, y2] (list or 1D numpy array of length 4)
                    # - Polygon: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] (list of 4 points or 2D numpy array shape (4,2))
                    # - Sometimes, rec_boxes is a numpy array of shape (N, 4) or (N, 4, 2)
                    # This code converts any numpy array to a list before passing to _standardize_bbox,
                    # which handles both rectangle and polygon formats robustly.
                    box = rec_boxes[i]
                    if hasattr(box, "tolist"):
                        box = box.tolist()
                    bbox = self._standardize_bbox(box)
                    if detect_only:
                        standardized_regions.append(TextRegion(bbox, "", 0.0))
                    elif confidence_value >= min_confidence:
                        standardized_regions.append(TextRegion(bbox, text_value, confidence_value))
            return standardized_regions

        # Old format fallback (list of lists/tuples)
        page_results = raw_results[0] if raw_results and raw_results[0] is not None else []
        for detection in page_results:
            # Initialize text and confidence
            text_value = ""
            confidence_value = 0.0

            # Paddle always seems to return the tuple structure [bbox, (text, conf)]
            # even if rec=False. We need to parse this structure regardless.
            if len(detection) == 4:  # Handle potential alternative format?
                detection = [detection, ("", 1.0)]  # Treat as bbox + dummy text/conf

            if not isinstance(detection, (list, tuple)) or len(detection) < 2:
                raise ValueError(f"Invalid detection format from PaddleOCR: {detection}")

            bbox_raw = detection[0]
            text_confidence = detection[1]

            if not isinstance(text_confidence, tuple) or len(text_confidence) < 2:
                # Even if detect_only, we expect the (text, conf) structure,
                # it might just contain dummy values.
                raise ValueError(
                    f"Invalid text/confidence structure from PaddleOCR: {text_confidence}"
                )

            # Extract text/conf only if not detect_only
            if not detect_only:
                text_value = str(text_confidence[0])
                confidence_value = float(text_confidence[1])

            # Standardize the bbox (always needed)
            try:
                bbox = self._standardize_bbox(bbox_raw)
            except ValueError as e:
                raise ValueError(
                    f"Could not standardize bounding box from PaddleOCR: {bbox_raw}"
                ) from e

            # Append based on mode
            if detect_only:
                standardized_regions.append(TextRegion(bbox, "", 0.0))
            elif confidence_value >= min_confidence:
                standardized_regions.append(TextRegion(bbox, text_value, confidence_value))

        return standardized_regions
