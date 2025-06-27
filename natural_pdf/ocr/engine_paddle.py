# ocr_engine_paddleocr.py
import importlib.util
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .engine import OCREngine, TextRegion
from .ocr_options import BaseOCROptions, PaddleOCROptions

logger = logging.getLogger(__name__)


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
            import paddleocr

            self.logger.info("PaddleOCR module imported successfully.")
        except ImportError as e:
            self.logger.error(f"Failed to import PaddleOCR/PaddlePaddle: {e}")
            raise RuntimeError(
                "paddleocr is not available. Install via: npdf install paddle"
            ) from e

        paddle_options = options if isinstance(options, PaddleOCROptions) else PaddleOCROptions()

        if len(languages) > 1:
            self.logger.warning(
                "PaddleOCR >= 3.0.0 only supports one language at a time. "
                "Using the first language provided: '%s'",
                languages[0],
            )
        primary_lang = languages[0] if languages else "en"

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
                    self.logger.error(
                        "Language '%s' is not supported by any available PaddleOCR version (v3, v4, v5). "
                        "Proceeding without a specific version, but this is likely to fail.",
                        primary_lang,
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
        self, image: np.ndarray, detect_only: bool, options: Optional[PaddleOCROptions]
    ) -> Any:
        """Process a single image with PaddleOCR using the .predict() method."""
        if self._model is None:
            raise RuntimeError("PaddleOCR model not initialized")

        # Prepare arguments for the .predict() method from PaddleOCROptions.
        # See: https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/OCR.html
        predict_args = {}
        if options and isinstance(options, PaddleOCROptions):
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
                if hasattr(options, arg) and getattr(options, arg) is not None:
                    predict_args[arg] = getattr(options, arg)

        # The `detect_only` flag is handled in `_standardize_results` by ignoring
        # the recognized text and confidence, as the new .predict() API does not
        # have a direct flag to disable only the recognition step.

        # Run OCR using the new .predict() method.
        raw_results = self._model.predict(image)
        return raw_results

    def _standardize_results(
        self, raw_results: Any, min_confidence: float, detect_only: bool
    ) -> List[TextRegion]:
        """Convert PaddleOCR results to standardized TextRegion objects."""
        standardized_regions = []

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
                    text = str(rec_texts[i]) if not detect_only else None
                    confidence = float(rec_scores[i]) if not detect_only else None
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
                        standardized_regions.append(TextRegion(bbox, text=None, confidence=None))
                    elif confidence is not None and confidence >= min_confidence:
                        standardized_regions.append(TextRegion(bbox, text, confidence))
            return standardized_regions

        # Old format fallback (list of lists/tuples)
        page_results = raw_results[0] if raw_results[0] is not None else []
        for detection in page_results:
            # Initialize text and confidence
            text = None
            confidence = None
            bbox_raw = None

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
                text = str(text_confidence[0])
                confidence = float(text_confidence[1])

            # Standardize the bbox (always needed)
            try:
                bbox = self._standardize_bbox(bbox_raw)
            except ValueError as e:
                raise ValueError(
                    f"Could not standardize bounding box from PaddleOCR: {bbox_raw}"
                ) from e

            # Append based on mode
            if detect_only:
                # Append regardless of dummy confidence value, set text/conf to None
                standardized_regions.append(TextRegion(bbox, text=None, confidence=None))
            elif confidence >= min_confidence:
                # Only append if confidence meets threshold in full OCR mode
                standardized_regions.append(TextRegion(bbox, text, confidence))

        return standardized_regions
