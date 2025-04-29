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
        """Initialize the PaddleOCR model."""
        try:
            import paddleocr

            self.logger.info("PaddleOCR module imported successfully.")
        except ImportError as e:
            self.logger.error(f"Failed to import PaddleOCR/PaddlePaddle: {e}")
            raise

        # Cast to PaddleOCROptions if possible
        paddle_options = options if isinstance(options, PaddleOCROptions) else PaddleOCROptions()

        # Determine parameters
        primary_lang = languages[0] if languages else "en"
        use_gpu = "cuda" in str(device).lower()

        # Create constructor arguments
        constructor_args = {
            "lang": primary_lang,
            "use_gpu": use_gpu,
            "use_angle_cls": paddle_options.use_angle_cls,
            "det": True,
            "rec": True,  # We'll control recognition at process time
        }

        # Add optional parameters if available
        for param in ["det_model_dir", "rec_model_dir", "cls_model_dir", "show_log", "use_onnx"]:
            if hasattr(paddle_options, param):
                val = getattr(paddle_options, param)
                if val is not None:
                    constructor_args[param] = val

        self.logger.debug(f"PaddleOCR constructor args: {constructor_args}")

        # Create the model
        try:
            self._model = paddleocr.PaddleOCR(**constructor_args)
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
        """Process a single image with PaddleOCR."""
        if self._model is None:
            raise RuntimeError("PaddleOCR model not initialized")

        # Prepare OCR arguments
        ocr_args = {}
        if options and isinstance(options, PaddleOCROptions):
            ocr_args["cls"] = options.cls if options.cls is not None else options.use_angle_cls
            ocr_args["det"] = options.det
            ocr_args["rec"] = not detect_only  # Control recognition based on detect_only flag

        # Run OCR
        raw_results = self._model.ocr(image, **ocr_args)
        return raw_results

    def _standardize_results(
        self, raw_results: Any, min_confidence: float, detect_only: bool
    ) -> List[TextRegion]:
        """Convert PaddleOCR results to standardized TextRegion objects."""
        standardized_regions = []

        if not raw_results or not isinstance(raw_results, list) or len(raw_results) == 0:
            return standardized_regions

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
