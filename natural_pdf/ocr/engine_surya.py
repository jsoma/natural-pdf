# ocr_engine_surya.py
import importlib.util
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

<<<<<<< HEAD
from .engine import OCREngine, TextRegion
=======
from .engine import OCREngine
>>>>>>> ea72b84d (A hundred updates, a thousand updates)
from .ocr_options import BaseOCROptions, SuryaOCROptions



class SuryaOCREngine(OCREngine):
    """Surya OCR engine implementation."""

    def __init__(self):
        super().__init__()
        self._recognition_predictor = None
        self._detection_predictor = None
        self._surya_recognition = None
        self._surya_detection = None

    def _initialize_model(
        self, languages: List[str], device: str, options: Optional[BaseOCROptions]
    ):
        """Initialize Surya predictors."""
        if not self.is_available():
            raise ImportError("Surya OCR library is not installed or available.")

<<<<<<< HEAD
        # Store languages for use in _process_single_image
        self._langs = languages

        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
=======
        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor

            self._surya_recognition = RecognitionPredictor
            self._surya_detection = DetectionPredictor
            logger.info("Surya modules imported successfully.")

            # --- Instantiate Predictors ---
            # Add arguments from options if Surya supports them
            # Example: device = options.device or 'cuda' if torch.cuda.is_available() else 'cpu'
            # predictor_args = {'device': options.device} # If applicable
            predictor_args = {}  # Assuming parameterless init based on example
>>>>>>> ea72b84d (A hundred updates, a thousand updates)

        self._surya_recognition = RecognitionPredictor
        self._surya_detection = DetectionPredictor
        self.logger.info("Surya modules imported successfully.")

        predictor_args = {}  # Configure if needed

        self.logger.info("Instantiating Surya DetectionPredictor...")
        self._detection_predictor = self._surya_detection(**predictor_args)
        self.logger.info("Instantiating Surya RecognitionPredictor...")
        self._recognition_predictor = self._surya_recognition(**predictor_args)

        self.logger.info("Surya predictors initialized.")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Surya uses PIL images directly, so just return the image."""
        return image

    def _process_single_image(
        self, image: Image.Image, detect_only: bool, options: Optional[SuryaOCROptions]
    ) -> Any:
        """Process a single image with Surya OCR."""
        if not self._recognition_predictor or not self._detection_predictor:
            raise RuntimeError("Surya predictors are not initialized.")

        # Store languages instance variable during initialization to use here
        langs = (
            [[lang] for lang in self._langs]
            if hasattr(self, "_langs")
            else [[self.DEFAULT_LANGUAGES[0]]]
        )

        # Surya expects lists of images, so we need to wrap our single image
        if detect_only:
            results = self._detection_predictor(images=[image])
        else:
            results = self._recognition_predictor(
                images=[image],
                langs=langs,  # Use the languages set during initialization
                det_predictor=self._detection_predictor,
            )

        # Surya may return a list with one result per image or a single result object
        # Return the result as-is and handle the extraction in _standardize_results
        return results

    def _standardize_results(
        self, raw_results: Any, min_confidence: float, detect_only: bool
    ) -> List[TextRegion]:
        """Convert Surya results to standardized TextRegion objects."""
        standardized_regions = []

        raw_result = raw_results
        if isinstance(raw_results, list) and len(raw_results) > 0:
            raw_result = raw_results[0]

        results = (
            raw_result.text_lines
            if hasattr(raw_result, "text_lines") and not detect_only
            else raw_result.bboxes
        )

        for line in results:
            # Always extract bbox first
            try:
                # Prioritize line.bbox, fallback to line.polygon
                bbox_raw = line.bbox if hasattr(line, "bbox") else getattr(line, "polygon", None)
                if bbox_raw is None:
                    raise ValueError("Missing bbox/polygon data")
                bbox = self._standardize_bbox(bbox_raw)
            except ValueError as e:
                raise ValueError(
                    f"Could not standardize bounding box from Surya result: {bbox_raw}"
                ) from e

            if detect_only:
                # For detect_only, text and confidence are None
                standardized_regions.append(TextRegion(bbox, text=None, confidence=None))
            else:
                # For full OCR, extract text and confidence, then filter
                text = line.text if hasattr(line, "text") else ""
                confidence = line.confidence
                if confidence >= min_confidence:
                    standardized_regions.append(TextRegion(bbox, text, confidence))

        return standardized_regions

    def is_available(self) -> bool:
        """Check if the surya library is installed."""
        return importlib.util.find_spec("surya") is not None
<<<<<<< HEAD
=======

    def _standardize_results(
        self, raw_ocr_result: Any, options: SuryaOCROptions
    ) -> List[Dict[str, Any]]:
        """Standardizes raw results from a single image from Surya."""
        standardized_page = []
        min_confidence = options.min_confidence

        # Check if the result has the expected structure (OCRResult with text_lines)
        if not hasattr(raw_ocr_result, "text_lines") or not isinstance(
            raw_ocr_result.text_lines, list
        ):
            logger.warning(f"Unexpected Surya result format: {type(raw_ocr_result)}. Skipping.")
            return standardized_page

        for line in raw_ocr_result.text_lines:
            try:
                # Extract data from Surya's TextLine object
                text = line.text
                confidence = line.confidence
                # Surya provides both polygon and bbox, bbox is already (x0, y0, x1, y1)
                bbox_raw = line.bbox  # Use bbox directly if available and correct format

                if confidence >= min_confidence:
                    bbox = self._standardize_bbox(bbox_raw)  # Validate/convert format
                    if bbox:
                        standardized_page.append(
                            {"bbox": bbox, "text": text, "confidence": confidence, "source": "ocr"}
                        )
                    else:
                        # Try polygon if bbox failed standardization
                        bbox_poly = self._standardize_bbox(line.polygon)
                        if bbox_poly:
                            standardized_page.append(
                                {
                                    "bbox": bbox_poly,
                                    "text": text,
                                    "confidence": confidence,
                                    "source": "ocr",
                                }
                            )
                        else:
                            logger.warning(
                                f"Skipping Surya line due to invalid bbox/polygon: {line}"
                            )

            except (AttributeError, ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid Surya TextLine format: {line}. Error: {e}")
                continue
        return standardized_page

    def process_image(
        self, images: Union[Image.Image, List[Image.Image]], options: BaseOCROptions
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Processes a single image or a batch of images with Surya OCR."""

        if not isinstance(options, SuryaOCROptions):
            logger.warning("Received BaseOCROptions, expected SuryaOCROptions. Using defaults.")
            options = SuryaOCROptions(
                languages=options.languages,
                min_confidence=options.min_confidence,
                device=options.device,
                extra_args=options.extra_args,
            )

        # Ensure predictors are loaded/initialized
        self._lazy_load_predictors(options)
        if not self._recognition_predictor or not self._detection_predictor:
            raise RuntimeError("Surya predictors could not be initialized.")

        # --- Prepare inputs for Surya ---
        is_batch = isinstance(images, list)
        input_images: List[Image.Image] = images if is_batch else [images]
        # Surya expects a list of language lists, one per image
        input_langs: List[List[str]] = [options.languages for _ in input_images]

        if not input_images:
            logger.warning("No images provided for Surya processing.")
            return [] if not is_batch else [[]]

        # --- Run Surya Prediction ---
        try:
            processing_mode = "batch" if is_batch else "single image"
            logger.info(f"Processing {processing_mode} ({len(input_images)} images) with Surya...")
            # Call Surya's predictor
            # It returns a list of OCRResult objects, one per input image
            predictions = self._recognition_predictor(
                images=input_images, langs=input_langs, det_predictor=self._detection_predictor
            )
            logger.info(f"Surya prediction complete. Received {len(predictions)} results.")

            # --- Standardize Results ---
            if len(predictions) != len(input_images):
                logger.error(
                    f"Surya result count ({len(predictions)}) does not match input count ({len(input_images)}). Returning empty results."
                )
                # Decide on error handling: raise error or return empty structure
                return [[] for _ in input_images] if is_batch else []

            all_standardized_results = [
                self._standardize_results(res, options) for res in predictions
            ]

            if is_batch:
                return all_standardized_results  # Return List[List[Dict]]
            else:
                return all_standardized_results[0]  # Return List[Dict] for single image

        except Exception as e:
            logger.error(f"Error during Surya OCR processing: {e}", exc_info=True)
            # Return empty structure matching input type on failure
            return [[] for _ in input_images] if is_batch else []

    # Note: Caching is handled differently for Surya as predictors are stateful
    # and initialized once. The base class _reader_cache is not used here.
    # If predictors could be configured per-run, caching would need rethinking.
>>>>>>> ea72b84d (A hundred updates, a thousand updates)
