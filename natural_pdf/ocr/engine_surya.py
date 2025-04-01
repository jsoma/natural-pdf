# ocr_engine_surya.py
import logging
import importlib.util
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

from .engine import OCREngine
from .ocr_options import SuryaOCROptions, BaseOCROptions

logger = logging.getLogger(__name__)

class SuryaOCREngine(OCREngine):
    """Surya OCR engine implementation."""

    def __init__(self):
        super().__init__()
        self._recognition_predictor = None
        self._detection_predictor = None
        self._surya_recognition = None
        self._surya_detection = None
        self._initialized = False

    def _lazy_load_predictors(self, options: SuryaOCROptions):
        """Initializes Surya predictors when first needed."""
        if self._initialized:
            return

        if not self.is_available():
            raise ImportError("Surya OCR library is not installed or available.")

        try:
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            self._surya_recognition = RecognitionPredictor
            self._surya_detection = DetectionPredictor
            logger.info("Surya modules imported successfully.")

            # --- Instantiate Predictors ---
            # Add arguments from options if Surya supports them
            # Example: device = options.device or 'cuda' if torch.cuda.is_available() else 'cpu'
            # predictor_args = {'device': options.device} # If applicable
            predictor_args = {} # Assuming parameterless init based on example

            logger.info("Instantiating Surya DetectionPredictor...")
            self._detection_predictor = self._surya_detection(**predictor_args)
            logger.info("Instantiating Surya RecognitionPredictor...")
            self._recognition_predictor = self._surya_recognition(**predictor_args)

            self._initialized = True
            logger.info("Surya predictors initialized.")

        except ImportError as e:
            logger.error(f"Failed to import Surya modules: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Surya predictors: {e}", exc_info=True)
            raise

    def is_available(self) -> bool:
        """Check if the surya library is installed."""
        return importlib.util.find_spec("surya") is not None

    def _standardize_results(self, raw_ocr_result: Any, options: SuryaOCROptions) -> List[Dict[str, Any]]:
        """Standardizes raw results from a single image from Surya."""
        standardized_page = []
        min_confidence = options.min_confidence

        # Check if the result has the expected structure (OCRResult with text_lines)
        if not hasattr(raw_ocr_result, 'text_lines') or not isinstance(raw_ocr_result.text_lines, list):
            logger.warning(f"Unexpected Surya result format: {type(raw_ocr_result)}. Skipping.")
            return standardized_page

        for line in raw_ocr_result.text_lines:
            try:
                # Extract data from Surya's TextLine object
                text = line.text
                confidence = line.confidence
                # Surya provides both polygon and bbox, bbox is already (x0, y0, x1, y1)
                bbox_raw = line.bbox # Use bbox directly if available and correct format

                if confidence >= min_confidence:
                    bbox = self._standardize_bbox(bbox_raw) # Validate/convert format
                    if bbox:
                        standardized_page.append({
                            'bbox': bbox,
                            'text': text,
                            'confidence': confidence,
                            'source': 'ocr'
                        })
                    else:
                        # Try polygon if bbox failed standardization
                        bbox_poly = self._standardize_bbox(line.polygon)
                        if bbox_poly:
                             standardized_page.append({
                                'bbox': bbox_poly, 'text': text, 'confidence': confidence, 'source': 'ocr'
                             })
                        else:
                             logger.warning(f"Skipping Surya line due to invalid bbox/polygon: {line}")

            except (AttributeError, ValueError, TypeError) as e:
                 logger.warning(f"Skipping invalid Surya TextLine format: {line}. Error: {e}")
                 continue
        return standardized_page

    def process_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        options: BaseOCROptions
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Processes a single image or a batch of images with Surya OCR."""

        if not isinstance(options, SuryaOCROptions):
             logger.warning("Received BaseOCROptions, expected SuryaOCROptions. Using defaults.")
             options = SuryaOCROptions(
                 languages=options.languages,
                 min_confidence=options.min_confidence,
                 device=options.device,
                 extra_args=options.extra_args
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
                images=input_images,
                langs=input_langs,
                det_predictor=self._detection_predictor
            )
            logger.info(f"Surya prediction complete. Received {len(predictions)} results.")

            # --- Standardize Results ---
            if len(predictions) != len(input_images):
                 logger.error(f"Surya result count ({len(predictions)}) does not match input count ({len(input_images)}). Returning empty results.")
                 # Decide on error handling: raise error or return empty structure
                 return [[] for _ in input_images] if is_batch else []

            all_standardized_results = [self._standardize_results(res, options) for res in predictions]

            if is_batch:
                return all_standardized_results # Return List[List[Dict]]
            else:
                return all_standardized_results[0] # Return List[Dict] for single image

        except Exception as e:
            logger.error(f"Error during Surya OCR processing: {e}", exc_info=True)
            # Return empty structure matching input type on failure
            return [[] for _ in input_images] if is_batch else []

    # Note: Caching is handled differently for Surya as predictors are stateful
    # and initialized once. The base class _reader_cache is not used here.
    # If predictors could be configured per-run, caching would need rethinking.

