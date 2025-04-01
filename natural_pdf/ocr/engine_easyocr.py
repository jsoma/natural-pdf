# ocr_engine_easyocr.py
import logging
import importlib.util
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import inspect # Used for dynamic parameter passing

from .engine import OCREngine
from .ocr_options import EasyOCROptions, BaseOCROptions

logger = logging.getLogger(__name__)

class EasyOCREngine(OCREngine):
    """EasyOCR engine implementation."""

    def __init__(self):
        super().__init__()
        self._easyocr = None # Lazy load easyocr module

    def _lazy_import_easyocr(self):
        """Imports easyocr only when needed."""
        if self._easyocr is None:
            if not self.is_available():
                raise ImportError("EasyOCR is not installed or available.")
            try:
                import easyocr
                self._easyocr = easyocr
                logger.info("EasyOCR module imported successfully.")
            except ImportError as e:
                logger.error(f"Failed to import EasyOCR: {e}")
                raise
        return self._easyocr

    def is_available(self) -> bool:
        """Check if EasyOCR is installed."""
        return importlib.util.find_spec("easyocr") is not None

    def _get_cache_key(self, options: EasyOCROptions) -> str:
        """Generate a more specific cache key for EasyOCR."""
        base_key = super()._get_cache_key(options)
        recog_key = options.recog_network
        detect_key = options.detect_network
        quantize_key = str(options.quantize)
        return f"{base_key}_{recog_key}_{detect_key}_{quantize_key}"

    def _get_reader(self, options: EasyOCROptions):
        """Get or initialize an EasyOCR reader based on options."""
        cache_key = self._get_cache_key(options)
        if cache_key in self._reader_cache:
            logger.debug(f"Using cached EasyOCR reader for key: {cache_key}")
            return self._reader_cache[cache_key]

        logger.info(f"Creating new EasyOCR reader for key: {cache_key}")
        easyocr = self._lazy_import_easyocr()

        constructor_sig = inspect.signature(easyocr.Reader.__init__)
        constructor_args = {}
        constructor_args['lang_list'] = options.languages
        constructor_args['gpu'] = 'cuda' in str(options.device).lower() or 'mps' in str(options.device).lower()

        for field_name, param in constructor_sig.parameters.items():
            if field_name in ['self', 'lang_list', 'gpu']: continue
            if hasattr(options, field_name):
                 constructor_args[field_name] = getattr(options, field_name)
            elif field_name in options.extra_args:
                 constructor_args[field_name] = options.extra_args[field_name]

        logger.debug(f"EasyOCR Reader constructor args: {constructor_args}")
        try:
            reader = easyocr.Reader(**constructor_args)
            self._reader_cache[cache_key] = reader
            logger.info("EasyOCR reader created successfully.")
            return reader
        except Exception as e:
            logger.error(f"Failed to create EasyOCR reader: {e}", exc_info=True)
            raise

    def _prepare_readtext_args(self, options: EasyOCROptions, reader) -> Dict[str, Any]:
        """Helper to prepare arguments for the readtext method."""
        readtext_sig = inspect.signature(reader.readtext)
        readtext_args = {}
        for field_name, param in readtext_sig.parameters.items():
             if field_name == 'image': continue
             if hasattr(options, field_name):
                 readtext_args[field_name] = getattr(options, field_name)
             elif field_name in options.extra_args:
                 readtext_args[field_name] = options.extra_args[field_name]
        logger.debug(f"EasyOCR readtext args: {readtext_args}")
        return readtext_args

    def _standardize_results(self, raw_results: List[Any], options: EasyOCROptions) -> List[Dict[str, Any]]:
        """Standardizes raw results from EasyOCR's readtext."""
        standardized_results = []
        min_confidence = options.min_confidence

        for detection in raw_results:
            try:
                if options.detail == 1 and isinstance(detection, (list, tuple)) and len(detection) >= 3:
                    bbox_raw = detection[0]
                    text = str(detection[1])
                    confidence = float(detection[2])

                    if confidence >= min_confidence:
                        bbox = self._standardize_bbox(bbox_raw)
                        if bbox:
                            standardized_results.append({
                                'bbox': bbox, 'text': text, 'confidence': confidence, 'source': 'ocr'
                            })
                        else:
                             logger.warning(f"Skipping result due to invalid bbox: {bbox_raw}")

                elif options.detail == 0 and isinstance(detection, str):
                     standardized_results.append({
                         'bbox': None, 'text': detection, 'confidence': 1.0, 'source': 'ocr'
                     })
            except (IndexError, ValueError, TypeError) as e:
                 logger.warning(f"Skipping invalid detection format: {detection}. Error: {e}")
                 continue
        return standardized_results


    def process_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        options: BaseOCROptions
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Processes a single image or a batch of images with EasyOCR."""

        if not isinstance(options, EasyOCROptions):
             logger.warning("Received BaseOCROptions, expected EasyOCROptions. Using defaults.")
             # Create default EasyOCR options if base was passed, preserving base settings
             options = EasyOCROptions(
                 languages=options.languages,
                 min_confidence=options.min_confidence,
                 device=options.device,
                 extra_args=options.extra_args # Pass along any extra args
             )

        reader = self._get_reader(options)
        readtext_args = self._prepare_readtext_args(options, reader)

        # --- Handle single image or batch ---
        if isinstance(images, list):
            # --- Batch Processing (Iterative for EasyOCR) ---
            all_results = []
            logger.info(f"Processing batch of {len(images)} images with EasyOCR (iteratively)...")
            for i, img in enumerate(images):
                if not isinstance(img, Image.Image):
                     logger.warning(f"Item at index {i} in batch is not a PIL Image. Skipping.")
                     all_results.append([])
                     continue
                img_array = np.array(img)
                try:
                    logger.debug(f"Processing image {i+1}/{len(images)} in batch.")
                    raw_results = reader.readtext(img_array, **readtext_args)
                    standardized = self._standardize_results(raw_results, options)
                    all_results.append(standardized)
                except Exception as e:
                    logger.error(f"Error processing image {i+1} in EasyOCR batch: {e}", exc_info=True)
                    all_results.append([]) # Append empty list for failed image
            logger.info(f"Finished processing batch with EasyOCR.")
            return all_results # Return List[List[Dict]]

        elif isinstance(images, Image.Image):
            # --- Single Image Processing ---
            logger.info("Processing single image with EasyOCR...")
            img_array = np.array(images)
            try:
                raw_results = reader.readtext(img_array, **readtext_args)
                standardized = self._standardize_results(raw_results, options)
                logger.info(f"Finished processing single image. Found {len(standardized)} results.")
                return standardized # Return List[Dict]
            except Exception as e:
                logger.error(f"Error processing single image with EasyOCR: {e}", exc_info=True)
                return [] # Return empty list on failure
        else:
            raise TypeError("Input 'images' must be a PIL Image or a list of PIL Images.")

