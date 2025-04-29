# ocr_engine_easyocr.py
import importlib.util
<<<<<<< HEAD
=======
import inspect  # Used for dynamic parameter passing
>>>>>>> ea72b84d (A hundred updates, a thousand updates)
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

<<<<<<< HEAD
from .engine import OCREngine, TextRegion
=======
from .engine import OCREngine
>>>>>>> ea72b84d (A hundred updates, a thousand updates)
from .ocr_options import BaseOCROptions, EasyOCROptions

logger = logging.getLogger(__name__)


class EasyOCREngine(OCREngine):
    """EasyOCR engine implementation."""

    def __init__(self):
        super().__init__()
<<<<<<< HEAD
        # No longer need _easyocr attribute
        # self._easyocr = None
=======
        self._easyocr = None  # Lazy load easyocr module

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
>>>>>>> ea72b84d (A hundred updates, a thousand updates)

    def is_available(self) -> bool:
        """Check if EasyOCR is installed."""
        return importlib.util.find_spec("easyocr") is not None

<<<<<<< HEAD
    def _initialize_model(
        self, languages: List[str], device: str, options: Optional[BaseOCROptions]
    ):
        """Initialize the EasyOCR model."""
        # Import directly here
=======
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
        constructor_args["lang_list"] = options.languages
        constructor_args["gpu"] = (
            "cuda" in str(options.device).lower() or "mps" in str(options.device).lower()
        )

        for field_name, param in constructor_sig.parameters.items():
            if field_name in ["self", "lang_list", "gpu"]:
                continue
            if hasattr(options, field_name):
                constructor_args[field_name] = getattr(options, field_name)
            elif field_name in options.extra_args:
                constructor_args[field_name] = options.extra_args[field_name]

        logger.debug(f"EasyOCR Reader constructor args: {constructor_args}")
>>>>>>> ea72b84d (A hundred updates, a thousand updates)
        try:
            import easyocr

            self.logger.info("EasyOCR module imported successfully.")
        except ImportError as e:
            self.logger.error(f"Failed to import EasyOCR: {e}")
            raise

        # Cast to EasyOCROptions if possible, otherwise use default
        easy_options = options if isinstance(options, EasyOCROptions) else EasyOCROptions()

        # Prepare constructor arguments
        use_gpu = "cuda" in device.lower() or "mps" in device.lower()

        constructor_args = {
            "lang_list": languages,
            "gpu": use_gpu,
            # Explicitly map relevant options
            "model_storage_directory": easy_options.model_storage_directory,
            "user_network_directory": easy_options.user_network_directory,
            "recog_network": easy_options.recog_network,
            "detect_network": easy_options.detect_network,
            "download_enabled": easy_options.download_enabled,
            "detector": easy_options.detector,
            "recognizer": easy_options.recognizer,
            "verbose": easy_options.verbose,
            "quantize": easy_options.quantize,
            "cudnn_benchmark": easy_options.cudnn_benchmark,
        }

        # Filter out None values, as EasyOCR expects non-None or default behaviour
        constructor_args = {k: v for k, v in constructor_args.items() if v is not None}

        self.logger.debug(f"EasyOCR Reader constructor args: {constructor_args}")

        # Create the reader
        try:
            self._model = easyocr.Reader(**constructor_args)
            self.logger.info("EasyOCR reader created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create EasyOCR reader: {e}")
            raise

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array for EasyOCR."""
        return np.array(image)

    def _process_single_image(
        self, image: np.ndarray, detect_only: bool, options: Optional[EasyOCROptions]
    ) -> Any:
        """Process a single image with EasyOCR."""
        if self._model is None:
            raise RuntimeError("EasyOCR model not initialized")

        # Cast options to proper type if provided
        easy_options = options if isinstance(options, EasyOCROptions) else EasyOCROptions()

        # Prepare readtext arguments (only needed if not detect_only)
        readtext_args = {}
<<<<<<< HEAD
        if not detect_only:
            for param in [
                "detail",
                "paragraph",
                "min_size",
                "contrast_ths",
                "adjust_contrast",
                "filter_ths",
                "text_threshold",
                "low_text",
                "link_threshold",
                "canvas_size",
                "mag_ratio",
                "slope_ths",
                "ycenter_ths",
                "height_ths",
                "width_ths",
                "y_ths",
                "x_ths",
                "add_margin",
                "output_format",
            ]:
                if hasattr(easy_options, param):
                    val = getattr(easy_options, param)
                    if val is not None:
                        readtext_args[param] = val

        # Process differently based on detect_only flag
        if detect_only:
            # Returns tuple (horizontal_list, free_list)
            # horizontal_list is a list containing one item: the list of boxes
            # Each box is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            bboxes_tuple = self._model.detect(
                image, **readtext_args
            )  # Pass args here too? Check EasyOCR docs if needed.
            if (
                bboxes_tuple
                and isinstance(bboxes_tuple, tuple)
                and len(bboxes_tuple) > 0
                and isinstance(bboxes_tuple[0], list)
            ):
                return bboxes_tuple[0]  # Return the list of polygons directly
            else:
                self.logger.warning(f"EasyOCR detect returned unexpected format: {bboxes_tuple}")
                return []  # Return empty list on unexpected format
        else:
            return self._model.readtext(image, **readtext_args)
=======
        for field_name, param in readtext_sig.parameters.items():
            if field_name == "image":
                continue
            if hasattr(options, field_name):
                readtext_args[field_name] = getattr(options, field_name)
            elif field_name in options.extra_args:
                readtext_args[field_name] = options.extra_args[field_name]
        logger.debug(f"EasyOCR readtext args: {readtext_args}")
        return readtext_args

    def _standardize_results(
        self, raw_results: List[Any], options: EasyOCROptions
    ) -> List[Dict[str, Any]]:
        """Standardizes raw results from EasyOCR's readtext."""
        standardized_results = []
        min_confidence = options.min_confidence
>>>>>>> ea72b84d (A hundred updates, a thousand updates)

    def _standardize_results(
        self, raw_results: Any, min_confidence: float, detect_only: bool
    ) -> List[TextRegion]:
        """Convert EasyOCR results to standardized TextRegion objects."""
        standardized_regions = []

        if detect_only:
            results = raw_results[0]
            # In detect_only mode, raw_results is already a list of bounding boxes
            # Each bbox is in [x_min, x_max, y_min, y_max] format
            if isinstance(results, list):
                for detection in results:
                    try:
                        # This block expects 'detection' to be a list/tuple of 4 numbers
                        if isinstance(detection, (list, tuple)) and len(detection) == 4:
                            x_min, x_max, y_min, y_max = detection
                            # Convert to standardized (x0, y0, x1, y1) format
                            try:
                                bbox = (float(x_min), float(y_min), float(x_max), float(y_max))
                                standardized_regions.append(
                                    TextRegion(bbox, text=None, confidence=None)
                                )
                            except (ValueError, TypeError) as e:
                                raise ValueError(
                                    f"Invalid number format in EasyOCR detect bbox: {detection}"
                                ) from e
                        else:
                            # This is where the error is raised if 'detection' is not a list/tuple of 4 numbers
                            raise ValueError(f"Invalid detection format from EasyOCR: {detection}")
                    except ValueError as e:
                        # Re-raise any value errors from standardization or format checks
                        raise e
                    except Exception as e:
                        # Catch other potential processing errors
                        raise ValueError(
                            f"Error processing EasyOCR detection item: {detection}"
                        ) from e
            else:
                raise ValueError(
                    f"Expected list of bounding boxes in detect_only mode, got: {type(raw_results)}"
                )

            return standardized_regions

        # Full OCR mode (readtext results)
        for detection in raw_results:
            try:
<<<<<<< HEAD
                # Detail mode (list/tuple result)
                if isinstance(detection, (list, tuple)) and len(detection) >= 3:
                    bbox_raw = detection[0]  # This is usually a polygon [[x1,y1],...]
=======
                if (
                    options.detail == 1
                    and isinstance(detection, (list, tuple))
                    and len(detection) >= 3
                ):
                    bbox_raw = detection[0]
>>>>>>> ea72b84d (A hundred updates, a thousand updates)
                    text = str(detection[1])
                    confidence = float(detection[2])

                    if confidence >= min_confidence:
<<<<<<< HEAD
                        try:
                            # Use the standard helper for polygons
                            bbox = self._standardize_bbox(bbox_raw)
                            standardized_regions.append(TextRegion(bbox, text, confidence))
                        except ValueError as e:
                            raise ValueError(
                                f"Could not standardize bounding box from EasyOCR readtext: {bbox_raw}"
                            ) from e

                # Simple mode (string result)
                elif isinstance(detection, str):
                    if 0.0 >= min_confidence:  # Always include if min_confidence is 0
                        standardized_regions.append(TextRegion((0, 0, 0, 0), detection, 1.0))
                else:
                    # Handle unexpected format in OCR mode
                    raise ValueError(
                        f"Invalid OCR detection format from EasyOCR readtext: {detection}"
                    )

            except ValueError as e:
                # Re-raise any value errors from standardization or format checks
                raise e
            except Exception as e:
                # Catch other potential processing errors
                raise ValueError(f"Error processing EasyOCR detection item: {detection}") from e

        return standardized_regions
=======
                        bbox = self._standardize_bbox(bbox_raw)
                        if bbox:
                            standardized_results.append(
                                {
                                    "bbox": bbox,
                                    "text": text,
                                    "confidence": confidence,
                                    "source": "ocr",
                                }
                            )
                        else:
                            logger.warning(f"Skipping result due to invalid bbox: {bbox_raw}")

                elif options.detail == 0 and isinstance(detection, str):
                    standardized_results.append(
                        {"bbox": None, "text": detection, "confidence": 1.0, "source": "ocr"}
                    )
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid detection format: {detection}. Error: {e}")
                continue
        return standardized_results

    def process_image(
        self, images: Union[Image.Image, List[Image.Image]], options: BaseOCROptions
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Processes a single image or a batch of images with EasyOCR."""

        if not isinstance(options, EasyOCROptions):
            logger.warning("Received BaseOCROptions, expected EasyOCROptions. Using defaults.")
            # Create default EasyOCR options if base was passed, preserving base settings
            options = EasyOCROptions(
                languages=options.languages,
                min_confidence=options.min_confidence,
                device=options.device,
                extra_args=options.extra_args,  # Pass along any extra args
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
                    logger.error(
                        f"Error processing image {i+1} in EasyOCR batch: {e}", exc_info=True
                    )
                    all_results.append([])  # Append empty list for failed image
            logger.info(f"Finished processing batch with EasyOCR.")
            return all_results  # Return List[List[Dict]]

        elif isinstance(images, Image.Image):
            # --- Single Image Processing ---
            logger.info("Processing single image with EasyOCR...")
            img_array = np.array(images)
            try:
                raw_results = reader.readtext(img_array, **readtext_args)
                standardized = self._standardize_results(raw_results, options)
                logger.info(f"Finished processing single image. Found {len(standardized)} results.")
                return standardized  # Return List[Dict]
            except Exception as e:
                logger.error(f"Error processing single image with EasyOCR: {e}", exc_info=True)
                return []  # Return empty list on failure
        else:
            raise TypeError("Input 'images' must be a PIL Image or a list of PIL Images.")
>>>>>>> ea72b84d (A hundred updates, a thousand updates)
