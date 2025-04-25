# ocr_engine_easyocr.py
import importlib.util
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .engine import OCREngine, TextRegion
from .ocr_options import BaseOCROptions, EasyOCROptions

logger = logging.getLogger(__name__)


class EasyOCREngine(OCREngine):
    """EasyOCR engine implementation."""

    def __init__(self):
        super().__init__()
        # No longer need _easyocr attribute
        # self._easyocr = None

    def is_available(self) -> bool:
        """Check if EasyOCR is installed."""
        return importlib.util.find_spec("easyocr") is not None

    def _initialize_model(
        self, languages: List[str], device: str, options: Optional[BaseOCROptions]
    ):
        """Initialize the EasyOCR model."""
        # Import directly here
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
                # Detail mode (list/tuple result)
                if isinstance(detection, (list, tuple)) and len(detection) >= 3:
                    bbox_raw = detection[0]  # This is usually a polygon [[x1,y1],...]
                    text = str(detection[1])
                    confidence = float(detection[2])

                    if confidence >= min_confidence:
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
