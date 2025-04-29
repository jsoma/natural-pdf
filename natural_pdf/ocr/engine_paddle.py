# ocr_engine_paddleocr.py
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
from .ocr_options import BaseOCROptions, PaddleOCROptions

logger = logging.getLogger(__name__)


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine implementation."""

<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======
    LANGUAGE_MAP = {
        "en": "en",
        "zh": "ch",
        "zh-cn": "ch",
        "zh-tw": "chinese_cht",
        "ja": "japan",
        "ko": "korean",
        "th": "thai",
        "fr": "french",
        "de": "german",
        "ru": "russian",
        "ar": "arabic",
        "hi": "hindi",
        "vi": "vietnam",
        "fa": "cyrillic",
        "ur": "cyrillic",
        "rs": "serbian",
        "oc": "latin",
        "rsc": "cyrillic",
        "bg": "bulgarian",
        "uk": "cyrillic",
        "be": "cyrillic",
        "te": "telugu",
        "kn": "kannada",
        "ta": "tamil",
        "latin": "latin",
        "cyrillic": "cyrillic",
        "devanagari": "devanagari",
    }

    def __init__(self):
        super().__init__()
        self._paddleocr = None

    def _lazy_import_paddleocr(self):
        """Imports paddleocr only when needed."""
        if self._paddleocr is None:
            if not self.is_available():
                raise ImportError("PaddleOCR or PaddlePaddle is not installed or available.")
            try:
                import paddle
                import paddleocr

                self._paddleocr = paddleocr
                logger.info("PaddleOCR module imported successfully.")
            except ImportError as e:
                logger.error(f"Failed to import PaddleOCR/PaddlePaddle: {e}")
                raise
        return self._paddleocr
>>>>>>> ea72b84d (A hundred updates, a thousand updates)

    def is_available(self) -> bool:
        """Check if PaddleOCR and paddlepaddle are installed."""
        paddle_installed = (
            importlib.util.find_spec("paddle") is not None
            or importlib.util.find_spec("paddlepaddle") is not None
        )
        paddleocr_installed = importlib.util.find_spec("paddleocr") is not None
        return paddle_installed and paddleocr_installed

<<<<<<< HEAD
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
=======
    def _map_language(self, iso_lang: str) -> str:
        """Map ISO language code to PaddleOCR language code."""
        return self.LANGUAGE_MAP.get(iso_lang.lower(), "en")

    def _get_cache_key(self, options: PaddleOCROptions) -> str:
        """Generate a more specific cache key for PaddleOCR."""
        base_key = super()._get_cache_key(options)
        primary_lang = self._map_language(options.languages[0]) if options.languages else "en"
        angle_cls_key = str(options.use_angle_cls)
        precision_key = options.precision
        return f"{base_key}_{primary_lang}_{angle_cls_key}_{precision_key}"

    def _get_reader(self, options: PaddleOCROptions):
        """Get or initialize a PaddleOCR reader based on options."""
        cache_key = self._get_cache_key(options)
        if cache_key in self._reader_cache:
            logger.debug(f"Using cached PaddleOCR reader for key: {cache_key}")
            return self._reader_cache[cache_key]

        logger.info(f"Creating new PaddleOCR reader for key: {cache_key}")
        paddleocr = self._lazy_import_paddleocr()

        constructor_sig = inspect.signature(paddleocr.PaddleOCR.__init__)
        constructor_args = {}
        constructor_args["lang"] = (
            self._map_language(options.languages[0]) if options.languages else "en"
        )

        for field_name, param in constructor_sig.parameters.items():
            if field_name in ["self", "lang"]:
                continue
            if field_name == "use_gpu":
                constructor_args["use_gpu"] = options.use_gpu
                continue
            if hasattr(options, field_name):
                constructor_args[field_name] = getattr(options, field_name)
            elif field_name in options.extra_args:
                constructor_args[field_name] = options.extra_args[field_name]

        constructor_args.pop("device", None)
        logger.debug(f"PaddleOCR constructor args: {constructor_args}")

        try:
            show_log = constructor_args.get("show_log", False)
            original_log_level = logging.getLogger("ppocr").level
            if not show_log:
                logging.getLogger("ppocr").setLevel(logging.ERROR)

            reader = paddleocr.PaddleOCR(**constructor_args)

            if not show_log:
                logging.getLogger("ppocr").setLevel(original_log_level)

            self._reader_cache[cache_key] = reader
            logger.info("PaddleOCR reader created successfully.")
            return reader
        except Exception as e:
            logger.error(f"Failed to create PaddleOCR reader: {e}", exc_info=True)
            raise

    def _prepare_ocr_args(self, options: PaddleOCROptions) -> Dict[str, Any]:
        """Helper to prepare arguments for the ocr method (excluding image)."""
        ocr_args = {}
        # Determine 'cls' value based on options precedence
        ocr_args["cls"] = options.cls if options.cls is not None else options.use_angle_cls
        ocr_args["det"] = options.det
        ocr_args["rec"] = options.rec
        # Add extra args if needed (less common for ocr method itself)
        # for field_name in options.extra_args:
        #      if field_name in ['cls', 'det', 'rec']: # Check against known ocr args
        #          ocr_args[field_name] = options.extra_args[field_name]
        logger.debug(f"PaddleOCR ocr args (excluding image): {ocr_args}")
        return ocr_args

    def _standardize_results(
        self, raw_page_results: Optional[List[Any]], options: PaddleOCROptions
    ) -> List[Dict[str, Any]]:
        """Standardizes raw results from a single page/image from PaddleOCR."""
        standardized_page = []
        if not raw_page_results:  # Handle None or empty list
            return standardized_page

        min_confidence = options.min_confidence
        for detection in raw_page_results:
            try:
                if not isinstance(detection, (list, tuple)) or len(detection) < 2:
                    continue
                bbox_raw = detection[0]
                text_confidence = detection[1]
                if not isinstance(text_confidence, tuple) or len(text_confidence) < 2:
                    continue
>>>>>>> ea72b84d (A hundred updates, a thousand updates)

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

<<<<<<< HEAD
            # Standardize the bbox (always needed)
            try:
                bbox = self._standardize_bbox(bbox_raw)
            except ValueError as e:
                raise ValueError(
                    f"Could not standardize bounding box from PaddleOCR: {bbox_raw}"
                ) from e
=======
                if confidence >= min_confidence:
                    bbox = self._standardize_bbox(bbox_raw)
                    if bbox:
                        standardized_page.append(
                            {"bbox": bbox, "text": text, "confidence": confidence, "source": "ocr"}
                        )
                    else:
                        logger.warning(f"Skipping result due to invalid bbox: {bbox_raw}")
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid detection format: {detection}. Error: {e}")
                continue
        return standardized_page

    def _pil_to_bgr(self, image: Image.Image) -> np.ndarray:
        """Converts PIL Image to BGR numpy array."""
        if image.mode == "BGR":  # Already BGR
            return np.array(image)
        img_rgb = image.convert("RGB")
        img_array_rgb = np.array(img_rgb)
        img_array_bgr = img_array_rgb[:, :, ::-1]  # Convert RGB to BGR
        return img_array_bgr

    def process_image(
        self, images: Union[Image.Image, List[Image.Image]], options: BaseOCROptions
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Processes a single image or a batch of images with PaddleOCR."""

        if not isinstance(options, PaddleOCROptions):
            logger.warning("Received BaseOCROptions, expected PaddleOCROptions. Using defaults.")
            options = PaddleOCROptions(
                languages=options.languages,
                min_confidence=options.min_confidence,
                device=options.device,
                extra_args=options.extra_args,
            )

        reader = self._get_reader(options)
        ocr_args = self._prepare_ocr_args(options)

        # Helper function to process one image
        def process_one(img):
            try:
                img_array_bgr = self._pil_to_bgr(img)
                raw_results = reader.ocr(img_array_bgr, **ocr_args)

                page_results = []
                if raw_results and isinstance(raw_results, list) and len(raw_results) > 0:
                    page_results = raw_results[0]

                return self._standardize_results(page_results, options)
            except Exception as e:
                logger.error(f"Error processing image with PaddleOCR: {e}")
                return []
>>>>>>> ea72b84d (A hundred updates, a thousand updates)

            # Append based on mode
            if detect_only:
                # Append regardless of dummy confidence value, set text/conf to None
                standardized_regions.append(TextRegion(bbox, text=None, confidence=None))
            elif confidence >= min_confidence:
                # Only append if confidence meets threshold in full OCR mode
                standardized_regions.append(TextRegion(bbox, text, confidence))

        return standardized_regions
