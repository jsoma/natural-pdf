# ocr_engine_surya.py
import importlib.util
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

from PIL import Image

from .engine import OCREngine, TextRegion
from .ocr_options import BaseOCROptions, SuryaOCROptions


class SuryaOCREngine(OCREngine):
    """Surya OCR engine implementation."""

    def __init__(self):
        super().__init__()
        self._recognition_predictor: Optional[Callable[..., Any]] = None
        self._detection_predictor: Optional[Callable[..., Any]] = None
        self._surya_recognition: Optional[Callable[..., Any]] = None
        self._surya_detection: Optional[Callable[..., Any]] = None
        self._langs: Sequence[str] = self.DEFAULT_LANGUAGES

    def _initialize_model(
        self, languages: List[str], device: str, options: Optional[BaseOCROptions]
    ):
        """Initialize Surya predictors."""
        if not self.is_available():
            raise ImportError("Surya OCR library is not installed or available.")

        self._langs = languages or self.DEFAULT_LANGUAGES

        from surya.detection import DetectionPredictor  # type: ignore[import-untyped]
        from surya.recognition import RecognitionPredictor  # type: ignore[import-untyped]

        self._surya_recognition = RecognitionPredictor
        self._surya_detection = DetectionPredictor
        self.logger.info("Surya modules imported successfully.")

        predictor_args: Dict[str, Any] = {}
        # Filter only allowed Surya args (currently none, but placeholder for future)
        allowed_args: Set[str] = set()
        filtered_args = {k: v for k, v in predictor_args.items() if k in allowed_args}
        dropped = set(predictor_args) - allowed_args
        if dropped:
            self.logger.warning(f"Dropped unsupported Surya args: {dropped}")

        if self._surya_detection is None or self._surya_recognition is None:
            raise RuntimeError("Failed to load Surya predictors.")

        self.logger.info("Instantiating Surya DetectionPredictor...")
        self._detection_predictor = self._surya_detection(**filtered_args)
        self.logger.info("Instantiating Surya RecognitionPredictor...")
        self._recognition_predictor = self._surya_recognition(**filtered_args)

        self.logger.info("Surya predictors initialized.")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Surya uses PIL images directly, so just return the image."""
        return image

    def _process_single_image(
        self, image: Any, detect_only: bool, options: Optional[BaseOCROptions]
    ) -> Any:
        """Process a single image with Surya OCR."""
        if self._recognition_predictor is None or self._detection_predictor is None:
            raise RuntimeError("Surya predictors are not initialized.")

        if not isinstance(image, Image.Image):
            raise TypeError("SuryaOCREngine expects PIL images after preprocessing")

        langs = [list(self._langs)]
        surya_options = options if isinstance(options, SuryaOCROptions) else None

        # Surya expects lists of images, so we need to wrap our single image
        if detect_only:
            detection_predictor = self._detection_predictor
            assert detection_predictor is not None
            results = detection_predictor(images=[image])
        else:
            # Some Surya versions require 'langs' parameter in the __call__ while
            # others assume the predictor was initialized with languages already.
            # Inspect the callable signature to decide what to pass.
            import inspect

            recog_callable = self._recognition_predictor
            try:
                sig = inspect.signature(recog_callable)
                has_langs_param = "langs" in sig.parameters
            except (TypeError, ValueError):
                # Fallback: assume langs not required if signature cannot be inspected
                has_langs_param = False

            recognition_predictor = self._recognition_predictor
            detection_predictor = self._detection_predictor
            assert recognition_predictor is not None
            assert detection_predictor is not None

            if has_langs_param:
                results = recognition_predictor(
                    langs=langs,
                    images=[image],
                    det_predictor=detection_predictor,
                )
            else:
                # Older/newer Surya versions that omit 'langs'
                results = recognition_predictor(
                    images=[image],
                    det_predictor=detection_predictor,
                )

        # Surya may return a list with one result per image or a single result object
        # Return the result as-is and handle the extraction in _standardize_results
        return results

    def _standardize_results(
        self, raw_results: Any, min_confidence: float, detect_only: bool
    ) -> List[TextRegion]:
        """Convert Surya results to standardized TextRegion objects."""
        standardized_regions: List[TextRegion] = []

        raw_result: Any
        if isinstance(raw_results, list) and raw_results:
            raw_result = raw_results[0]
        else:
            raw_result = raw_results

        if raw_result is None:
            return standardized_regions

        if not detect_only and hasattr(raw_result, "text_lines"):
            results_iter = getattr(raw_result, "text_lines", None)
        else:
            results_iter = getattr(raw_result, "bboxes", None)

        if results_iter is None:
            return standardized_regions

        if isinstance(results_iter, Iterable):
            results_iterable = results_iter
        else:
            results_iterable = [results_iter]

        for line in results_iterable:
            # Always extract bbox first
            bbox_raw: Any = None
            try:
                # Prioritize line.bbox, fallback to line.polygon
                bbox_raw = getattr(line, "bbox", None)
                if bbox_raw is None:
                    bbox_raw = getattr(line, "polygon", None)
                if bbox_raw is None:
                    raise ValueError("Missing bbox/polygon data")
                bbox = self._standardize_bbox(bbox_raw)
            except ValueError as e:
                raise ValueError(
                    f"Could not standardize bounding box from Surya result: {bbox_raw}"
                ) from e

            if detect_only:
                standardized_regions.append(TextRegion(bbox, "", 0.0))
            else:
                text = getattr(line, "text", "")
                confidence = float(getattr(line, "confidence", 0.0))
                if confidence >= min_confidence:
                    standardized_regions.append(TextRegion(bbox, text, confidence))

        return standardized_regions

    def is_available(self) -> bool:
        """Check if the surya library is installed."""
        return importlib.util.find_spec("surya") is not None
