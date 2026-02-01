# ocr_engine_rapidocr.py
"""RapidOCR engine implementation.

RapidOCR uses PaddleOCR models converted to ONNX format, providing
the same accuracy with simpler installation (~15MB vs ~500MB).
"""
import importlib.util
import logging
from typing import Any, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

from .engine import OCREngine, TextRegion
from .ocr_options import BaseOCROptions, RapidOCROptions


class RapidOCREngine(OCREngine):
    """RapidOCR engine implementation.

    RapidOCR provides PaddleOCR models via ONNX runtime, offering:
    - Simpler installation (~15MB vs ~500MB for PaddlePaddle)
    - Same model accuracy as PaddleOCR
    - Cross-platform compatibility via ONNX
    """

    def __init__(self):
        super().__init__()
        self._engine: Optional[Any] = None

    def is_available(self) -> bool:
        """Check if the rapidocr library is installed."""
        return importlib.util.find_spec("rapidocr") is not None

    def _initialize_model(
        self, languages: List[str], device: str, options: Optional[BaseOCROptions]
    ):
        """Initialize RapidOCR engine."""
        if not self.is_available():
            raise ImportError("RapidOCR library is not installed or available.")

        from rapidocr import RapidOCR

        self.logger.info("Initializing RapidOCR engine...")

        # Build initialization parameters
        params = {}

        if options and isinstance(options, RapidOCROptions):
            # If custom config path is provided, use that directly
            if options.config_path:
                self._engine = RapidOCR(config_path=options.config_path)
                self.logger.info(f"RapidOCR initialized with custom config: {options.config_path}")
                return

            # Map options to RapidOCR params
            # Note: RapidOCR uses params dict for configuration
            if options.det_model_type:
                params["det_model_type"] = options.det_model_type
            if options.rec_model_type:
                params["rec_model_type"] = options.rec_model_type

        # Initialize with params if any, otherwise use defaults
        if params:
            self._engine = RapidOCR(params=params)
        else:
            self._engine = RapidOCR()

        self.logger.info("RapidOCR engine initialized successfully.")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """RapidOCR accepts PIL images directly."""
        return image

    def _process_single_image(
        self, image: Any, detect_only: bool, options: Optional[BaseOCROptions]
    ) -> Any:
        """Process a single image with RapidOCR."""
        if self._engine is None:
            raise RuntimeError("RapidOCR engine is not initialized.")

        if not isinstance(image, Image.Image):
            raise TypeError("RapidOCREngine expects PIL images after preprocessing")

        # Determine engine settings
        use_det = True
        use_cls = not detect_only
        use_rec = not detect_only

        if options and isinstance(options, RapidOCROptions):
            use_det = options.use_det
            if not detect_only:
                use_cls = options.use_cls
                use_rec = options.use_rec

        # RapidOCR accepts PIL images directly
        result = self._engine(
            image,
            use_det=use_det,
            use_cls=use_cls,
            use_rec=use_rec,
        )

        return result

    def _standardize_results(
        self, raw_results: Any, min_confidence: float, detect_only: bool
    ) -> List[TextRegion]:
        """Convert RapidOCR results to standardized TextRegion objects.

        RapidOCR returns a RapidOCROutput dataclass with:
        - .boxes: shape (N, 4, 2) polygon coordinates (numpy array)
        - .txts: tuple of recognized text strings
        - .scores: tuple of confidence scores
        """
        regions: List[TextRegion] = []

        if raw_results is None:
            return regions

        # RapidOCR returns a dataclass-like object with boxes, txts, scores
        boxes = getattr(raw_results, "boxes", None)
        txts = getattr(raw_results, "txts", None)
        scores = getattr(raw_results, "scores", None)

        if boxes is None:
            return regions

        for i, box in enumerate(boxes):
            # Get text and score for this detection
            text = txts[i] if txts and i < len(txts) else ""
            score = float(scores[i]) if scores and i < len(scores) else 0.0

            # Skip low confidence results (unless detect_only)
            if not detect_only and score < min_confidence:
                continue

            # Convert polygon to bbox using TextRegion.from_polygon
            # box is typically shape (4, 2) - four corner points
            try:
                polygon = box.tolist() if hasattr(box, "tolist") else list(box)
                region = TextRegion.from_polygon(polygon, text, score)
                regions.append(region)
            except (ValueError, TypeError, AttributeError) as e:
                self.logger.warning(f"Failed to process box {i}: {e}")
                continue

        return regions
