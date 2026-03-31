# engine_paddleocr_vl.py
import importlib.util
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from natural_pdf.utils.option_validation import validate_option_type

from .engine import OCREngine, TextRegion
from .engine_paddle import _translate_device_for_paddle
from .ocr_options import BaseOCROptions, PaddleOCRVLOptions

logger = logging.getLogger(__name__)

# Block labels to skip — purely visual content with no text to extract.
_SKIP_LABELS = {"image", "figure", "chart", "seal"}

# Pre-compiled regexes for HTML stripping
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Default confidence score for PaddleOCR-VL blocks (no per-block scores available)
_DEFAULT_BLOCK_CONFIDENCE = 0.99


def _strip_html_tags(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = _TAG_RE.sub(" ", html)
    return _WS_RE.sub(" ", text).strip()


class PaddleOCRVLEngine(OCREngine):
    """PaddleOCR-VL engine implementation.

    PaddleOCR-VL is a VLM-based document understanding engine that returns
    block-level results with layout labels (text, paragraph_title, header,
    footer, table, image, etc.) and bounding boxes.
    """

    def __init__(self):
        super().__init__()

    def is_available(self) -> bool:
        """Check if PaddleOCR-VL dependencies are installed."""
        paddle_installed = (
            importlib.util.find_spec("paddle") is not None
            or importlib.util.find_spec("paddlepaddle") is not None
        )
        paddleocr_installed = importlib.util.find_spec("paddleocr") is not None
        paddlex_installed = importlib.util.find_spec("paddlex") is not None
        return paddle_installed and paddleocr_installed and paddlex_installed

    def _initialize_model(
        self, languages: List[str], device: str, options: Optional[BaseOCROptions]
    ):
        """Initialize the PaddleOCR-VL model."""
        if languages and languages != self.DEFAULT_LANGUAGES:
            self.logger.warning(
                "PaddleOCR-VL does not support language selection; ignoring languages=%s",
                languages,
            )

        try:
            from paddleocr import PaddleOCRVL  # type: ignore[import-untyped]

            self.logger.info("PaddleOCR-VL module imported successfully.")
        except ImportError as e:
            self.logger.error(f"Failed to import PaddleOCR-VL: {e}")
            raise RuntimeError(
                'PaddleOCR-VL is not available. Install via: pip install "natural-pdf[paddle]"'
            ) from e

        vl_options, _ = validate_option_type(options, PaddleOCRVLOptions, "PaddleOCRVLEngine")

        # Translate device string for PaddleOCR compatibility
        # (e.g. "cuda" -> "gpu:0", "mps" -> "cpu")
        paddle_device = _translate_device_for_paddle(device, self.logger)

        # Build constructor kwargs from options
        init_kwargs: Dict[str, Any] = {}
        if paddle_device:
            init_kwargs["device"] = paddle_device

        option_fields = {
            "pipeline_version",
            "use_layout_detection",
            "use_chart_recognition",
            "use_seal_recognition",
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "format_block_content",
        }
        for field_name in option_fields:
            value = getattr(vl_options, field_name, None)
            if value is not None:
                init_kwargs[field_name] = value

        # Pass through any extra_args
        if vl_options.extra_args:
            init_kwargs.update(vl_options.extra_args)

        try:
            self._model = PaddleOCRVL(**init_kwargs)
            self.logger.info("PaddleOCR-VL model created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create PaddleOCR-VL model: {e}")
            raise

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to RGB numpy array for PaddleOCR-VL."""
        img_rgb = image.convert("RGB")
        return np.array(img_rgb)

    def _process_single_image(
        self, image: Any, detect_only: bool, options: Optional[BaseOCROptions]
    ) -> Any:
        """Process a single image with PaddleOCR-VL."""
        if self._model is None:
            raise RuntimeError("PaddleOCR-VL model not initialized")

        if not isinstance(image, np.ndarray):
            raise TypeError("PaddleOCRVLEngine expects preprocessed numpy arrays")

        raw_results = self._model.predict(image)
        return raw_results

    def _standardize_results(
        self, raw_results: Any, min_confidence: float, detect_only: bool, **kwargs
    ) -> List[TextRegion]:
        """Convert PaddleOCR-VL block results to standardized TextRegion objects.

        PaddleOCR-VL returns a list of result objects, each containing blocks.
        Each block has a label, bbox, and content.
        """
        standardized_regions: List[TextRegion] = []

        if raw_results is None:
            return standardized_regions

        # Normalize: if a single result object was returned, wrap in a list
        if (
            isinstance(raw_results, dict)
            or hasattr(raw_results, "blocks")
            or hasattr(raw_results, "parsing_res_list")
        ):
            raw_results = [raw_results]
        elif not isinstance(raw_results, list):
            self.logger.warning(
                "Unexpected raw_results type from PaddleOCR-VL: %s", type(raw_results).__name__
            )
            return standardized_regions

        # PaddleOCR-VL returns a list of result objects (one per image)
        for result in raw_results:
            blocks = getattr(result, "blocks", None)
            if blocks is None:
                # Try dict-style access; PaddleOCR-VL 3.4+ uses "parsing_res_list"
                if isinstance(result, dict):
                    blocks = result.get("parsing_res_list") or result.get("blocks", [])
                else:
                    continue

            for block in blocks:
                # Get block label
                label = getattr(block, "label", None)
                if label is None and isinstance(block, dict):
                    label = block.get("label", "")
                label = str(label).lower() if label else ""

                # Skip purely visual labels
                if label in _SKIP_LABELS:
                    continue
                is_table = label == "table"

                # Get bounding box
                bbox_raw = getattr(block, "bbox", None)
                if bbox_raw is None and isinstance(block, dict):
                    bbox_raw = block.get("bbox")
                if bbox_raw is None:
                    continue

                if hasattr(bbox_raw, "tolist"):
                    bbox_raw = bbox_raw.tolist()

                try:
                    bbox = self._standardize_bbox(bbox_raw)
                except ValueError:
                    self.logger.warning("Skipping block with invalid bbox: %s", bbox_raw)
                    continue

                # Get text content
                if detect_only:
                    standardized_regions.append(TextRegion(bbox, "", 0.0))
                    continue

                content = getattr(block, "content", None)
                if content is None and isinstance(block, dict):
                    content = block.get("content", "")
                content = str(content) if content else ""

                # For table blocks, strip HTML to get plain text
                if is_table and content:
                    content = _strip_html_tags(content)

                if not content.strip():
                    continue

                confidence = _DEFAULT_BLOCK_CONFIDENCE

                if confidence >= min_confidence:
                    standardized_regions.append(TextRegion(bbox, content, confidence))

        return standardized_regions
