"""Default checkbox detector — wendys-llc/checkbox-detector YOLO12n.

A 2-class YOLO12n model (checked/unchecked) trained on full-page
document images (~1000px max side, letterboxed to 1024x1024).
Detects and classifies in one pass — no separate classifier step.
Renders at 72 DPI to match training scale; SAHI tiles when the
rendered image exceeds 1024px.

Only needs onnxruntime + numpy + huggingface_hub.
"""

import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from .base import DetectionContext
from .checkbox_options import BaseCheckboxOptions, DefaultCheckboxOptions
from .onnx_engine import OnnxCheckboxDetector

logger = logging.getLogger(__name__)


class DefaultCheckboxDetector(OnnxCheckboxDetector):
    """wendys-llc/checkbox-detector — detects checked/unchecked checkboxes via ONNX."""

    def detect(
        self,
        image: Image.Image,
        options: BaseCheckboxOptions,
        context: Optional[DetectionContext] = None,
    ) -> List[Dict[str, Any]]:
        # Ensure default-model-specific defaults
        if not isinstance(options, DefaultCheckboxOptions):
            opts = DefaultCheckboxOptions(
                confidence=options.confidence,
                device=options.device,
            )
        else:
            opts = options

        detections = super().detect(image, opts, context)

        # Override engine name
        for det in detections:
            det["engine"] = "default"

        return detections
