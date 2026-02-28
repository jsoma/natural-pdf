"""Default checkbox detector — jsoma/checkbox-detector YOLO12n.

A 2-class YOLO12n model (checked/unchecked) trained on ~16k tiled
document images. Detects and classifies in one pass — no separate
classifier step needed. Uses SAHI tiling at inference for reliable
detection of small checkboxes on full-page images.

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
    """jsoma/checkbox-detector — detects checked/unchecked checkboxes via ONNX."""

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
                resolution=options.resolution,
                device=options.device,
            )
        else:
            opts = options

        detections = super().detect(image, opts, context)

        # Override engine name
        for det in detections:
            det["engine"] = "default"

        return detections
