# layout_detector_doclayout.py
"""PP-DocLayout-V3 layout detector via HuggingFace transformers."""

import importlib.util
import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from natural_pdf.utils.option_validation import validate_option_type

from .base import LayoutDetector
from .layout_options import BaseLayoutOptions, DocLayoutOptions

logger = logging.getLogger(__name__)


class DocLayoutDetector(LayoutDetector):
    """Document layout detector using PP-DocLayout-V3.

    A lightweight (~45MB) RT-DETR model from PaddlePaddle that detects 25
    document region classes. Runs via HuggingFace transformers — no
    PaddlePaddle framework needed.

    Install: pip install transformers torch
    """

    TYPE_MAP: Dict[str, str] = {
        "doc_title": "title",
        "paragraph_title": "title",
        "figure_title": "caption",
        "content": "text",
        "aside_text": "text",
        "abstract": "text",
        "reference_content": "text",
        "vision_footnote": "footnote",
        "formula_number": "text",
        "number": "text",
        "seal": "unknown",
        "algorithm": "text",
        # These pass through as-is: text, table, figure, image, formula,
        # header, footer, footnote, reference, chart
    }

    def __init__(self):
        super().__init__()
        self.supported_classes = {
            "abstract",
            "algorithm",
            "aside_text",
            "chart",
            "content",
            "formula",
            "doc_title",
            "figure_title",
            "footer",
            "footnote",
            "formula_number",
            "header",
            "image",
            "number",
            "paragraph_title",
            "reference",
            "reference_content",
            "seal",
            "table",
            "text",
            "vision_footnote",
        }

    def is_available(self) -> bool:
        """Check if transformers and torch are installed."""
        return (
            importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("torch") is not None
        )

    def _get_cache_key(self, options: BaseLayoutOptions) -> str:
        options, _ = validate_option_type(options, DocLayoutOptions, "DocLayoutDetector")
        device_key = str(options.device).lower()
        model_key = options.model_name.replace("/", "_")
        return f"{self.__class__.__name__}_{device_key}_{model_key}"

    def _load_model_from_options(self, options: BaseLayoutOptions) -> Any:
        """Load the PP-DocLayout-V3 model."""
        if not self.is_available():
            raise RuntimeError(
                "DocLayout dependencies not installed. Install via: pip install transformers torch"
            )

        options, _ = validate_option_type(options, DocLayoutOptions, "DocLayoutDetector")

        import torch
        from transformers import PPDocLayoutV3ForObjectDetection, PPDocLayoutV3ImageProcessor

        self.logger.info("Loading DocLayout model: %s", options.model_name)

        processor = PPDocLayoutV3ImageProcessor.from_pretrained(options.model_name)
        model = PPDocLayoutV3ForObjectDetection.from_pretrained(options.model_name)
        model.eval()

        device = options.device or "cpu"
        if device == "cpu":
            pass
        elif device == "cuda" and torch.cuda.is_available():
            pass
        elif (
            device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            pass
        else:
            device = "cpu"

        model = model.to(device)
        self.logger.info("DocLayout model loaded on %s.", device)

        return {"processor": processor, "model": model, "device": device}

    def detect(
        self, image: Image.Image, options: BaseLayoutOptions, context=None
    ) -> List[Dict[str, Any]]:
        """Detect layout regions in an image."""
        if not self.is_available():
            raise RuntimeError(
                "DocLayout dependencies not installed. Install via: pip install transformers torch"
            )

        options, _ = validate_option_type(options, DocLayoutOptions, "DocLayoutDetector")

        self.validate_classes(options.classes or [])
        if options.exclude_classes:
            self.validate_classes(options.exclude_classes)

        loaded = self._get_model(options)
        processor = loaded["processor"]
        model = loaded["model"]
        device = loaded["device"]

        normalized_classes_req, normalized_classes_excl = self._build_class_filters(options)

        import torch

        inputs = processor(images=[image.convert("RGB")], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        raw = processor.post_process_object_detection(
            outputs, threshold=options.confidence, target_sizes=target_sizes
        )[0]

        id2label = model.config.id2label

        detections = []
        for score, label_id, box in zip(
            raw["scores"].tolist(), raw["labels"].tolist(), raw["boxes"].tolist()
        ):
            label_name = id2label.get(label_id, str(label_id))
            normalized_class = self._normalize_class_name(label_name)

            if normalized_classes_req and normalized_class not in normalized_classes_req:
                continue
            if normalized_class in normalized_classes_excl:
                continue

            x0, y0, x1, y1 = box
            detections.append(
                {
                    "bbox": (float(x0), float(y0), float(x1), float(y1)),
                    "class": label_name,
                    "confidence": float(score),
                    "normalized_class": normalized_class,
                    "source": "layout",
                    "model": "doclayout",
                }
            )

        self.logger.info("DocLayout detected %d regions matching criteria.", len(detections))
        return detections
