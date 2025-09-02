"""RT-DETR based checkbox detector implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .base import CheckboxDetector
from .checkbox_options import RTDETRCheckboxOptions

logger = logging.getLogger(__name__)

# Lazy imports cache
_transformers_cache = None


def _get_transformers():
    """Lazy import transformers to avoid heavy dependency at module load."""
    global _transformers_cache
    if _transformers_cache is None:
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            _transformers_cache = (AutoImageProcessor, AutoModelForObjectDetection)
        except ImportError:
            raise ImportError(
                "transformers library is required for RT-DETR checkbox detection. "
                "Install it with: pip install transformers"
            )
    return _transformers_cache


def _get_torch():
    """Lazy import torch."""
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError(
            "torch is required for RT-DETR checkbox detection. "
            "Install it with: pip install torch"
        )


class RTDETRCheckboxDetector(CheckboxDetector):
    """RT-DETR based checkbox detector using HuggingFace transformers."""

    def __init__(self):
        """Initialize the RT-DETR checkbox detector."""
        super().__init__()

    @classmethod
    def is_available(cls) -> bool:
        """Check if transformers and torch are available."""
        try:
            _get_transformers()
            _get_torch()
            return True
        except ImportError:
            return False

    def _get_cache_key(self, options: RTDETRCheckboxOptions) -> str:
        """Generate cache key including model repo and revision."""
        base_key = super()._get_cache_key(options)
        model_key = options.model_repo.replace("/", "_")
        revision_key = options.model_revision or "default"
        return f"{base_key}_{model_key}_{revision_key}"

    def _load_model_from_options(self, options: RTDETRCheckboxOptions) -> Dict[str, Any]:
        """Load RT-DETR model and processor from HuggingFace."""
        AutoImageProcessor, AutoModelForObjectDetection = _get_transformers()
        torch = _get_torch()

        try:
            # Load image processor
            if options.image_processor_repo:
                image_processor = AutoImageProcessor.from_pretrained(
                    options.image_processor_repo, revision=options.model_revision
                )
            else:
                image_processor = AutoImageProcessor.from_pretrained(
                    options.model_repo, revision=options.model_revision
                )

            # Load model
            model = AutoModelForObjectDetection.from_pretrained(
                options.model_repo, revision=options.model_revision
            )

            # Move to device
            if options.device and options.device != "cpu":
                if options.device == "cuda" and torch.cuda.is_available():
                    model = model.to("cuda")
                elif options.device == "mps" and torch.backends.mps.is_available():
                    model = model.to("mps")
                else:
                    self.logger.warning(
                        f"Requested device '{options.device}' not available, using CPU"
                    )
                    model = model.to("cpu")
            else:
                model = model.to("cpu")

            # Set to eval mode
            model.eval()

            return {
                "model": model,
                "processor": image_processor,
                "device": next(model.parameters()).device,
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkbox model '{options.model_repo}'. "
                f"This may be due to network issues or missing credentials. "
                f"Original error: {e}"
            )

    def detect(self, image: Image.Image, options: RTDETRCheckboxOptions) -> List[Dict[str, Any]]:
        """
        Detect checkboxes in the given image using RT-DETR.

        Args:
            image: PIL Image to analyze
            options: RT-DETR specific options

        Returns:
            List of standardized detection dictionaries
        """
        torch = _get_torch()

        # Get cached model
        model_dict = self._get_model(options)
        model = model_dict["model"]
        processor = model_dict["processor"]
        device = model_dict["device"]

        # Prepare inputs
        inputs = processor(images=[image], return_tensors="pt")
        if device.type != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        if device.type != "cpu":
            target_sizes = target_sizes.to(device)

        results = processor.post_process_object_detection(
            outputs, threshold=options.post_process_threshold, target_sizes=target_sizes
        )[0]

        # Convert to standardized format
        detections = []
        for i in range(len(results["scores"])):
            score = results["scores"][i].item()

            # Apply confidence threshold
            if score < options.confidence:
                continue

            label = results["labels"][i].item()
            box = results["boxes"][i].tolist()

            # Get label text from model config
            if hasattr(model.config, "id2label") and label in model.config.id2label:
                label_text = model.config.id2label[label]
            else:
                label_text = str(label)

            # Map to checkbox state
            is_checked, state = self._map_label_to_state(label_text, options)

            detection = {
                "bbox": tuple(box),  # (x0, y0, x1, y1)
                "class": label_text,
                "normalized_class": "checkbox",
                "is_checked": is_checked,
                "checkbox_state": state,
                "confidence": score,
                "model": options.model_repo.split("/")[-1],  # Short model name
                "source": "checkbox",
            }
            detections.append(detection)

        # Apply NMS if needed
        if options.nms_threshold > 0:
            detections = self._apply_nms(detections, options.nms_threshold)

        # Limit detections if specified
        if options.max_detections > 0 and len(detections) > options.max_detections:
            # Sort by confidence and keep top N
            detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
            detections = detections[: options.max_detections]

        return detections
