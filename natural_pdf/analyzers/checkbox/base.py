"""Base class for checkbox detection engines."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set

from PIL import Image

from .checkbox_options import CheckboxOptions

logger = logging.getLogger(__name__)


class CheckboxDetector(ABC):
    """Abstract base class for checkbox detection engines.

    This class defines the standard interface that all checkbox detection engines
    must implement in natural-pdf. Checkbox detectors analyze document images to
    identify checkboxes and their states (checked/unchecked).

    Subclasses must implement:
    - detect(): Core checkbox detection for a single image
    - is_available(): Check if engine dependencies are installed
    - _load_model_from_options(): Load and configure the detection model
    - _get_cache_key(): Generate cache keys for model instances

    Attributes:
        logger: Logger instance for the specific detector.
        _model_cache: Dictionary cache for loaded model instances.
    """

    def __init__(self):
        """Initialize the base checkbox detector."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self._model_cache: Dict[str, Any] = {}  # Cache for initialized models

    @abstractmethod
    def detect(self, image: Image.Image, options: CheckboxOptions) -> List[Dict[str, Any]]:
        """
        Detect checkboxes in a given PIL Image.

        Args:
            image: PIL Image of the page/region to analyze.
            options: Instance of CheckboxOptions with configuration.

        Returns:
            List of detection dictionaries with:
            - 'bbox': Tuple[float, float, float, float] - (x0, y0, x1, y1) relative to image
            - 'class': str - Original class name from model (e.g., 'checkbox', 'checked_checkbox')
            - 'normalized_class': str - Always 'checkbox'
            - 'is_checked': bool - Whether checkbox is checked
            - 'checkbox_state': str - 'checked' or 'unchecked'
            - 'confidence': float - Confidence score (0.0-1.0)
            - 'model': str - Name of the model used
            - 'source': str - Always 'checkbox'
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Check if the detector's dependencies are installed and usable.

        Returns:
            True if the detector is available, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_cache_key(self, options: CheckboxOptions) -> str:
        """
        Generate a cache key for model loading based on relevant options.

        Args:
            options: The options dataclass instance.

        Returns:
            A string cache key.
        """
        # Base key includes device, subclasses should add model specifics
        device_key = str(options.device).lower()
        return f"{self.__class__.__name__}_{device_key}"

    def _get_model(self, options: CheckboxOptions) -> Any:
        """
        Get or initialize the underlying model based on options, using caching.
        """
        cache_key = self._get_cache_key(options)
        if cache_key not in self._model_cache:
            self.logger.info(f"Loading model for cache key: {cache_key}")
            try:
                # Ensure dependencies are met before loading
                if not self.is_available():
                    raise RuntimeError(f"{self.__class__.__name__} dependencies are not met.")
                self._model_cache[cache_key] = self._load_model_from_options(options)
                self.logger.info(f"Model loaded successfully for key: {cache_key}")
            except Exception as e:
                self.logger.error(f"Failed to load model for key {cache_key}: {e}", exc_info=True)
                # Remove potentially corrupted cache entry
                self._model_cache.pop(cache_key, None)
                raise
        else:
            self.logger.debug(f"Using cached model for key: {cache_key}")
        return self._model_cache[cache_key]

    @abstractmethod
    def _load_model_from_options(self, options: CheckboxOptions) -> Any:
        """
        Load and configure the detection model based on provided options.

        Args:
            options: The options dataclass instance.

        Returns:
            The loaded model object(s).
        """
        raise NotImplementedError("Subclasses must implement _load_model_from_options")

    def _map_label_to_state(self, label: str, options: CheckboxOptions) -> tuple[bool, str]:
        """
        Map model output label to checkbox state.

        Args:
            label: Raw label from model (e.g., 'checked_checkbox', '1')
            options: Options containing label mapping

        Returns:
            Tuple of (is_checked: bool, state: str)
        """
        # Normalize label
        normalized_label = str(label).lower().strip()

        # Check mapping
        if normalized_label in options.label_mapping:
            state = options.label_mapping[normalized_label]
            is_checked = state == "checked"
            return is_checked, state

        # Default heuristic if not in mapping
        if any(term in normalized_label for term in ["checked", "tick", "filled", "1"]):
            return True, "checked"
        else:
            return False, "unchecked"

    def _apply_nms(
        self, detections: List[Dict[str, Any]], iou_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Apply non-maximum suppression to remove overlapping detections.
        For checkboxes, we reject ANY meaningful overlap.

        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression (ignored for checkboxes - we use stricter rules)

        Returns:
            Filtered list of detections
        """
        if not detections:
            return detections

        # Sort by confidence (descending), then by area (ascending) to prefer smaller boxes
        def sort_key(det):
            bbox = det["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            return (-det["confidence"], area)

        sorted_detections = sorted(detections, key=sort_key)

        keep = []
        for i, det in enumerate(sorted_detections):
            should_keep = True
            det_bbox = det["bbox"]

            for kept_det in keep:
                kept_bbox = kept_det["bbox"]

                # Check for ANY overlap at all
                if self._boxes_overlap(det_bbox, kept_bbox):
                    should_keep = False
                    logger.debug(f"Rejecting box {det_bbox} due to overlap with {kept_bbox}")
                    break

            if should_keep:
                keep.append(det)
                logger.debug(f"Keeping box {det_bbox} with confidence {det['confidence']}")

        logger.info(f"NMS: Reduced {len(detections)} detections to {len(keep)}")
        return keep

    def _boxes_overlap(self, box1: tuple, box2: tuple) -> bool:
        """Check if two boxes have any overlap at all."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Check if boxes are separated
        if x1_max <= x2_min or x2_max <= x1_min:
            return False
        if y1_max <= y2_min or y2_max <= y1_min:
            return False

        # If we get here, boxes overlap
        return True

    def _compute_intersection_ratio(self, box1: tuple, box2: tuple) -> float:
        """
        Compute intersection ratio relative to the smaller box.
        This is more aggressive than IoU for checkbox detection.
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

        # Areas of both boxes
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        # Ratio relative to smaller box
        smaller_area = min(area1, area2)
        if smaller_area == 0:
            return 0.0

        return inter_area / smaller_area

    def _compute_iou(self, box1: tuple, box2: tuple) -> float:
        """Compute IoU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def __del__(self):
        """Cleanup resources."""
        self.logger.info(f"Cleaning up {self.__class__.__name__} resources.")
        self._model_cache.clear()
