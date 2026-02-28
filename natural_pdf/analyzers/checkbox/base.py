"""Base class for checkbox detection engines."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .checkbox_options import BaseCheckboxOptions

logger = logging.getLogger(__name__)


@dataclass
class DetectionContext:
    """Context passed to detect() — not user-facing."""

    page: Any = None
    img_scale_x: float = 1.0
    img_scale_y: float = 1.0


class CheckboxDetector(ABC):
    """Abstract base class for checkbox detection engines.

    Subclasses must implement:
    - detect(): Core checkbox detection
    - is_available(): Check if engine dependencies are installed

    All engines return a list of canonical detection dicts with:
    - bbox: (x0, y0, x1, y1)
    - coord_space: "pdf" or "image"
    - confidence: 0.0-1.0
    - label: "checkbox"
    - engine: str
    - is_checked: bool or None
    - checkbox_state: "checked" / "unchecked" / "unknown"
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def detect(
        self,
        image: Image.Image,
        options: BaseCheckboxOptions,
        context: Optional[DetectionContext] = None,
    ) -> List[Dict[str, Any]]:
        """Detect checkboxes in a given PIL Image.

        Args:
            image: PIL Image of the page/region to analyze.
            options: Options instance with configuration.
            context: Detection context with page reference and scale factors.

        Returns:
            List of canonical detection dicts.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the detector's dependencies are installed."""
        ...

    @staticmethod
    def nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        threshold: float,
    ) -> List[int]:
        """Pure-numpy NMS shared by all engines.

        Args:
            boxes: (N, 4) array of (x0, y0, x1, y1) boxes.
            scores: (N,) array of confidence scores.
            threshold: IoU threshold for suppression.

        Returns:
            List of indices to keep.
        """
        if len(boxes) == 0:
            return []

        boxes = np.asarray(boxes, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)

        x0 = boxes[:, 0]
        y0 = boxes[:, 1]
        x1 = boxes[:, 2]
        y1 = boxes[:, 3]

        areas = (x1 - x0) * (y1 - y0)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            rest = order[1:]
            xx0 = np.maximum(x0[i], x0[rest])
            yy0 = np.maximum(y0[i], y0[rest])
            xx1 = np.minimum(x1[i], x1[rest])
            yy1 = np.minimum(y1[i], y1[rest])

            inter_w = np.maximum(0.0, xx1 - xx0)
            inter_h = np.maximum(0.0, yy1 - yy0)
            inter_area = inter_w * inter_h

            union_area = areas[i] + areas[rest] - inter_area
            iou = np.where(union_area > 0, inter_area / union_area, 0.0)

            inds = np.where(iou <= threshold)[0]
            order = rest[inds]

        return keep
