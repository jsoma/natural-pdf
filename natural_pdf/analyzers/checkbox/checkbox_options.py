"""Options classes for checkbox detection engines."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckboxOptions:
    """Base options for checkbox detection engines."""

    confidence: float = 0.02  # Default very low confidence for DETR models
    resolution: int = 150  # DPI for rendering pages to images
    device: Optional[str] = "cpu"  # Preferred device ('cpu', 'cuda', 'mps', etc.)

    # Label mapping from model outputs to standard states
    label_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            # Common mappings
            "checkbox": "unchecked",
            "checked_checkbox": "checked",
            "checkbox_checked": "checked",
            "unchecked_checkbox": "unchecked",
            "checkbox_unchecked": "unchecked",
            # Numeric mappings
            "0": "unchecked",
            "1": "checked",
            # Descriptive mappings
            "empty": "unchecked",
            "tick": "checked",
            "filled": "checked",
            "blank": "unchecked",
        }
    )

    # Non-max suppression parameters
    nms_threshold: float = 0.1  # IoU threshold for overlapping boxes (low for checkboxes)

    # Text filtering
    reject_with_text: bool = (
        True  # Reject detections that contain text (checkboxes should be empty)
    )

    # Extra arguments for engine-specific parameters
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RTDETRCheckboxOptions(CheckboxOptions):
    """Options specific to RT-DETR checkbox detection models."""

    model_repo: str = "wendys-llc/rtdetr-v2-r50-chkbx"  # Default checkbox model
    model_revision: Optional[str] = None  # Specific model revision
    image_processor_repo: Optional[str] = None  # Override image processor if needed

    # RT-DETR specific parameters
    max_detections: int = 100  # Maximum number of detections per image
    post_process_threshold: float = 0.0  # Threshold for post-processing (0.0 for all)
