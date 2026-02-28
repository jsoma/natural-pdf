"""Options classes for checkbox detection engines."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BaseCheckboxOptions:
    """Base options shared by all checkbox detection engines."""

    confidence: float = 0.3
    resolution: int = 150
    device: Optional[str] = "cpu"
    classify: bool = True
    classify_with: Optional[Any] = None  # Judge instance
    reject_with_text: bool = True
    existing: str = "replace"
    limit: Optional[int] = None
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorCheckboxOptions(BaseCheckboxOptions):
    """Options for vector (native PDF rect) checkbox detection."""

    min_size: float = 6.0  # PDF points
    max_size: float = 25.0
    max_aspect_ratio: float = 1.5
    require_stroke: bool = False


@dataclass
class OnnxCheckboxOptions(BaseCheckboxOptions):
    """Options for generic YOLO-format ONNX checkbox detection."""

    model_path: Optional[str] = None  # Local .onnx file
    model_repo: Optional[str] = None  # HuggingFace repo
    model_file: Optional[str] = None  # File within repo
    model_revision: Optional[str] = None  # HuggingFace repo revision/tag
    input_size: int = 640
    nms_threshold: float = 0.45
    class_names: Optional[List[str]] = None
    checkbox_class_indices: Optional[List[int]] = None

    # SAHI tiling — improves detection of small checkboxes on full pages.
    # Enabled by default; tiles the image into overlapping crops and merges.
    sahi_enabled: bool = True
    sahi_overlap: float = 0.2  # Fraction of tile overlap (0.0-0.5)
    sahi_min_image_ratio: float = 1.0  # Tile whenever image exceeds model input size


@dataclass
class DefaultCheckboxOptions(OnnxCheckboxOptions):
    """Options for the default jsoma/checkbox-detector YOLO12n model."""

    model_repo: str = "jsoma/checkbox-detector"
    model_file: str = "checkbox_yolo12n.onnx"
    model_revision: Optional[str] = "v1"
    input_size: int = 1024


@dataclass
class VLMCheckboxOptions(BaseCheckboxOptions):
    """Options for VLM-based checkbox detection."""

    model_name: str = "gemini-2.0-flash"
    client: Optional[Any] = None
    languages: Optional[List[str]] = None
