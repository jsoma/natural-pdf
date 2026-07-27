# layout_options.py
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from typing_extensions import TypedDict

from natural_pdf.utils.option_validation import (
    validate_confidence,
    validate_device,
    validate_non_empty_string,
    validate_positive_int,
)

logger = logging.getLogger(__name__)


# --- LayoutDetection TypedDict ---
# Functional form needed because "class" is a reserved word.
LayoutDetection = TypedDict(
    "LayoutDetection",
    {
        "bbox": Tuple,
        "class": str,
        "confidence": float,
        "normalized_class": str,
        "canonical_type": str,
        "source": str,
        "model": str,
    },
    total=False,
)


# --- DetectionContext ---
@dataclass
class DetectionContext:
    """Context passed to detect() — not user-facing."""

    layout_host: Any = None  # Page reference (used by Surya for high-res crops)
    img_scale_x: float = 1.0  # Image-to-PDF X scale factor
    img_scale_y: float = 1.0  # Image-to-PDF Y scale factor


# --- Base Layout Options ---
@dataclass
class BaseLayoutOptions:
    """Base options for layout detection engines."""

    confidence: float = 0.5  # Minimum confidence threshold for detections
    classes: Optional[List[str]] = None  # Specific classes to detect (None for all)
    exclude_classes: Optional[List[str]] = None  # Classes to exclude
    device: Optional[str] = "cpu"  # Preferred device ('cpu', 'cuda', 'mps', etc.)
    extra_args: Dict[str, Any] = field(
        default_factory=dict
    )  # For engine-specific args not yet fields

    def __post_init__(self):
        """Validate base layout options."""
        self.confidence = validate_confidence(
            self.confidence, "confidence", self.__class__.__name__
        )
        self.device = validate_device(self.device, "device", self.__class__.__name__)


# --- YOLO Specific Options ---
@dataclass
class YOLOLayoutOptions(BaseLayoutOptions):
    """Options specific to YOLO-based layout detection."""

    model_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    model_file: str = "doclayout_yolo_docstructbench_imgsz1024.pt"
    image_size: int = 1024  # Input image size for the model

    def __post_init__(self):
        """Validate YOLO layout options."""
        super().__post_init__()
        self.image_size = validate_positive_int(
            self.image_size, "image_size", "YOLOLayoutOptions", default=1024
        )


# --- TATR Specific Options ---
@dataclass
class TATRLayoutOptions(BaseLayoutOptions):
    """Options specific to Table Transformer (TATR) layout detection."""

    # Which models to use (can be local paths or HF identifiers)
    detection_model: str = "microsoft/table-transformer-detection"
    structure_model: str = "microsoft/table-transformer-structure-recognition-v1.1-all"
    # Input image resizing parameters
    max_detection_size: int = 800
    max_structure_size: int = 1000
    # Whether to create cell regions (can be slow)
    create_cells: bool = True
    # Image enhancement options
    enhance_contrast: float = 1.5  # Contrast enhancement factor (1.0 = no change)
    # Special thresholds for specific elements
    column_threshold: Optional[float] = (
        None  # Lower threshold for columns (default: confidence * 0.8)
    )

    def __post_init__(self):
        """Validate TATR layout options."""
        super().__post_init__()
        self.max_detection_size = validate_positive_int(
            self.max_detection_size, "max_detection_size", "TATRLayoutOptions", default=800
        )
        self.max_structure_size = validate_positive_int(
            self.max_structure_size, "max_structure_size", "TATRLayoutOptions", default=1000
        )
        if self.column_threshold is not None:
            self.column_threshold = validate_confidence(
                self.column_threshold, "column_threshold", "TATRLayoutOptions"
            )


# --- Paddle Specific Options ---
@dataclass
class PaddleLayoutOptions(BaseLayoutOptions):
    """
    Options specific to PaddlePaddle PP-StructureV3 layout detection.

    Commonly-used parameters are exposed as fields. All other PP-StructureV3
    constructor arguments (model names/dirs, thresholds, batch sizes, etc.)
    can be passed via ``extra_args`` and will be forwarded directly.

    See: https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-StructureV3.html
    """

    lang: Optional[str] = None  # For English model selection
    create_cells: Optional[bool] = True
    verbose: bool = False
    # Module usage flags (commonly toggled)
    use_doc_orientation_classify: Optional[bool] = True
    use_doc_unwarping: Optional[bool] = True
    use_textline_orientation: Optional[bool] = True
    use_table_recognition: Optional[bool] = True
    use_formula_recognition: Optional[bool] = False
    use_chart_recognition: Optional[bool] = True
    use_region_detection: Optional[bool] = True
    use_seal_recognition: Optional[bool] = False

    def __post_init__(self):
        """Validate Paddle layout options."""
        super().__post_init__()


# --- Surya Specific Options ---
@dataclass
class SuryaLayoutOptions(BaseLayoutOptions):
    """Options specific to Surya layout detection."""

    model_name: str = "default"  # Placeholder if different models become available
    recognize_table_structure: bool = True  # Automatically run table structure recognition?

    def __post_init__(self):
        """Validate Surya layout options."""
        super().__post_init__()
        # Surya has minimal configuration - validation reserved for future expansion


# --- DocLayout (PP-DocLayoutV3) Specific Options ---
@dataclass
class DocLayoutOptions(BaseLayoutOptions):
    """Options specific to PP-DocLayout-V3 layout detection.

    A lightweight (~45MB) transformers model that detects 25 document
    region classes. No PaddlePaddle framework needed.

    Install: pip install transformers torch
    """

    model_name: str = "PaddlePaddle/PP-DocLayoutV3_safetensors"

    def __post_init__(self):
        """Validate DocLayout options."""
        super().__post_init__()


# --- VLM Specific Options ---
@dataclass
class VLMLayoutOptions(BaseLayoutOptions):
    """Options specific to VLM-based layout detection (any OpenAI-compatible API or local model)."""

    model_name: str = "gemini-2.0-flash"
    client: Optional[Any] = None  # Allow passing a pre-configured client
    languages: Optional[List[str]] = None  # Language hints for VLM prompt

    def __post_init__(self):
        """Validate VLM layout options."""
        super().__post_init__()
        self.model_name = validate_non_empty_string(
            self.model_name, "model_name", "VLMLayoutOptions", default="gemini-2.0-flash"
        )


# Backward-compatible alias
GeminiLayoutOptions = VLMLayoutOptions


# --- Union Type ---
LayoutOptions = Union[
    YOLOLayoutOptions,
    TATRLayoutOptions,
    PaddleLayoutOptions,
    SuryaLayoutOptions,
    DocLayoutOptions,
    VLMLayoutOptions,
    BaseLayoutOptions,  # Include base for typing flexibility
]
