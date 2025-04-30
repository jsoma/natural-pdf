# layout_options.py
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


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


# --- YOLO Specific Options ---
@dataclass
class YOLOLayoutOptions(BaseLayoutOptions):
    """Options specific to YOLO-based layout detection."""

    model_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    model_file: str = "doclayout_yolo_docstructbench_imgsz1024.pt"
    image_size: int = 1024  # Input image size for the model


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


# --- Paddle Specific Options ---
@dataclass
class PaddleLayoutOptions(BaseLayoutOptions):
    """Options specific to PaddlePaddle PP-Structure layout detection."""

    lang: str = "en"  # Language ('en', 'ch', etc.)
    use_angle_cls: bool = False  # Use text angle classification?
    enable_table: bool = True  # Enable table structure detection?
    show_log: bool = False  # Show Paddle internal logs?
    detect_text: bool = True  # Also detect raw text boxes using PaddleOCR?
    verbose: bool = False  # Verbose logging for the detector class


# --- Surya Specific Options ---
@dataclass
class SuryaLayoutOptions(BaseLayoutOptions):
    """Options specific to Surya layout detection."""

    model_name: str = "default"  # Placeholder if different models become available
    recognize_table_structure: bool = True  # Automatically run table structure recognition?


# --- Docling Specific Options ---
@dataclass
class DoclingLayoutOptions(BaseLayoutOptions):
    """Options specific to Docling layout detection."""

    # Pass kwargs directly to Docling's DocumentConverter via extra_args
    # Common examples shown here for documentation, add others as needed to extra_args
    # model_name: str = "ds4sd/SmolDocling-256M-preview" # Example model (pass via extra_args)
    # prompt_text: Optional[str] = None # Optional prompt (pass via extra_args)
    verbose: bool = False  # Verbose logging for the detector class
    # Other kwargs like 'device', 'batch_size' can go in extra_args


# --- Gemini Specific Options ---
@dataclass
class GeminiLayoutOptions(BaseLayoutOptions):
    """Options specific to Gemini-based layout detection (using OpenAI compatibility)."""

    model_name: str = "gemini-2.0-flash"
    client: Optional[Any] = None  # Allow passing a pre-configured client
    # Removed: prompt_template, temperature, top_p, max_output_tokens
    # These are typically passed directly to the chat completion call or via extra_args


# --- Union Type ---
LayoutOptions = Union[
    YOLOLayoutOptions,
    TATRLayoutOptions,
    PaddleLayoutOptions,
    SuryaLayoutOptions,
    DoclingLayoutOptions,
    GeminiLayoutOptions,
    BaseLayoutOptions,  # Include base for typing flexibility
]
