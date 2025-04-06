# layout_options.py
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union

logger = logging.getLogger(__name__)

# --- Base Layout Options ---
@dataclass
class BaseLayoutOptions:
    """Base options for layout detection engines."""
    confidence: float = 0.5 # Minimum confidence threshold for detections
    classes: Optional[List[str]] = None # Specific classes to detect (None for all)
    exclude_classes: Optional[List[str]] = None # Classes to exclude
    device: Optional[str] = 'cpu' # Preferred device ('cpu', 'cuda', 'mps', etc.)
    extra_args: Dict[str, Any] = field(default_factory=dict) # For engine-specific args not yet fields

# --- YOLO Specific Options ---
@dataclass
class YOLOLayoutOptions(BaseLayoutOptions):
    """Options specific to YOLO-based layout detection."""
    model_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    model_file: str = "doclayout_yolo_docstructbench_imgsz1024.pt"
    image_size: int = 1024 # Input image size for the model

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

# --- Paddle Specific Options ---
@dataclass
class PaddleLayoutOptions(BaseLayoutOptions):
    """Options specific to PaddlePaddle PP-Structure layout detection."""
    lang: str = "en" # Language ('en', 'ch', etc.)
    use_angle_cls: bool = False # Use text angle classification?
    enable_table: bool = True # Enable table structure detection?
    show_log: bool = False # Show Paddle internal logs?
    detect_text: bool = True # Also detect raw text boxes using PaddleOCR?
    verbose: bool = False # Verbose logging for the detector class

# --- Surya Specific Options ---
@dataclass
class SuryaLayoutOptions(BaseLayoutOptions):
    """Options specific to Surya layout detection."""
    model_name: str = "default" # Placeholder if different models become available
    recognize_table_structure: bool = True # Automatically run table structure recognition?

# --- Docling Specific Options ---
@dataclass
class DoclingLayoutOptions(BaseLayoutOptions):
    """Options specific to Docling layout detection."""
    # Pass kwargs directly to Docling's DocumentConverter via extra_args
    # Common examples shown here for documentation, add others as needed to extra_args
    # model_name: str = "ds4sd/SmolDocling-256M-preview" # Example model (pass via extra_args)
    # prompt_text: Optional[str] = None # Optional prompt (pass via extra_args)
    verbose: bool = False # Verbose logging for the detector class
    # Other kwargs like 'device', 'batch_size' can go in extra_args

# --- Union Type ---
LayoutOptions = Union[
    YOLOLayoutOptions,
    TATRLayoutOptions,
    PaddleLayoutOptions,
    SuryaLayoutOptions,
    DoclingLayoutOptions,
    BaseLayoutOptions # Include base for typing flexibility
]
