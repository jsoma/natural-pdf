# layout_options.py
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from natural_pdf.utils.option_validation import (
    validate_confidence,
    validate_device,
    validate_non_empty_string,
    validate_positive_int,
)

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
    See: https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-StructureV3.html
    """

    # Model paths and names
    layout_detection_model_name: Optional[str] = None
    layout_detection_model_dir: Optional[str] = None
    layout_threshold: Optional[float] = None
    layout_nms: Optional[bool] = None
    layout_unclip_ratio: Optional[float] = None
    layout_merge_bboxes_mode: Optional[str] = None
    chart_recognition_model_name: Optional[str] = None
    chart_recognition_model_dir: Optional[str] = None
    chart_recognition_batch_size: Optional[int] = None
    region_detection_model_name: Optional[str] = None
    region_detection_model_dir: Optional[str] = None
    doc_orientation_classify_model_name: Optional[str] = None
    doc_orientation_classify_model_dir: Optional[str] = None
    doc_unwarping_model_name: Optional[str] = None
    doc_unwarping_model_dir: Optional[str] = None
    text_detection_model_name: Optional[str] = None
    text_detection_model_dir: Optional[str] = None
    text_det_limit_side_len: Optional[int] = None
    text_det_limit_type: Optional[str] = None
    text_det_thresh: Optional[float] = None
    text_det_box_thresh: Optional[float] = None
    text_det_unclip_ratio: Optional[float] = None
    textline_orientation_model_name: Optional[str] = None
    textline_orientation_model_dir: Optional[str] = None
    textline_orientation_batch_size: Optional[int] = None
    text_recognition_model_name: Optional[str] = None
    text_recognition_model_dir: Optional[str] = None
    text_recognition_batch_size: Optional[int] = None
    text_rec_score_thresh: Optional[float] = None
    table_classification_model_name: Optional[str] = None
    table_classification_model_dir: Optional[str] = None
    wired_table_structure_recognition_model_name: Optional[str] = None
    wired_table_structure_recognition_model_dir: Optional[str] = None
    wireless_table_structure_recognition_model_name: Optional[str] = None
    wireless_table_structure_recognition_model_dir: Optional[str] = None
    wired_table_cells_detection_model_name: Optional[str] = None
    wired_table_cells_detection_model_dir: Optional[str] = None
    wireless_table_cells_detection_model_name: Optional[str] = None
    wireless_table_cells_detection_model_dir: Optional[str] = None
    seal_text_detection_model_name: Optional[str] = None
    seal_text_detection_model_dir: Optional[str] = None
    seal_det_limit_side_len: Optional[int] = None
    seal_det_limit_type: Optional[str] = None
    seal_det_thresh: Optional[float] = None
    seal_det_box_thresh: Optional[float] = None
    seal_det_unclip_ratio: Optional[float] = None
    seal_text_recognition_model_name: Optional[str] = None
    seal_text_recognition_model_dir: Optional[str] = None
    seal_text_recognition_batch_size: Optional[int] = None
    seal_rec_score_thresh: Optional[float] = None
    formula_recognition_model_name: Optional[str] = None
    formula_recognition_model_dir: Optional[str] = None
    formula_recognition_batch_size: Optional[int] = None
    # Module usage flags
    use_doc_orientation_classify: Optional[bool] = True
    use_doc_unwarping: Optional[bool] = True
    use_textline_orientation: Optional[bool] = True
    use_seal_recognition: Optional[bool] = False
    use_table_recognition: Optional[bool] = True
    use_formula_recognition: Optional[bool] = False
    use_chart_recognition: Optional[bool] = True
    use_region_detection: Optional[bool] = True
    # General parameters
    device: Optional[str] = None
    enable_hpi: Optional[bool] = None
    use_tensorrt: Optional[bool] = None
    precision: Optional[str] = None
    enable_mkldnn: Optional[bool] = False
    cpu_threads: Optional[int] = None
    paddlex_config: Optional[str] = None
    lang: Optional[str] = None  # For English model selection
    verbose: bool = False  # Verbose logging for the detector class
    create_cells: Optional[bool] = True

    def __post_init__(self):
        """Validate Paddle layout options."""
        super().__post_init__()
        # Validate threshold parameters if set
        if self.layout_threshold is not None:
            self.layout_threshold = validate_confidence(
                self.layout_threshold, "layout_threshold", "PaddleLayoutOptions"
            )
        if self.text_det_thresh is not None:
            self.text_det_thresh = validate_confidence(
                self.text_det_thresh, "text_det_thresh", "PaddleLayoutOptions"
            )
        if self.text_det_box_thresh is not None:
            self.text_det_box_thresh = validate_confidence(
                self.text_det_box_thresh, "text_det_box_thresh", "PaddleLayoutOptions"
            )
        if self.text_rec_score_thresh is not None:
            self.text_rec_score_thresh = validate_confidence(
                self.text_rec_score_thresh, "text_rec_score_thresh", "PaddleLayoutOptions"
            )
        if self.seal_det_thresh is not None:
            self.seal_det_thresh = validate_confidence(
                self.seal_det_thresh, "seal_det_thresh", "PaddleLayoutOptions"
            )
        if self.seal_det_box_thresh is not None:
            self.seal_det_box_thresh = validate_confidence(
                self.seal_det_box_thresh, "seal_det_box_thresh", "PaddleLayoutOptions"
            )
        if self.seal_rec_score_thresh is not None:
            self.seal_rec_score_thresh = validate_confidence(
                self.seal_rec_score_thresh, "seal_rec_score_thresh", "PaddleLayoutOptions"
            )
        # Validate batch sizes if set
        if self.chart_recognition_batch_size is not None:
            self.chart_recognition_batch_size = validate_positive_int(
                self.chart_recognition_batch_size,
                "chart_recognition_batch_size",
                "PaddleLayoutOptions",
            )
        if self.textline_orientation_batch_size is not None:
            self.textline_orientation_batch_size = validate_positive_int(
                self.textline_orientation_batch_size,
                "textline_orientation_batch_size",
                "PaddleLayoutOptions",
            )
        if self.text_recognition_batch_size is not None:
            self.text_recognition_batch_size = validate_positive_int(
                self.text_recognition_batch_size,
                "text_recognition_batch_size",
                "PaddleLayoutOptions",
            )
        if self.seal_text_recognition_batch_size is not None:
            self.seal_text_recognition_batch_size = validate_positive_int(
                self.seal_text_recognition_batch_size,
                "seal_text_recognition_batch_size",
                "PaddleLayoutOptions",
            )
        if self.formula_recognition_batch_size is not None:
            self.formula_recognition_batch_size = validate_positive_int(
                self.formula_recognition_batch_size,
                "formula_recognition_batch_size",
                "PaddleLayoutOptions",
            )


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

    def __post_init__(self):
        """Validate Docling layout options."""
        super().__post_init__()
        # Docling has minimal configuration - validation reserved for future expansion


# --- Gemini Specific Options ---
@dataclass
class GeminiLayoutOptions(BaseLayoutOptions):
    """Options specific to Gemini-based layout detection (using OpenAI compatibility)."""

    model_name: str = "gemini-2.0-flash"
    client: Optional[Any] = None  # Allow passing a pre-configured client
    # Removed: prompt_template, temperature, top_p, max_output_tokens
    # These are typically passed directly to the chat completion call or via extra_args

    def __post_init__(self):
        """Validate Gemini layout options."""
        super().__post_init__()
        self.model_name = validate_non_empty_string(
            self.model_name, "model_name", "GeminiLayoutOptions", default="gemini-2.0-flash"
        )


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
