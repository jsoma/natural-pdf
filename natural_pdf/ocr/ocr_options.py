# ocr_options.py
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

from natural_pdf.utils.option_validation import (
    validate_confidence,
    validate_device,
    validate_positive_int,
)


# --- Base Options ---
@dataclass
class BaseOCROptions:
    """Base class for OCR engine options."""

    extra_args: Dict[str, Any] = field(default_factory=dict)


# --- EasyOCR Specific Options ---
@dataclass
class EasyOCROptions(BaseOCROptions):
    """Specific options for the EasyOCR engine."""

    model_storage_directory: Optional[str] = None
    user_network_directory: Optional[str] = None
    recog_network: str = "english_g2"
    detect_network: str = "craft"
    download_enabled: bool = True
    detector: bool = True
    recognizer: bool = True
    verbose: bool = True
    quantize: bool = True
    cudnn_benchmark: bool = False
    detail: int = 1
    decoder: str = "greedy"
    beamWidth: int = 5
    batch_size: int = 1
    workers: int = 0
    allowlist: Optional[str] = None
    blocklist: Optional[str] = None
    paragraph: bool = False
    min_size: int = 10
    contrast_ths: float = 0.1
    adjust_contrast: float = 0.5
    filter_ths: float = 0.0
    text_threshold: float = 0.7
    low_text: float = 0.4
    link_threshold: float = 0.4
    canvas_size: int = 2560
    mag_ratio: float = 1.0
    slope_ths: float = 0.1
    ycenter_ths: float = 0.5
    height_ths: float = 0.5
    width_ths: float = 0.5
    y_ths: float = 0.5
    x_ths: float = 1.0
    add_margin: float = 0.1
    output_format: str = "standard"

    def __post_init__(self):
        """Validate EasyOCR options."""
        self.batch_size = validate_positive_int(self.batch_size, "batch_size", "EasyOCROptions")
        self.workers = (
            validate_positive_int(self.workers, "workers", "EasyOCROptions", default=0)
            if self.workers != 0
            else 0
        )
        self.min_size = validate_positive_int(
            self.min_size, "min_size", "EasyOCROptions", default=10
        )
        self.beamWidth = validate_positive_int(
            self.beamWidth, "beamWidth", "EasyOCROptions", default=5
        )
        self.canvas_size = validate_positive_int(
            self.canvas_size, "canvas_size", "EasyOCROptions", default=2560
        )


# --- PaddleOCR Specific Options ---
@dataclass
class PaddleOCROptions(BaseOCROptions):
    """
    Specific options for the PaddleOCR engine, reflecting the paddleocr>=3.0.0 API.
    See: https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/OCR.html
    """

    # --- Constructor Parameters ---

    # Model paths and names
    doc_orientation_classify_model_name: Optional[str] = None
    doc_orientation_classify_model_dir: Optional[str] = None
    doc_unwarping_model_name: Optional[str] = None
    doc_unwarping_model_dir: Optional[str] = None
    text_detection_model_name: Optional[str] = None
    text_detection_model_dir: Optional[str] = None
    textline_orientation_model_name: Optional[str] = None
    textline_orientation_model_dir: Optional[str] = None
    text_recognition_model_name: Optional[str] = None
    text_recognition_model_dir: Optional[str] = None

    # Module usage flags (can be overridden at predict time)
    use_doc_orientation_classify: Optional[bool] = False
    use_doc_unwarping: Optional[bool] = False
    use_textline_orientation: Optional[bool] = False

    # Batch sizes
    textline_orientation_batch_size: Optional[int] = None
    text_recognition_batch_size: Optional[int] = None

    # Detection parameters (can be overridden at predict time)
    # https://github.com/PaddlePaddle/PaddleOCR/issues/15424
    text_det_limit_side_len: Optional[int] = 736  # WAITING FOR FIX
    text_det_limit_type: Optional[str] = "max"  # WAITING FOR FIX
    text_det_thresh: Optional[float] = None
    text_det_box_thresh: Optional[float] = None
    text_det_unclip_ratio: Optional[float] = None
    text_det_input_shape: Optional[Tuple[int, int]] = None

    # Recognition parameters (can be overridden at predict time)
    text_rec_score_thresh: Optional[float] = None
    text_rec_input_shape: Optional[Tuple[int, int, int]] = None

    # General parameters
    lang: Optional[str] = None
    ocr_version: Optional[str] = None
    device: Optional[str] = None
    enable_hpi: Optional[bool] = None
    use_tensorrt: Optional[bool] = None
    precision: Optional[str] = None
    enable_mkldnn: Optional[bool] = False  # https://github.com/PaddlePaddle/PaddleOCR/issues/15294
    # mkldnn_cache_capacity: Optional[int] = None
    cpu_threads: Optional[int] = None
    paddlex_config: Optional[str] = None

    def __post_init__(self):
        """Validate PaddleOCR options."""
        self.device = validate_device(self.device, "device", "PaddleOCROptions")
        if self.textline_orientation_batch_size is not None:
            self.textline_orientation_batch_size = validate_positive_int(
                self.textline_orientation_batch_size,
                "textline_orientation_batch_size",
                "PaddleOCROptions",
            )
        if self.text_recognition_batch_size is not None:
            self.text_recognition_batch_size = validate_positive_int(
                self.text_recognition_batch_size,
                "text_recognition_batch_size",
                "PaddleOCROptions",
            )
        if self.text_det_thresh is not None:
            self.text_det_thresh = validate_confidence(
                self.text_det_thresh, "text_det_thresh", "PaddleOCROptions"
            )
        if self.text_det_box_thresh is not None:
            self.text_det_box_thresh = validate_confidence(
                self.text_det_box_thresh, "text_det_box_thresh", "PaddleOCROptions"
            )
        if self.text_rec_score_thresh is not None:
            self.text_rec_score_thresh = validate_confidence(
                self.text_rec_score_thresh, "text_rec_score_thresh", "PaddleOCROptions"
            )


# --- PaddleOCR-VL Specific Options ---
@dataclass
class PaddleOCRVLOptions(BaseOCROptions):
    """
    Specific options for the PaddleOCR-VL engine (VLM-based document understanding).
    See: https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-ChatOCRv4.html
    """

    pipeline_version: Optional[str] = None
    use_layout_detection: Optional[bool] = None
    use_chart_recognition: Optional[bool] = None
    use_seal_recognition: Optional[bool] = None
    use_doc_orientation_classify: Optional[bool] = None
    use_doc_unwarping: Optional[bool] = None
    format_block_content: Optional[bool] = None


# --- Surya Specific Options ---
@dataclass
class SuryaOCROptions(BaseOCROptions):
    """Specific options for the Surya OCR engine."""

    # Currently, Surya example shows languages passed at prediction time.

    def __post_init__(self):
        """Validate Surya OCR options."""
        # Surya has minimal options - validation reserved for future expansion
        pass


# --- Doctr Specific Options ---
@dataclass
class DoctrOCROptions(BaseOCROptions):
    """Specific options for the doctr engine."""

    # OCR predictor options
    det_arch: str = "db_resnet50"
    reco_arch: str = "crnn_vgg16_bn"
    pretrained: bool = True
    assume_straight_pages: bool = True  # Faster if pages are straight
    export_as_straight_boxes: bool = False  # Output straight boxes even if rotated text is detected

    # Additional options from standalone predictors
    # Detection predictor options
    symmetric_pad: bool = True
    preserve_aspect_ratio: bool = True
    batch_size: int = 1

    # Postprocessing parameters
    bin_thresh: Optional[float] = None  # Default is usually 0.3
    box_thresh: Optional[float] = None  # Default is usually 0.1

    # Options for orientation predictors
    use_orientation_predictor: bool = False  # Whether to use page orientation predictor

    def __post_init__(self):
        """Validate DocTR options."""
        self.batch_size = validate_positive_int(self.batch_size, "batch_size", "DoctrOCROptions")
        if self.bin_thresh is not None:
            self.bin_thresh = validate_confidence(self.bin_thresh, "bin_thresh", "DoctrOCROptions")
        if self.box_thresh is not None:
            self.box_thresh = validate_confidence(self.box_thresh, "box_thresh", "DoctrOCROptions")


# --- RapidOCR Specific Options ---
@dataclass
class RapidOCROptions(BaseOCROptions):
    """
    Specific options for the RapidOCR engine.

    RapidOCR uses PaddleOCR models converted to ONNX format, providing
    the same accuracy with simpler installation (~15MB vs ~500MB).
    """

    # Constructor settings
    det_model_type: str = "mobile"  # "mobile" or "server"
    rec_model_type: str = "mobile"  # "mobile" or "server"
    config_path: Optional[str] = None  # Path to custom config.yaml

    # Runtime settings (passed to __call__)
    use_det: bool = True
    use_cls: bool = True
    use_rec: bool = True
    return_word_box: bool = False  # Split lines into individual word boxes
    return_single_char_box: bool = False  # Return per-character boxes
    text_score: Optional[float] = None  # Text confidence filter (default 0.5)
    box_thresh: Optional[float] = None  # Detection box threshold (default 0.5)
    unclip_ratio: Optional[float] = None  # Box expansion ratio (default 1.6)

    def __post_init__(self):
        """Validate RapidOCR options."""
        if self.text_score is not None:
            self.text_score = validate_confidence(self.text_score, "text_score", "RapidOCROptions")
        if self.box_thresh is not None:
            self.box_thresh = validate_confidence(self.box_thresh, "box_thresh", "RapidOCROptions")


# --- Union type for type hinting ---
OCROptions = Union[
    EasyOCROptions,
    PaddleOCROptions,
    PaddleOCRVLOptions,
    SuryaOCROptions,
    DoctrOCROptions,
    RapidOCROptions,
    BaseOCROptions,
]
