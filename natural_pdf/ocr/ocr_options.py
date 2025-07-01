# ocr_options.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


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
        pass


# --- Surya Specific Options ---
@dataclass
class SuryaOCROptions(BaseOCROptions):
    """Specific options for the Surya OCR engine."""

    # Currently, Surya example shows languages passed at prediction time.
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


# --- Union type for type hinting ---
OCROptions = Union[
    EasyOCROptions, PaddleOCROptions, SuryaOCROptions, DoctrOCROptions, BaseOCROptions
]
