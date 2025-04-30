# ocr_options.py
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# Assume logger is configured elsewhere or remove if not needed globally


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

    # def __post_init__(self):
    #     logger.debug(f"Initialized EasyOCROptions: {self}")


# --- PaddleOCR Specific Options ---
@dataclass
class PaddleOCROptions(BaseOCROptions):
    """Specific options for the PaddleOCR engine."""

    # General
    use_gpu: Optional[bool] = None
    gpu_mem: int = 8000  # Default from Paddle documentation
    gpu_mem: int = 8000  # Default from Paddle documentation
    ir_optim: bool = True
    use_tensorrt: bool = False
    min_subgraph_size: int = 15
    precision: str = "fp32"
    enable_mkldnn: bool = False
    cpu_threads: int = 10
    use_fp16: bool = False
    show_log: bool = False
    use_onnx: bool = False
    use_zero_copy_run: bool = False

    # Detection
    det: bool = True
    det_algorithm: str = "DB"
    show_log: bool = False
    use_onnx: bool = False
    use_zero_copy_run: bool = False

    # Detection
    det: bool = True
    det_algorithm: str = "DB"
    det_model_dir: Optional[str] = None
    det_limit_side_len: int = 960  # Corresponds to det_max_side_len
    # DB specific
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.5
    det_db_unclip_ratio: float = 2.0
    # EAST specific
    det_east_score_thresh: float = 0.8
    det_east_cover_thresh: float = 0.1
    det_east_nms_thresh: float = 0.2

    # Recognition
    rec: bool = True
    rec_algorithm: str = "CRNN"
    det_limit_side_len: int = 960  # Corresponds to det_max_side_len
    # DB specific
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.5
    det_db_unclip_ratio: float = 2.0
    # EAST specific
    det_east_score_thresh: float = 0.8
    det_east_cover_thresh: float = 0.1
    det_east_nms_thresh: float = 0.2

    # Recognition
    rec: bool = True
    rec_algorithm: str = "CRNN"
    rec_model_dir: Optional[str] = None
    rec_image_shape: str = "3, 32, 320"  # Kept as string per Paddle examples
    rec_batch_num: int = 30  # Default from Paddle documentation
    rec_image_shape: str = "3, 32, 320"  # Kept as string per Paddle examples
    rec_batch_num: int = 30  # Default from Paddle documentation
    max_text_length: int = 25
    rec_char_dict_path: Optional[str] = None  # Path to char dictionary file
    rec_char_dict_path: Optional[str] = None  # Path to char dictionary file
    use_space_char: bool = True
    drop_score: float = 0.5

    # Classification
    cls: Optional[bool] = None  # Often inferred from use_angle_cls
    use_angle_cls: bool = False  # Default from Paddle documentation
    cls_model_dir: Optional[str] = None
    cls_image_shape: str = "3, 48, 192"  # Kept as string per Paddle examples
    label_list: List[str] = field(default_factory=lambda: ["0", "180"])  # Default from Paddle doc
    cls_batch_num: int = 30

    # Classification
    cls: Optional[bool] = None  # Often inferred from use_angle_cls
    use_angle_cls: bool = False  # Default from Paddle documentation
    cls_model_dir: Optional[str] = None
    cls_image_shape: str = "3, 48, 192"  # Kept as string per Paddle examples
    label_list: List[str] = field(default_factory=lambda: ["0", "180"])  # Default from Paddle doc
    cls_batch_num: int = 30

    def __post_init__(self):
        pass

    #     if self.use_gpu is None:
    #         if self.device and "cuda" in self.device.lower():
    #             self.use_gpu = True
    #         else:
    #             self.use_gpu = False
    #     # logger.debug(f"Initialized PaddleOCROptions: {self}")


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
