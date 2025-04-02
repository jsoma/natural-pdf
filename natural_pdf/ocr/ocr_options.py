# ocr_options.py
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# Assume logger is configured elsewhere or remove if not needed globally

# --- Base Options ---
@dataclass
class BaseOCROptions:
    """Base class for OCR engine options."""
    languages: List[str] = field(default_factory=lambda: ['en'])
    min_confidence: float = 0.5
    device: Optional[str] = 'cpu' # Suggestion, actual device usage depends on engine impl.
    extra_args: Dict[str, Any] = field(default_factory=dict)

# --- EasyOCR Specific Options ---
@dataclass
class EasyOCROptions(BaseOCROptions):
    """Specific options for the EasyOCR engine."""
    model_storage_directory: Optional[str] = None
    user_network_directory: Optional[str] = None
    recog_network: str = 'english_g2'
    detect_network: str = 'craft'
    download_enabled: bool = True
    detector: bool = True
    recognizer: bool = True
    verbose: bool = True
    quantize: bool = True
    cudnn_benchmark: bool = False
    detail: int = 1
    decoder: str = 'greedy'
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
    output_format: str = 'standard'

    # def __post_init__(self):
    #     logger.debug(f"Initialized EasyOCROptions: {self}")


# --- PaddleOCR Specific Options ---
@dataclass
class PaddleOCROptions(BaseOCROptions):
    """Specific options for the PaddleOCR engine."""
    use_angle_cls: bool = True
    use_gpu: Optional[bool] = None
    gpu_mem: int = 500
    ir_optim: bool = True
    use_tensorrt: bool = False
    min_subgraph_size: int = 15
    precision: str = 'fp32'
    enable_mkldnn: bool = False
    cpu_threads: int = 10
    use_fp16: bool = False
    det_model_dir: Optional[str] = None
    rec_model_dir: Optional[str] = None
    cls_model_dir: Optional[str] = None
    det_limit_side_len: int = 960
    rec_batch_num: int = 6
    max_text_length: int = 25
    use_space_char: bool = True
    drop_score: float = 0.5
    show_log: bool = False
    use_onnx: bool = False
    det: bool = True
    rec: bool = True
    cls: Optional[bool] = None

    def __post_init__(self):
        if self.use_gpu is None:
            if self.device and 'cuda' in self.device.lower():
                self.use_gpu = True
            else:
                self.use_gpu = False
        # logger.debug(f"Initialized PaddleOCROptions: {self}")

# --- Surya Specific Options ---
@dataclass
class SuryaOCROptions(BaseOCROptions):
    """Specific options for the Surya OCR engine."""
    # Currently, Surya example shows languages passed at prediction time.
    # Add fields here if Surya's RecognitionPredictor or DetectionPredictor
    # constructors accept relevant arguments (e.g., model paths, device settings).
    # For now, it primarily uses the base options like 'languages' and 'min_confidence'.
    # Configuration like batch sizes are often set via environment variables for Surya.
    pass


# --- Union type for type hinting ---
OCROptions = Union[EasyOCROptions, PaddleOCROptions, SuryaOCROptions, BaseOCROptions]

