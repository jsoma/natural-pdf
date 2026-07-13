# ocr_options.py
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
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

    def _init_key(self) -> Optional[str]:
        """Return a hashable string of init-time fields for engine caching.

        The cache key determines when a cached engine instance can be reused.
        Override in subclasses to include fields that affect model initialization
        (not runtime inference params like thresholds or batch sizes).
        """
        return _canonical_option_key({"extra_args": self.extra_args})

    def _cache_key(self) -> Optional[str]:
        """Return a stable string of fields that can affect OCR output.

        This is intentionally broader than ``_init_key()``. Engine instances
        should be reused across runtime-only changes, but cached OCR results
        must be invalidated when thresholds, batching, generation settings, or
        provider-specific extra args can change the emitted text or boxes.
        """
        try:
            data = asdict(self) if is_dataclass(self) else dict(getattr(self, "__dict__", {}))
        except Exception:
            # Dataclasses.asdict deep-copies nested values. An option object
            # that cannot be copied cannot prove a stable cache identity.
            return None
        normalized = _json_safe(data)
        if normalized is _UNCACHEABLE:
            return None
        payload = {
            "class": self.__class__.__qualname__,
            "options": normalized,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


_UNCACHEABLE = object()


def _json_safe(value: Any) -> Any:
    """Normalize canonical option values, or return an uncacheable sentinel.

    Option values can be forwarded to model constructors and inference calls.
    ``repr()`` is not an identity: custom mutable objects can keep a stable
    representation while changing OCR output.  Callers must therefore bypass
    result and engine caches when a value cannot be represented canonically.
    """

    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            return _UNCACHEABLE
        normalized: Dict[str, Any] = {}
        for key in sorted(value):
            item = _json_safe(value[key])
            if item is _UNCACHEABLE:
                return _UNCACHEABLE
            normalized[key] = item
        return normalized
    if isinstance(value, (list, tuple)):
        normalized_items = []
        for item in value:
            normalized = _json_safe(item)
            if normalized is _UNCACHEABLE:
                return _UNCACHEABLE
            normalized_items.append(normalized)
        return normalized_items
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _UNCACHEABLE
    return _UNCACHEABLE


def _canonical_option_key(value: Any) -> Optional[str]:
    """Return a canonical JSON identity for option fields, if one exists."""

    normalized = _json_safe(value)
    if normalized is _UNCACHEABLE:
        return None
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


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

    def _init_key(self) -> Optional[str]:
        return _canonical_option_key(
            {
                "recog_network": self.recog_network,
                "detect_network": self.detect_network,
                "quantize": self.quantize,
                "cudnn_benchmark": self.cudnn_benchmark,
                "model_storage_directory": self.model_storage_directory,
                "user_network_directory": self.user_network_directory,
                "download_enabled": self.download_enabled,
                "detector": self.detector,
                "recognizer": self.recognizer,
            }
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

    def _init_key(self) -> Optional[str]:
        # PaddleOCR's pipeline accepts every declared field below at
        # construction time (including ``lang`` and ``device``).  Keep this
        # identity in lockstep with the constructor rather than maintaining a
        # fragile hand-picked string: reusing a pipeline built for a different
        # language/device/model setting produces incorrect results.
        constructor_fields = {
            name: getattr(self, name)
            for name in (
                "doc_orientation_classify_model_name",
                "doc_orientation_classify_model_dir",
                "doc_unwarping_model_name",
                "doc_unwarping_model_dir",
                "text_detection_model_name",
                "text_detection_model_dir",
                "textline_orientation_model_name",
                "textline_orientation_model_dir",
                "text_recognition_model_name",
                "text_recognition_model_dir",
                "use_doc_orientation_classify",
                "use_doc_unwarping",
                "use_textline_orientation",
                "textline_orientation_batch_size",
                "text_recognition_batch_size",
                "text_det_limit_side_len",
                "text_det_limit_type",
                "text_det_thresh",
                "text_det_box_thresh",
                "text_det_unclip_ratio",
                "text_det_input_shape",
                "text_rec_score_thresh",
                "text_rec_input_shape",
                "lang",
                "ocr_version",
                "device",
                "enable_hpi",
                "use_tensorrt",
                "precision",
                "enable_mkldnn",
                "cpu_threads",
                "paddlex_config",
            )
        }
        return _canonical_option_key(constructor_fields)


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

    # Predict-time VLM generation parameters
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None

    def _init_key(self) -> Optional[str]:
        # ``extra_args`` is forwarded to PaddleOCRVL's constructor, so it is
        # part of the model identity rather than merely an inference setting.
        constructor_fields = {
            "pipeline_version": self.pipeline_version,
            "use_layout_detection": self.use_layout_detection,
            "use_chart_recognition": self.use_chart_recognition,
            "use_seal_recognition": self.use_seal_recognition,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "format_block_content": self.format_block_content,
            "extra_args": self.extra_args,
        }
        return _canonical_option_key(constructor_fields)


# --- Surya Specific Options ---
@dataclass
class SuryaOCROptions(BaseOCROptions):
    """Specific options for the Surya OCR engine."""

    strip_math: bool = True
    """Remove ``<math>…</math>`` tags and their content from OCR output.
    HTML formatting tags (``<b>``, etc.) are always stripped."""


# --- Chandra Specific Options ---
@dataclass
class ChandraOCROptions(BaseOCROptions):
    """Specific options for the Chandra OCR engine (VLM-based document OCR).

    Install: ``pip install chandra-ocr[hf]``
    """

    method: str = "hf"
    """Inference backend: ``"hf"`` for local HuggingFace, ``"vllm"`` for remote vLLM server."""

    vllm_url: Optional[str] = None
    """URL for the vLLM server when ``method="vllm"``."""

    max_output_tokens: int = 12384
    """Maximum number of tokens to generate per page."""

    def _init_key(self) -> Optional[str]:
        return _canonical_option_key({"method": self.method, "vllm_url": self.vllm_url})


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

    def _init_key(self) -> Optional[str]:
        return _canonical_option_key(
            {
                "det_arch": self.det_arch,
                "reco_arch": self.reco_arch,
                "pretrained": self.pretrained,
                "assume_straight_pages": self.assume_straight_pages,
                "export_as_straight_boxes": self.export_as_straight_boxes,
                "symmetric_pad": self.symmetric_pad,
                "preserve_aspect_ratio": self.preserve_aspect_ratio,
                "batch_size": self.batch_size,
                "bin_thresh": self.bin_thresh,
                "box_thresh": self.box_thresh,
                "use_orientation_predictor": self.use_orientation_predictor,
            }
        )


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

    def _init_key(self) -> Optional[str]:
        return _canonical_option_key(
            {
                "det_model_type": self.det_model_type,
                "rec_model_type": self.rec_model_type,
                "config_path": self.config_path,
            }
        )


# --- Union type for type hinting ---
OCROptions = Union[
    EasyOCROptions,
    PaddleOCROptions,
    PaddleOCRVLOptions,
    SuryaOCROptions,
    ChandraOCROptions,
    DoctrOCROptions,
    RapidOCROptions,
    BaseOCROptions,
]
