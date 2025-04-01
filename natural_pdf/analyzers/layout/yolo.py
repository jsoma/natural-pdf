# layout_detector_yolo.py
import logging
import importlib.util
import os
import tempfile
from typing import List, Dict, Any, Optional
from PIL import Image

# Assuming base class and options are importable
try:
    from .base import LayoutDetector
    from .layout_options import YOLOLayoutOptions, BaseLayoutOptions
except ImportError:
    # Placeholders if run standalone or imports fail
    class BaseLayoutOptions: pass
    class YOLOLayoutOptions(BaseLayoutOptions): pass
    class LayoutDetector:
         def __init__(self): self.logger=logging.getLogger(); self.supported_classes=set()
         def _get_model(self, options): raise NotImplementedError
         def _normalize_class_name(self, n): return n
         def validate_classes(self, c): pass
    logging.basicConfig()

logger = logging.getLogger(__name__)

# Check for dependencies
yolo_spec = importlib.util.find_spec("doclayout_yolo")
hf_spec = importlib.util.find_spec("huggingface_hub")
YOLOv10 = None
hf_hub_download = None

if yolo_spec and hf_spec:
    try:
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        logger.warning(f"Could not import YOLO dependencies: {e}")
else:
    logger.warning("doclayout_yolo or huggingface_hub not found. YOLODocLayoutDetector will not be available.")


class YOLODocLayoutDetector(LayoutDetector):
    """Document layout detector using YOLO model."""

    def __init__(self):
        super().__init__()
        self.supported_classes = {
            'title', 'plain text', 'abandon', 'figure', 'figure_caption',
            'table', 'table_caption', 'table_footnote', 'isolate_formula',
            'formula_caption'
        }

    def is_available(self) -> bool:
        """Check if dependencies are installed."""
        return YOLOv10 is not None and hf_hub_download is not None

    def _get_cache_key(self, options: YOLOLayoutOptions) -> str:
        """Generate cache key based on model repo/file and device."""
        # Ensure options is the correct type
        if not isinstance(options, YOLOLayoutOptions):
             # This shouldn't happen if called correctly, but handle defensively
             options = YOLOLayoutOptions(device=options.device) # Use base device

        device_key = str(options.device).lower()
        model_key = f"{options.model_repo.replace('/','_')}_{options.model_file}"
        return f"{self.__class__.__name__}_{device_key}_{model_key}"

    def _load_model_from_options(self, options: YOLOLayoutOptions) -> Any:
        """Load the YOLOv10 model based on options."""
        if not self.is_available():
             raise RuntimeError("YOLO dependencies (doclayout_yolo, huggingface_hub) not installed.")
        self.logger.info(f"Loading YOLO model: {options.model_repo}/{options.model_file}")
        try:
            model_path = hf_hub_download(repo_id=options.model_repo, filename=options.model_file)
            model = YOLOv10(model_path)
            self.logger.info("YOLO model loaded.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to download or load YOLO model: {e}", exc_info=True)
            raise

    def detect(self, image: Image.Image, options: BaseLayoutOptions) -> List[Dict[str, Any]]:
        """Detect layout elements in an image using YOLO."""
        if not self.is_available():
            raise RuntimeError("YOLO dependencies (doclayout_yolo, huggingface_hub) not installed.")

        # Ensure options are the correct type, falling back to defaults if base type passed
        if not isinstance(options, YOLOLayoutOptions):
             self.logger.warning("Received BaseLayoutOptions, expected YOLOLayoutOptions. Using defaults.")
             options = YOLOLayoutOptions(
                 confidence=options.confidence, classes=options.classes,
                 exclude_classes=options.exclude_classes, device=options.device,
                 extra_args=options.extra_args
             )

        # Validate classes before proceeding
        self.validate_classes(options.classes or [])
        if options.exclude_classes:
            self.validate_classes(options.exclude_classes)

        # Get the cached/loaded model
        model = self._get_model(options)

        # YOLOv10 predict method requires an image path. Save temp file.
        detections = []
        # Use a context manager for robust temp file handling
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_image_path = os.path.join(temp_dir, "temp_layout_image.png")
            try:
                self.logger.debug(f"Saving temporary image for YOLO detector to: {temp_image_path}")
                image.convert("RGB").save(temp_image_path) # Ensure RGB

                # Run model prediction
                self.logger.debug(f"Running YOLO prediction (imgsz={options.image_size}, conf={options.confidence}, device={options.device})...")
                results = model.predict(
                    temp_image_path,
                    imgsz=options.image_size,
                    conf=options.confidence,
                    device=options.device or 'cpu' # Default to cpu if None
                    # Add other predict args from options.extra_args if needed
                    # **options.extra_args
                )
                self.logger.debug(f"YOLO prediction returned {len(results)} result objects.")

                # Process results into standardized format
                img_width, img_height = image.size # Get original image size for context if needed
                for result in results:
                    if result.boxes is None: continue
                    boxes = result.boxes.xyxy
                    labels = result.boxes.cls
                    scores = result.boxes.conf
                    class_names = result.names # Dictionary mapping index to name

                    for box, label_idx_tensor, score_tensor in zip(boxes, labels, scores):
                        x_min, y_min, x_max, y_max = map(float, box.tolist())
                        label_idx = int(label_idx_tensor.item()) # Get int index
                        score = float(score_tensor.item()) # Get float score

                        if label_idx not in class_names:
                             self.logger.warning(f"Label index {label_idx} not found in model names dict. Skipping.")
                             continue
                        label_name = class_names[label_idx]
                        normalized_class = self._normalize_class_name(label_name)

                        # Apply class filtering (using normalized names)
                        if options.classes and normalized_class not in [self._normalize_class_name(c) for c in options.classes]:
                            continue
                        if options.exclude_classes and normalized_class in [self._normalize_class_name(c) for c in options.exclude_classes]:
                            continue

                        detections.append({
                            'bbox': (x_min, y_min, x_max, y_max),
                            'class': label_name,
                            'confidence': score,
                            'normalized_class': normalized_class,
                            'source': 'layout',
                            'model': 'yolo'
                        })
                self.logger.info(f"YOLO detected {len(detections)} layout elements matching criteria.")

            except Exception as e:
                self.logger.error(f"Error during YOLO detection: {e}", exc_info=True)
                raise # Re-raise the exception

        return detections
