# layout_detector_surya.py
import logging
import importlib.util
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from .base import LayoutDetector
from .layout_options import SuryaLayoutOptions, BaseLayoutOptions

logger = logging.getLogger(__name__)

# Check for dependency
surya_spec = importlib.util.find_spec("surya")
LayoutPredictor = None
if surya_spec:
    try:
        from surya.layout import LayoutPredictor
    except ImportError as e:
        logger.warning(f"Could not import Surya dependencies: {e}")
else:
    logger.warning("surya not found. SuryaLayoutDetector will not be available.")


class SuryaLayoutDetector(LayoutDetector):
    """Document layout detector using Surya models."""

    def __init__(self):
        super().__init__()
        self.supported_classes = {
            'text', 'pageheader', 'pagefooter', 'sectionheader',
            'table', 'tableofcontents', 'picture', 'caption',
            'heading', 'title', 'list', 'listitem', 'code',
            'textinlinemath', 'mathformula', 'form'
        }
        # Predictor instance is cached via _get_model

    def is_available(self) -> bool:
        """Check if surya is installed."""
        return LayoutPredictor is not None

    def _get_cache_key(self, options: BaseLayoutOptions) -> str:
        """Generate cache key based on model name and device."""
        if not isinstance(options, SuryaLayoutOptions):
             options = SuryaLayoutOptions(device=options.device) # Use base device

        device_key = str(options.device).lower() if options.device else 'default_device'
        # Include model_name if it affects loading, otherwise device might be enough
        model_key = options.model_name
        return f"{self.__class__.__name__}_{device_key}_{model_key}"

    def _load_model_from_options(self, options: BaseLayoutOptions) -> Any:
        """Load the Surya LayoutPredictor model."""
        if not self.is_available():
            raise RuntimeError("Surya dependency (surya-ocr) not installed.")

        if not isinstance(options, SuryaLayoutOptions):
            raise TypeError("Incorrect options type provided for Surya model loading.")

        self.logger.info(f"Loading Surya LayoutPredictor (device={options.device})...")
        try:
            # Pass device and potentially other init args from options.extra_args
            predictor_args = {'device': options.device} if options.device else {}
            predictor_args.update(options.extra_args) # Add any extra init args

            predictor = LayoutPredictor(**predictor_args)
            self.logger.info("Surya LayoutPredictor loaded.")
            return predictor
        except Exception as e:
            self.logger.error(f"Failed to load Surya LayoutPredictor: {e}", exc_info=True)
            raise

    def detect(self, image: Image.Image, options: BaseLayoutOptions) -> List[Dict[str, Any]]:
        """Detect layout elements in an image using Surya."""
        if not self.is_available():
            raise RuntimeError("Surya dependency (surya-ocr) not installed.")

        if not isinstance(options, SuryaLayoutOptions):
             self.logger.warning("Received BaseLayoutOptions, expected SuryaLayoutOptions. Using defaults.")
             options = SuryaLayoutOptions(
                 confidence=options.confidence, classes=options.classes,
                 exclude_classes=options.exclude_classes, device=options.device,
                 extra_args=options.extra_args
             )

        self.validate_classes(options.classes or [])
        if options.exclude_classes:
            self.validate_classes(options.exclude_classes)

        # Get the cached/loaded predictor instance
        layout_predictor = self._get_model(options)

        # Surya predictor takes a list of images
        input_image_list = [image.convert("RGB")] # Ensure RGB

        detections = []
        try:
            self.logger.debug("Running Surya layout prediction...")
            # Call the predictor (returns a list of LayoutResult objects)
            layout_predictions = layout_predictor(input_image_list)
            self.logger.debug(f"Surya prediction returned {len(layout_predictions)} results.")

            if not layout_predictions:
                self.logger.warning("Surya returned empty predictions list.")
                return []

            # Process results for the first (and only) image
            prediction = layout_predictions[0] # LayoutResult object

            # Prepare normalized class filters once
            normalized_classes_req = {self._normalize_class_name(c) for c in options.classes} if options.classes else None
            normalized_classes_excl = {self._normalize_class_name(c) for c in options.exclude_classes} if options.exclude_classes else set()

            for layout_box in prediction.bboxes:
                # Extract the class name and normalize it
                class_name_orig = layout_box.label
                normalized_class = self._normalize_class_name(class_name_orig)
                score = float(layout_box.confidence)

                # Apply confidence threshold
                if score < options.confidence: continue

                # Apply class filtering
                if normalized_classes_req and normalized_class not in normalized_classes_req: continue
                if normalized_class in normalized_classes_excl: continue

                # Extract bbox coordinates (Surya provides [x_min, y_min, x_max, y_max])
                x_min, y_min, x_max, y_max = map(float, layout_box.bbox)

                # Add detection
                detection_data = {
                    'bbox': (x_min, y_min, x_max, y_max),
                    'class': class_name_orig,
                    'confidence': score,
                    'normalized_class': normalized_class,
                    'source': 'layout',
                    'model': 'surya'
                    # Add polygon etc. if needed, check attributes on layout_box
                    # 'polygon': layout_box.polygon if hasattr(layout_box, 'polygon') else None,
                }
                detections.append(detection_data)

            self.logger.info(f"Surya detected {len(detections)} layout elements matching criteria.")

        except Exception as e:
            self.logger.error(f"Error during Surya layout detection: {e}", exc_info=True)
            raise

        return detections

