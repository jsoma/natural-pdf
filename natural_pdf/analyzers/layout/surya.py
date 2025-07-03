# layout_detector_surya.py
import copy
import importlib.util
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .base import LayoutDetector
from .layout_options import BaseLayoutOptions, SuryaLayoutOptions

logger = logging.getLogger(__name__)

# Check for dependencies
surya_spec = importlib.util.find_spec("surya")
LayoutPredictor = None
TableRecPredictor = None

if surya_spec:
    try:
        from surya.common.util import expand_bbox, rescale_bbox
        from surya.layout import LayoutPredictor
        from surya.table_rec import TableRecPredictor
    except ImportError as e:
        logger.warning(f"Could not import Surya dependencies (layout and/or table_rec): {e}")
else:
    logger.warning("surya not found. SuryaLayoutDetector will not be available.")


class SuryaLayoutDetector(LayoutDetector):
    """Document layout and table structure detector using Surya models."""

    def __init__(self):
        super().__init__()
        self.supported_classes = {
            "text",
            "pageheader",
            "pagefooter",
            "sectionheader",
            "table",
            "tableofcontents",
            "picture",
            "caption",
            "heading",
            "title",
            "list",
            "listitem",
            "code",
            "textinlinemath",
            "mathformula",
            "form",
            "table-row",
            "table-column",
        }
        self._page_ref = None  # To store page reference from options

    def is_available(self) -> bool:
        return LayoutPredictor is not None and TableRecPredictor is not None

    def _get_cache_key(self, options: BaseLayoutOptions) -> str:
        if not isinstance(options, SuryaLayoutOptions):
            options = SuryaLayoutOptions(device=options.device)
        device_key = str(options.device).lower() if options.device else "default_device"
        model_key = options.model_name
        return f"{self.__class__.__name__}_{device_key}_{model_key}"

    def _load_model_from_options(self, options: BaseLayoutOptions) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError(
                "Surya dependencies (surya.layout and surya.table_rec) not installed."
            )
        if not isinstance(options, SuryaLayoutOptions):
            raise TypeError("Incorrect options type provided for Surya model loading.")
        self.logger.info(f"Loading Surya models (device={options.device})...")
        models = {}
        models["layout"] = LayoutPredictor()
        models["table_rec"] = TableRecPredictor()
        self.logger.info("Surya LayoutPredictor and TableRecPredictor loaded.")
        return models

    def detect(self, image: Image.Image, options: BaseLayoutOptions) -> List[Dict[str, Any]]:
        """Detect layout elements and optionally table structure in an image using Surya."""
        if not self.is_available():
            raise RuntimeError("Surya dependencies (layout and table_rec) not installed.")

        if not isinstance(options, SuryaLayoutOptions):
            self.logger.warning(
                "Received BaseLayoutOptions, expected SuryaLayoutOptions. Using defaults."
            )
            options = SuryaLayoutOptions(
                confidence=options.confidence,
                classes=options.classes,
                exclude_classes=options.exclude_classes,
                device=options.device,
                extra_args=options.extra_args,
                recognize_table_structure=True,
            )

        # Extract page reference and scaling factors from extra_args (passed by LayoutAnalyzer)
        self._page_ref = options.extra_args.get("_page_ref")

        # We still need this check, otherwise later steps that need these vars will fail
        can_do_table_rec = options.recognize_table_structure
        if options.recognize_table_structure and not can_do_table_rec:
            logger.warning(
                "Surya table recognition cannot proceed without page reference. Disabling."
            )
            options.recognize_table_structure = False

        # Validate classes
        if options.classes:
            self.validate_classes(options.classes)
        if options.exclude_classes:
            self.validate_classes(options.exclude_classes)

        models = self._get_model(options)
        layout_predictor = models["layout"]
        table_rec_predictor = models["table_rec"]

        input_image = image.convert("RGB")

        initial_layout_detections = []
        tables_to_process = []

        self.logger.debug("Running Surya layout prediction...")
        layout_predictions = layout_predictor([input_image])
        self.logger.debug(f"Surya prediction returned {len(layout_predictions)} results.")
        if not layout_predictions:
            return []
        prediction = layout_predictions[0]

        normalized_classes_req = (
            {self._normalize_class_name(c) for c in options.classes} if options.classes else None
        )
        normalized_classes_excl = (
            {self._normalize_class_name(c) for c in options.exclude_classes}
            if options.exclude_classes
            else set()
        )

        for layout_box in prediction.bboxes:

            class_name_orig = layout_box.label
            normalized_class = self._normalize_class_name(class_name_orig)
            score = float(layout_box.confidence)

            if score < options.confidence:
                continue
            if normalized_classes_req and normalized_class not in normalized_classes_req:
                continue
            if normalized_class in normalized_classes_excl:
                continue

            x_min, y_min, x_max, y_max = map(float, layout_box.bbox)
            detection_data = {
                "bbox": (x_min, y_min, x_max, y_max),
                "class": class_name_orig,
                "confidence": score,
                "normalized_class": normalized_class,
                "source": "layout",
                "model": "surya",
            }
            initial_layout_detections.append(detection_data)

            if options.recognize_table_structure and normalized_class in (
                "table",
                "tableofcontents",
            ):
                tables_to_process.append(detection_data)

        self.logger.info(
            f"Surya initially detected {len(initial_layout_detections)} layout elements matching criteria."
        )

        if not options.recognize_table_structure or not tables_to_process:
            self.logger.debug(
                "Skipping Surya table structure recognition (disabled or no tables found)."
            )
            return initial_layout_detections

        self.logger.info(
            f"Attempting Surya table structure recognition for {len(tables_to_process)} tables..."
        )
        high_res_crops = []

        high_res_dpi = getattr(self._page_ref._parent, "_config", {}).get(
            "surya_table_rec_dpi", 192
        )
        # Use render() for clean image without highlights
        high_res_page_image = self._page_ref.render(resolution=high_res_dpi)

        # Render high-res page ONCE
        self.logger.debug(
            f"Rendering page {self._page_ref.number} at {high_res_dpi} DPI for table recognition, size {high_res_page_image.width}x{high_res_page_image.height}."
        )

        source_tables = []
        for i, table_detection in enumerate(tables_to_process):
            highres_bbox = rescale_bbox(
                list(table_detection["bbox"]), image.size, high_res_page_image.size
            )
            highres_bbox = expand_bbox(highres_bbox)

            crop = high_res_page_image.crop(highres_bbox)
            high_res_crops.append(crop)
            source_tables.append(highres_bbox)

        if not high_res_crops:
            self.logger.info("No valid high-resolution table crops generated.")
            return initial_layout_detections

        structure_detections = []  # Detections relative to std_res input_image

        self.logger.debug(
            f"Running Surya table recognition on {len(high_res_crops)} high-res images..."
        )
        table_predictions = table_rec_predictor(high_res_crops)
        self.logger.debug(f"Surya table recognition returned {len(table_predictions)} results.")

        def build_row_item(element, source_table_bbox, label):
            adjusted_bbox = [
                float(element.bbox[0] + source_table_bbox[0]),
                float(element.bbox[1] + source_table_bbox[1]),
                float(element.bbox[2] + source_table_bbox[0]),
                float(element.bbox[3] + source_table_bbox[1]),
            ]

            adjusted_bbox = rescale_bbox(adjusted_bbox, high_res_page_image.size, image.size)

            return {
                "bbox": adjusted_bbox,
                "class": label,
                "confidence": 1.0,
                "normalized_class": label,
                "source": "layout",
                "model": "surya",
            }

        for table_pred, source_table_bbox in zip(table_predictions, source_tables):
            for box in table_pred.rows:
                structure_detections.append(build_row_item(box, source_table_bbox, "table-row"))

            for box in table_pred.cols:
                structure_detections.append(build_row_item(box, source_table_bbox, "table-column"))

            for box in table_pred.cells:
                structure_detections.append(build_row_item(box, source_table_bbox, "table-cell"))

        self.logger.info(f"Added {len(structure_detections)} table structure elements.")

        return initial_layout_detections + structure_detections
