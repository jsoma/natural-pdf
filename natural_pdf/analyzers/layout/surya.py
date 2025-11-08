# layout_detector_surya.py
import importlib
import importlib.util
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, cast

from PIL import Image

from natural_pdf.utils.page_context import resolve_page_context

from .base import LayoutDetector
from .layout_options import BaseLayoutOptions, SuryaLayoutOptions

logger = logging.getLogger(__name__)

# Check for dependencies
surya_spec = importlib.util.find_spec("surya")
LayoutPredictor: Optional[Type[Any]] = None
TableRecPredictor: Optional[Type[Any]] = None
expand_bbox: Optional[Callable[[List[float]], List[float]]] = None
rescale_bbox: Optional[Callable[[List[float], Tuple[int, int], Tuple[int, int]], List[float]]] = (
    None
)

if surya_spec:
    try:
        surya_common = importlib.import_module("surya.common.util")
        expand_bbox = cast(
            Callable[[List[float]], List[float]],
            getattr(surya_common, "expand_bbox", None),
        )
        rescale_bbox = cast(
            Callable[[List[float], Tuple[int, int], Tuple[int, int]], List[float]],
            getattr(surya_common, "rescale_bbox", None),
        )
        surya_layout = importlib.import_module("surya.layout")
        surya_table_rec = importlib.import_module("surya.table_rec")

        LayoutPredictor = cast(Type[Any], getattr(surya_layout, "LayoutPredictor", None))
        TableRecPredictor = cast(Type[Any], getattr(surya_table_rec, "TableRecPredictor", None))
    except ImportError as e:  # pragma: no cover - optional dependency
        logger.warning(f"Could not import Surya dependencies (layout and/or table_rec): {e}")
else:  # pragma: no cover - optional dependency
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
        assert LayoutPredictor is not None
        assert TableRecPredictor is not None
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

        # Extract page reference passed through extra_args (from LayoutAnalyzer)
        host_obj = options.extra_args.get("_layout_host") or options.extra_args.get("_page_ref")
        page_ref = None
        context_bounds: Optional[Tuple[float, float, float, float]] = None
        if host_obj is not None:
            try:
                page_ref, context_bounds = resolve_page_context(host_obj)
            except ValueError as exc:
                self.logger.debug("Unable to resolve page context from %s: %s", host_obj, exc)

        # We still need this check, otherwise later steps that need these vars will fail
        can_do_table_rec = options.recognize_table_structure
        if options.recognize_table_structure and not can_do_table_rec:
            logger.warning(
                "Surya table recognition cannot proceed without page reference. Disabling."
            )
            options.recognize_table_structure = False
        if options.recognize_table_structure and (expand_bbox is None or rescale_bbox is None):
            logger.warning(
                "Surya table recognition functions unavailable; disabling table recognition."
            )
            options.recognize_table_structure = False

        # Validate classes
        if options.classes:
            self.validate_classes(options.classes)
        if options.exclude_classes:
            self.validate_classes(options.exclude_classes)

        models = self._get_model(options)
        layout_predictor = cast(Any, models["layout"])
        table_rec_predictor = cast(Any, models["table_rec"])

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

        if page_ref is None:
            self.logger.warning(
                "Page reference not available; skipping Surya table structure recognition."
            )
            return initial_layout_detections

        self.logger.info(
            f"Attempting Surya table structure recognition for {len(tables_to_process)} tables..."
        )
        high_res_crops: List[Image.Image] = []

        parent_doc = getattr(page_ref, "_parent", None)
        config = parent_doc._config if parent_doc is not None else {}

        high_res_dpi = config.get("surya_table_rec_dpi", 192)
        # Use render() for clean image without highlights
        high_res_page_image = page_ref.render(resolution=high_res_dpi)
        if high_res_page_image is None:
            self.logger.warning(
                "Could not render high-resolution page image; skipping table recognition."
            )
            return initial_layout_detections

        # Render high-res page ONCE
        self.logger.debug(
            "Rendering page %s at %s DPI for table recognition, size %sx%s.",
            getattr(page_ref, "number", "unknown"),
            high_res_dpi,
            high_res_page_image.width,
            high_res_page_image.height,
        )

        source_tables: List[List[float]] = []
        local_rescale = rescale_bbox
        local_expand = expand_bbox
        if local_rescale is None or local_expand is None:
            self.logger.warning(
                "Surya table recognition helpers unavailable; skipping structure extraction."
            )
            return initial_layout_detections

        for table_detection in tables_to_process:
            highres_bbox = local_rescale(
                list(table_detection["bbox"]), image.size, high_res_page_image.size
            )
            expanded_bbox = local_expand(highres_bbox)
            if not isinstance(expanded_bbox, (list, tuple)) or len(expanded_bbox) != 4:
                self.logger.debug(
                    "Skipping table detection with invalid expanded bbox: %s", expanded_bbox
                )
                continue

            crop_bbox = (
                float(expanded_bbox[0]),
                float(expanded_bbox[1]),
                float(expanded_bbox[2]),
                float(expanded_bbox[3]),
            )

            crop = high_res_page_image.crop(crop_bbox)
            high_res_crops.append(crop)
            source_tables.append([float(v) for v in crop_bbox])

        if not high_res_crops:
            self.logger.info("No valid high-resolution table crops generated.")
            return initial_layout_detections

        structure_detections = []  # Detections relative to std_res input_image

        self.logger.debug(
            f"Running Surya table recognition on {len(high_res_crops)} high-res images..."
        )
        table_predictions = table_rec_predictor(high_res_crops)
        self.logger.debug(f"Surya table recognition returned {len(table_predictions)} results.")

        def build_row_item(
            element: Any, source_table_bbox: List[float], label: str
        ) -> Dict[str, Any]:
            adjusted_bbox = [
                float(element.bbox[0] + source_table_bbox[0]),
                float(element.bbox[1] + source_table_bbox[1]),
                float(element.bbox[2] + source_table_bbox[0]),
                float(element.bbox[3] + source_table_bbox[1]),
            ]

            adjusted_bbox = local_rescale(adjusted_bbox, high_res_page_image.size, image.size)

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
