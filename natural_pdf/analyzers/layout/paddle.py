# layout_detector_paddle.py
import importlib.util
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# Assuming base class and options are importable
try:
    from .base import LayoutDetector
    from .layout_options import BaseLayoutOptions, PaddleLayoutOptions
except ImportError:
    # Placeholders if run standalone or imports fail
    class BaseLayoutOptions:
        pass

    class PaddleLayoutOptions(BaseLayoutOptions):
        pass

    class LayoutDetector:
        def __init__(self):
            self.logger = logging.getLogger()
            self.supported_classes = set()

        def _get_model(self, options):
            raise NotImplementedError

        def _normalize_class_name(self, n):
            return n

        def validate_classes(self, c):
            pass

    logging.basicConfig()

logger = logging.getLogger(__name__)

# Check for dependencies
paddle_spec = importlib.util.find_spec("paddle") or importlib.util.find_spec("paddlepaddle")
paddleocr_spec = importlib.util.find_spec("paddleocr")
PPStructureV3 = None
_paddle_import_error = None  # Store the import error for debugging

if paddle_spec and paddleocr_spec:
    try:
        from paddleocr import PPStructureV3
    except ImportError as e:
        _paddle_import_error = str(e)
        logger.warning(f"Could not import Paddle dependencies: {e}")
else:
    if not paddle_spec:
        _paddle_import_error = "paddlepaddle not found"
    elif not paddleocr_spec:
        _paddle_import_error = "paddleocr not found"
    else:
        _paddle_import_error = "Unknown import issue"
    logger.warning(
        "paddlepaddle or paddleocr not found. PaddleLayoutDetector will not be available."
    )

from .table_structure_utils import group_cells_into_rows_and_columns


class PaddleLayoutDetector(LayoutDetector):
    """Document layout and table structure detector using PaddlePaddle's PP-StructureV3."""

    def __init__(self):
        super().__init__()
        # Supported classes by PP-StructureV3 (based on docs and common usage)
        self.supported_classes = {
            "text",
            "title",
            "figure",
            "table",
            "header",
            "footer",
            "reference",
            "equation",
            # New labels from V3
            "image",
            "paragraph_title",
            "doc_title",
            "figure_title",
            "table_cell",
        }
        # Models are loaded via _get_model

    def is_available(self) -> bool:
        """Check if dependencies are installed."""
        if PPStructureV3 is None and _paddle_import_error:
            # Raise an informative error instead of just returning False
            raise RuntimeError(f"Paddle dependencies check failed: {_paddle_import_error}")
        return PPStructureV3 is not None

    def _get_cache_key(self, options: BaseLayoutOptions) -> str:
        """Generate cache key based on model configuration."""
        if not isinstance(options, PaddleLayoutOptions):
            options = PaddleLayoutOptions(device=options.device)

        device_key = str(options.device).lower() if options.device else "default_device"
        lang_key = options.lang
        table_key = str(options.use_table_recognition)
        orientation_key = str(options.use_textline_orientation)

        return f"{self.__class__.__name__}_{device_key}_{lang_key}_{table_key}_{orientation_key}"

    def _load_model_from_options(self, options: BaseLayoutOptions) -> Any:
        """Load the PPStructureV3 model based on options."""
        if not self.is_available():
            raise RuntimeError("Paddle dependencies (paddlepaddle, paddleocr) not installed.")

        if not isinstance(options, PaddleLayoutOptions):
            raise TypeError("Incorrect options type provided for Paddle model loading.")

        self.logger.info(f"Loading PP-StructureV3 model with options: {options}")

        # List of valid PPStructureV3 constructor arguments (from official docs)
        valid_init_args = {
            "layout_detection_model_name",
            "layout_detection_model_dir",
            "layout_threshold",
            "layout_nms",
            "layout_unclip_ratio",
            "layout_merge_bboxes_mode",
            "chart_recognition_model_name",
            "chart_recognition_model_dir",
            "chart_recognition_batch_size",
            "region_detection_model_name",
            "region_detection_model_dir",
            "doc_orientation_classify_model_name",
            "doc_orientation_classify_model_dir",
            "doc_unwarping_model_name",
            "doc_unwarping_model_dir",
            "text_detection_model_name",
            "text_detection_model_dir",
            "text_det_limit_side_len",
            "text_det_limit_type",
            "text_det_thresh",
            "text_det_box_thresh",
            "text_det_unclip_ratio",
            "textline_orientation_model_name",
            "textline_orientation_model_dir",
            "textline_orientation_batch_size",
            "text_recognition_model_name",
            "text_recognition_model_dir",
            "text_recognition_batch_size",
            "text_rec_score_thresh",
            "table_classification_model_name",
            "table_classification_model_dir",
            "wired_table_structure_recognition_model_name",
            "wired_table_structure_recognition_model_dir",
            "wireless_table_structure_recognition_model_name",
            "wireless_table_structure_recognition_model_dir",
            "wired_table_cells_detection_model_name",
            "wired_table_cells_detection_model_dir",
            "wireless_table_cells_detection_model_name",
            "wireless_table_cells_detection_model_dir",
            "seal_text_detection_model_name",
            "seal_text_detection_model_dir",
            "seal_det_limit_side_len",
            "seal_det_limit_type",
            "seal_det_thresh",
            "seal_det_box_thresh",
            "seal_det_unclip_ratio",
            "seal_text_recognition_model_name",
            "seal_text_recognition_model_dir",
            "seal_text_recognition_batch_size",
            "seal_rec_score_thresh",
            "formula_recognition_model_name",
            "formula_recognition_model_dir",
            "formula_recognition_batch_size",
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_textline_orientation",
            "use_seal_recognition",
            "use_table_recognition",
            "use_formula_recognition",
            "use_chart_recognition",
            "use_region_detection",
            "device",
            "enable_hpi",
            "use_tensorrt",
            "precision",
            "enable_mkldnn",
            "cpu_threads",
            "paddlex_config",
        }

        # Build init_args from dataclass fields and filtered extra_args
        init_args = {}
        # Add all dataclass fields that are in the valid set and not None
        for field_name in options.__dataclass_fields__:
            if field_name in valid_init_args:
                value = getattr(options, field_name)
                if value is not None:
                    init_args[field_name] = value
        # Add filtered extra_args (not starting with '_' and in valid set)
        filtered_extra_args = {
            k: v
            for k, v in options.extra_args.items()
            if not k.startswith("_") and k in valid_init_args
        }
        init_args.update(filtered_extra_args)

        # Special handling for English model selection
        if getattr(options, "lang", None) == "en":
            init_args["text_recognition_model_name"] = "en_PP-OCRv4_mobile_rec"

        try:
            model_instance = PPStructureV3(**init_args)
            self.logger.info("PP-StructureV3 model loaded.")
            return model_instance
        except Exception as e:
            self.logger.error(f"Failed to load PP-StructureV3 model: {e}", exc_info=True)
            raise

    def detect(self, image: Image.Image, options: BaseLayoutOptions) -> List[Dict[str, Any]]:
        """Detect layout elements in an image using PP-StructureV3."""
        if not self.is_available():
            raise RuntimeError("Paddle dependencies (paddlepaddle, paddleocr) not installed.")

        if not isinstance(options, PaddleLayoutOptions):
            self.logger.warning(
                "Received BaseLayoutOptions, expected PaddleLayoutOptions. Using defaults."
            )
            options = PaddleLayoutOptions(
                confidence=options.confidence,
                classes=options.classes,
                exclude_classes=options.exclude_classes,
                device=options.device,
                extra_args=options.extra_args,
            )

        # --- Backward compatibility for renamed options passed via extra_args ---
        if "use_angle_cls" in options.extra_args:
            self.logger.warning(
                "Parameter 'use_angle_cls' is deprecated for Paddle. Use 'use_textline_orientation' instead."
            )
            options.use_textline_orientation = options.extra_args.pop("use_angle_cls")
        if "enable_table" in options.extra_args:
            self.logger.warning(
                "Parameter 'enable_table' is deprecated for Paddle. Use 'use_table_recognition' instead."
            )
            options.use_table_recognition = options.extra_args.pop("enable_table")

        self.validate_classes(options.classes or [])
        if options.exclude_classes:
            self.validate_classes(options.exclude_classes)

        # Get the cached/loaded PP-StructureV3 instance
        ppstructure_instance = self._get_model(options)

        # Convert PIL image to numpy array for prediction
        img_np = np.array(image.convert("RGB"))
        self.logger.debug("Running PP-StructureV3 analysis...")
        try:
            results = ppstructure_instance.predict(img_np)
        except Exception as e:
            self.logger.error(f"Error during PP-StructureV3 analysis: {e}", exc_info=True)
            raise

        self.logger.debug(f"PP-StructureV3 returned {len(results)} result objects.")

        # --- Process Results ---
        detections = []
        if not results:
            self.logger.warning("PaddleLayout returned empty results")
            return []

        # Prepare normalized class filters once
        normalized_classes_req = (
            {self._normalize_class_name(c) for c in options.classes} if options.classes else None
        )
        normalized_classes_excl = (
            {self._normalize_class_name(c) for c in options.exclude_classes}
            if options.exclude_classes
            else set()
        )

        # Debug counters
        table_count = 0
        cell_count = 0
        row_count = 0
        col_count = 0
        matched_table_structures = 0

        # A single image input returns a list with one result object
        for res in results:
            # Handle both possible result structures (with or without 'res' key)
            if isinstance(res, dict) and "res" in res:
                result_data = res["res"]
            elif isinstance(res, dict):
                result_data = res
            else:
                self.logger.warning(f"Skipping result with unexpected structure: {res}")
                continue

            # --- Process Layout Regions ---
            layout_res = result_data.get("layout_det_res", {})
            table_res_list = result_data.get("table_res_list", [])
            # Build a map of table_region_id to structure info for fast lookup
            table_structures_by_id = {}
            for t in table_res_list:
                if "table_region_id" in t:
                    table_structures_by_id[t["table_region_id"]] = t
            table_structures = table_res_list or []
            table_idx = 0  # fallback index if no region_id
            if table_res_list:
                self.logger.debug(
                    f"Found {len(table_res_list)} table structure(s) in table_res_list."
                )

            if not layout_res or "boxes" not in layout_res:
                self.logger.debug("No layout detection boxes found in result.")
            else:
                for region in layout_res["boxes"]:
                    try:
                        region_type_orig = region.get("label", "unknown")
                        region_type = region_type_orig.lower()
                        normalized_class = self._normalize_class_name(region_type)

                        # Apply class filtering
                        if (
                            normalized_classes_req
                            and normalized_class not in normalized_classes_req
                        ):
                            continue
                        if normalized_class in normalized_classes_excl:
                            continue

                        confidence_score = region.get("score", 1.0)
                        if confidence_score < options.confidence:
                            continue

                        bbox = region.get("coordinate")
                        if not bbox or len(bbox) != 4:
                            self.logger.warning(f"Skipping region with invalid bbox: {region}")
                            continue
                        x_min, y_min, x_max, y_max = map(float, bbox)

                        detection_data = {
                            "bbox": (x_min, y_min, x_max, y_max),
                            "class": region_type_orig,
                            "confidence": confidence_score,
                            "normalized_class": normalized_class,
                            "source": "layout",
                            "model": "paddle_v3",
                        }

                        # --- Table structure parsing ---
                        if normalized_class == "table" and options.create_cells:
                            table_count += 1
                            # Try to match by region_id, else by order
                            table_struct = None
                            region_id = region.get("table_region_id")
                            if region_id is not None and region_id in table_structures_by_id:
                                table_struct = table_structures_by_id[region_id]
                            elif table_idx < len(table_structures):
                                table_struct = table_structures[table_idx]
                                table_idx += 1

                            if table_struct:
                                matched_table_structures += 1
                                self.logger.debug(
                                    f"Matched table structure for table_region_id {region_id} or index {table_idx-1}."
                                )
                                # Attach structure info as metadata
                                detection_data["metadata"] = {
                                    k: v
                                    for k, v in table_struct.items()
                                    if k not in ("cell_box_list", "table_ocr_pred", "pred_html")
                                }
                                detection_data["html"] = table_struct.get("pred_html")
                                # Add cell regions
                                cell_boxes = []
                                for cell_bbox in table_struct.get("cell_box_list", []):
                                    if cell_bbox is None or len(cell_bbox) != 4:
                                        continue
                                    sx0, sy0, sx1, sy1 = map(float, cell_bbox)
                                    cell_boxes.append((sx0, sy0, sx1, sy1))
                                    detections.append(
                                        {
                                            "bbox": (sx0, sy0, sx1, sy1),
                                            "class": "table_cell",
                                            "confidence": confidence_score,
                                            "normalized_class": self._normalize_class_name(
                                                "table_cell"
                                            ),
                                            "source": "layout",
                                            "model": "paddle_v3",
                                            "parent_bbox": (x_min, y_min, x_max, y_max),
                                        }
                                    )
                                    cell_count += 1
                                    self.logger.debug(
                                        f"Created table_cell region for bbox {(sx0, sy0, sx1, sy1)}."
                                    )
                                # Add row/col regions if not present in Paddle output
                                if not table_struct.get("row_box_list") and not table_struct.get(
                                    "col_box_list"
                                ):
                                    row_boxes, col_boxes = group_cells_into_rows_and_columns(
                                        cell_boxes
                                    )
                                    for row_bbox in row_boxes:
                                        rx0, ry0, rx1, ry1 = row_bbox
                                        detections.append(
                                            {
                                                "bbox": (rx0, ry0, rx1, ry1),
                                                "class": "table_row",
                                                "confidence": confidence_score,
                                                "normalized_class": self._normalize_class_name(
                                                    "table_row"
                                                ),
                                                "source": "layout",
                                                "model": "paddle_v3",
                                                "parent_bbox": (x_min, y_min, x_max, y_max),
                                            }
                                        )
                                        row_count += 1
                                        self.logger.debug(
                                            f"[UTIL] Created table_row region for bbox {(rx0, ry0, rx1, ry1)}."
                                        )
                                    for col_bbox in col_boxes:
                                        cx0, cy0, cx1, cy1 = col_bbox
                                        detections.append(
                                            {
                                                "bbox": (cx0, cy0, cx1, cy1),
                                                "class": "table_column",
                                                "confidence": confidence_score,
                                                "normalized_class": self._normalize_class_name(
                                                    "table_column"
                                                ),
                                                "source": "layout",
                                                "model": "paddle_v3",
                                                "parent_bbox": (x_min, y_min, x_max, y_max),
                                            }
                                        )
                                        col_count += 1
                                        self.logger.debug(
                                            f"[UTIL] Created table_column region for bbox {(cx0, cy0, cx1, cy1)}."
                                        )
                                else:
                                    # Add row regions from Paddle output if present
                                    for row_bbox in table_struct.get("row_box_list", []):
                                        if row_bbox is None or len(row_bbox) != 4:
                                            continue
                                        rx0, ry0, rx1, ry1 = map(float, row_bbox)
                                        detections.append(
                                            {
                                                "bbox": (rx0, ry0, rx1, ry1),
                                                "class": "table_row",
                                                "confidence": confidence_score,
                                                "normalized_class": self._normalize_class_name(
                                                    "table_row"
                                                ),
                                                "source": "layout",
                                                "model": "paddle_v3",
                                                "parent_bbox": (x_min, y_min, x_max, y_max),
                                            }
                                        )
                                        row_count += 1
                                        self.logger.debug(
                                            f"Created table_row region for bbox {(rx0, ry0, rx1, ry1)}."
                                        )
                                    # Add column regions from Paddle output if present
                                    for col_bbox in table_struct.get("col_box_list", []):
                                        if col_bbox is None or len(col_bbox) != 4:
                                            continue
                                        cx0, cy0, cx1, cy1 = map(float, col_bbox)
                                        detections.append(
                                            {
                                                "bbox": (cx0, cy0, cx1, cy1),
                                                "class": "table_column",
                                                "confidence": confidence_score,
                                                "normalized_class": self._normalize_class_name(
                                                    "table_column"
                                                ),
                                                "source": "layout",
                                                "model": "paddle_v3",
                                                "parent_bbox": (x_min, y_min, x_max, y_max),
                                            }
                                        )
                                        col_count += 1
                                        self.logger.debug(
                                            f"Created table_column region for bbox {(cx0, cy0, cx1, cy1)}."
                                        )
                        detections.append(detection_data)
                    except (TypeError, KeyError, IndexError, ValueError) as e:
                        self.logger.warning(f"Error processing Paddle region: {region}. Error: {e}")
                        continue

        self.logger.info(
            f"PaddleLayout detected {len(detections)} layout elements matching criteria. Tables: {table_count}, matched structures: {matched_table_structures}, cells: {cell_count}, rows: {row_count}, columns: {col_count}."
        )
        return detections
