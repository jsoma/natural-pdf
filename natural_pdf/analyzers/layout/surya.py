# layout_detector_surya.py
import logging
import importlib.util
import os
import tempfile
import copy
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from .base import LayoutDetector
from .layout_options import SuryaLayoutOptions, BaseLayoutOptions

logger = logging.getLogger(__name__)

# Check for dependencies
surya_spec = importlib.util.find_spec("surya")
LayoutPredictor = None
TableRecPredictor = None

if surya_spec:
    try:
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
            'text', 'pageheader', 'pagefooter', 'sectionheader',
            'table', 'tableofcontents', 'picture', 'caption',
            'heading', 'title', 'list', 'listitem', 'code',
            'textinlinemath', 'mathformula', 'form',
            'table-row', 'table-column'
        }
        self._page_ref = None # To store page reference from options

    def is_available(self) -> bool:
        return LayoutPredictor is not None and TableRecPredictor is not None

    def _get_cache_key(self, options: BaseLayoutOptions) -> str:
        if not isinstance(options, SuryaLayoutOptions):
             options = SuryaLayoutOptions(device=options.device)
        device_key = str(options.device).lower() if options.device else 'default_device'
        model_key = options.model_name
        return f"{self.__class__.__name__}_{device_key}_{model_key}"

    def _load_model_from_options(self, options: BaseLayoutOptions) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("Surya dependencies (surya.layout and surya.table_rec) not installed.")
        if not isinstance(options, SuryaLayoutOptions):
            raise TypeError("Incorrect options type provided for Surya model loading.")
        self.logger.info(f"Loading Surya models (device={options.device})...")
        models = {}
        try:            
            models['layout'] = LayoutPredictor()
            models['table_rec'] = TableRecPredictor()
            self.logger.info("Surya LayoutPredictor and TableRecPredictor loaded.")
            return models
        except Exception as e:
            self.logger.error(f"Failed to load Surya models: {e}", exc_info=True)
            raise
            
    def _expand_bbox(self, bbox: Tuple[float, float, float, float], 
                     padding: int, max_width: int, max_height: int) -> Tuple[int, int, int, int]:
        """Expand bbox by padding, clamping to max dimensions."""
        x0, y0, x1, y1 = bbox
        x0 = max(0, int(x0 - padding))
        y0 = max(0, int(y0 - padding))
        x1 = min(max_width, int(x1 + padding))
        y1 = min(max_height, int(y1 + padding))
        return x0, y0, x1, y1

    def detect(self, image: Image.Image, options: BaseLayoutOptions) -> List[Dict[str, Any]]:
        """Detect layout elements and optionally table structure in an image using Surya."""
        if not self.is_available():
            raise RuntimeError("Surya dependencies (layout and table_rec) not installed.")

        if not isinstance(options, SuryaLayoutOptions):
             self.logger.warning("Received BaseLayoutOptions, expected SuryaLayoutOptions. Using defaults.")
             options = SuryaLayoutOptions(
                 confidence=options.confidence, classes=options.classes,
                 exclude_classes=options.exclude_classes, device=options.device,
                 extra_args=options.extra_args,
                 recognize_table_structure=True
             )
             
        # Extract page reference and scaling factors from extra_args (passed by LayoutAnalyzer)
        self._page_ref = options.extra_args.get('_page_ref')
        img_scale_x = options.extra_args.get('_img_scale_x')
        img_scale_y = options.extra_args.get('_img_scale_y')
        
        # We still need this check, otherwise later steps that need these vars will fail
        can_do_table_rec = options.recognize_table_structure and self._page_ref and img_scale_x is not None and img_scale_y is not None
        if options.recognize_table_structure and not can_do_table_rec:
             logger.warning("Surya table recognition cannot proceed without page reference and scaling factors. Disabling.")
             options.recognize_table_structure = False

        # Validate classes
        if options.classes: self.validate_classes(options.classes)
        if options.exclude_classes: self.validate_classes(options.exclude_classes)

        models = self._get_model(options)
        layout_predictor = models['layout']
        table_rec_predictor = models['table_rec']

        input_image = image.convert("RGB")
        input_image_list = [input_image]

        initial_layout_detections = [] # Detections relative to input_image
        tables_to_process = []
        
        # --- Initial Layout Detection --- 
        self.logger.debug("Running Surya layout prediction...")
        layout_predictions = layout_predictor(input_image_list)
        self.logger.debug(f"Surya prediction returned {len(layout_predictions)} results.")
        if not layout_predictions: return []
        prediction = layout_predictions[0]

        normalized_classes_req = {self._normalize_class_name(c) for c in options.classes} if options.classes else None
        normalized_classes_excl = {self._normalize_class_name(c) for c in options.exclude_classes} if options.exclude_classes else set()

        for layout_box in prediction.bboxes:
            class_name_orig = layout_box.label
            normalized_class = self._normalize_class_name(class_name_orig)
            score = float(layout_box.confidence)

            if score < options.confidence: continue
            if normalized_classes_req and normalized_class not in normalized_classes_req: continue
            if normalized_class in normalized_classes_excl: continue

            x_min, y_min, x_max, y_max = map(float, layout_box.bbox)
            detection_data = {
                'bbox': (x_min, y_min, x_max, y_max),
                'class': class_name_orig,
                'confidence': score,
                'normalized_class': normalized_class,
                'source': 'layout',
                'model': 'surya'
            }
            initial_layout_detections.append(detection_data)

            if options.recognize_table_structure and normalized_class in ('table', 'tableofcontents'):
                tables_to_process.append(detection_data)

        self.logger.info(f"Surya initially detected {len(initial_layout_detections)} layout elements matching criteria.")

        # --- Table Structure Recognition (Optional) ---
        if not options.recognize_table_structure or not tables_to_process:
            self.logger.debug("Skipping Surya table structure recognition (disabled or no tables found).")
            return initial_layout_detections

        self.logger.info(f"Attempting Surya table structure recognition for {len(tables_to_process)} tables...")
        high_res_crops = []
        pdf_offsets = [] # Store (pdf_x0, pdf_y0) for each crop
        
        high_res_dpi = getattr(self._page_ref._parent, '_config', {}).get('surya_table_rec_dpi', 192)
        bbox_padding = getattr(self._page_ref._parent, '_config', {}).get('surya_table_bbox_padding', 10)
        pdf_to_highres_scale = high_res_dpi / 72.0
        
        # Render high-res page ONCE
        self.logger.debug(f"Rendering page {self._page_ref.number} at {high_res_dpi} DPI for table recognition...")
        high_res_page_image = self._page_ref.to_image(resolution=high_res_dpi, include_highlights=False)
        if not high_res_page_image:
            raise RuntimeError(f"Failed to render page {self._page_ref.number} at high resolution.")
        self.logger.debug(f"  High-res image size: {high_res_page_image.width}x{high_res_page_image.height}")

        for i, table_detection in enumerate(tables_to_process):
            img_x0, img_y0, img_x1, img_y1 = table_detection['bbox']
            
            # PDF coords
            pdf_x0 = img_x0 * img_scale_x
            pdf_y0 = img_y0 * img_scale_y
            pdf_x1 = img_x1 * img_scale_x
            pdf_y1 = img_y1 * img_scale_y
            pdf_x0 = max(0, pdf_x0)
            pdf_y0 = max(0, pdf_y0)
            pdf_x1 = min(self._page_ref.width, pdf_x1)
            pdf_y1 = min(self._page_ref.height, pdf_y1)
            
            # High-res image coords
            hr_x0 = pdf_x0 * pdf_to_highres_scale
            hr_y0 = pdf_y0 * pdf_to_highres_scale
            hr_x1 = pdf_x1 * pdf_to_highres_scale
            hr_y1 = pdf_y1 * pdf_to_highres_scale
            
            # Expand high-res bbox
            hr_x0_exp, hr_y0_exp, hr_x1_exp, hr_y1_exp = self._expand_bbox(
                (hr_x0, hr_y0, hr_x1, hr_y1), 
                padding=bbox_padding, 
                max_width=high_res_page_image.width, 
                max_height=high_res_page_image.height
            )

            crop = high_res_page_image.crop((hr_x0_exp, hr_y0_exp, hr_x1_exp, hr_y1_exp))
            high_res_crops.append(crop)
            pdf_offsets.append((pdf_x0, pdf_y0)) 
        
        if not high_res_crops:
            self.logger.info("No valid high-resolution table crops generated.")
            return initial_layout_detections

        structure_detections = [] # Detections relative to std_res input_image

        # --- Run Table Recognition (will raise error on failure) --- 
        self.logger.debug(f"Running Surya table recognition on {len(high_res_crops)} high-res images...")
        table_predictions = table_rec_predictor(high_res_crops)
        self.logger.debug(f"Surya table recognition returned {len(table_predictions)} results.")

        # --- Process Results --- 
        if len(table_predictions) != len(pdf_offsets):
             # This case is less likely if predictor didn't error, but good sanity check
             raise RuntimeError(f"Mismatch between table inputs ({len(pdf_offsets)}) and predictions ({len(table_predictions)}).")

        for table_pred, (offset_pdf_x0, offset_pdf_y0) in zip(table_predictions, pdf_offsets):
            # Process Rows
            for row_box in table_pred.rows:
                crop_rx0, crop_ry0, crop_rx1, crop_ry1 = map(float, row_box.bbox)
                pdf_row_x0 = offset_pdf_x0 + crop_rx0 / pdf_to_highres_scale
                pdf_row_y0 = offset_pdf_y0 + crop_ry0 / pdf_to_highres_scale
                pdf_row_x1 = offset_pdf_x0 + crop_rx1 / pdf_to_highres_scale
                pdf_row_y1 = offset_pdf_y0 + crop_ry1 / pdf_to_highres_scale
                img_row_x0 = pdf_row_x0 / img_scale_x
                img_row_y0 = pdf_row_y0 / img_scale_y
                img_row_x1 = pdf_row_x1 / img_scale_x
                img_row_y1 = pdf_row_y1 / img_scale_y
                structure_detections.append({
                    'bbox': (img_row_x0, img_row_y0, img_row_x1, img_row_y1),
                    'class': 'table-row', 'confidence': 1.0, 'normalized_class': 'table-row',
                    'source': 'layout', 'model': 'surya'
                })

            # Process Columns
            for col_box in table_pred.cols:
                crop_cx0, crop_cy0, crop_cx1, crop_cy1 = map(float, col_box.bbox)
                pdf_col_x0 = offset_pdf_x0 + crop_cx0 / pdf_to_highres_scale
                pdf_col_y0 = offset_pdf_y0 + crop_cy0 / pdf_to_highres_scale
                pdf_col_x1 = offset_pdf_x0 + crop_cx1 / pdf_to_highres_scale
                pdf_col_y1 = offset_pdf_y0 + crop_cy1 / pdf_to_highres_scale
                img_col_x0 = pdf_col_x0 / img_scale_x
                img_col_y0 = pdf_col_y0 / img_scale_y
                img_col_x1 = pdf_col_x1 / img_scale_x
                img_col_y1 = pdf_col_y1 / img_scale_y
                structure_detections.append({
                    'bbox': (img_col_x0, img_col_y0, img_col_x1, img_col_y1),
                    'class': 'table-column', 'confidence': 1.0, 'normalized_class': 'table-column',
                    'source': 'layout', 'model': 'surya'
                })

        self.logger.info(f"Added {len(structure_detections)} table structure elements.")

        return initial_layout_detections + structure_detections

