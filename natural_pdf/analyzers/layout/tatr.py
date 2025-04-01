# layout_detector_tatr.py
import logging
import importlib.util
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

# Assuming base class and options are importable
from .base import LayoutDetector
from .layout_options import TATRLayoutOptions, BaseLayoutOptions

logger = logging.getLogger(__name__)

# Check for dependencies
torch_spec = importlib.util.find_spec("torch")
torchvision_spec = importlib.util.find_spec("torchvision")
transformers_spec = importlib.util.find_spec("transformers")
torch = None
transforms = None
AutoModelForObjectDetection = None

if torch_spec and torchvision_spec and transformers_spec:
    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForObjectDetection
    except ImportError as e:
        logger.warning(f"Could not import TATR dependencies (torch, torchvision, transformers): {e}")
else:
    logger.warning("torch, torchvision, or transformers not found. TableTransformerDetector will not be available.")


class TableTransformerDetector(LayoutDetector):
    """Table structure detector using Microsoft's Table Transformer (TATR) models."""

    # Custom resize transform (keep as nested class or move outside)
    class MaxResize(object):
        def __init__(self, max_size=800):
            self.max_size = max_size
        def __call__(self, image):
            width, height = image.size
            current_max_size = max(width, height)
            scale = self.max_size / current_max_size
            # Use LANCZOS for resizing
            resized_image = image.resize((int(round(scale*width)), int(round(scale*height))), Image.Resampling.LANCZOS)
            return resized_image

    def __init__(self):
        super().__init__()
        self.supported_classes = {
            'table', 'table row', 'table column', 'table column header', 'table projected row header', 'table spanning cell' # Add others if supported by models used
        }
        # Models are loaded via _get_model

    def is_available(self) -> bool:
        """Check if dependencies are installed."""
        return torch is not None and transforms is not None and AutoModelForObjectDetection is not None

    def _get_cache_key(self, options: TATRLayoutOptions) -> str:
        """Generate cache key based on model IDs and device."""
        if not isinstance(options, TATRLayoutOptions):
            options = TATRLayoutOptions(device=options.device)

        device_key = str(options.device).lower()
        det_model_key = options.detection_model.replace('/','_')
        struct_model_key = options.structure_model.replace('/','_')
        return f"{self.__class__.__name__}_{device_key}_{det_model_key}_{struct_model_key}"

    def _load_model_from_options(self, options: TATRLayoutOptions) -> Dict[str, Any]:
        """Load the TATR detection and structure models."""
        if not self.is_available():
             raise RuntimeError("TATR dependencies (torch, torchvision, transformers) not installed.")

        device = options.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Loading TATR models: Detection='{options.detection_model}', Structure='{options.structure_model}' onto device='{device}'")
        try:
            detection_model = AutoModelForObjectDetection.from_pretrained(
                options.detection_model, revision="no_timm" # Important revision for some versions
            ).to(device)
            structure_model = AutoModelForObjectDetection.from_pretrained(
                options.structure_model
            ).to(device)
            self.logger.info("TATR models loaded.")
            return {'detection': detection_model, 'structure': structure_model}
        except Exception as e:
            self.logger.error(f"Failed to load TATR models: {e}", exc_info=True)
            raise

    # --- Helper methods (box_cxcywh_to_xyxy, rescale_bboxes, outputs_to_objects) ---
    # Keep these as defined in the original tatr.txt file, making them instance methods
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(out_bbox.device) # Ensure tensor on correct device
        return boxes

    def outputs_to_objects(self, outputs, img_size, id2label):
        logits = outputs.logits
        bboxes = outputs.pred_boxes
        # Use softmax activation function
        prob = logits.softmax(-1)[0, :, :-1] # Exclude the "no object" class
        scores, labels = prob.max(-1)

        # Convert to absolute coordinates
        img_w, img_h = img_size
        boxes = self.rescale_bboxes(bboxes[0, ...], (img_w, img_h)) # Pass tuple size

        # Move results to CPU for list comprehension
        scores = scores.cpu().tolist()
        labels = labels.cpu().tolist()
        boxes = boxes.cpu().tolist()

        objects = []
        for score, label_idx, bbox in zip(scores, labels, boxes):
            class_label = id2label.get(label_idx, 'unknown') # Use get with default
            if class_label != 'no object' and class_label != 'unknown':
                objects.append({
                    'label': class_label,
                    'score': float(score),
                    'bbox': [round(float(c), 2) for c in bbox] # Round coordinates
                })
        return objects
    # --- End Helper Methods ---

    def detect(self, image: Image.Image, options: BaseLayoutOptions) -> List[Dict[str, Any]]:
        """Detect tables and their structure in an image."""
        if not self.is_available():
            raise RuntimeError("TATR dependencies (torch, torchvision, transformers) not installed.")

        if not isinstance(options, TATRLayoutOptions):
             self.logger.warning("Received BaseLayoutOptions, expected TATRLayoutOptions. Using defaults.")
             options = TATRLayoutOptions(
                 confidence=options.confidence, classes=options.classes,
                 exclude_classes=options.exclude_classes, device=options.device,
                 extra_args=options.extra_args
             )

        self.validate_classes(options.classes or [])
        if options.exclude_classes:
            self.validate_classes(options.exclude_classes)

        models = self._get_model(options)
        detection_model = models['detection']
        structure_model = models['structure']
        device = options.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare transforms based on options
        detection_transform = transforms.Compose([
            self.MaxResize(options.max_detection_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        structure_transform = transforms.Compose([
            self.MaxResize(options.max_structure_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # --- Detect Tables ---
        self.logger.debug("Running TATR table detection...")
        pixel_values = detection_transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = detection_model(pixel_values)

        id2label_det = detection_model.config.id2label
        id2label_det[detection_model.config.num_labels] = "no object" # Add no object class
        tables = self.outputs_to_objects(outputs, image.size, id2label_det)
        tables = [t for t in tables if t['score'] >= options.confidence and t['label'] == 'table'] # Filter for tables
        self.logger.debug(f"Detected {len(tables)} table regions.")

        all_detections = []

        # Add table detections if requested
        normalized_classes_req = {self._normalize_class_name(c) for c in options.classes} if options.classes else None
        normalized_classes_excl = {self._normalize_class_name(c) for c in options.exclude_classes} if options.exclude_classes else set()

        if normalized_classes_req is None or 'table' in normalized_classes_req:
             if 'table' not in normalized_classes_excl:
                 for table in tables:
                     all_detections.append({
                         'bbox': tuple(table['bbox']),
                         'class': 'table',
                         'confidence': float(table['score']),
                         'normalized_class': 'table',
                         'source': 'layout',
                         'model': 'tatr'
                     })

        # --- Process Structure ---
        structure_class_names = {'table row', 'table column', 'table column header', 'table projected row header', 'table spanning cell'}
        normalized_structure_classes = {self._normalize_class_name(c) for c in structure_class_names}

        needed_structure = False
        if normalized_classes_req is None: # If no specific classes requested
             needed_structure = any(norm_cls not in normalized_classes_excl for norm_cls in normalized_structure_classes)
        else: # Specific classes requested
             needed_structure = any(norm_cls in normalized_classes_req for norm_cls in normalized_structure_classes)

        if needed_structure and tables:
            self.logger.debug("Running TATR structure recognition...")
            id2label_struct = structure_model.config.id2label
            id2label_struct[structure_model.config.num_labels] = "no object"

            for table in tables:
                x_min, y_min, x_max, y_max = map(int, table['bbox'])
                # Ensure coordinates are within image bounds
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(image.width, x_max), min(image.height, y_max)
                if x_max <= x_min or y_max <= y_min: continue # Skip invalid crop

                cropped_table = image.crop((x_min, y_min, x_max, y_max))
                if cropped_table.width == 0 or cropped_table.height == 0: continue # Skip empty crop

                pixel_values_struct = structure_transform(cropped_table).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs_struct = structure_model(pixel_values_struct)

                structure_elements = self.outputs_to_objects(outputs_struct, cropped_table.size, id2label_struct)
                structure_elements = [e for e in structure_elements if e['score'] >= options.confidence]

                for element in structure_elements:
                    element_class_orig = element['label']
                    normalized_class = self._normalize_class_name(element_class_orig)

                    # Apply class filtering
                    if normalized_classes_req and normalized_class not in normalized_classes_req: continue
                    if normalized_class in normalized_classes_excl: continue

                    # Adjust coordinates
                    ex0, ey0, ex1, ey1 = element['bbox']
                    adj_bbox = (ex0 + x_min, ey0 + y_min, ex1 + x_min, ey1 + y_min)

                    all_detections.append({
                        'bbox': adj_bbox,
                        'class': element_class_orig,
                        'confidence': float(element['score']),
                        'normalized_class': normalized_class,
                        'source': 'layout',
                        'model': 'tatr'
                    })
            self.logger.debug(f"Added {len(all_detections) - len(tables)} structure elements.")

        self.logger.info(f"TATR detected {len(all_detections)} layout elements matching criteria.")
        return all_detections

