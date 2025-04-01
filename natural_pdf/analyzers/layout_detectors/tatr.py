import torch
from torch import nn
from PIL import Image
from typing import List, Dict, Any, Optional
from torchvision import transforms
from transformers import AutoModelForObjectDetection

from .base import LayoutDetector

class TableTransformerDetector(LayoutDetector):
    """
    Table structure detector using Microsoft's Table Transformer (TATR) models.
    """
    
    # Custom resize transform
    class MaxResize(object):
        def __init__(self, max_size=800):
            self.max_size = max_size

        def __call__(self, image):
            width, height = image.size
            current_max_size = max(width, height)
            scale = self.max_size / current_max_size
            resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
            return resized_image
    
    def __init__(self, 
                detection_model: str = "microsoft/table-transformer-detection",
                structure_model: str = "microsoft/table-transformer-structure-recognition-v1.1-all",
                max_detection_size: int = 800,
                max_structure_size: int = 1000,
                device: str = None):
        """
        Initialize the Table Transformer detector.
        
        Args:
            detection_model: HuggingFace model ID for table detection
            structure_model: HuggingFace model ID for table structure recognition
            max_detection_size: Maximum size for detection model input
            max_structure_size: Maximum size for structure model input
            device: Device to run inference on (None for auto-detection)
        """
        super().__init__()
        self.detection_model_id = detection_model
        self.structure_model_id = structure_model
        self.max_detection_size = max_detection_size
        self.max_structure_size = max_structure_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models will be lazy-loaded
        self._detection_model = None
        self._structure_model = None
        
        # Transforms for detection and structure recognition
        self.detection_transform = transforms.Compose([
            self.MaxResize(max_detection_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.structure_transform = transforms.Compose([
            self.MaxResize(max_structure_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Supported classes
        self.supported_classes = {
            'table', 'table row', 'table column', 'table column header'
        }
    
    @property
    def detection_model(self):
        """Lazy-load the table detection model."""
        if self._detection_model is None:
            self._detection_model = AutoModelForObjectDetection.from_pretrained(
                self.detection_model_id, revision="no_timm"
            ).to(self.device)
        return self._detection_model
    
    @property
    def structure_model(self):
        """Lazy-load the table structure recognition model."""
        if self._structure_model is None:
            self._structure_model = AutoModelForObjectDetection.from_pretrained(
                self.structure_model_id
            ).to(self.device)
        return self._structure_model
    
    def box_cxcywh_to_xyxy(self, x):
        """Convert bounding box from center-width format to corner format."""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)
    
    def rescale_bboxes(self, out_bbox, size):
        """Rescale bounding boxes to image size."""
        width, height = size
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
        return boxes
    
    def outputs_to_objects(self, outputs, img_size, id2label):
        """Convert model outputs to structured objects."""
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == 'no object':
                objects.append({
                    'label': class_label, 
                    'score': float(score), 
                    'bbox': [float(elem) for elem in bbox]
                })
        return objects
    
    def detect(self, image_path: str, confidence: float = 0.5,
               classes: Optional[List[str]] = None,
               exclude_classes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Detect tables and their structure in an image.
        
        Args:
            image_path: Path to the image to analyze
            confidence: Minimum confidence threshold for detections
            classes: Specific classes to detect, or None for all supported classes
            exclude_classes: Classes to exclude from detection
            
        Returns:
            List of detected regions with their properties
        """
        # Validate requested classes
        self.validate_classes(classes or [])
        
        # Validate excluded classes
        if exclude_classes:
            self.validate_classes(exclude_classes)
        
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Detect tables
        pixel_values = self.detection_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.detection_model(pixel_values)
        
        id2label = self.detection_model.config.id2label
        id2label[len(id2label)] = "no object"
        tables = self.outputs_to_objects(outputs, image.size, id2label)
        
        # Filter by confidence
        tables = [t for t in tables if t['score'] >= confidence]
        
        # If no tables found, return empty list
        if not tables:
            return []
        
        # Process each table to find its structure
        all_detections = []
        
        # Add tables to detections if requested
        if not classes or 'table' in classes:
            if not exclude_classes or 'table' not in exclude_classes:
                for table in tables:
                    all_detections.append({
                        'bbox': tuple(table['bbox']),
                        'class': 'table',
                        'confidence': float(table['score']),
                        'normalized_class': 'table'
                    })
        
        # Process table structure if needed
        structure_classes = {'table row', 'table column', 'table column header'}
        needed_structure = False
        
        # Check if we need to process structure
        if not classes:
            # No classes specified, detect all non-excluded
            needed_structure = any(c not in (exclude_classes or []) for c in structure_classes)
        else:
            # Specific classes requested
            needed_structure = any(c in classes for c in structure_classes)
        
        if needed_structure:
            for table in tables:
                # Crop the table
                x_min, y_min, x_max, y_max = table['bbox']
                cropped_table = image.crop((x_min, y_min, x_max, y_max))
                
                # Recognize table structure
                structure_pixel_values = self.structure_transform(cropped_table).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    structure_outputs = self.structure_model(structure_pixel_values)
                
                structure_id2label = self.structure_model.config.id2label
                structure_id2label[len(structure_id2label)] = "no object"
                
                # Get table structure elements
                structure_elements = self.outputs_to_objects(structure_outputs, cropped_table.size, structure_id2label)
                
                # Filter by confidence
                structure_elements = [e for e in structure_elements if e['score'] >= confidence]
                
                # Process each structure element
                for element in structure_elements:
                    element_class = element['label']
                    
                    # Skip if specific classes requested and this isn't one of them
                    if classes and element_class not in classes:
                        continue
                        
                    # Skip if this class is in the excluded classes
                    if exclude_classes and element_class in exclude_classes:
                        continue
                    
                    # Adjust coordinates to the original image (add table's top-left corner)
                    x_min_struct, y_min_struct, x_max_struct, y_max_struct = element['bbox']
                    adjusted_bbox = (
                        x_min_struct + x_min,
                        y_min_struct + y_min,
                        x_max_struct + x_min,
                        y_max_struct + y_min
                    )
                    
                    all_detections.append({
                        'bbox': adjusted_bbox,
                        'class': element_class,
                        'confidence': float(element['score']),
                        'normalized_class': self._normalize_class_name(element_class)
                    })
        
        return all_detections
