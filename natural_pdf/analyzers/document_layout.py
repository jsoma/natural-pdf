"""
Document layout analysis for natural-pdf.

This module provides functionality for detecting and analyzing the layout
of PDF documents using machine learning models.
"""
import os
import tempfile
import importlib.util
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
import torch
from PIL import Image

from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from torchvision import transforms
from transformers import AutoModelForObjectDetection

from natural_pdf.elements.region import Region

# Set up module logger
logger = logging.getLogger("natural_pdf.analyzers.layout")


class LayoutDetector:
    """
    Base class for document layout detection.
    """
    def __init__(self):
        self.supported_classes: Set[str] = set()
        
    def detect(self, image_path: str, confidence: float = 0.5, 
               classes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Detect layout elements in an image.
        
        Args:
            image_path: Path to the image to analyze
            confidence: Minimum confidence threshold for detections
            classes: Specific classes to detect, or None for all supported classes
            
        Returns:
            List of detected regions with their properties
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _normalize_class_name(self, name: str) -> str:
        """Convert class names with spaces to hyphenated format for selectors."""
        return name.lower().replace(' ', '-')
        
    def validate_classes(self, classes: List[str]) -> None:
        """
        Validate that the requested classes are supported by this detector.
        
        Args:
            classes: List of class names to validate
            
        Raises:
            ValueError: If any class is not supported
        """
        if classes:
            normalized_supported = {self._normalize_class_name(c) for c in self.supported_classes}
            unsupported = [c for c in classes if self._normalize_class_name(c) not in normalized_supported]
            if unsupported:
                raise ValueError(f"Classes not supported by this detector: {unsupported}. "
                               f"Supported classes: {sorted(self.supported_classes)}")


class YOLODocLayoutDetector(LayoutDetector):
    """
    Document layout detector using YOLO model.
    """
    def __init__(self, 
                model_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench",
                model_file: str = "doclayout_yolo_docstructbench_imgsz1024.pt",
                device: str = "cpu"):
        """
        Initialize the YOLO document layout detector.
        
        Args:
            model_repo: Hugging Face repository ID for the model
            model_file: Filename of the model in the repository
            device: Device to use for inference ('cpu' or 'cuda:0', etc.)
        """
        super().__init__()
        self.model_repo = model_repo
        self.model_file = model_file
        self.device = device
        self._model = None
        self._model_path = None
        
        # DocLayout YOLO classes
        self.supported_classes = {
            'title', 'plain text', 'abandon', 'figure', 'figure_caption', 
            'table', 'table_caption', 'table_footnote', 'isolate_formula', 
            'formula_caption'
        }
        
    @property
    def model(self) -> YOLOv10:
        """Lazy-load the model when first needed."""
        if self._model is None:
            self._model_path = hf_hub_download(repo_id=self.model_repo, filename=self.model_file)
            self._model = YOLOv10(self._model_path)
        return self._model
    
    def detect(self, image_path: str, confidence: float = 0.2, 
              classes: Optional[List[str]] = None, 
              exclude_classes: Optional[List[str]] = None,
              image_size: int = 1024) -> List[Dict[str, Any]]:
        """
        Detect layout elements in an image using YOLO.
        
        Args:
            image_path: Path to the image to analyze
            confidence: Minimum confidence threshold for detections
            classes: Specific classes to detect, or None for all supported classes
            exclude_classes: Classes to exclude from detection
            image_size: Size to resize the image to before detection
            
        Returns:
            List of detected regions with their properties
        """
        # Validate requested classes
        self.validate_classes(classes or [])
        
        # Validate excluded classes
        if exclude_classes:
            self.validate_classes(exclude_classes)
        
        # Run model prediction
        results = self.model.predict(
            image_path,
            imgsz=image_size,
            conf=confidence,
            device=self.device
        )
        
        # Process results into standardized format
        detections = []
        for result in results:
            boxes = result.boxes.xyxy  # [x_min, y_min, x_max, y_max]
            labels = result.boxes.cls
            scores = result.boxes.conf
            class_names = result.names
            
            for box, label, score in zip(boxes, labels, scores):
                x_min, y_min, x_max, y_max = box.tolist()
                label_idx = int(label)
                label_name = class_names[label_idx]
                
                # Skip if specific classes requested and this isn't one of them
                if classes and label_name not in classes:
                    continue
                    
                # Skip if this class is in the excluded classes
                if exclude_classes and label_name in exclude_classes:
                    continue
                    
                detections.append({
                    'bbox': (x_min, y_min, x_max, y_max),
                    'class': label_name,
                    'confidence': float(score),
                    'normalized_class': self._normalize_class_name(label_name)
                })
                
        return detections


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


class DoclingLayoutDetector(LayoutDetector):
    """
    Document layout and text recognition using Docling.
    
    Docling provides a hierarchical document understanding system that can analyze:
    - Document structure (headers, text, figures, tables)
    - Text content via integrated OCR
    - Hierarchical relationships between document elements
    """
    
    def __init__(self, verbose=False, **kwargs):
        """
        Initialize the Docling document analyzer.
        
        Args:
            verbose: Whether to enable verbose logging
            **kwargs: Additional parameters to pass to DocumentConverter
        """
        # Set up logger with optional verbose mode
        import logging
        self.logger = logging.getLogger("natural_pdf.analyzers.layout.docling")
        self.original_level = self.logger.level
        if verbose:
            self.logger.setLevel(logging.DEBUG)
            
        super().__init__()
        self.verbose = verbose
        self.converter_kwargs = kwargs
        self._docling_document = None
        self._converter = None
        
    def __del__(self):
        # Restore the original logging level when done
        if hasattr(self, 'logger') and hasattr(self, 'original_level'):
            self.logger.setLevel(self.original_level)
            
    @property
    def converter(self):
        """Lazy-load the DocumentConverter on first use."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                self.logger.debug("Initializing Docling DocumentConverter")
                self._converter = DocumentConverter(**self.converter_kwargs)
            except ImportError:
                raise ImportError(
                    "Docling integration requires docling. "
                    "Install with: pip install docling"
                )
        return self._converter
    
    def detect(self, image_path, confidence=0.5, classes=None, exclude_classes=None):
        """
        Detect document structure and text using Docling.
        
        Args:
            image_path: Path to the image or PDF to analyze
            confidence: Minimum confidence threshold for detections (not used by Docling)
            classes: Specific classes to detect (used for filtering)
            exclude_classes: Classes to exclude from detection (used for filtering)
            
        Returns:
            List of detection dictionaries with hierarchical information
        """
        self.logger.info(f"Processing {image_path} with Docling")
        
        try:
            # Convert the document using Docling's DocumentConverter
            result = self.converter.convert(image_path)
            doc = result.document
            
            # Store for later use
            self._docling_document = doc
            self.logger.info(f"Docling document created with {len(doc.body.children)} top-level elements")
            
            # Convert Docling document to our detection format
            detections = self._convert_docling_to_detections(doc, confidence, classes, exclude_classes)
            
            return detections
        except Exception as e:
            self.logger.error(f"Error processing with Docling: {e}")
            raise
    
    def _convert_docling_to_detections(self, doc, confidence, classes, exclude_classes):
        """
        Convert a Docling document to our standard detection format.
        
        Args:
            doc: DoclingDocument object
            confidence: Confidence threshold to apply (not used by Docling)
            classes: Classes to include (if specified)
            exclude_classes: Classes to exclude
            
        Returns:
            List of detection dictionaries with hierarchy information
        """
        if not doc or not hasattr(doc, 'body') or not hasattr(doc.body, 'children'):
            self.logger.warning("Invalid or empty Docling document")
            return []
            
        detections = []
        id_to_detection = {}  # Map from Docling ID to detection index
        
        # Process text elements
        if hasattr(doc, 'texts') and doc.texts:
            self.logger.debug(f"Processing {len(doc.texts)} text elements")
            
            # First pass: create detections for all text elements
            for text_elem in doc.texts:
                # Skip if no provenance information
                if not hasattr(text_elem, 'prov') or not text_elem.prov:
                    continue
                    
                # Get the bounding box
                prov = text_elem.prov[0]  # Take first provenance entry
                if not hasattr(prov, 'bbox') or not prov.bbox:
                    continue
                
                bbox = prov.bbox
                
                page_height = doc.pages.get(prov.page_no).size.height if hasattr(doc, 'pages') else 792  # Default letter size
                # Already in top-left coordinates
                t = page_height - bbox.t
                b = page_height - bbox.b
                
                # Ensure top is always less than bottom for PIL coordinates
                if t > b:
                    t, b = b, t
                
                # Get the label and normalize it
                label = str(text_elem.label) if hasattr(text_elem, 'label') else 'text'
                normalized_label = self._normalize_class_name(label)
                
                # Skip if filtered by class
                if classes and normalized_label not in classes:
                    continue
                if exclude_classes and normalized_label in exclude_classes:
                    continue
                
                # Create detection
                detection = {
                    'bbox': (bbox.l, t, bbox.r, b),
                    'class': label,
                    'normalized_class': normalized_label,
                    'confidence': 0.95,  # Default confidence for Docling
                    'text': text_elem.text if hasattr(text_elem, 'text') else None,
                    'docling_id': text_elem.self_ref if hasattr(text_elem, 'self_ref') else None,
                    'parent_id': text_elem.parent.self_ref if hasattr(text_elem, 'parent') and hasattr(text_elem.parent, 'self_ref') else None,
                    'model': 'docling'
                }
                
                detections.append(detection)
                
                # Track by ID for hierarchy reconstruction
                if detection['docling_id']:
                    id_to_detection[detection['docling_id']] = len(detections) - 1
        
        # Process pictures if available
        if hasattr(doc, 'pictures') and doc.pictures:
            self.logger.debug(f"Processing {len(doc.pictures)} picture elements")
            
            for pic_elem in doc.pictures:
                # Skip if no provenance information
                if not hasattr(pic_elem, 'prov') or not pic_elem.prov:
                    continue
                    
                # Get the bounding box
                prov = pic_elem.prov[0]  # Take first provenance entry
                if not hasattr(prov, 'bbox') or not prov.bbox:
                    continue
                
                bbox = prov.bbox
                
                page_height = doc.pages.get(prov.page_no).size.height if hasattr(doc, 'pages') else 792
                # In BOTTOMLEFT system, bbox.t is distance from bottom (higher value = higher on page)
                # In TOPLEFT system, we need distance from top (convert using page_height)
                t = page_height - bbox.t  # Correct: Top is page_height minus the top in BOTTOMLEFT
                b = page_height - bbox.b  # Correct: Bottom is page_height minus the bottom in BOTTOMLEFT
                
                # Ensure top is always less than bottom for PIL coordinates
                if t > b:
                    t, b = b, t
                
                label = 'figure'  # Default label for pictures
                normalized_label = 'figure'
                
                # Skip if filtered by class
                if classes and normalized_label not in classes:
                    continue
                if exclude_classes and normalized_label in exclude_classes:
                    continue
                
                # Create detection
                detection = {
                    'bbox': (bbox.l, t, bbox.r, b),
                    'class': label,
                    'normalized_class': normalized_label,
                    'confidence': 0.95,  # Default confidence
                    'docling_id': pic_elem.self_ref if hasattr(pic_elem, 'self_ref') else None,
                    'parent_id': pic_elem.parent.self_ref if hasattr(pic_elem, 'parent') and hasattr(pic_elem.parent, 'self_ref') else None,
                    'model': 'docling'
                }
                
                detections.append(detection)
                
                # Track by ID for hierarchy reconstruction
                if detection['docling_id']:
                    id_to_detection[detection['docling_id']] = len(detections) - 1
                    
        # Process tables if available
        if hasattr(doc, 'tables') and doc.tables:
            self.logger.debug(f"Processing {len(doc.tables)} table elements")
            
            for table_elem in doc.tables:
                # Skip if no provenance information
                if not hasattr(table_elem, 'prov') or not table_elem.prov:
                    continue
                    
                # Get the bounding box
                prov = table_elem.prov[0]  # Take first provenance entry
                if not hasattr(prov, 'bbox') or not prov.bbox:
                    continue
                
                bbox = prov.bbox
                
                # Convert from bottom-left to top-left coordinates
                page_height = doc.pages.get(prov.page_no).size.height if hasattr(doc, 'pages') else 792
                # In BOTTOMLEFT system, bbox.t is distance from bottom (higher value = higher on page)
                # In TOPLEFT system, we need distance from top (convert using page_height)
                t = page_height - bbox.t  # Correct: Top is page_height minus the top in BOTTOMLEFT
                b = page_height - bbox.b  # Correct: Bottom is page_height minus the bottom in BOTTOMLEFT
                
                # Ensure top is always less than bottom for PIL coordinates
                if t > b:
                    t, b = b, t
                
                label = 'table'  # Default label for tables
                normalized_label = 'table'
                
                # Skip if filtered by class
                if classes and normalized_label not in classes:
                    continue
                if exclude_classes and normalized_label in exclude_classes:
                    continue
                
                # Create detection
                detection = {
                    'bbox': (bbox.l, t, bbox.r, b),
                    'class': label,
                    'normalized_class': normalized_label,
                    'confidence': 0.95,  # Default confidence
                    'docling_id': table_elem.self_ref if hasattr(table_elem, 'self_ref') else None,
                    'parent_id': table_elem.parent.self_ref if hasattr(table_elem, 'parent') and hasattr(table_elem.parent, 'self_ref') else None,
                    'model': 'docling'
                }
                
                detections.append(detection)
                
                # Track by ID for hierarchy reconstruction
                if detection['docling_id']:
                    id_to_detection[detection['docling_id']] = len(detections) - 1
        
        self.logger.info(f"Created {len(detections)} detections from Docling document")
        return detections
    
    def get_docling_document(self):
        """Get the original Docling document for advanced usage."""
        return self._docling_document


class PaddleLayoutDetector(LayoutDetector):
    """
    Document layout and table structure detector using PaddlePaddle's PP-Structure.
    """
    def __init__(self, 
                lang: str = "en",
                use_angle_cls: bool = False,
                device: str = "cpu",
                enable_table: bool = True,
                show_log: bool = False,
                detect_text: bool = True,
                verbose: bool = False):
        """
        Initialize the PaddlePaddle layout detector.
        
        Args:
            lang: Language code for the detector ('en', 'ch', etc.)
            use_angle_cls: Whether to use text orientation detection
            device: Device to run inference on ('cpu' or 'gpu')
            enable_table: Whether to use PP-Structure table detection
            show_log: Whether to show PaddleOCR logs
            detect_text: Whether to use direct text detection in addition to layout
            verbose: Whether to show detailed detection information
        """
        # Set a module-specific logger
        self.logger = logging.getLogger("natural_pdf.analyzers.layout.paddle")
        # Store current level to restore it later
        self.original_level = self.logger.level
        # Set to DEBUG if verbose is True
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        super().__init__()
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.device = device
        self.enable_table = enable_table
        self.show_log = show_log
        self.detect_text = detect_text
        self.verbose = verbose
        self._ppstructure = None
        
    def __del__(self):
        # Restore the original logging level
        self.logger.setLevel(self.original_level)
        
        # Validate PaddlePaddle availability
        if not self._is_paddle_available():
            raise ImportError(
                "PaddlePaddle and PaddleOCR are required for PaddleLayoutDetector. "
                "Please install them with: pip install paddlepaddle paddleocr"
            )
        
        # Supported classes by PP-Structure
        self.supported_classes = {
            'text', 'title', 'figure', 'figure_caption', 
            'table', 'table_caption', 'table_cell', 'table_row', 'table_column',
            'header', 'footer', 'reference', 'equation'
        }
    
    def _is_paddle_available(self) -> bool:
        """Check if PaddlePaddle and PaddleOCR are installed."""
        paddle_spec = importlib.util.find_spec("paddle")
        paddleocr_spec = importlib.util.find_spec("paddleocr")
        return paddle_spec is not None and paddleocr_spec is not None
    
    @property
    def ppstructure(self):
        """Lazy-load the PP-Structure model."""
        if self._ppstructure is None:
            # Import here to avoid dependency if not used
            from paddleocr import PPStructure
            
            # Initialize PP-Structure with minimal settings
            # Note: Paddleocr's PPStructure requires minimal parameters to work correctly
            layout_config = {
                'show_log': self.show_log,
                'lang': self.lang
            }
            
            # Initialize PP-Structure with enhanced settings
            self._ppstructure = PPStructure(**layout_config)
        return self._ppstructure
    
    def detect(self, image_path: str, confidence: float = 0.5,
              classes: Optional[List[str]] = None,
              exclude_classes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Detect layout elements in an image using PaddlePaddle.
        
        Args:
            image_path: Path to the image to analyze
            confidence: Minimum confidence threshold for detections
            classes: Specific classes to detect, or None for all supported classes
            exclude_classes: Classes to exclude from detection
            
        Returns:
            List of detected regions with their properties
        """
        self.logger.info(f"Starting PaddleLayout detection on {image_path}")
        self.logger.debug(f"Parameters: confidence={confidence}, classes={classes}, exclude_classes={exclude_classes}, detect_text={self.detect_text}")
        # Validate requested classes
        self.validate_classes(classes or [])
        
        # Validate excluded classes
        if exclude_classes:
            self.validate_classes(exclude_classes)
        
        # Convert classes to lowercase for matching
        classes_lower = [c.lower() for c in (classes or [])]
        exclude_classes_lower = [c.lower() for c in (exclude_classes or [])]
        
        # Process image with PP-Structure
        try:
            # Try to run PPStructure on the image directly
            result = self.ppstructure(image_path)
            
            # Debug output for troubleshooting
            self.logger.debug(f"PaddleLayout detected {len(result)} regions")
            for i, reg in enumerate(result):
                self.logger.debug(f"  Region {i+1}: type={reg.get('type', 'unknown')}, "
                                 f"confidence={reg.get('score', 0.0)}, "
                                 f"bbox={reg.get('bbox', [])}")
        except Exception as e:
            self.logger.error(f"Error in PaddleLayout detection: {e}")
            return []
            
        # If no results, return empty list
        if not result:
            self.logger.warning("PaddleLayout returned empty results")
            return []
            
        # Create detections list with the layout regions
        detections = []
        
        # Process standard layout results
        for region in result:
            try:
                region_type = region.get('type', '').lower()
                
                # Skip if specific classes requested and this isn't one of them
                if classes and region_type not in classes_lower:
                    continue
                
                # Skip if this class is in the excluded classes
                if exclude_classes and region_type in exclude_classes_lower:
                    continue
                
                # Get confidence score (default to 0.99 if not provided)
                confidence_score = region.get('score', 0.99)
                
                # Skip if confidence is below threshold
                if confidence_score < confidence:
                    continue
                
                # Get bounding box
                bbox = region.get('bbox', [0, 0, 0, 0])
                if len(bbox) < 4:
                    print(f"Invalid bbox format: {bbox}, skipping region")
                    continue
                    
                x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # Normalize the class name for our system
                if region_type == 'figure':
                    normalized_type = 'figure'
                elif region_type in ('text', 'header', 'footer', 'reference'):
                    normalized_type = 'plain-text'
                elif region_type == 'table':
                    normalized_type = 'table'
                elif region_type == 'title':
                    normalized_type = 'title'
                elif region_type == 'equation':
                    normalized_type = 'isolate-formula'
                else:
                    normalized_type = region_type.replace(' ', '-')
                
                # Add detection
                detections.append({
                    'bbox': (x_min, y_min, x_max, y_max),
                    'class': region_type,
                    'confidence': confidence_score,
                    'normalized_class': normalized_type,
                    'source': 'layout',
                    'model': 'paddle'
                })
            except Exception as e:
                self.logger.error(f"Error processing layout region: {e}, region data: {region}")
        
        # Always add text box regions from the direct OCR if detect_text is enabled
        if self.detect_text:
            try:
                # Import PaddleOCR
                from paddleocr import PaddleOCR
                
                # Use PaddleOCR directly for text detection only (no recognition for speed)
                ocr = PaddleOCR(lang=self.lang, show_log=self.show_log)
                ocr_result = ocr.ocr(image_path, det=True, rec=False, cls=False)
                
                # Now add text box regions if available
                if ocr_result and len(ocr_result) > 0 and len(ocr_result[0]) > 0:
                    text_boxes = ocr_result[0]
                    self.logger.debug(f"Adding {len(text_boxes)} text box regions from OCR detection")
                    
                    for text_box in text_boxes:
                        try:
                            # Get box coordinates - these are actually lists of points, not lists of [box, text, confidence]
                            # when using det=True, rec=False
                            points = text_box
                            
                            # When using det=True, rec=False, there's no text or confidence
                            # Just the polygon points, so we use default values
                            text = ""
                            text_confidence = 0.95  # High default confidence for detection
                            
                            # Skip if confidence is below threshold
                            if text_confidence < confidence:
                                continue
                            
                            # Calculate bounding box
                            x_coords = [p[0] for p in points]
                            y_coords = [p[1] for p in points]
                            x0, y0 = min(x_coords), min(y_coords)
                            x1, y1 = max(x_coords), max(y_coords)
                            
                            # Add detection with original polygon points
                            detections.append({
                                'bbox': (x0, y0, x1, y1),
                                'class': 'text',
                                'confidence': text_confidence,
                                'normalized_class': 'plain-text',
                                'polygon': points,
                                'text': text,
                                'source': 'ocr',
                                'model': 'paddle'
                            })
                        except Exception as e:
                            self.logger.error(f"Error processing text box: {e}, box data: {text_box}")
            except Exception as e:
                self.logger.error(f"Error adding OCR text boxes: {e}")
                # Continue with standard layout detection only
        
        # Process table cells if available and not excluded
        for region in result:
            region_type = region.get('type', '').lower()
            
            # Skip if not a table or table handling is disabled
            if region_type != 'table' or not self.enable_table:
                continue
                
            # Get confidence score (default to 0.99 if not provided)
            confidence_score = region.get('score', 0.99)
            
            # Get bounding box for coordinate translation
            bbox = region.get('bbox', [0, 0, 0, 0])
            x_min, y_min = bbox[0], bbox[1]
            
            # Process cells if available
            if 'res' in region and isinstance(region['res'], dict) and 'cells' in region['res']:
                cells = region['res']['cells']
                
                # Process cells, rows, and columns if requested
                process_cells = not classes or 'table_cell' in classes_lower
                process_cells = process_cells and ('table_cell' not in exclude_classes_lower)
                
                if process_cells:
                    for cell in cells:
                        # Convert cell coordinates to global coordinates
                        cell_bbox = cell.get('bbox', [0, 0, 0, 0])
                        cell_x_min = cell_bbox[0] + x_min
                        cell_y_min = cell_bbox[1] + y_min
                        cell_x_max = cell_bbox[2] + x_min
                        cell_y_max = cell_bbox[3] + y_min
                        
                        # Add cell detection
                        detections.append({
                            'bbox': (cell_x_min, cell_y_min, cell_x_max, cell_y_max),
                            'class': 'table_cell',
                            'confidence': confidence_score * 0.9,  # Slightly lower confidence for cells
                            'normalized_class': 'table-cell',
                            'row_idx': cell.get('row_idx', 0),
                            'col_idx': cell.get('col_idx', 0),
                            'source': 'layout'
                        })
        
        self.logger.info(f"PaddleLayout detection completed with {len(detections)} regions")
        return detections


def convert_to_regions(page: Any, detections: List[Dict[str, Any]], 
                      scale_factor: float = 1.0) -> List[Region]:
    """
    Convert layout detections to Region objects.
    
    Args:
        page: Page object to create regions for
        detections: List of detection dictionaries
        scale_factor: Factor to scale coordinates from image to PDF space
        
    Returns:
        List of Region objects with layout metadata
    """
    conversion_logger = logging.getLogger("natural_pdf.analyzers.layout.convert")
    conversion_logger.debug(f"Converting {len(detections)} detections to regions with scale {scale_factor}")
    regions = []
    
    for det in detections:
        # Extract detection info
        x_min, y_min, x_max, y_max = det['bbox']
        
        # Ensure coordinates are in proper order (min values are smaller)
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        
        # Scale coordinates from image to PDF space
        if scale_factor != 1.0:
            x_min *= scale_factor
            y_min *= scale_factor
            x_max *= scale_factor
            y_max *= scale_factor
            
        # Create region with metadata
        region = Region(page, (x_min, y_min, x_max, y_max))
        region.region_type = det['class']
        region.confidence = det['confidence']
        region.normalized_type = det['normalized_class']
        
        # Add source info - important for filtering
        region.source = det.get('source', 'detected')
        region.model = det.get('model', 'unknown')
        
        # Add additional metadata if available
        for key, value in det.items():
            if key not in ('bbox', 'class', 'confidence', 'normalized_class', 'source', 'model'):
                setattr(region, key, value)
        
        regions.append(region)
        
    conversion_logger.debug(f"Created {len(regions)} region objects from {len(detections)} detections")
    return regions