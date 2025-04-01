import logging
import importlib.util
from typing import List, Dict, Any, Optional

from .base import LayoutDetector

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
