from typing import List, Dict, Any, Optional
import logging
from PIL import Image

from .base import LayoutDetector

class SuryaLayoutDetector(LayoutDetector):
    """
    Document layout detector using Surya models.
    
    Surya provides high-quality layout detection with detailed labeling
    for academic and technical documents.
    """
    
    def __init__(self, 
                model_name: str = "default",
                device: str = None,
                verbose: bool = False):
        """
        Initialize the Surya layout detector.
        
        Args:
            model_name: Name of the layout model to use (default uses the standard model)
            device: Device to run inference on (None for auto-detection)
            verbose: Whether to show detailed logs
        """
        self.logger = logging.getLogger("natural_pdf.analyzers.layout.surya")
        self.original_level = self.logger.level
        if verbose:
            self.logger.setLevel(logging.DEBUG)
            
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        self._layout_predictor = None
        
        # Supported classes based on Surya's layout model
        self.supported_classes = {
            'text', 'pageheader', 'pagefooter', 'sectionheader', 
            'table', 'tableofcontents', 'picture', 'caption', 
            'heading', 'title', 'list', 'listitem', 'code', 
            'textinlinemath', 'mathformula', 'form'
        }
    
    def __del__(self):
        # Restore the original logging level
        if hasattr(self, 'logger') and hasattr(self, 'original_level'):
            self.logger.setLevel(self.original_level)
    
    @property
    def layout_predictor(self):
        """Lazy-load the layout predictor model."""
        if self._layout_predictor is None:
            try:
                from surya.layout import LayoutPredictor
                self.logger.debug("Initializing Surya LayoutPredictor")
                
                # Configure device if provided
                kwargs = {}
                if self.device:
                    kwargs['device'] = self.device
                
                # Initialize the layout predictor
                self._layout_predictor = LayoutPredictor(**kwargs)
            except ImportError:
                raise ImportError(
                    "Surya integration requires surya. "
                    "Install with: pip install surya"
                )
        return self._layout_predictor
    
    def detect(self, image_path: str, confidence: float = 0.5,
              classes: Optional[List[str]] = None,
              exclude_classes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Detect layout elements in an image using Surya.
        
        Args:
            image_path: Path to the image to analyze
            confidence: Minimum confidence threshold for detections
            classes: Specific classes to detect, or None for all supported classes
            exclude_classes: Classes to exclude from detection
            
        Returns:
            List of detected regions with their properties
        """
        self.logger.info(f"Starting Surya layout detection on {image_path}")
        
        # Validate requested classes
        self.validate_classes(classes or [])
        
        # Validate excluded classes
        if exclude_classes:
            self.validate_classes(exclude_classes)
        
        # Convert classes to lowercase for matching
        classes_lower = [c.lower() for c in (classes or [])]
        exclude_classes_lower = [c.lower() for c in (exclude_classes or [])]
        
        try:
            # Load the image using PIL
            image = Image.open(image_path)
            
            # Process with Surya's layout predictor
            layout_predictions = self.layout_predictor([image])
            
            if not layout_predictions:
                self.logger.warning("Surya returned empty predictions")
                return []
                
            # Process results into standardized format
            detections = []
            
            # Get the first prediction (for single image input)
            prediction = layout_predictions[0]
            
            for layout_box in prediction.bboxes:
                # Extract the class name and normalize it
                class_name = layout_box.label.lower()
                normalized_class = self._normalize_class_name(class_name)
                
                # Skip if specific classes requested and this isn't one of them
                if classes and normalized_class not in classes_lower:
                    continue
                
                # Skip if this class is in the excluded classes
                if exclude_classes and normalized_class in exclude_classes_lower:
                    continue
                
                # Skip if confidence is below threshold
                if layout_box.confidence < confidence:
                    continue
                
                # Extract bbox coordinates
                x_min, y_min, x_max, y_max = layout_box.bbox
                
                # Add detection
                detections.append({
                    'bbox': (x_min, y_min, x_max, y_max),
                    'class': class_name,
                    'confidence': layout_box.confidence,
                    'normalized_class': normalized_class,
                    'polygon': layout_box.polygon if hasattr(layout_box, 'polygon') else None,
                    'position': layout_box.position if hasattr(layout_box, 'position') else None,
                    'top_k': layout_box.top_k if hasattr(layout_box, 'top_k') else None,
                    'source': 'layout',
                    'model': 'surya'
                })
                
            self.logger.info(f"Surya detected {len(detections)} layout elements")
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in Surya layout detection: {e}")
            raise