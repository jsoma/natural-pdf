from typing import List, Dict, Any, Optional
from huggingface_hub import hf_hub_download
from .base import LayoutDetector
from doclayout_yolo import YOLOv10

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
