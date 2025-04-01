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
