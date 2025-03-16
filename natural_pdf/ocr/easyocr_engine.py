"""
EasyOCR engine implementation.
"""
import importlib.util
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image

from .engine import OCREngine


class EasyOCREngine(OCREngine):
    """EasyOCR implementation."""
    
    def __init__(self, **kwargs):
        """
        Initialize EasyOCR engine with optional settings.
        
        Args:
            **kwargs: Engine-specific settings
        """
        super().__init__(**kwargs)
        self._readers = {}  # Cache for readers
        
        # Store initialization settings to use in model initialization
        self._init_settings = kwargs
        
    def is_available(self) -> bool:
        """
        Check if EasyOCR is installed.
        
        Returns:
            True if EasyOCR is available, False otherwise
        """
        try:
            import easyocr
            return True
        except ImportError:
            return False
    
    def get_reader(self, config: Dict[str, Any]):
        """
        Get or initialize an EasyOCR reader based on configuration.
        
        Args:
            config: OCR configuration
            
        Returns:
            EasyOCR reader instance
        """
        print(f"EasyOCR.get_reader: Config = {config}")
        
        # Get languages from config
        languages = config.get("languages", ["en"])
        print(f"EasyOCR.get_reader: Languages = {languages}")
        
        # Create a cache key from languages
        cache_key = f"easyocr_{'-'.join(languages)}"
        print(f"EasyOCR.get_reader: Cache key = {cache_key}")
        
        # Return cached reader if available
        if cache_key in self._readers:
            print(f"EasyOCR.get_reader: Using cached reader")
            return self._readers[cache_key]
        
        # Check if easyocr is installed
        if not importlib.util.find_spec("easyocr"):
            print(f"EasyOCR.get_reader: EasyOCR not installed")
            raise ImportError(
                "EasyOCR is not installed. Please install it with: pip install easyocr"
            )
        
        # Import easyocr
        print(f"EasyOCR.get_reader: Importing easyocr")
        import easyocr
        
        # Start with initialization settings
        reader_kwargs = self._init_settings.copy()
        print(f"EasyOCR.get_reader: Init settings = {reader_kwargs}")
        
        # Add languages parameter
        reader_kwargs["lang_list"] = languages
        
        # Handle device parameter mapping
        if "device" in config:
            device = config["device"]
            if device.startswith("cuda"):
                reader_kwargs["gpu"] = True
            else:
                reader_kwargs["gpu"] = False
            print(f"EasyOCR.get_reader: Set gpu={reader_kwargs.get('gpu', False)} from device={device}")
        
        # Apply model_settings if provided
        model_settings = config.get("model_settings", {})
        reader_kwargs.update(model_settings)
        print(f"EasyOCR.get_reader: Final kwargs = {reader_kwargs}")
        
        # Create reader with specified settings
        print(f"EasyOCR.get_reader: Creating EasyOCR.Reader with lang_list={languages}")
        try:
            reader = easyocr.Reader(**reader_kwargs)
            print(f"EasyOCR.get_reader: Successfully created reader")
        except Exception as e:
            print(f"EasyOCR.get_reader: Error creating reader: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Cache reader
        self._readers[cache_key] = reader
        print(f"EasyOCR.get_reader: Reader cached with key {cache_key}")
        return reader
    
    def process_image(self, image: Image.Image, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process an image with EasyOCR.
        
        Args:
            image: PIL Image to process
            config: OCR configuration
            
        Returns:
            List of standardized OCR results
        """
        print(f"EasyOCR.process_image: Starting with image type {type(image)}, size {image.width}x{image.height if hasattr(image, 'height') else 'unknown'}")
        
        # Save image for debugging
        try:
            import os
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "easyocr_debug_input.png")
            if isinstance(image, Image.Image):
                image.save(debug_path)
                print(f"EasyOCR.process_image: Saved input image to {debug_path}")
        except Exception as e:
            print(f"EasyOCR.process_image: Could not save debug image: {e}")
            
        # Normalize config
        if config is None:
            config = {}
        print(f"EasyOCR.process_image: Raw config = {config}")
        config = self.normalize_config(config)
        print(f"EasyOCR.process_image: Normalized config = {config}")
        
        # Skip if OCR is disabled
        if not config.get("enabled"):
            print(f"EasyOCR.process_image: OCR is disabled in config, returning empty list")
            return []
            
        # Direct test with known working code for debug
        print(f"EasyOCR.process_image: Running direct test with EasyOCR")
        try:
            import easyocr
            raw_reader = easyocr.Reader(['en'])
            import numpy as np
            img_array = np.array(image)
            direct_result = raw_reader.readtext(img_array)
            print(f"EasyOCR.process_image: Direct test got {len(direct_result)} results")
        except Exception as e:
            print(f"EasyOCR.process_image: Direct test failed: {e}")
        
        # Get reader
        reader = self.get_reader(config)
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Extract model_settings for readtext parameters
        model_settings = config.get("model_settings", {})
        
        # For backward compatibility, handle both flattened and nested parameters
        readtext_kwargs = {}
        
        # Add all model_settings to kwargs
        readtext_kwargs.update(model_settings)
        
        # For backward compatibility, also check nested structures
        detection_params = config.get("detection_params", {})
        recognition_params = config.get("recognition_params", {})
        
        # Add nested params if provided
        if detection_params:
            for key, value in detection_params.items():
                if key not in readtext_kwargs:
                    readtext_kwargs[key] = value
                    
        if recognition_params:
            for key, value in recognition_params.items():
                if key not in readtext_kwargs:
                    readtext_kwargs[key] = value
        
        # Run OCR with all parameters
        print(f"EasyOCR: Running OCR with parameters: {readtext_kwargs}")
        try:
            result = reader.readtext(img_array, **readtext_kwargs)
            print(f"EasyOCR: Got {len(result)} results")
        except Exception as e:
            print(f"EasyOCR error: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        # Apply minimum confidence threshold
        min_confidence = config.get("min_confidence", 0.5)
        
        # Convert to standardized format
        standardized_results = []
        
        for detection in result:
            # Check the format based on what was returned
            if isinstance(detection, list) and len(detection) >= 3:
                # This is the detailed format (detail=1)
                bbox = detection[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text = detection[1]
                confidence = detection[2]
                
                # Skip if confidence is below threshold
                if confidence < min_confidence:
                    continue
                
                # Convert polygon bbox to rectangle (x0, y0, x1, y1)
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x0 = min(x_coords)
                y0 = min(y_coords)
                x1 = max(x_coords)
                y1 = max(y_coords)
                
                standardized_results.append({
                    'bbox': (x0, y0, x1, y1),
                    'text': text,
                    'confidence': confidence,
                    'source': 'ocr'
                })
            elif isinstance(detection, str):
                # Simple format (detail=0), no bbox or confidence
                standardized_results.append({
                    'bbox': (0, 0, 1, 1),  # Dummy bbox
                    'text': detection,
                    'confidence': 1.0,  # Default confidence
                    'source': 'ocr'
                })
        
        return standardized_results
        
    def __del__(self):
        """Cleanup resources when the engine is deleted."""
        # Clear reader cache to free up memory
        self._readers.clear()