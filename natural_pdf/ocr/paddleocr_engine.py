"""
PaddleOCR engine implementation.
"""
import importlib.util
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image

from .engine import OCREngine


class PaddleOCREngine(OCREngine):
    """PaddleOCR implementation."""
    
    # Language code mapping from ISO to PaddleOCR codes
    LANGUAGE_MAP = {
        'en': 'en',
        'zh': 'ch',
        'zh-cn': 'ch',
        'zh-tw': 'chinese_cht',
        'ja': 'japan',
        'ko': 'korean',
        'th': 'thai',
        'fr': 'french',
        'de': 'german',
        'ru': 'russian',
        'ar': 'arabic',
        'hi': 'hindi',
        'vi': 'vietnam',
        'fa': 'cyrillic',
        'ur': 'cyrillic',
        'rs': 'serbian',
        'oc': 'latin',
        'rsc': 'cyrillic',
        'bg': 'bulgarian',
        'uk': 'cyrillic',
        'be': 'cyrillic',
        'te': 'telugu',
        'kn': 'kannada',
        'ta': 'tamil',
        'latin': 'latin',   # Direct mapping for some codes
        'cyrillic': 'cyrillic',
        'devanagari': 'devanagari',
    }
    
    def __init__(self, **kwargs):
        """
        Initialize PaddleOCR engine.
        
        Args:
            **kwargs: Engine-specific settings
        """
        super().__init__(**kwargs)
        self._readers = {}  # Cache for readers
        
        # Store initialization settings to use in model initialization
        self._init_settings = kwargs
        
    def is_available(self) -> bool:
        """
        Check if PaddleOCR is installed.
        
        Returns:
            True if PaddleOCR is available, False otherwise
        """
        try:
            import paddleocr
            import paddle
            return True
        except ImportError:
            return False
    
    def map_language(self, language: str) -> str:
        """
        Map ISO language code to PaddleOCR language code.
        
        Args:
            language: ISO language code (e.g., 'en', 'zh-cn')
            
        Returns:
            PaddleOCR language code (e.g., 'en', 'ch')
        """
        return self.LANGUAGE_MAP.get(language.lower(), 'en')
    
    def get_reader(self, config: Dict[str, Any]):
        """
        Get or initialize a PaddleOCR reader based on configuration.
        
        Args:
            config: OCR configuration
            
        Returns:
            PaddleOCR reader instance
        """
        # Get primary language from config and map it to PaddleOCR format
        languages = config.get("languages", ["en"])
        primary_lang = self.map_language(languages[0]) if languages else 'en'
        
        # Handle device parameter mapping
        device = config.get("device", "cpu")
        
        # Create a cache key from configuration
        use_angle_cls = config.get("model_settings", {}).get("use_angle_cls", False)
        cache_key = f"paddleocr_{primary_lang}_{device}_{use_angle_cls}"
        
        # Return cached reader if available
        if cache_key in self._readers:
            return self._readers[cache_key]
        
        # Check if paddleocr is installed
        if not importlib.util.find_spec("paddleocr"):
            raise ImportError(
                "PaddleOCR is not installed. Please install it with: pip install paddlepaddle paddleocr"
            )
        
        # Import paddleocr
        import paddleocr
        
        # Start with initialization settings
        reader_kwargs = self._init_settings.copy()
        
        # Set the language
        reader_kwargs["lang"] = primary_lang
        
        # Apply model_settings if provided
        model_settings = config.get("model_settings", {})
        reader_kwargs.update(model_settings)
        
        # Create reader with specified settings
        reader = paddleocr.PaddleOCR(**reader_kwargs)
        
        # Cache reader
        self._readers[cache_key] = reader
        return reader
    
    def process_image(self, image: Image.Image, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process an image with PaddleOCR.
        
        Args:
            image: PIL Image to process
            config: OCR configuration
            
        Returns:
            List of standardized OCR results
        """
        # Normalize config
        if config is None:
            config = {}
        config = self.normalize_config(config)
        
        # Skip if OCR is disabled
        if not config.get("enabled"):
            return []
        
        # Get reader
        reader = self.get_reader(config)
        
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            # PaddleOCR expects BGR format, but PIL is RGB
            # Make a copy to preserve the original image
            img_array = np.array(image.copy())
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # Convert RGB to BGR for PaddleOCR
                img_array = img_array[:, :, ::-1]
        else:
            img_array = image
        
        # Run OCR
        # PaddleOCR result format: 
        # [
        #   [
        #     [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  # Detection box
        #     ('text', confidence)  # Recognition result
        #   ],
        #   # More results...
        # ]
        print(f"PaddleOCR: Running OCR with cls={config.get('model_settings', {}).get('cls', False)}")
        try:
            # Use cls parameter from config or model_settings
            cls = config.get("model_settings", {}).get("cls", False)
            result = reader.ocr(img_array, cls=cls)
            print(f"PaddleOCR: Got result type {type(result)}")
            if result is not None:
                if isinstance(result, list) and len(result) > 0:
                    page_result = result[0] if isinstance(result[0], list) else result
                    print(f"PaddleOCR: Got {len(page_result)} results")
                else:
                    print(f"PaddleOCR: Got empty result list")
            else:
                print(f"PaddleOCR: Got None result")
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        # Apply minimum confidence threshold
        min_confidence = config.get("min_confidence", 0.5)
        
        # Convert to standardized format
        standardized_results = []
        
        # PaddleOCR might return None if no text is detected
        if result is None:
            return []
        
        # PaddleOCR might return a list of page results or a single page result
        # Handle both cases
        if isinstance(result, list) and len(result) > 0:
            # If it's a list of pages (multi-page input), use the first page result
            # Since we're processing a single image, there should only be one page
            page_result = result[0] if isinstance(result[0], list) else result
            
            for detection in page_result:
                # Check if the detection has the expected structure
                if not isinstance(detection, list) or len(detection) < 2:
                    continue
                
                # Extract the detection box and recognition result
                try:
                    bbox = detection[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    text_confidence = detection[1]  # ('text', confidence)
                    
                    # Extract text and confidence
                    if isinstance(text_confidence, tuple) and len(text_confidence) >= 2:
                        text = text_confidence[0]
                        confidence = float(text_confidence[1])  # Convert to float
                    else:
                        # Fallback if the format is unexpected
                        text = str(text_confidence)
                        confidence = 1.0
                    
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
                except Exception as e:
                    print(f"Error processing PaddleOCR detection: {e}")
                    continue
        
        return standardized_results
    
    def __del__(self):
        """Cleanup resources when the engine is deleted."""
        # Clear reader cache to free up memory
        self._readers.clear()