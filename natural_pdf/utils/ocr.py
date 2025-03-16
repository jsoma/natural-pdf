"""
OCR utilities for natural-pdf.
"""
import base64
import io
import json
import os
import importlib.util
import importlib.resources
import webbrowser
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from PIL import Image


class OCRManager:
    """
    Manager class for OCR operations.
    
    This singleton class handles:
    - OCR engine initialization and caching
    - OCR parameter normalization
    - Detection and recognition operations
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of OCRManager."""
        if cls._instance is None:
            cls._instance = OCRManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the OCR manager."""
        self._readers = {}  # Cache for initialized OCR engines
        self._default_config = {
            "engine": "paddleocr",  # Default to PaddleOCR
            "languages": ["en"],
            "min_confidence": 0.5
            # Engine-specific parameters can be passed directly
        }
    
    def normalize_config(self, config: Optional[Union[bool, str, List, Dict]] = None) -> Dict[str, Any]:
        """
        Normalize OCR configuration from various formats.
        
        Args:
            config: OCR configuration in various formats:
                - None: OCR disabled
                - True: OCR enabled with defaults
                - "auto": Auto OCR mode
                - "easyocr": Use EasyOCR with defaults
                - ["en", "fr"]: Use default engine with these languages
                - {"languages": ["en"]}: Detailed configuration
                
        Returns:
            Normalized configuration dictionary
        """
        if config is None:
            return {"enabled": False}
            
        if config is True:
            return {"enabled": True, **self._default_config}
            
        if isinstance(config, str):
            if config.lower() == "auto":
                return {"enabled": "auto", **self._default_config}
            else:
                # Assume it's an engine name
                return {"enabled": True, "engine": config.lower(), **self._default_config}
                
        if isinstance(config, list):
            # Assume it's a list of languages
            return {"enabled": True, "languages": config, **self._default_config}
            
        if isinstance(config, dict):
            # Start with enabled=True and defaults
            result = {"enabled": True, **self._default_config}
            # Then override with provided values
            result.update(config)
            return result
            
        # Fallback for unknown types
        return {"enabled": False}
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge OCR configurations, with override_config taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base with
            
        Returns:
            Merged configuration
        """
        result = base_config.copy()
        
        # Simple override for top-level keys, except for nested dicts
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Merge nested dicts
                result[key].update(value)
            else:
                # Replace value
                result[key] = value
                
        return result
    
    def get_reader(self, config: Dict[str, Any]) -> Any:
        """
        Get or initialize an OCR reader based on configuration.
        
        Args:
            config: OCR configuration
            
        Returns:
            OCR reader instance
        """
        engine = config.get("engine", "easyocr")
        languages = config.get("languages", ["en"])
        
        # Create a cache key from engine and languages
        cache_key = f"{engine}_{'-'.join(languages)}"
        
        # Return cached reader if available
        if cache_key in self._readers:
            return self._readers[cache_key]
            
        # Initialize new reader based on engine
        if engine == "easyocr":
            # Check if easyocr is installed
            if not importlib.util.find_spec("easyocr"):
                raise ImportError(
                    "EasyOCR is not installed. Please install it with: pip install easyocr"
                )
                
            # Import easyocr
            import easyocr
            
            # Get GPU flag (use GPU if available)
            gpu = config.get("gpu", None)  # None means auto-detect
            
            # Create reader
            reader = easyocr.Reader(
                languages,
                gpu=False,
                download_enabled=config.get("download_enabled", True),
                model_storage_directory=config.get("model_storage_directory", None),
                user_network_directory=config.get("user_network_directory", None),
                recog_network=config.get("recog_network", "standard"),
                detector=config.get("detector", True),
                recognizer=config.get("recognizer", True)
            )
            
            # Cache reader
            self._readers[cache_key] = reader
            return reader
            
        # Add other OCR engines here (tesseract, etc.)
        
        raise ValueError(f"Unsupported OCR engine: {engine}")
    
    def detect_and_recognize(self, image: Image.Image, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run full OCR pipeline on an image (detection + recognition).
        
        Args:
            image: PIL Image to process
            config: OCR configuration
            
        Returns:
            List of OCR results with text, bbox and confidence
        """
        engine = config.get("engine", "easyocr")
        
        if engine == "easyocr":
            return self._easyocr_detect_and_recognize(image, config)
            
        # Add other engines here
        
        raise ValueError(f"Unsupported OCR engine: {engine}")
    
    def _easyocr_detect_and_recognize(self, image: Image.Image, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run EasyOCR on an image.
        
        Args:
            image: PIL Image to process
            config: OCR configuration
            
        Returns:
            List of OCR results with text, bbox and confidence
        """
        # Get reader
        reader = self.get_reader(config)
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # Get parameters directly from config (flatten structure)
        # Default values are based on EasyOCR's defaults
        
        # Detection parameters
        text_threshold = config.get("text_threshold", 0.7)
        low_text = config.get("low_text", 0.4)
        link_threshold = config.get("link_threshold", 0.4)
        canvas_size = config.get("canvas_size", 2560)
        mag_ratio = config.get("mag_ratio", 1.0)
        slope_ths = config.get("slope_ths", 0.1)
        ycenter_ths = config.get("ycenter_ths", 0.5)
        height_ths = config.get("height_ths", 0.5)
        width_ths = config.get("width_ths", 0.5)
        add_margin = config.get("add_margin", 0.1)
        
        # Recognition parameters
        decoder = config.get("decoder", "greedy")
        beamWidth = config.get("beamWidth", 5)
        batch_size = config.get("batch_size", 1)
        workers = config.get("workers", 0)
        allowlist = config.get("allowlist", None)
        blocklist = config.get("blocklist", None)
        detail = config.get("detail", 1)
        paragraph = config.get("paragraph", False)
        min_size = config.get("min_size", 10)
        contrast_ths = config.get("contrast_ths", 0.1)
        adjust_contrast = config.get("adjust_contrast", 0.5)
        
        # For backward compatibility, also check nested structures
        detection_params = config.get("detection_params", {})
        recognition_params = config.get("recognition_params", {})
        
        # Override with nested params if provided (backward compatibility)
        if detection_params:
            text_threshold = detection_params.get("text_threshold", text_threshold)
            low_text = detection_params.get("low_text", low_text)
            link_threshold = detection_params.get("link_threshold", link_threshold)
            canvas_size = detection_params.get("canvas_size", canvas_size)
            mag_ratio = detection_params.get("mag_ratio", mag_ratio)
            slope_ths = detection_params.get("slope_ths", slope_ths)
            ycenter_ths = detection_params.get("ycenter_ths", ycenter_ths)
            height_ths = detection_params.get("height_ths", height_ths)
            width_ths = detection_params.get("width_ths", width_ths)
            add_margin = detection_params.get("add_margin", add_margin)
            
        if recognition_params:
            decoder = recognition_params.get("decoder", decoder)
            beamWidth = recognition_params.get("beamWidth", beamWidth)
            batch_size = recognition_params.get("batch_size", batch_size)
            workers = recognition_params.get("workers", workers)
            allowlist = recognition_params.get("allowlist", allowlist)
            blocklist = recognition_params.get("blocklist", blocklist)
            detail = recognition_params.get("detail", detail)
            paragraph = recognition_params.get("paragraph", paragraph)
            min_size = recognition_params.get("min_size", min_size)
            contrast_ths = recognition_params.get("contrast_ths", contrast_ths)
            adjust_contrast = recognition_params.get("adjust_contrast", adjust_contrast)
        
        # Run OCR
        result = reader.readtext(
            img_array,
            decoder=decoder,
            beamWidth=beamWidth,
            batch_size=batch_size,
            workers=workers,
            allowlist=allowlist,
            blocklist=blocklist,
            detail=detail,
            paragraph=paragraph,
            min_size=min_size,
            contrast_ths=contrast_ths,
            adjust_contrast=adjust_contrast,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
            slope_ths=slope_ths,
            ycenter_ths=ycenter_ths,
            height_ths=height_ths,
            width_ths=width_ths,
            add_margin=add_margin
        )
        
        # Convert to standardized format
        # EasyOCR format depends on the 'detail' parameter:
        # With detail=1 (default): [[bbox, text, confidence], ...]
        # With detail=0: [text, ...]
        # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (clockwise from top-left)
        # We convert to our format: [{'bbox': (x0,y0,x1,y1), 'text': text, 'confidence': conf}, ...]
        
        standardized_results = []
        
        for detection in result:
            # Check the format based on what was returned
            if isinstance(detection, list) and len(detection) >= 3:
                # This is the detailed format (detail=1)
                bbox = detection[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text = detection[1]
                confidence = detection[2]
                
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
                    'confidence': confidence
                })
            elif isinstance(detection, str):
                # Simple format (detail=0), no bbox or confidence
                # This shouldn't happen if we're controlling the detail parameter,
                # but handle it just in case
                standardized_results.append({
                    'bbox': (0, 0, 1, 1),  # Dummy bbox
                    'text': detection,
                    'confidence': 1.0  # Default confidence
                })
            else:
                # Check if it's the polygon format [polygon_points, text, confidence]
                if (hasattr(detection, '__getitem__') and 
                    len(detection) >= 3 and 
                    isinstance(detection[0], list) and 
                    len(detection[0]) >= 4 and 
                    all(isinstance(pt, list) and len(pt) == 2 for pt in detection[0])):
                    
                    # Extract polygon points, text, and confidence
                    polygon = detection[0]  # List of [x,y] points
                    text = detection[1] if isinstance(detection[1], str) else str(detection[1])
                    confidence = float(detection[2]) if len(detection) > 2 else 0.5
                    
                    # Convert polygon to rectangular bbox
                    x_coords = [point[0] for point in polygon]
                    y_coords = [point[1] for point in polygon]
                    x0 = min(x_coords)
                    y0 = min(y_coords)
                    x1 = max(x_coords)
                    y1 = max(y_coords)
                    
                    # Convert the polygon points to tuples for consistency
                    polygon_tuples = [(float(point[0]), float(point[1])) for point in polygon]
                    
                    standardized_results.append({
                        'bbox': (x0, y0, x1, y1),
                        'text': text,
                        'confidence': confidence,
                        'polygon': polygon_tuples  # Store the original polygon points
                    })
                # Handle other unknown formats
                elif hasattr(detection, '__getitem__'):
                    # It's some kind of sequence but not a full polygon format
                    if len(detection) >= 2:
                        text = detection[1] if isinstance(detection[1], str) else str(detection[1])
                        confidence = float(detection[2]) if len(detection) > 2 else 0.5
                        
                        # Try to extract bbox if first element looks like coordinates
                        if isinstance(detection[0], list) and all(isinstance(x, (int, float)) for x in detection[0]):
                            # Just a warning, not an error
                            print(f"Note: Using non-standard OCR format: {detection}")
                            standardized_results.append({
                                'bbox': (0, 0, 1, 1),  # Dummy bbox
                                'text': text,
                                'confidence': confidence
                            })
                        else:
                            standardized_results.append({
                                'bbox': (0, 0, 1, 1),  # Dummy bbox
                                'text': text,
                                'confidence': confidence
                            })
                    else:
                        # Just a warning for truly unknown formats
                        print(f"Warning: Unexpected OCR result format: {detection}")
            
        return standardized_results
        
    def recognize_region(self, image: Image.Image, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run OCR recognition on a specific region.
        
        Args:
            image: PIL Image of the region to process
            config: OCR configuration
            
        Returns:
            List of OCR results with text, bbox and confidence
        """
        # For most OCR engines, we can just use detect_and_recognize on the cropped image
        # Since the region is already extracted, it will just detect/recognize within it
        return self.detect_and_recognize(image, config)


# Function to load the OCR debug HTML template
def get_ocr_debug_template():
    """
    Load the OCR debug HTML template.
    
    Returns:
        str: The HTML template as a string
    """
    try:
        # Try using importlib.resources (Python 3.7+)
        try:
            # For Python 3.9+
            with importlib.resources.files('natural_pdf.templates').joinpath('ocr_debug.html').open('r', encoding='utf-8') as f:
                return f.read()
        except (AttributeError, TypeError):
            # Fallback for Python 3.7-3.8
            return importlib.resources.read_text('natural_pdf.templates', 'ocr_debug.html')
    except (ImportError, FileNotFoundError):
        # Fallback for direct file access (development)
        import os
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        template_path = os.path.join(package_dir, 'templates', 'ocr_debug.html')
        
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"OCR debug template not found at {template_path}")


def debug_ocr_to_html(pages, output_path=None):
    """
    Generate an HTML debug report for OCR results.
    
    Args:
        pages: List of Page objects or a PageCollection
        output_path: Path to save the HTML report (optional)
        
    Returns:
        Path to the generated HTML file, or HTML string if no path provided
    """
    # Prepare the data structure
    pages_data = {"pages": []}
    
    # Process each page
    for i, page in enumerate(pages):
        # Extract OCR elements
        try:
            ocr_elements = page.find_all('text[source=ocr]')
            if not ocr_elements:
                ocr_elements = page.extract_ocr_elements()
        except Exception as e:
            print(f"Error extracting OCR from page {i}: {e}")
            continue
            
        # Skip if no OCR elements found
        if not ocr_elements:
            continue
            
        # Get page image as base64
        img_data = _get_page_image_base64(page)
        
        # Create page data
        page_data = {
            "page_number": page.number,
            "image": img_data,
            "regions": []
        }
        
        # Process OCR elements
        for j, elem in enumerate(ocr_elements):
            region = {
                "id": f"region_{j}",
                "bbox": [elem.x0, elem.top, elem.x1, elem.bottom],
                "ocr_text": elem.text,
                "corrected_text": elem.text,
                "confidence": getattr(elem, 'confidence', 0.0),
                "modified": False
            }
            page_data["regions"].append(region)
            
        pages_data["pages"].append(page_data)
    
    # Get the HTML template and generate the final HTML
    template = get_ocr_debug_template()
    html = template.format(
        pages_data=json.dumps(pages_data)
    )
    
    # Save to file if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        # Try to open the file in browser
        try:
            webbrowser.open('file://' + os.path.abspath(output_path))
        except Exception:
            pass
        return output_path
    
    # Return as string otherwise
    return html


def _get_page_image_base64(page):
    """Generate a base64 encoded image of the page."""
    # Create a clean image of the page
    img = page.show(scale=2.0)
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"