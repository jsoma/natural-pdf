"""
Base OCR engine interface.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from PIL import Image

# Set up module logger
logger = logging.getLogger("natural_pdf.ocr.engine")


class OCREngine(ABC):
    """Base OCR engine interface."""
    
    def __init__(self, **kwargs):
        """
        Initialize with engine-specific settings.
        
        Args:
            **kwargs: Engine-specific settings
        """
        self.logger = logging.getLogger(f"natural_pdf.ocr.{self.__class__.__name__}")
        self.logger.debug(f"Initializing {self.__class__.__name__} with settings: {kwargs}")
    
    @abstractmethod
    def process_image(self, image: Image.Image, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process an image and return standardized results.
        
        Args:
            image: PIL Image to process
            config: OCR configuration:
                - enabled: Whether OCR is enabled
                - languages: List of language codes (ISO format)
                - device: Device to use (e.g., 'cpu', 'cuda')
                - min_confidence: Threshold for result filtering
                - model_settings: Engine-specific settings
            
        Returns:
            List of standardized result dictionaries with:
            - 'bbox': (x0, y0, x1, y1) - Rectangle coordinates
            - 'text': Recognized text
            - 'confidence': Confidence score (0.0-1.0)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this engine's dependencies are installed.
        
        Returns:
            True if the engine can be used, False otherwise
        """
        return False
    
    def normalize_config(self, config: Optional[Union[bool, str, List, Dict]] = None) -> Dict[str, Any]:
        """
        Normalize OCR configuration from various formats.
        
        Args:
            config: OCR configuration in various formats:
                - None: OCR disabled
                - True: OCR enabled with defaults
                - "auto": Auto OCR mode
                - ["en", "fr"]: Use these languages
                - {"languages": ["en"]}: Detailed configuration
                
        Returns:
            Normalized configuration dictionary
        """
        logger.debug(f"Normalizing OCR config: {config}")
        # Base config - Note: default is now enabled=True except for None
        result = {
            "enabled": False,  # Will be updated below for different config types
            "languages": ["en"],
            "device": "cpu",
            "min_confidence": 0.5,
            "model_settings": {}
        }
        
        # Handle simple cases
        if config is None:
            # Keep default of disabled for None
            return result
            
        if config is True:
            result["enabled"] = True
            return result
            
        if isinstance(config, str):
            if config.lower() == "auto":
                result["enabled"] = "auto"
                return result
            else:
                # Assume it's a language code
                result["enabled"] = True
                result["languages"] = [config]
                return result
                
        if isinstance(config, list):
            # Assume it's a list of languages
            result["enabled"] = True
            result["languages"] = config
            return result
            
        if isinstance(config, dict):
            # If enabled isn't explicitly set and we have contents, assume enabled
            if "enabled" not in config:
                # Enable by default if we have settings
                has_settings = (
                    ("languages" in config and config["languages"]) or
                    ("model_settings" in config and config["model_settings"])
                )
                if has_settings:
                    result["enabled"] = True
                
            # Update with provided values
            result.update(config)
            
            # Ensure model_settings exists
            result.setdefault("model_settings", {})
            
            return result
            
        # Fallback for unknown types - enable by default
        result["enabled"] = True
        logger.debug(f"Normalized OCR config: {result}")
        return result
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge OCR configurations, with override_config taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base with
            
        Returns:
            Merged configuration
        """
        logger.debug(f"Merging OCR configs: base={base_config}, override={override_config}")
        result = base_config.copy()
        
        # Special handling for model_settings to ensure deep merge
        if "model_settings" in override_config:
            if "model_settings" not in result:
                result["model_settings"] = {}
            result["model_settings"].update(override_config["model_settings"])
            
        # Merge other top-level keys
        for key, value in override_config.items():
            if key != "model_settings":  # Already handled above
                result[key] = value
                
        logger.debug(f"Merged OCR config result: {result}")
        return result