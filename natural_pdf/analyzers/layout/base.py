# layout_detector_base.py
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Union
from PIL import Image

# Assuming layout_options defines BaseLayoutOptions
try:
    from .layout_options import BaseLayoutOptions
except ImportError:
    # Placeholder if run standalone or options not found
    class BaseLayoutOptions: pass

logger = logging.getLogger(__name__)

class LayoutDetector(ABC):
    """
    Abstract Base Class for layout detection engines.

    Subclasses should implement is_available, _load_model_from_options, detect,
    and override _get_cache_key if model loading depends on options beyond device.
    They should also populate the 'supported_classes' set.
    """

    def __init__(self):
        """Initializes the base layout detector."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.supported_classes: Set[str] = set() # Subclasses should populate this
        self._model_cache: Dict[str, Any] = {} # Cache for initialized models

    @abstractmethod
    def detect(self, image: Image.Image, options: BaseLayoutOptions) -> List[Dict[str, Any]]:
        """
        Detect layout elements in a given PIL Image.

        Args:
            image: PIL Image of the page to analyze.
            options: An instance of a dataclass inheriting from BaseLayoutOptions
                     containing configuration for this run.

        Returns:
            List of standardized detection dictionaries with at least:
            - 'bbox': Tuple[float, float, float, float] - (x0, y0, x1, y1) relative to image size
            - 'class': str - Original class name from the model
            - 'confidence': float - Confidence score (0.0-1.0)
            - 'normalized_class': str - Hyphenated, lowercase class name
            - 'model': str - Name of the model used (e.g., 'yolo', 'tatr')
            - 'source': str - Usually 'layout'
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the detector's dependencies are installed and usable.

        Returns:
            True if the detector is available, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_cache_key(self, options: BaseLayoutOptions) -> str:
        """
        Generates a cache key for model loading based on relevant options.
        Subclasses MUST override this to include options that change the loaded model
        (e.g., model path, model name, specific configurations like TATR structure model).

        Args:
            options: The options dataclass instance.

        Returns:
            A string cache key.
        """
        # Base key only includes device, subclasses MUST add model specifics
        device_key = str(options.device).lower()
        return f"{self.__class__.__name__}_{device_key}"

    def _get_model(self, options: BaseLayoutOptions) -> Any:
        """
        Gets or initializes the underlying model based on options, using caching.
        Subclasses must implement _load_model_from_options.
        """
        cache_key = self._get_cache_key(options)
        if cache_key not in self._model_cache:
             self.logger.info(f"Loading model for cache key: {cache_key}")
             try:
                 # Ensure dependencies are met before loading
                 if not self.is_available():
                      raise RuntimeError(f"{self.__class__.__name__} dependencies are not met.")
                 self._model_cache[cache_key] = self._load_model_from_options(options)
                 self.logger.info(f"Model loaded successfully for key: {cache_key}")
             except Exception as e:
                  self.logger.error(f"Failed to load model for key {cache_key}: {e}", exc_info=True)
                  # Remove potentially corrupted cache entry
                  self._model_cache.pop(cache_key, None)
                  raise # Re-raise exception after logging
        else:
             self.logger.debug(f"Using cached model for key: {cache_key}")
        return self._model_cache[cache_key]

    @abstractmethod
    def _load_model_from_options(self, options: BaseLayoutOptions) -> Any:
        """
        Abstract method for subclasses to implement the actual model loading logic
        based on the provided options. Should return the loaded model object(s).
        Should handle necessary imports internally.
        """
        raise NotImplementedError("Subclasses must implement _load_model_from_options")

    def _normalize_class_name(self, name: str) -> str:
        """Convert class names with spaces/underscores to hyphenated lowercase format."""
        if not isinstance(name, str): name = str(name) # Ensure string
        return name.lower().replace(' ', '-').replace('_', '-')

    def validate_classes(self, classes: List[str]) -> None:
        """
        Validate that the requested classes are supported by this detector.

        Args:
            classes: List of class names to validate.

        Raises:
            ValueError: If any class is not supported.
        """
        if not self.supported_classes:
             self.logger.warning("Supported classes not defined for this detector. Skipping class validation.")
             return

        if classes:
            # Normalize both requested and supported classes for comparison
            normalized_supported = {self._normalize_class_name(c) for c in self.supported_classes}
            normalized_requested = {self._normalize_class_name(c) for c in classes}
            unsupported_normalized = normalized_requested - normalized_supported

            if unsupported_normalized:
                # Find original names of unsupported classes for better error message
                unsupported_original = [
                    c for c in classes if self._normalize_class_name(c) in unsupported_normalized
                ]
                raise ValueError(f"Classes not supported by {self.__class__.__name__}: {unsupported_original}. "
                               f"Supported (normalized): {sorted(list(normalized_supported))}")

    def __del__(self):
        """Cleanup resources."""
        self.logger.info(f"Cleaning up {self.__class__.__name__} resources.")
        # Clear model cache to free up memory/GPU resources if models are large
        # Consider implications if models are shared or expensive to reload
        # del self._model_cache # Optional: uncomment if models should be released aggressively
        self._model_cache.clear()

