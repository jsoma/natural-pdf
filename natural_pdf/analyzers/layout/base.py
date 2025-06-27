# layout_detector_base.py
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

from PIL import Image

# Assuming layout_options defines BaseLayoutOptions
try:
    from .layout_options import BaseLayoutOptions
except ImportError:
    # Placeholder if run standalone or options not found
    class BaseLayoutOptions:
        pass


logger = logging.getLogger(__name__)


class LayoutDetector(ABC):
    """Abstract base class for layout detection engines.

    This class defines the standard interface that all layout detection engines
    must implement in natural-pdf. Layout detectors analyze document images to
    identify structural elements like tables, figures, headers, paragraphs, etc.

    The base class provides common functionality including model caching, result
    standardization, and configuration management, while concrete implementations
    handle engine-specific detection logic for different models (YOLO, TATR, Surya, etc.).

    Subclasses must implement:
    - detect(): Core layout detection for a single image
    - is_available(): Check if engine dependencies are installed
    - _load_model_from_options(): Load and configure the detection model
    - _get_cache_key(): Generate cache keys for model instances

    Subclasses should also populate the 'supported_classes' set with the document
    element types they can detect (e.g., 'table', 'figure', 'text', 'title').

    Attributes:
        logger: Logger instance for the specific detector.
        supported_classes: Set of document element types this detector can identify.
        _model_cache: Dictionary cache for loaded model instances.

    Example:
        Implementing a custom layout detector:
        ```python
        class MyLayoutDetector(LayoutDetector):
            def __init__(self):
                super().__init__()
                self.supported_classes = {'table', 'figure', 'text'}

            @classmethod
            def is_available(cls) -> bool:
                try:
                    import my_layout_library
                    return True
                except ImportError:
                    return False

            def detect(self, image, options):
                # Implement layout detection
                return detection_results
        ```

        Using a layout detector:
        ```python
        if YOLODetector.is_available():
            detector = YOLODetector()
            results = detector.detect(page_image, options)
            for result in results:
                print(f"Found {result['class']} at {result['bbox']}")
        ```
    """

    def __init__(self):
        """Initializes the base layout detector."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.supported_classes: Set[str] = set()  # Subclasses should populate this
        self._model_cache: Dict[str, Any] = {}  # Cache for initialized models

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
                raise
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
        if not isinstance(name, str):
            name = str(name)  # Ensure string
        return name.lower().replace(" ", "-").replace("_", "-")

    def validate_classes(self, classes: List[str]) -> None:
        """
        Validate that the requested classes are supported by this detector.

        Args:
            classes: List of class names to validate.

        Raises:
            ValueError: If any class is not supported.
        """
        if not self.supported_classes:
            self.logger.warning(
                "Supported classes not defined for this detector. Skipping class validation."
            )
            return

        if classes:
            normalized_supported = {self._normalize_class_name(c) for c in self.supported_classes}
            normalized_requested = {self._normalize_class_name(c) for c in classes}
            unsupported_normalized = normalized_requested - normalized_supported

            if unsupported_normalized:
                # Find original names of unsupported classes for better error message
                unsupported_original = [
                    c for c in classes if self._normalize_class_name(c) in unsupported_normalized
                ]
                raise ValueError(
                    f"Classes not supported by {self.__class__.__name__}: {unsupported_original}. "
                    f"Supported (normalized): {sorted(list(normalized_supported))}"
                )

    def __del__(self):
        """Cleanup resources."""
        self.logger.info(f"Cleaning up {self.__class__.__name__} resources.")
        self._model_cache.clear()
