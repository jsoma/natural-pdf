# ocr_engine_base.py
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

# Assuming ocr_options defines BaseOCROptions
from .ocr_options import BaseOCROptions

logger = logging.getLogger(__name__)


class TextRegion:
    """Standard representation of an OCR text region.

    TextRegion provides a standardized format for representing text detected by
    OCR engines, regardless of the underlying engine implementation. This ensures
    consistent interfaces across different OCR backends (EasyOCR, Surya, PaddleOCR, etc.).

    The class handles coordinate normalization and provides utilities for converting
    between different coordinate formats (bounding boxes vs. polygons).

    Attributes:
        bbox: Bounding box coordinates as (x0, y0, x1, y1) tuple.
        text: The recognized text content.
        confidence: Confidence score from 0.0 (low) to 1.0 (high).
        source: Source identifier, typically "ocr" or engine name.

    Example:
        ```python
        # Create from bounding box
        region = TextRegion(
            bbox=(100, 200, 300, 250),
            text="Hello World",
            confidence=0.95
        )

        # Create from polygon coordinates
        polygon = [[100, 200], [300, 200], [300, 250], [100, 250]]
        region = TextRegion.from_polygon(polygon, "Hello World", 0.95)

        # Convert to dictionary for processing
        data = region.to_dict()
        ```
    """

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        text: str,
        confidence: float,
        source: str = "ocr",
    ):
        """
        Initialize a text region.

        Args:
            bbox: Tuple of (x0, y0, x1, y1) coordinates
            text: The recognized text
            confidence: Confidence score (0.0-1.0)
            source: Source of the text region (default: "ocr")
        """
        self.bbox = bbox
        self.text = text
        self.confidence = confidence
        self.source = source

    @classmethod
    def from_polygon(cls, polygon: List[List[float]], text: str, confidence: float):
        """Create from polygon coordinates [[x1,y1], [x2,y2], ...]"""
        x_coords = [float(point[0]) for point in polygon]
        y_coords = [float(point[1]) for point in polygon]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        return cls(bbox, text, confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for compatibility."""
        return {
            "bbox": self.bbox,
            "text": self.text,
            "confidence": self.confidence,
            "source": self.source,
        }


class OCREngine(ABC):
    """Abstract base class for OCR engines.

    This class defines the standard interface that all OCR engines must implement
    in natural-pdf. It provides a consistent API for text recognition regardless
    of the underlying OCR technology (EasyOCR, Surya, PaddleOCR, DocTR, etc.).

    The base class handles common functionality like model caching, parameter
    validation, and result standardization, while concrete implementations
    provide engine-specific processing logic.

    Subclasses must implement:
    - process_single_image(): Core OCR processing for a single image
    - is_available(): Check if the engine dependencies are installed
    - get_supported_languages(): Return list of supported language codes

    Class Attributes:
        DEFAULT_MIN_CONFIDENCE: Default confidence threshold (0.2).
        DEFAULT_LANGUAGES: Default language list (["en"]).
        DEFAULT_DEVICE: Default processing device ("cpu").

    Attributes:
        logger: Logger instance for the specific engine.
        _model: Cached model instance for the engine.
        _initialized: Whether the engine has been initialized.
        _reader_cache: Cache for initialized models/readers.

    Example:
        Implementing a custom OCR engine:
        ```python
        class MyOCREngine(OCREngine):
            @classmethod
            def is_available(cls) -> bool:
                try:
                    import my_ocr_library
                    return True
                except ImportError:
                    return False

            def process_single_image(self, image, languages, min_confidence,
                                   device, detect_only, options):
                # Implement OCR processing
                return text_regions
        ```

        Using an OCR engine:
        ```python
        if EasyOCREngine.is_available():
            engine = EasyOCREngine()
            results = engine.process_image(image, languages=['en', 'es'])
        ```
    """

    # Default values as class constants
    DEFAULT_MIN_CONFIDENCE = 0.2
    DEFAULT_LANGUAGES = ["en"]
    DEFAULT_DEVICE = "cpu"

    def __init__(self):
        """Initializes the base OCR engine."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self._model = None
        self._initialized = False
        self._reader_cache = {}  # Cache for initialized models/readers

    def process_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        detect_only: bool = False,
        options: Optional[BaseOCROptions] = None,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Process a single image or batch of images with OCR.

        Args:
            images: A single PIL Image or a list of PIL Images
            languages: List of languages to use (default: ['en'])
            min_confidence: Minimum confidence threshold (default: 0.2)
            device: Device to use for processing (default: 'cpu')
            detect_only: Whether to only detect text regions without recognition
            options: Engine-specific options

        Returns:
            For a single image: List of text region dictionaries
            For a batch: List of lists of text region dictionaries
        """
        # Convert single image to batch format
        single_image = not isinstance(images, list)
        image_batch = [images] if single_image else images

        # Use default values where parameters are not provided
        effective_languages = languages or self.DEFAULT_LANGUAGES
        effective_confidence = (
            min_confidence if min_confidence is not None else self.DEFAULT_MIN_CONFIDENCE
        )
        effective_device = device or self.DEFAULT_DEVICE

        # Ensure the model is initialized
        self._ensure_initialized(effective_languages, effective_device, options)

        # Process each image in the batch
        results = []
        for img in image_batch:
            # Preprocess the image for the specific engine
            processed_img = self._preprocess_image(img)

            # Process the image with the engine-specific implementation
            raw_results = self._process_single_image(processed_img, detect_only, options)

            # Convert results to standardized format
            text_regions = self._standardize_results(raw_results, effective_confidence, detect_only)

            # Convert TextRegion objects to dictionaries for backward compatibility
            region_dicts = [region.to_dict() for region in text_regions]
            results.append(region_dicts)

        # Return results in the appropriate format
        return results[0] if single_image else results

    def _ensure_initialized(
        self, languages: List[str], device: str, options: Optional[BaseOCROptions]
    ):
        """Ensure the model is initialized with the correct parameters."""
        if not self._initialized:
            self._initialize_model(languages, device, options)
            self._initialized = True

    @abstractmethod
    def _initialize_model(
        self, languages: List[str], device: str, options: Optional[BaseOCROptions]
    ):
        """Initialize the OCR model with the given parameters."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _preprocess_image(self, image: Image.Image) -> Any:
        """Convert PIL Image to engine-specific format."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _process_single_image(
        self, image: Any, detect_only: bool, options: Optional[BaseOCROptions]
    ) -> Any:
        """Process a single image with the initialized model."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _standardize_results(
        self, raw_results: Any, min_confidence: float, detect_only: bool
    ) -> List[TextRegion]:
        """Convert engine-specific results to standardized TextRegion objects."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the engine's dependencies are installed and usable.

        Returns:
            True if the engine is available, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_cache_key(self, options: BaseOCROptions) -> str:
        """
        Generates a cache key based on relevant options.
        Subclasses should override if more specific key generation is needed.

        Args:
            options: The options dataclass instance.

        Returns:
            A string cache key.
        """
        lang_key = "-".join(sorted(getattr(options, "languages", self.DEFAULT_LANGUAGES)))
        device_key = str(getattr(options, "device", self.DEFAULT_DEVICE)).lower()
        return f"{self.__class__.__name__}_{lang_key}_{device_key}"

    def _standardize_bbox(self, bbox: Any) -> Tuple[float, float, float, float]:
        """Standardizes bounding boxes to (x0, y0, x1, y1) format. Raises ValueError if standardization fails."""
        # Check if it's already in the correct tuple/list format
        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(isinstance(n, (int, float)) for n in bbox)
        ):
            try:
                return tuple(float(c) for c in bbox[:4])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid number format in bbox: {bbox}") from e

        # Check if it's in polygon format [[x1,y1],[x2,y2],...]
        elif (
            isinstance(bbox, (list, tuple))
            and len(bbox) > 0
            and isinstance(bbox[0], (list, tuple))
            and len(bbox[0]) == 2  # Ensure points are pairs
        ):
            try:
                x_coords = [float(point[0]) for point in bbox]
                y_coords = [float(point[1]) for point in bbox]
                if not x_coords or not y_coords:  # Handle empty polygon case
                    raise ValueError("Empty polygon provided")
                return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            except (ValueError, TypeError, IndexError) as e:
                raise ValueError(f"Invalid polygon format or values: {bbox}") from e

        # If it's neither format, raise an error
        raise ValueError(f"Could not standardize bounding box from unexpected format: {bbox}")

    def __del__(self):
        """Cleanup resources when the engine is deleted."""
        self.logger.info(f"Cleaning up {self.__class__.__name__} resources.")
        # Clear reader cache to free up memory/GPU resources
        self._reader_cache.clear()
