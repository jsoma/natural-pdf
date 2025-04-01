# ocr_engine_base.py
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image

# Assuming ocr_options defines BaseOCROptions
from .ocr_options import BaseOCROptions

logger = logging.getLogger(__name__)

class OCREngine(ABC):
    """Abstract Base Class for OCR engines."""

    def __init__(self):
        """Initializes the base OCR engine."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self._reader_cache = {} # Cache for initialized models/readers

    @abstractmethod
    def process_image(
        self,
        images: Union[Image.Image, List[Image.Image]], # Accept single or list
        options: BaseOCROptions
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]: # Return single or list of lists
        """
        Processes a single image or a batch of images using the specific engine and options.

        Args:
            images: A single PIL Image or a list of PIL Images.
            options: An instance of a dataclass inheriting from BaseOCROptions
                     containing configuration for this run.

        Returns:
            If input is a single image: List of result dictionaries.
            If input is a list of images: List of lists of result dictionaries,
                                          corresponding to each input image.
                                          An empty list indicates failure for that image.
        """
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
        # Basic key includes languages and device
        lang_key = "-".join(sorted(options.languages))
        device_key = str(options.device).lower()
        return f"{self.__class__.__name__}_{lang_key}_{device_key}"

    def _standardize_bbox(self, bbox: Any) -> Optional[Tuple[float, float, float, float]]:
        """
        Helper to standardize bounding boxes to (x0, y0, x1, y1) format.

        Args:
            bbox: The bounding box in the engine's native format.
                  Expected formats:
                  - List/Tuple of 4 numbers: (x0, y0, x1, y1)
                  - List of points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (polygon)

        Returns:
            Tuple[float, float, float, float] or None if conversion fails.
        """
        try:
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(n, (int, float)) for n in bbox):
                # Already in (x0, y0, x1, y1) format (or similar)
                return tuple(float(c) for c in bbox[:4])
            elif isinstance(bbox, (list, tuple)) and len(bbox) > 0 and isinstance(bbox[0], (list, tuple)):
                # Polygon format [[x1,y1],[x2,y2],...]
                x_coords = [float(point[0]) for point in bbox]
                y_coords = [float(point[1]) for point in bbox]
                x0 = min(x_coords)
                y0 = min(y_coords)
                x1 = max(x_coords)
                y1 = max(y_coords)
                return (x0, y0, x1, y1)
        except Exception as e:
            self.logger.warning(f"Could not standardize bounding box: {bbox}. Error: {e}")
        return None

    def __del__(self):
        """Cleanup resources when the engine is deleted."""
        self.logger.info(f"Cleaning up {self.__class__.__name__} resources.")
        # Clear reader cache to free up memory/GPU resources
        self._reader_cache.clear()

