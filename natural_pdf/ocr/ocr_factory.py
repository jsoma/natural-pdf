import logging
from typing import Dict

from natural_pdf.engine_provider import get_provider

from .engine import OCREngine

logger = logging.getLogger(__name__)

# Preference order for recommended engine selection
_ENGINE_PREFERENCE = ["easyocr", "doctr", "paddle", "surya"]


class OCRFactory:
    """Factory for creating and managing OCR engines.

    This is a user-facing facade that delegates to the EngineProvider for
    actual engine management and caching.
    """

    @staticmethod
    def create_engine(engine_type: str, **kwargs) -> OCREngine:
        """Create and return an OCR engine instance.

        Args:
            engine_type: One of 'surya', 'easyocr', 'paddle', 'doctr'
            **kwargs: Arguments to pass to the engine constructor

        Returns:
            An initialized OCR engine

        Raises:
            RuntimeError: If the required dependencies aren't installed
            LookupError: If the engine_type is unknown
        """
        provider = get_provider()
        engine_name = engine_type.lower()

        try:
            return provider.get("ocr", context=None, name=engine_name, **kwargs)
        except LookupError:
            raise ValueError(f"Unknown engine type: {engine_type}")

    @staticmethod
    def list_available_engines() -> Dict[str, bool]:
        """Returns a dictionary of engine names and their availability status."""
        from .ocr_provider import ENGINE_REGISTRY, _instantiate_engine_provider

        engines = {}
        for name in ENGINE_REGISTRY:
            try:
                registry_entry = ENGINE_REGISTRY[name]
                engine_instance = _instantiate_engine_provider(registry_entry["provider"])
                engines[name] = engine_instance.is_available()
            except Exception:
                engines[name] = False
        return engines

    @staticmethod
    def get_recommended_engine(**kwargs) -> OCREngine:
        """Returns the best available OCR engine based on what's installed.

        First tries engines in order of preference: EasyOCR, Doctr, Paddle, Surya.
        If none are available, raises ImportError with installation instructions.

        Args:
            **kwargs: Arguments to pass to the engine constructor

        Returns:
            The best available OCR engine instance

        Raises:
            ImportError: If no engines are available
        """
        available = OCRFactory.list_available_engines()

        # Try engines in order of recommendation
        for engine_name in _ENGINE_PREFERENCE:
            if available.get(engine_name, False):
                logger.info(f"Using {engine_name} OCR engine")
                return OCRFactory.create_engine(engine_name, **kwargs)

        # If we get here, no engines are available
        raise ImportError(
            "No OCR engines are installed. You can add one via the npdf installer, e.g.:\n"
            "  npdf install easyocr   # fastest to set up\n"
            "  npdf install paddle    # best Asian-language accuracy\n"
            "  npdf install surya     # Surya OCR engine\n"
            "  npdf install yolo      # Layout detection (YOLO)\n"
        )
