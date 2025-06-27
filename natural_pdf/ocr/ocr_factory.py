import importlib.util
import logging
from typing import Any, Dict, List, Optional, Type, Union

from .engine import OCREngine

logger = logging.getLogger(__name__)


class OCRFactory:
    """Factory for creating and managing OCR engines with optional dependencies."""

    @staticmethod
    def create_engine(engine_type: str, **kwargs) -> OCREngine:
        """Create and return an OCR engine instance.

        Args:
            engine_type: One of 'surya', 'easyocr', 'paddle', 'doctr'
            **kwargs: Arguments to pass to the engine constructor

        Returns:
            An initialized OCR engine

        Raises:
            ImportError: If the required dependencies aren't installed
            ValueError: If the engine_type is unknown
        """
        if engine_type == "surya":
            try:
                from .engine_surya import SuryaOCREngine

                return SuryaOCREngine(**kwargs)
            except ImportError:
                raise ImportError(
                    "Surya engine requires additional dependencies. "
                    "Install with: npdf install surya"
                )
        elif engine_type == "easyocr":
            try:
                from .engine_easyocr import EasyOCREngine

                return EasyOCREngine(**kwargs)
            except ImportError:
                raise ImportError(
                    "EasyOCR engine requires the 'easyocr' package. "
                    "Install with: pip install easyocr (or npdf install easyocr when available)"
                )
        elif engine_type == "paddle":
            try:
                from .engine_paddle import PaddleOCREngine

                return PaddleOCREngine(**kwargs)
            except ImportError:
                raise ImportError(
                    "PaddleOCR engine requires 'paddleocr' and 'paddlepaddle'. "
                    "Install with: npdf install paddle"
                )
        elif engine_type == "doctr":
            try:
                from .engine_doctr import DoctrOCREngine

                return DoctrOCREngine(**kwargs)
            except ImportError:
                raise ImportError(
                    "Doctr engine requires the 'python-doctr' package. "
                    "Install with: pip install python-doctr[torch]"
                )
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

    @staticmethod
    def list_available_engines() -> Dict[str, bool]:
        """Returns a dictionary of engine names and their availability status."""
        engines = {}

        # Check Surya
        try:
            engines["surya"] = importlib.util.find_spec("surya") is not None
        except ImportError:
            engines["surya"] = False

        # Check EasyOCR
        try:
            engines["easyocr"] = importlib.util.find_spec("easyocr") is not None
        except ImportError:
            engines["easyocr"] = False

        # Check PaddleOCR
        try:
            paddle = (
                importlib.util.find_spec("paddle") is not None
                or importlib.util.find_spec("paddlepaddle") is not None
            )
            paddleocr = importlib.util.find_spec("paddleocr") is not None
            engines["paddle"] = paddle and paddleocr
        except ImportError:
            engines["paddle"] = False

        # Check Doctr
        try:
            engines["doctr"] = importlib.util.find_spec("doctr") is not None
        except ImportError:
            engines["doctr"] = False

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
        if available.get("easyocr", False):
            logger.info("Using EasyOCR engine (recommended)")
            return OCRFactory.create_engine("easyocr", **kwargs)
        elif available.get("doctr", False):
            logger.info("Using Doctr engine")
            return OCRFactory.create_engine("doctr", **kwargs)
        elif available.get("paddle", False):
            logger.info("Using PaddleOCR engine")
            return OCRFactory.create_engine("paddle", **kwargs)
        elif available.get("surya", False):
            logger.info("Using Surya OCR engine")
            return OCRFactory.create_engine("surya", **kwargs)

        # If we get here, no engines are available
        raise ImportError(
            "No OCR engines are installed. You can add one via the npdf installer, e.g.:\n"
            "  npdf install easyocr   # fastest to set up\n"
            "  npdf install paddle    # best Asian-language accuracy\n"
            "  npdf install surya     # Surya OCR engine\n"
            "  npdf install yolo      # Layout detection (YOLO)\n"
        )
