"""
OCR engines for natural-pdf.

This module provides different OCR engines that can be used with natural-pdf.
"""
import logging

# Set up module logger
logger = logging.getLogger("natural_pdf.ocr")
from .ocr_manager import OCRManager
from .engine import OCREngine
from .ocr_options import OCROptions
from .engine import OCREngine
from .engine_paddle import PaddleOCREngine
from .engine_surya import SuryaOCREngine

__all__ = ['OCRManager', 'OCREngine', 'OCROptions', 'EasyOCREngine', 'PaddleOCREngine', 'SuryaOCREngine']

DEFAULT_ENGINE = SuryaOCREngine

def get_engine(engine_name=None, **kwargs):
    """
    Get OCR engine by name.
    
    Args:
        engine_name: Name of the engine to use ('easyocr', 'paddleocr', etc.)
                     If None, the default engine is used (PaddleOCR if available, otherwise EasyOCR)
        **kwargs: Additional arguments to pass to the engine constructor
        
    Returns:
        OCREngine instance
    """
    logger.debug(f"Initializing OCR engine: {engine_name or 'default'}")
    
    if engine_name is None or engine_name == 'default':
        engine = DEFAULT_ENGINE(**kwargs)
        logger.info(f"Using default OCR engine: {engine.__class__.__name__}")
        return engine
    
    if engine_name.lower() == 'easyocr':
        logger.info("Initializing EasyOCR engine")
        return EasyOCREngine(**kwargs)
    
    if engine_name.lower() == 'paddleocr':
        try:
            from .engine_paddle import PaddleOCREngine
            logger.info("Initializing PaddleOCR engine")
            return PaddleOCREngine(**kwargs)
        except ImportError:
            logger.error("PaddleOCR is not installed")
            raise ImportError(
                "PaddleOCR is not installed. Please install it with: pip install paddlepaddle paddleocr"
            )
    
    logger.error(f"Unknown OCR engine: {engine_name}")
    raise ValueError(f"Unknown OCR engine: {engine_name}")