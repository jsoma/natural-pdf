"""
Natural PDF - A more intuitive interface for working with PDFs.
"""
import logging

# Create library logger
logger = logging.getLogger("natural_pdf")

# Add a NullHandler to prevent "No handler found" warnings
# (Best practice for libraries)
logger.addHandler(logging.NullHandler())

# Utility function for users to easily configure logging
def configure_logging(level=logging.INFO, handler=None):
    """Configure Natural PDF's logging.
    
    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG)
        handler: A custom handler, or None to use StreamHandler
    """
    # Remove NullHandler if present
    if logger.handlers and isinstance(logger.handlers[0], logging.NullHandler):
        logger.removeHandler(logger.handlers[0])
    
    if handler is None:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.setLevel(level)
    
    # Propagate level to all child loggers
    for name in logging.root.manager.loggerDict:
        if name.startswith("natural_pdf."):
            logging.getLogger(name).setLevel(level)

from natural_pdf.core.pdf import PDF
from natural_pdf.core.page import Page
from natural_pdf.elements.region import Region
from natural_pdf.elements.collections import ElementCollection

# Import QA module if available
try:
    from natural_pdf.qa import DocumentQA, get_qa_engine
    HAS_QA = True
except ImportError:
    HAS_QA = False

__version__ = "0.1.0"

if HAS_QA:
    __all__ = ["PDF", "Page", "Region", "ElementCollection", "configure_logging", "DocumentQA", "get_qa_engine"]
else:
    __all__ = ["PDF", "Page", "Region", "ElementCollection", "configure_logging"]