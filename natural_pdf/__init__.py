"""
Natural PDF - A more intuitive interface for working with PDFs.
"""

import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create library logger
logger = logging.getLogger("natural_pdf")

# Add a NullHandler to prevent "No handler found" warnings
# (Best practice for libraries)
logger.addHandler(logging.NullHandler())


def configure_logging(level=logging.INFO, handler=None):
    """Configure logging for the natural_pdf package.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        handler: Optional custom handler. Defaults to a StreamHandler.
    """
    # Avoid adding duplicate handlers
    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        return

    if handler is None:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)

    logger.propagate = False


from natural_pdf.core.page import Page
from natural_pdf.core.pdf import PDF
from natural_pdf.elements.collections import ElementCollection
from natural_pdf.elements.region import Region

# Import QA module if available
try:
    from natural_pdf.qa import DocumentQA, get_qa_engine

    HAS_QA = True
except ImportError:
    HAS_QA = False

__version__ = "0.1.1"

__all__ = [
    "PDF",
    "PDFCollection",
    "Page",
    "Region",
    "ElementCollection",
    "TextSearchOptions",
    "MultiModalSearchOptions",
    "BaseSearchOptions",
    "configure_logging",
]

if HAS_QA:
    __all__.extend(["DocumentQA", "get_qa_engine"])


from .collections.pdf_collection import PDFCollection

# Core classes
from .core.pdf import PDF
from .elements.region import Region

# Search options (if extras installed)
try:
    from .search.search_options import BaseSearchOptions, MultiModalSearchOptions, TextSearchOptions
except ImportError:
    # Define dummy classes if extras not installed, so imports don't break
    # but using them will raise the ImportError from check_haystack_availability
    class TextSearchOptions:
        def __init__(self, *args, **kwargs):
            pass

    class MultiModalSearchOptions:
        def __init__(self, *args, **kwargs):
            pass

    class BaseSearchOptions:
        def __init__(self, *args, **kwargs):
            pass


# Expose logging setup? (Optional)
# from . import logging_config
# logging_config.setup_logging()

# Explicitly define what gets imported with 'from natural_pdf import *'
__all__ = [
    "PDF",
    "PDFCollection",
    "Region",
    "TextSearchOptions",  # Include search options
    "MultiModalSearchOptions",
    "BaseSearchOptions",
]
