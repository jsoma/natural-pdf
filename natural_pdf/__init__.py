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


# Global options system
class ConfigSection:
    """A configuration section that holds key-value option pairs."""

    def __init__(self, **defaults):
        self.__dict__.update(defaults)

    def __repr__(self):
        items = [f"{k}={v!r}" for k, v in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(items)})"


class Options:
    """Global options for natural-pdf, similar to pandas options."""

    def __init__(self):
        # Image rendering defaults
        self.image = ConfigSection(width=None, resolution=150)

        # OCR defaults
        self.ocr = ConfigSection(engine="easyocr", languages=["en"], min_confidence=0.5)

        # Text extraction defaults (empty for now)
        self.text = ConfigSection()


# Create global options instance
options = Options()


# Version
__version__ = "0.1.1"

from natural_pdf.analyzers.guides import Guides
from natural_pdf.core.page import Page
from natural_pdf.core.page_collection import PageCollection
from natural_pdf.core.pdf import PDF

# Core imports
from natural_pdf.core.pdf_collection import PDFCollection
from natural_pdf.elements.region import Region
from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion

# Search options (if extras installed)
try:
    from natural_pdf.search.search_options import (
        BaseSearchOptions,
        MultiModalSearchOptions,
        TextSearchOptions,
    )
except ImportError:
    # Define dummy classes if extras not installed, so imports don't break
    class BaseSearchOptions:
        def __init__(self, *args, **kwargs):
            pass

    class TextSearchOptions:
        def __init__(self, *args, **kwargs):
            pass

    class MultiModalSearchOptions:
        def __init__(self, *args, **kwargs):
            pass


# Import QA module if available
try:
    from natural_pdf.qa import DocumentQA, get_qa_engine

    HAS_QA = True
except ImportError:
    HAS_QA = False

# Explicitly define what gets imported with 'from natural_pdf import *'
__all__ = [
    "PDF",
    "PDFCollection",
    "Page",
    "Region",
    "Flow",
    "FlowRegion",
    "Guides",
    "TextSearchOptions",
    "MultiModalSearchOptions",
    "BaseSearchOptions",
    "configure_logging",
    "options",
    "PageCollection",
]

# Add QA components to __all__ if available
if HAS_QA:
    __all__.extend(["DocumentQA", "get_qa_engine"])
