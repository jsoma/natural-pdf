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

        # Layout and navigation defaults
        self.layout = ConfigSection(
            directional_offset=0.01,  # Offset in points when using directional methods
            auto_multipage=False,  # Whether directional methods span pages by default
            directional_within=None,  # Region to constrain directional operations to
        )


# Create global options instance
options = Options()


def set_option(name: str, value):
    """
    Set a global Natural PDF option.

    Args:
        name: Option name in dot notation (e.g., 'layout.auto_multipage')
        value: New value for the option

    Example:
        import natural_pdf as npdf
        npdf.set_option('layout.auto_multipage', True)
        npdf.set_option('ocr.engine', 'surya')
    """
    parts = name.split(".")
    obj = options

    # Navigate to the right section
    for part in parts[:-1]:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            raise KeyError(f"Unknown option section: {part}")

    # Set the final value
    final_key = parts[-1]
    if hasattr(obj, final_key):
        setattr(obj, final_key, value)
    else:
        raise KeyError(f"Unknown option: {name}")


# Version
__version__ = "0.1.1"

# Apply pdfminer patches for known bugs
try:
    from natural_pdf.utils.pdfminer_patches import apply_patches

    apply_patches()
except Exception as e:
    logger.warning(f"Failed to apply pdfminer patches: {e}")

from natural_pdf.analyzers.guides import Guides
from natural_pdf.core.page import Page
from natural_pdf.core.page_collection import PageCollection
from natural_pdf.core.pdf import PDF

# Core imports
from natural_pdf.core.pdf_collection import PDFCollection
from natural_pdf.elements.region import Region
from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion

# Judge for visual classification
from natural_pdf.judge import Decision, Judge, JudgeError, PickResult

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
    "Judge",
    "Decision",
    "PickResult",
    "JudgeError",
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
