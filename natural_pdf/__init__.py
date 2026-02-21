"""
Natural PDF - A more intuitive interface for working with PDFs.
"""

import logging
import os
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Type

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
            engine="yolo",
            directional_offset=0.01,  # Offset in points when using directional methods
            auto_multipage=False,  # Whether directional methods span pages by default
            directional_within=None,  # Region to constrain directional operations to
        )

        # Table extraction defaults
        self.tables = ConfigSection(engine="pdfplumber")

        # Selector defaults (None = native engine)
        self.selectors = ConfigSection(engine=None)


# Create global options instance
options = Options()

# Option validators: (validator_func, warning_message, corrector_func or None)
# If corrector_func is None, the validator raises an error instead of correcting
_OPTION_VALIDATORS = {
    "ocr.min_confidence": (
        lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0,
        "must be a number between 0.0 and 1.0",
        lambda v: max(0.0, min(1.0, float(v))) if isinstance(v, (int, float)) else 0.5,
    ),
    "image.resolution": (
        lambda v: isinstance(v, (int, float)) and v > 0,
        "must be a positive number",
        lambda v: max(1, int(v)) if isinstance(v, (int, float)) else 150,
    ),
}


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
    from natural_pdf.utils.option_validation import is_strict_mode

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
    if not hasattr(obj, final_key):
        raise KeyError(f"Unknown option: {name}")

    # Validate if validator exists for this option
    if name in _OPTION_VALIDATORS:
        validator, message, corrector = _OPTION_VALIDATORS[name]
        if not validator(value):
            if is_strict_mode():
                from natural_pdf.exceptions import InvalidOptionError

                raise InvalidOptionError(f"[set_option] {name}={value!r} {message}")
            else:
                corrected = corrector(value)
                logger.warning(f"[set_option] {name}={value!r} {message}, using {corrected!r}")
                value = corrected

    setattr(obj, final_key, value)


# Version (surfaced from installed metadata when possible)
try:
    __version__ = importlib_metadata.version("natural-pdf")
except importlib_metadata.PackageNotFoundError:
    try:
        from setuptools_scm import get_version  # type: ignore[import]

        __version__ = get_version(root=Path(__file__).resolve().parents[1])
    except Exception:  # pragma: no cover - SCM fallback is best-effort
        __version__ = "0.0.0"

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
from natural_pdf.core.vlm_client import set_default_client
from natural_pdf.elements.region import Region

# Unified exception hierarchy
from natural_pdf.exceptions import (
    ClassificationError,
    ConfigurationError,
    ExportError,
    InvalidOptionError,
    LayoutEngineNotAvailableError,
    LayoutError,
    NaturalPDFError,
    OCREngineNotAvailableError,
    OCRError,
    QAError,
    SearchError,
    SelectorError,
    SelectorMatchError,
    SelectorParseError,
)
from natural_pdf.exporters import export_training_data
from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion

# Judge for visual classification
from natural_pdf.judge import Decision, Judge, JudgeError, PickResult

# Explicitly define what gets imported with 'from natural_pdf import *'
__all__ = [
    # Core classes
    "PDF",
    "PDFCollection",
    "Page",
    "Region",
    "Flow",
    "FlowRegion",
    "Guides",
    "PageCollection",
    # Judge
    "Judge",
    "Decision",
    "PickResult",
    "JudgeError",
    # VLM client
    "set_default_client",
    # Configuration
    "configure_logging",
    "options",
    # Exceptions (unified hierarchy)
    "NaturalPDFError",
    "OCRError",
    "OCREngineNotAvailableError",
    "LayoutError",
    "LayoutEngineNotAvailableError",
    "SelectorError",
    "SelectorParseError",
    "SelectorMatchError",
    "ConfigurationError",
    "InvalidOptionError",
    "ExportError",
    "SearchError",
    "ClassificationError",
    "QAError",
    # Exporters
    "export_training_data",
]
