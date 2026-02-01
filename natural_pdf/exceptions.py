# exceptions.py
"""
Unified exception hierarchy for Natural PDF.

All domain-specific exceptions inherit from NaturalPDFError, providing:
- Consistent exception handling across the library
- Clear error categorization for users
- Proper exception chaining support

Usage:
    from natural_pdf.exceptions import OCRError, LayoutError

    try:
        engine.process_image(image)
    except OCRError as e:
        logger.error(f"OCR failed: {e}")
"""


class NaturalPDFError(Exception):
    """Base exception for all Natural PDF errors.

    All domain-specific exceptions should inherit from this class.
    This allows users to catch all Natural PDF errors with a single handler:

        try:
            pdf.apply_ocr()
        except NaturalPDFError as e:
            handle_error(e)
    """

    pass


# --- OCR Errors ---


class OCRError(NaturalPDFError):
    """Error during OCR processing.

    Raised when:
    - OCR engine initialization fails
    - Image processing fails
    - Text recognition fails
    - Engine is not available
    """

    pass


class OCREngineNotAvailableError(OCRError):
    """Raised when a requested OCR engine is not installed or available."""

    pass


# --- Layout Detection Errors ---


class LayoutError(NaturalPDFError):
    """Error during layout detection.

    Raised when:
    - Layout detector initialization fails
    - Model loading fails
    - Detection processing fails
    """

    pass


class LayoutEngineNotAvailableError(LayoutError):
    """Raised when a requested layout engine is not installed or available."""

    pass


# --- Selector Errors ---


class SelectorError(NaturalPDFError):
    """Error in selector parsing or matching.

    Raised when:
    - Selector syntax is invalid
    - Selector matching fails
    - Referenced elements not found
    """

    pass


class SelectorParseError(SelectorError):
    """Raised when a selector string cannot be parsed."""

    pass


class SelectorMatchError(SelectorError):
    """Raised when selector matching encounters an error."""

    pass


# --- Configuration Errors ---


class ConfigurationError(NaturalPDFError):
    """Error in configuration or options.

    Raised when:
    - Invalid option values provided
    - Required configuration missing
    - Incompatible options combination
    """

    pass


class InvalidOptionError(ConfigurationError):
    """Raised when an option value is invalid (wrong type, out of range, etc.)."""

    pass


# --- Export Errors ---


class ExportError(NaturalPDFError):
    """Error during export operations.

    Raised when:
    - Export format not supported
    - Export writing fails
    - Required data missing for export
    """

    pass


# --- Search/Index Errors ---
# Note: Existing IndexConfigurationError and IndexExistsError in search module
# should migrate to inherit from these


class SearchError(NaturalPDFError):
    """Error during search or indexing operations."""

    pass


class IndexError(SearchError):
    """Error during index operations (not Python's built-in IndexError)."""

    pass


# --- Classification Errors ---
# Note: Existing ClassificationError in classification/pipelines.py
# should migrate to inherit from this


class ClassificationError(NaturalPDFError):
    """Error during classification operations.

    Raised when:
    - Classification model loading fails
    - Classification inference fails
    - Invalid classification configuration
    """

    pass


# --- Q&A Errors ---


class QAError(NaturalPDFError):
    """Error during document Q&A operations.

    Raised when:
    - Q&A model initialization fails
    - Question answering fails
    - Context extraction fails
    """

    pass
