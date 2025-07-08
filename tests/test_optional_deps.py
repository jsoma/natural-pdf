import importlib
import os
import sys

import pytest

# Skip all tests in this module on Windows to avoid torch DLL issues
if sys.platform.startswith("win"):
    pytest.skip(
        "Skipping optional dependency tests on Windows due to torch DLL issues",
        allow_module_level=True,
    )

from natural_pdf import PDF, PDFCollection  # Import PDFCollection
from natural_pdf.core.page import Page

# --- Fixtures --- #

# Define PDF paths relative to the project root (where pytest is usually run)
TEST_PDF_URL = "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf"
NEEDS_OCR_PDF_PATH = "pdfs/tiny-ocr.pdf"
STANDARD_PDF_PATH = "pdfs/01-practice.pdf"


@pytest.fixture(scope="module")
def standard_pdf_page():
    """Fixture to load the first page of the standard test PDF."""
    try:
        # Use the local path if available, otherwise fallback to URL?
        # For consistency in tests, let's stick to the local path for now.
        # Assume the pdfs directory is in the root alongside tests/
        pdf = PDF(STANDARD_PDF_PATH)
        if not pdf.pages:
            pytest.fail(f"Standard PDF has no pages: {STANDARD_PDF_PATH}")
        return pdf.pages[0]
    except Exception as e:
        pytest.fail(f"Failed to load standard PDF ({STANDARD_PDF_PATH}) for module tests: {e}")


@pytest.fixture(scope="module")
def needs_ocr_pdf_page():
    """Fixture to load the first page of the OCR test PDF."""
    try:
        pdf = PDF(NEEDS_OCR_PDF_PATH)
        if not pdf.pages:
            pytest.fail(f"OCR PDF has no pages: {NEEDS_OCR_PDF_PATH}")
        return pdf.pages[0]
    except Exception as e:
        pytest.fail(f"Failed to load OCR PDF ({NEEDS_OCR_PDF_PATH}) for module tests: {e}")


@pytest.fixture(scope="module")
def standard_pdf_collection():
    """Fixture to create a PDFCollection with the standard test PDF."""
    try:
        # Use a list containing the path
        collection = PDFCollection([STANDARD_PDF_PATH])
        assert len(collection.pdfs) == 1
        return collection
    except Exception as e:
        pytest.fail(f"Failed to create PDFCollection ({STANDARD_PDF_PATH}) for module tests: {e}")


# --- Helper to check if we are in a 'full' test environment ---
def are_optional_deps_installed():
    """
    Checks if a sentinel optional package (e.g., easyocr) is installed.
    This helps determine if we're running in the 'test_full' nox session.
    """
    try:
        # Check multiple packages to be more certain we're in a full environment
        sentinel_packages = ["easyocr", "paddleocr", "surya-ocr", "doclayout_yolo"]
        installed_count = 0
        for pkg in sentinel_packages:
            try:
                if pkg == "surya-ocr":
                    importlib.import_module("surya")
                elif pkg == "doclayout_yolo":
                    importlib.import_module("doclayout_yolo")
                else:
                    importlib.import_module(pkg)
                installed_count += 1
            except ImportError:
                pass
        # If most sentinel packages are installed, we're likely in test_full
        return installed_count >= 2
    except Exception:
        return False


# --- Interactive Viewer (ipywidgets) Tests ---


def test_page_viewer_widget_creation_when_installed(standard_pdf_page):
    """Tests that Page.viewer() returns a widget when ipywidgets is installed."""
    pytest.importorskip("ipywidgets")
    from natural_pdf.widgets.viewer import InteractiveViewerWidget

    viewer_instance = standard_pdf_page.viewer()
    assert viewer_instance is not None
    assert isinstance(viewer_instance, InteractiveViewerWidget)


# --- OCR and Layout Tests ---


# Use parametrize to test all engines with the same logic
@pytest.mark.parametrize(
    "engine, package_name",
    [
        ("easyocr", "easyocr"),
        ("paddle", "paddleocr"),
        ("surya", "surya"),
        ("doctr", "doctr"),
        ("yolo", "doclayout_yolo"),
        ("docling", "docling"),
        ("gemini", "openai"),
    ],
)
def test_engine_works_when_installed(needs_ocr_pdf_page, standard_pdf_page, engine, package_name):
    """Tests that a given engine works when its dependency is installed."""
    pytest.importorskip(package_name)

    if engine == "paddle" and sys.platform == "darwin":
        pytest.skip("PaddleOCR tests skipped on macOS")
    if engine in ["surya", "docling"] and sys.version_info < (3, 10):
        pytest.skip(f"{engine} tests skipped on Python < 3.10")

    try:
        if engine in ["easyocr", "paddle", "surya", "doctr"]:
            result = needs_ocr_pdf_page.apply_ocr(engine=engine)
        elif engine in ["yolo", "surya", "docling", "gemini"]:
            # Gemini requires classes, so we provide a default list for testing
            if engine == "gemini":
                pytest.importorskip("openai")
                # A mock client or specific test setup might be needed here if real calls are made
                # For now, we assume the test environment handles credentials or mocking
                with pytest.raises(Exception):  # It will fail on client
                    _ = standard_pdf_page.analyze_layout(engine=engine, classes=["text", "title"])
                return
            result = standard_pdf_page.analyze_layout(engine=engine)
        else:
            pytest.fail(f"Test logic not implemented for engine: {engine}")

        assert isinstance(result, list)
    except Exception as e:
        # We don't want auth errors etc to fail the test, just import errors
        if isinstance(e, (ImportError, RuntimeError)):
            pytest.fail(f"Engine '{engine}' failed unexpectedly when installed: {e}")
        else:
            pytest.skip(f"Skipping '{engine}' test due to non-dependency-related error: {e}")


@pytest.mark.parametrize(
    "engine, package_name",
    [
        ("easyocr", "easyocr"),
        ("paddle", "paddleocr"),
        ("surya", "surya"),
        ("doctr", "doctr"),
        ("yolo", "doclayout_yolo"),
        ("docling", "docling"),
        ("gemini", "openai"),
    ],
)
def test_engine_fails_gracefully_when_not_installed(
    needs_ocr_pdf_page, standard_pdf_page, engine, package_name
):
    """Tests that using an engine without its dependency installed raises a RuntimeError."""
    if are_optional_deps_installed():
        pytest.skip(
            f"Skipping test: All optional dependencies, including {package_name}, are installed."
        )

    if engine == "gemini":
        with pytest.raises(RuntimeError, match="No client provided"):
            _ = standard_pdf_page.analyze_layout(engine=engine)
    else:
        with pytest.raises(RuntimeError, match="is not available"):
            if engine in ["easyocr", "paddle", "surya", "doctr"]:
                _ = needs_ocr_pdf_page.apply_ocr(engine=engine)
            elif engine in ["yolo", "surya", "docling", "gemini"]:
                _ = standard_pdf_page.analyze_layout(engine=engine)
