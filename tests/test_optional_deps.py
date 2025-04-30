import importlib  # Use importlib for checking
import os
import sys

import pytest

from natural_pdf import PDF, PDFCollection  # Import PDFCollection
from natural_pdf.core.page import Page

# --- Fixtures --- #

# Define PDF paths relative to the project root (where pytest is usually run)
TEST_PDF_URL = "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf"
NEEDS_OCR_PDF_PATH = "pdfs/needs-ocr.pdf"
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


# --- Helper ---
def is_extra_installed(extra_name):
    """Checks if packages associated with an extra appear importable."""
    extra_packages = {
        "interactive": ["ipywidgets"],
        "easyocr": ["easyocr"],
        "paddle": ["paddleocr"],
        "surya": ["surya"],
        "layout_yolo": ["doclayout_yolo"],
        "haystack": ["haystack", "lancedb_haystack"],
        "qa": ["transformers"],
    }
    if extra_name not in extra_packages:
        return False

    packages_to_check = extra_packages[extra_name]
    try:
        for pkg_name in packages_to_check:
            importlib.import_module(pkg_name)
        return True
    except ImportError:
        return False


# --- Interactive Viewer (ipywidgets) Tests --- (Existing)


def test_ipywidgets_availability_flag():
    """Tests the internal _IPYWIDGETS_AVAILABLE flag based on environment."""
    try:
        from natural_pdf.widgets.viewer import _IPYWIDGETS_AVAILABLE

        flag_value = _IPYWIDGETS_AVAILABLE
    except ImportError:
        pytest.fail("Could not import or find _IPYWIDGETS_AVAILABLE in natural_pdf.widgets.viewer")
    should_be_installed = is_extra_installed("interactive")
    assert (
        flag_value == should_be_installed
    ), f"_IPYWIDGETS_AVAILABLE flag mismatch. Expected {should_be_installed}, got {flag_value}."


def test_page_viewer_widget_creation_when_installed(standard_pdf_page):
    """Tests that Page.viewer() returns a widget when ipywidgets is installed."""
    pytest.importorskip("ipywidgets")
    from natural_pdf.widgets.viewer import SimpleInteractiveViewerWidget

    viewer_instance = standard_pdf_page.viewer()
    assert (
        viewer_instance is not None
    ), "Page.viewer() should return an object when ipywidgets is installed."
    assert isinstance(
        viewer_instance, SimpleInteractiveViewerWidget
    ), f"Page.viewer() returned type {type(viewer_instance)}, expected SimpleInteractiveViewerWidget."


def test_page_viewer_widget_creation_when_not_installed(standard_pdf_page):
    """Tests that Page.viewer() returns None when ipywidgets is missing."""
    if is_extra_installed("interactive"):
        pytest.skip("Skipping test: ipywidgets IS installed in this environment.")
    viewer_instance = standard_pdf_page.viewer()
    assert (
        viewer_instance is None
    ), "Page.viewer() should return None when ipywidgets is not installed."


# --- EasyOCR Tests --- #


def test_ocr_easyocr_works_when_installed(needs_ocr_pdf_page):
    """Test running EasyOCR when installed."""
    pytest.importorskip("easyocr")
    try:
        # Use extract_ocr_elements which doesn't modify the page state
        ocr_elements = needs_ocr_pdf_page.extract_ocr_elements(engine="easyocr")
        assert isinstance(ocr_elements, list)
        assert len(ocr_elements) > 0, "EasyOCR should find text elements on the OCR PDF."
        # Check if the first element looks like a TextElement (basic check)
        assert hasattr(ocr_elements[0], "text"), "OCR result should have text attribute."
        assert hasattr(ocr_elements[0], "bbox"), "OCR result should have bbox attribute."
    except Exception as e:
        pytest.fail(f"EasyOCR extraction failed when installed: {e}")


def test_ocr_easyocr_fails_gracefully_when_not_installed(needs_ocr_pdf_page):
    """Test calling EasyOCR when not installed."""
    if is_extra_installed("easyocr"):
        pytest.skip("Skipping test: EasyOCR IS installed.")
    # Check how OCRManager handles unavailable engines - assuming it returns empty list
    ocr_elements = needs_ocr_pdf_page.extract_ocr_elements(engine="easyocr")
    assert (
        ocr_elements == []
    ), "extract_ocr_elements should return empty list for unavailable engine."


# --- PaddleOCR Tests --- #


def test_ocr_paddle_works_when_installed(needs_ocr_pdf_page):
    """Test running PaddleOCR when installed."""
    pytest.importorskip("paddleocr")
    if sys.platform == "darwin":
        pytest.skip("PaddleOCR tests skipped on macOS")
    try:
        ocr_elements = needs_ocr_pdf_page.extract_ocr_elements(engine="paddle")
        assert isinstance(ocr_elements, list)
        assert len(ocr_elements) > 0, "PaddleOCR should find text elements on the OCR PDF."
        assert hasattr(ocr_elements[0], "text")
        assert hasattr(ocr_elements[0], "bbox")
    except Exception as e:
        pytest.fail(f"PaddleOCR extraction failed when installed: {e}")


def test_ocr_paddle_fails_gracefully_when_not_installed(needs_ocr_pdf_page):
    """Test calling PaddleOCR when not installed."""
    if is_extra_installed("paddle"):
        pytest.skip("Skipping test: PaddleOCR IS installed.")
    if sys.platform == "darwin":  # Also skip if check fails but platform is darwin
        pytest.skip("PaddleOCR tests skipped on macOS")
    # Check how OCRManager handles unavailable engines - assume it returns empty list (KEEPING THIS)
    # It might be reasonable for OCR manager to return [] if engine isn't there,
    # vs layout which is explicitly requested.
    # Alternatively, OCRManager could also be changed to raise errors.
    ocr_elements = needs_ocr_pdf_page.extract_ocr_elements(engine="paddle")
    assert (
        ocr_elements == []
    ), "extract_ocr_elements should return empty list for unavailable engine."


# --- Surya Tests --- #


def test_layout_surya_works_when_installed(standard_pdf_page):  # Use standard PDF for layout
    """Test running Surya layout analysis when installed."""
    pytest.importorskip("surya")
    if sys.version_info < (3, 10):
        pytest.skip("Surya tests skipped on Python < 3.10")
    try:
        layout_regions = standard_pdf_page.analyze_layout(engine="surya")
        from natural_pdf.elements.collections import ElementCollection  # Import needed for check

        assert isinstance(
            layout_regions, ElementCollection
        ), "analyze_layout should return an ElementCollection"
        # Layout might return empty list, check type
        # assert len(layout_regions) > 0, "Surya should find layout regions." # Keep commented unless specific PDF guarantees regions
    except Exception as e:
        pytest.fail(f"Surya layout analysis failed when installed: {e}")


def test_layout_surya_fails_gracefully_when_not_installed(standard_pdf_page):
    """Test calling Surya layout analysis when not installed raises error."""
    if is_extra_installed("surya"):
        pytest.skip("Skipping test: Surya IS installed.")
    if sys.version_info < (3, 10):  # Also skip if check fails but Python is < 3.10
        pytest.skip("Surya tests skipped on Python < 3.10")
    # Expect RuntimeError because engine is known but unavailable
    with pytest.raises(RuntimeError, match="not available"):
        _ = standard_pdf_page.analyze_layout(engine="surya")


# --- Layout YOLO Tests --- #


def test_layout_yolo_works_when_installed(standard_pdf_page):
    """Test running YOLO layout analysis when installed."""
    # Check for the *actual* package associated with the extra
    pytest.importorskip("doclayout_yolo")
    try:
        layout_regions = standard_pdf_page.analyze_layout(engine="yolo")
        from natural_pdf.elements.collections import ElementCollection

        assert isinstance(layout_regions, ElementCollection)
    except Exception as e:
        pytest.fail(f"YOLO layout analysis failed when installed: {e}")


def test_layout_yolo_fails_gracefully_when_not_installed(standard_pdf_page):
    """Test calling YOLO layout analysis when not installed raises error."""
    if is_extra_installed("layout_yolo"):
        pytest.skip("Skipping test: Layout YOLO IS installed.")
    # Expect RuntimeError because engine is known but unavailable
    with pytest.raises(RuntimeError, match="not available"):
        _ = standard_pdf_page.analyze_layout(engine="yolo")


# --- Haystack Tests --- #


def test_search_haystack_works_when_installed(standard_pdf_collection):
    """Test basic Haystack initialization via PDFCollection when installed."""
    pytest.importorskip("haystack")
    pytest.importorskip("lancedb")
    pytest.importorskip("lancedb_haystack")

    try:
        # Initialize search on the collection
        standard_pdf_collection.init_search(
            collection_name="test_lance_collection",
            persist=True,
            uri="./test_lance_index",
            index=False,
        )
        assert hasattr(standard_pdf_collection, "_search_service")
        assert standard_pdf_collection._search_service is not None
        # Check the service type and URI
        from natural_pdf.search.haystack_search_service import HaystackSearchService

        assert isinstance(standard_pdf_collection._search_service, HaystackSearchService)
        assert standard_pdf_collection._search_service._persist is True
        assert standard_pdf_collection._search_service._uri == "./test_lance_index"
        assert standard_pdf_collection._search_service.table_name == "test_lance_collection"

        # Cleanup the test directory
        import shutil

        if os.path.exists("./test_lance_index"):
            shutil.rmtree("./test_lance_index")

    except ImportError as ie:
        pytest.fail(f"Import failed even though haystack/lancedb seem installed: {ie}")
    except Exception as e:
        pytest.fail(f"Haystack/LanceDB integration failed when installed: {e}")
    finally:
        # Ensure cleanup even on failure
        import shutil

        if os.path.exists("./test_lance_index"):
            shutil.rmtree("./test_lance_index")


def test_search_haystack_fails_gracefully_when_not_installed(standard_pdf_collection):
    """Test initializing Haystack features via PDFCollection when not installed."""
    if is_extra_installed("haystack"):
        pytest.skip("Skipping test: Haystack/LanceDB extras ARE installed.")

    # Expect an ImportError
    with pytest.raises(
        ImportError,
        match="Search Service could not be created. Ensure Haystack extras are installed",
    ):
        # Pass parameters consistent with the factory/service
        standard_pdf_collection.init_search(
            collection_name="test_missing_deps",
            persist=True,
            uri="./test_missing_index",
            index=False,
        )


# --- QA Tests --- #


def test_qa_works_when_installed(standard_pdf_page):
    """Test basic QA functionality (requires transformers core dep)."""
    # No importorskip needed as transformers is core
    try:
        # Simple question
        result = standard_pdf_page.ask("What is this document about?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "confidence" in result
        # We don't know the answer, but it should run
    except Exception as e:
        pytest.fail(f"QA execution failed: {e}")


# No 'fails gracefully' test needed for QA as its core dep (transformers) is always installed.
# We might need tests for *specific models* if they require separate downloads/setup.
