import sys
import tempfile
from pathlib import Path

# Ensure the repository root is ahead of any installed natural_pdf package.
REPO_ROOT = Path(__file__).resolve().parents[1]
root_str = str(REPO_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

import pytest

# Common test PDF URLs from tutorials
SAMPLE_PDFS = {
    "practice": "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf",
    "atlanta": "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/Atlanta_Public_Schools_GA_sample.pdf",
    "needs_ocr": "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/needs-ocr.pdf",
    "cia_doc": "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/cia-doc.pdf",
    "geometry": "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/geometry.pdf",
}

# Local paths to PDF files in the repo
LOCAL_PDFS = {
    "practice": Path("pdfs/01-practice.pdf"),
    "atlanta": Path("pdfs/Atlanta_Public_Schools_GA_sample.pdf"),
    "needs_ocr": Path("pdfs/needs-ocr.pdf"),
    "cia_doc": Path("pdfs/cia-doc.pdf"),
    "geometry": Path("pdfs/geometry.pdf"),
}


def _load_pdf(name):
    """Helper to load a PDF by name, trying local first then URL."""
    from natural_pdf import PDF

    if LOCAL_PDFS[name].exists():
        return PDF(str(LOCAL_PDFS[name].resolve()))
    else:
        return PDF(SAMPLE_PDFS[name])


# =============================================================================
# SESSION-SCOPED FIXTURES (for read-only tests - much faster!)
# =============================================================================
# These fixtures load PDFs once per test session (per worker with pytest-xdist).
# Use these for tests that only READ from PDFs, not modify them.


@pytest.fixture(scope="session")
def practice_pdf():
    """Returns a loaded practice PDF object (session-scoped for performance)."""
    pdf = _load_pdf("practice")
    yield pdf
    pdf.close()


@pytest.fixture(scope="session")
def geometry_pdf():
    """Returns a loaded geometry PDF object (session-scoped for performance)."""
    pdf = _load_pdf("geometry")
    yield pdf
    pdf.close()


@pytest.fixture(scope="session")
def atlanta_pdf():
    """Returns a loaded Atlanta Public Schools PDF object (session-scoped for performance)."""
    pdf = _load_pdf("atlanta")
    yield pdf
    pdf.close()


@pytest.fixture(scope="session")
def needs_ocr_pdf():
    """Returns a loaded PDF that needs OCR (session-scoped for performance)."""
    pdf = _load_pdf("needs_ocr")
    yield pdf
    pdf.close()


@pytest.fixture(scope="session")
def cia_doc_pdf():
    """Returns a loaded CIA document PDF (session-scoped for performance)."""
    pdf = _load_pdf("cia_doc")
    yield pdf
    pdf.close()


# =============================================================================
# PDF FACTORY FIXTURE (for flexible, cached PDF loading)
# =============================================================================


@pytest.fixture(scope="session")
def pdf_factory():
    """
    Factory fixture that provides cached PDF loading.

    Usage:
        def test_something(pdf_factory):
            pdf = pdf_factory("pdfs/01-practice.pdf")
            # PDF is cached - subsequent calls return same instance

    This is useful when tests need PDFs not covered by the named fixtures,
    or when you want to load arbitrary PDF paths with caching.
    """
    from natural_pdf import PDF

    cache = {}

    def get_pdf(path):
        if path not in cache:
            cache[path] = PDF(path)
        return cache[path]

    yield get_pdf

    # Cleanup all cached PDFs at session end
    for pdf in cache.values():
        try:
            pdf.close()
        except Exception:
            pass


# =============================================================================
# FUNCTION-SCOPED FIXTURES (for tests that mutate PDF state)
# =============================================================================
# Use these when tests need to modify the PDF (add exclusions, apply OCR, etc.)


@pytest.fixture
def practice_pdf_fresh():
    """Returns a FRESH practice PDF (function-scoped for mutation tests)."""
    pdf = _load_pdf("practice")
    yield pdf
    pdf.close()


@pytest.fixture
def atlanta_pdf_fresh():
    """Returns a FRESH Atlanta PDF (function-scoped for mutation tests)."""
    pdf = _load_pdf("atlanta")
    yield pdf
    pdf.close()


@pytest.fixture
def needs_ocr_pdf_fresh():
    """Returns a FRESH needs-OCR PDF (function-scoped for mutation tests)."""
    pdf = _load_pdf("needs_ocr")
    yield pdf
    pdf.close()


# =============================================================================
# COLLECTION FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def pdf_collection():
    """Returns a collection of PDFs (session-scoped for performance)."""
    from natural_pdf import PDFCollection

    # Use a subset of PDFs to keep tests faster
    if LOCAL_PDFS["practice"].exists() and LOCAL_PDFS["atlanta"].exists():
        pdf_paths = [str(LOCAL_PDFS["practice"].resolve()), str(LOCAL_PDFS["atlanta"].resolve())]
    else:
        pdf_paths = [SAMPLE_PDFS["practice"], SAMPLE_PDFS["atlanta"]]

    collection = PDFCollection(pdf_paths)
    yield collection

    # Close each PDF in the collection
    for pdf in collection.pdfs:
        try:
            pdf.close()
        except Exception:
            pass


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def temp_output_dir():
    """Creates a temporary directory for test output files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# =============================================================================
# PYTEST HOOKS
# =============================================================================


# Hook to catch Windows DLL errors during test execution
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Convert specific Windows DLL errors to skips instead of failures."""
    outcome = yield
    report = outcome.get_result()

    # Only process on Windows
    if not sys.platform.startswith("win"):
        return

    # Check if the test failed with the specific DLL error
    if report.when in ("setup", "call") and report.failed:
        if hasattr(report, "longrepr") and report.longrepr:
            error_text = str(report.longrepr)
            # Check for the specific torch DLL error
            if (
                (
                    "shm.dll" in error_text
                    and ("WinError 127" in error_text or "WinError 126" in error_text)
                )
                or ("vcruntime140_1.dll" in error_text)
                or ("torch" in error_text and "DLL" in error_text)
            ):
                # Convert failure to skip
                report.outcome = "skipped"
                report.wasxfail = False
                report.longrepr = "Skipped due to Windows torch DLL issue"
