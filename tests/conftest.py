import os
import sys
import tempfile
from pathlib import Path

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


@pytest.fixture
def practice_pdf():
    """Returns a loaded practice PDF object"""
    from natural_pdf import PDF

    try:
        # Try local file first
        if LOCAL_PDFS["practice"].exists():
            pdf = PDF(str(LOCAL_PDFS["practice"].resolve()))
        else:
            # Fall back to URL
            pdf = PDF(SAMPLE_PDFS["practice"])
        yield pdf
    finally:
        if "pdf" in locals():
            pdf.close()


@pytest.fixture
def geometry_pdf():
    """Returns a loaded practice PDF object"""
    from natural_pdf import PDF

    try:
        # Try local file first
        if LOCAL_PDFS["geometry"].exists():
            pdf = PDF(str(LOCAL_PDFS["geometry"].resolve()))
        else:
            # Fall back to URL
            pdf = PDF(SAMPLE_PDFS["geometry"])
        yield pdf
    finally:
        if "pdf" in locals():
            pdf.close()


@pytest.fixture
def atlanta_pdf():
    """Returns a loaded Atlanta Public Schools PDF object"""
    from natural_pdf import PDF

    try:
        # Try local file first
        if LOCAL_PDFS["atlanta"].exists():
            pdf = PDF(str(LOCAL_PDFS["atlanta"].resolve()))
        else:
            # Fall back to URL
            pdf = PDF(SAMPLE_PDFS["atlanta"])
        yield pdf
    finally:
        if "pdf" in locals():
            pdf.close()


@pytest.fixture
def needs_ocr_pdf():
    """Returns a loaded PDF that needs OCR"""
    from natural_pdf import PDF

    try:
        # Try local file first
        if LOCAL_PDFS["needs_ocr"].exists():
            pdf = PDF(str(LOCAL_PDFS["needs_ocr"].resolve()))
        else:
            # Fall back to URL
            pdf = PDF(SAMPLE_PDFS["needs_ocr"])
        yield pdf
    finally:
        if "pdf" in locals():
            pdf.close()


@pytest.fixture
def cia_doc_pdf():
    """Returns a loaded CIA document PDF with various page types"""
    from natural_pdf import PDF

    try:
        # Try local file first
        if LOCAL_PDFS["cia_doc"].exists():
            pdf = PDF(str(LOCAL_PDFS["cia_doc"].resolve()))
        else:
            # Fall back to URL
            pdf = PDF(SAMPLE_PDFS["cia_doc"])
        yield pdf
    finally:
        if "pdf" in locals():
            pdf.close()


@pytest.fixture
def pdf_collection():
    """Returns a collection of PDFs"""
    from natural_pdf import PDFCollection

    # Use a subset of PDFs to keep tests faster
    pdf_paths = []
    # Try local paths first
    if LOCAL_PDFS["practice"].exists() and LOCAL_PDFS["atlanta"].exists():
        pdf_paths = [str(LOCAL_PDFS["practice"].resolve()), str(LOCAL_PDFS["atlanta"].resolve())]
    else:
        # Fall back to URLs
        pdf_paths = [SAMPLE_PDFS["practice"], SAMPLE_PDFS["atlanta"]]

    try:
        collection = PDFCollection(pdf_paths)
        yield collection
    finally:
        # Close each PDF in the collection explicitly instead
        if "collection" in locals():
            for pdf in collection.pdfs:
                try:
                    pdf.close()
                except:
                    pass


@pytest.fixture
def temp_output_dir():
    """Creates a temporary directory for test output files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


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
                report.longrepr = f"Skipped due to Windows torch DLL issue"
