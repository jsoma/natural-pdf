from pathlib import Path
from unittest.mock import Mock

import pytest

from natural_pdf.core.pdf import PDF
from natural_pdf.describe.summary import ElementSummary, InspectionSummary
from natural_pdf.engine_provider import get_provider

PDF_PATH = Path("pdfs/01-practice.pdf").resolve()


@pytest.fixture()
def sample_pdf():
    pdf = PDF(str(PDF_PATH))
    try:
        yield pdf
    finally:
        pdf.close()


class TestExtensions:
    def test_page_describe_inspect(self, sample_pdf):
        """Verify describe and inspect work end-to-end on a real page."""
        page = sample_pdf.pages[0]

        summary = page.describe()
        assert isinstance(summary, ElementSummary)

        inspection = page.inspect(limit=5)
        assert isinstance(inspection, InspectionSummary)

    def test_pdf_describe_inspect(self, sample_pdf):
        """Verify describe and inspect work at the PDF level."""
        summary = sample_pdf.describe()
        assert isinstance(summary, ElementSummary)

        inspection = sample_pdf.inspect(limit=5)
        assert isinstance(inspection, InspectionSummary)

    def test_region_describe_inspect(self, sample_pdf):
        """Regions should also produce summaries/inspections."""
        page = sample_pdf.pages[0]
        region = page.region(0, 0, page.width, page.height / 2)

        summary = region.describe()
        assert isinstance(summary, ElementSummary)

        inspection = region.inspect(limit=5)
        assert isinstance(inspection, InspectionSummary)

    def test_pdf_collection_describe_inspect(self):
        """PDFCollection.describe/inspect should operate on aggregate elements."""
        from natural_pdf.core.pdf_collection import PDFCollection

        collection = PDFCollection([str(PDF_PATH)])
        summary = collection.describe()
        assert isinstance(summary, ElementSummary)

        inspection = collection.inspect(limit=5)
        assert isinstance(inspection, InspectionSummary)

    def test_engine_registration(self):
        """Verify custom engines can be registered and retrieved."""

        mock_factory = Mock(return_value="engine_instance")

        # Register
        get_provider().register("ocr", "test_custom_engine", mock_factory)

        # Verify it's in the registry
        engines = get_provider().list("ocr")
        assert "test_custom_engine" in engines["ocr"]

        # Verify retrieval
        instance = get_provider().get("ocr", name="test_custom_engine", context=Mock())
        assert instance == "engine_instance"
