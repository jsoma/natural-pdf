"""Tests for Region PDF export and exclusion-aware saving."""

import os

import pytest

from natural_pdf import PDF

PDF_PATH = "pdfs/01-practice.pdf"
TEMP_DIR = "temp"


@pytest.fixture(autouse=True)
def ensure_temp_dir():
    os.makedirs(TEMP_DIR, exist_ok=True)


@pytest.fixture
def pdf():
    p = PDF(PDF_PATH)
    yield p
    p.close()


@pytest.fixture
def page(pdf):
    return pdf.pages[0]


# --- Feature A: Region / ElementCollection / FlowRegion save_pdf ---


class TestRegionSavePdfCrop:
    def test_region_save_pdf_crop(self, page):
        """Verify crop method produces a PDF with CropBox set."""
        pikepdf = pytest.importorskip("pikepdf")

        region = page.region(50, 50, page.width - 50, 200)
        out = os.path.join(TEMP_DIR, "test_region_crop.pdf")
        result = region.save_pdf(out)

        # Returns self for chaining
        assert result is region

        # Verify output
        assert os.path.exists(out)
        with pikepdf.Pdf.open(out) as doc:
            assert len(doc.pages) == 1
            # CropBox should be set
            assert "/CropBox" in doc.pages[0]

    def test_region_save_pdf_whiteout(self, page):
        """Verify whiteout method produces a full-size page with overlay."""
        pikepdf = pytest.importorskip("pikepdf")

        region = page.region(50, 50, page.width - 50, 200)
        out = os.path.join(TEMP_DIR, "test_region_whiteout.pdf")
        region.save_pdf(out, method="whiteout")

        assert os.path.exists(out)
        with pikepdf.Pdf.open(out) as doc:
            assert len(doc.pages) == 1
            # Should NOT have a modified CropBox (whiteout keeps full page)
            # The content stream should contain white fill operations
            p = doc.pages[0]
            contents = p.get("/Contents")
            assert contents is not None
            # Read all content streams and look for white fill command
            if isinstance(contents, pikepdf.Array):
                # Last stream should be the whiteout
                stream_data = contents[-1].read_bytes()
            else:
                stream_data = contents.read_bytes()
            assert b"1 1 1 rg" in stream_data  # White fill color

    def test_save_pdf_returns_self(self, page):
        """Method chaining should work."""
        region = page.region(0, 0, page.width, 100)
        out = os.path.join(TEMP_DIR, "test_chain.pdf")
        assert region.save_pdf(out) is region

    def test_save_pdf_invalid_method_raises(self, page):
        """Invalid method should raise ValueError."""
        region = page.region(0, 0, page.width, 100)
        out = os.path.join(TEMP_DIR, "test_invalid.pdf")
        with pytest.raises(ValueError, match="Invalid method"):
            region.save_pdf(out, method="invalid")


class TestCollectionSavePdf:
    def test_collection_save_pdf(self, page):
        """Multiple elements should produce multiple pages."""
        pikepdf = pytest.importorskip("pikepdf")

        elements = page.find_all("text:bold")
        assert len(elements) >= 2, "Need at least 2 bold elements for this test"

        subset = elements[:3]
        out = os.path.join(TEMP_DIR, "test_collection.pdf")
        result = subset.save_pdf(out)

        assert result is subset
        assert os.path.exists(out)
        with pikepdf.Pdf.open(out) as doc:
            assert len(doc.pages) == len(subset)

    def test_collection_save_pdf_empty_raises(self):
        """Empty collection should raise ValueError."""
        from natural_pdf.elements.element_collection import ElementCollection

        empty = ElementCollection([])
        out = os.path.join(TEMP_DIR, "test_empty.pdf")
        with pytest.raises(ValueError, match="empty"):
            empty.save_pdf(out)


# --- Feature B: apply_exclusions on PDF.save_pdf ---


class TestApplyExclusions:
    def test_pdf_save_pdf_apply_exclusions(self, pdf):
        """Adding an exclusion and saving with apply_exclusions should produce output."""
        pikepdf = pytest.importorskip("pikepdf")

        # Add an exclusion for large text (headers)
        pdf.add_exclusion(lambda p: p.find("text[size>=14]"))
        out = os.path.join(TEMP_DIR, "test_exclusions.pdf")
        pdf.save_pdf(out, apply_exclusions=True)

        assert os.path.exists(out)
        with pikepdf.Pdf.open(out) as doc:
            assert len(doc.pages) == len(pdf.pages)

    def test_apply_exclusions_with_ocr_raises(self, pdf):
        """Combining apply_exclusions with ocr should raise ValueError."""
        out = os.path.join(TEMP_DIR, "test_exc_ocr.pdf")
        with pytest.raises(ValueError, match="Cannot combine"):
            pdf.save_pdf(out, apply_exclusions=True, ocr=True)

    def test_apply_exclusions_no_exclusions(self, pdf):
        """When no exclusions are set, should still produce valid output."""
        pikepdf = pytest.importorskip("pikepdf")

        out = os.path.join(TEMP_DIR, "test_no_exclusions.pdf")
        pdf.save_pdf(out, apply_exclusions=True)

        assert os.path.exists(out)
        with pikepdf.Pdf.open(out) as doc:
            assert len(doc.pages) == len(pdf.pages)

    def test_page_collection_apply_exclusions(self, pdf):
        """PageCollection.save_pdf with apply_exclusions should work."""
        pikepdf = pytest.importorskip("pikepdf")

        pdf.add_exclusion(lambda p: p.find("text[size>=14]"))
        out = os.path.join(TEMP_DIR, "test_pc_exclusions.pdf")
        # 01-practice.pdf has 1 page, so slice [:1] gives us that page
        pdf.pages[:1].save_pdf(out, apply_exclusions=True)

        assert os.path.exists(out)
        with pikepdf.Pdf.open(out) as doc:
            assert len(doc.pages) == 1
