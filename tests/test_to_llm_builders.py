"""Tests for the non-page to_llm builders (region, collection, element, pdf).

These four had zero coverage; two P0 bugs lived here (the pdf.to_llm() header
never showing the filename, and unvalidated detail/include_hints values).
"""

import pytest

from natural_pdf import PDF


@pytest.fixture(scope="module")
def pdf():
    doc = PDF("pdfs/01-practice.pdf")
    yield doc
    doc.close()


@pytest.fixture(scope="module")
def page(pdf):
    return pdf.pages[0]


class TestPdfToLLM:
    def test_header_contains_filename(self, pdf):
        out = pdf.to_llm()
        assert "01-practice.pdf" in out.splitlines()[0]
        assert "(1 pages)" in out.splitlines()[0]

    def test_max_pages_bounds_iteration(self, pdf):
        out = pdf.to_llm(max_pages=1)
        assert "Page 1:" in out

    def test_max_chars_cap(self, pdf):
        out = pdf.to_llm(max_chars=50)
        assert len(out) <= 50
        assert "[output capped" in out


class TestValidation:
    def test_invalid_detail_raises(self, page):
        with pytest.raises(ValueError, match="detail"):
            page.to_llm(detail="verbose")

    def test_invalid_hints_raises(self, page):
        with pytest.raises(ValueError, match="include_hints"):
            page.to_llm(include_hints="apis")

    def test_invalid_detail_raises_on_pdf(self, pdf):
        with pytest.raises(ValueError, match="detail"):
            pdf.to_llm(detail="verbose")


class TestRegionToLLM:
    def test_basic_output(self, page):
        region = page.create_region(0, 0, page.width, page.height / 2)
        out = region.to_llm()
        assert out.startswith("=== Region")
        assert "TEXT CONTENT" in out

    def test_detail_levels(self, page):
        region = page.create_region(0, 0, page.width, page.height)
        brief = region.to_llm(detail="brief")
        full = region.to_llm(detail="full")
        assert len(brief) <= len(full)

    def test_max_chars(self, page):
        region = page.create_region(0, 0, page.width, page.height)
        out = region.to_llm(max_chars=100)
        assert len(out) <= 100


class TestCollectionToLLM:
    def test_basic_output(self, page):
        words = page.find_all("text")
        out = words.to_llm()
        assert out.startswith("=== ElementCollection")
        assert "EXTENT" in out

    def test_empty_collection(self, page):
        empty = page.find_all('text:contains("zzz-nothing-zzz")')
        out = empty.to_llm()
        assert "0 elements" in out


class TestElementToLLM:
    def test_basic_output(self, page):
        el = page.find("text")
        assert el is not None
        out = el.to_llm()
        assert out.startswith('=== "')

    def test_brief(self, page):
        el = page.find("text")
        out = el.to_llm(detail="brief")
        assert "NEARBY" not in out
