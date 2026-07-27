"""Round-trip tests for the searchable-PDF exporter.

These would have caught two long-standing output bugs:
- XML special characters being double-escaped into the text layer (``&amp;``)
- every word landing on its own hOCR line, which destroys inter-word spacing
"""

import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("pikepdf")

from natural_pdf import PDF
from natural_pdf.exceptions import ExportError
from natural_pdf.exporters.searchable_pdf import _generate_hocr_for_page, create_searchable_pdf


class FakeWord:
    def __init__(self, text, bbox, confidence=0.99):
        self.text = text
        self.bbox = bbox
        self.confidence = confidence


def _make_page_with_words(monkeypatch, page, words):
    class FakeCollection:
        def __init__(self, elements):
            self.elements = elements

    monkeypatch.setattr(page, "find_all", lambda selector, **kw: FakeCollection(list(words)))
    return page


@pytest.fixture
def practice_page():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        yield pdf.pages[0]
    finally:
        pdf.close()


def _hocr_words(hocr: str):
    root = ET.fromstring(hocr)
    ns = {"h": "http://www.w3.org/1999/xhtml"}
    spans = root.findall(".//h:span[@class='ocrx_word']", ns) or root.findall(
        ".//span[@class='ocrx_word']"
    )
    return spans


def _hocr_lines(hocr: str):
    root = ET.fromstring(hocr)
    ns = {"h": "http://www.w3.org/1999/xhtml"}
    return root.findall(".//h:span[@class='ocr_line']", ns) or root.findall(
        ".//span[@class='ocr_line']"
    )


def test_hocr_special_chars_not_double_escaped(monkeypatch, practice_page):
    """'A & B' must come back as 'A & B', not 'A &amp; B'."""
    page = _make_page_with_words(
        monkeypatch,
        practice_page,
        [FakeWord("A&B", (10, 10, 40, 20)), FakeWord("<tag>", (50, 10, 80, 20))],
    )
    hocr = _generate_hocr_for_page(page, 800, 1000)
    words = _hocr_words(hocr)
    texts = [w.text for w in words]
    assert texts == ["A&B", "<tag>"]
    # The serialized form must escape exactly once
    assert "&amp;amp;" not in hocr
    assert "&amp;" in hocr


def test_hocr_groups_words_on_same_line(monkeypatch, practice_page):
    """Words with the same baseline must share one ocr_line element."""
    same_row = [
        FakeWord("alpha", (10, 100, 50, 112)),
        FakeWord("beta", (60, 100, 100, 112)),
        FakeWord("gamma", (110, 101, 160, 113)),
    ]
    next_row = [FakeWord("delta", (10, 130, 60, 142))]
    page = _make_page_with_words(monkeypatch, practice_page, same_row + next_row)

    hocr = _generate_hocr_for_page(page, 800, 1000)
    lines = _hocr_lines(hocr)
    assert len(lines) == 2, "expected two visual lines, got one ocr_line per word"

    first_line_words = [w.text for w in lines[0].findall("*")]
    assert first_line_words == ["alpha", "beta", "gamma"]


def test_searchable_pdf_text_roundtrip(monkeypatch, practice_page, tmp_path):
    """Full export: text extracted from the output PDF keeps words and entities."""
    words = [
        FakeWord("Fish", (10, 100, 45, 112)),
        FakeWord("&", (50, 100, 58, 112)),
        FakeWord("Chips", (63, 100, 105, 112)),
    ]
    page = _make_page_with_words(monkeypatch, practice_page, words)

    out = tmp_path / "searchable.pdf"
    create_searchable_pdf(page, str(out), dpi=100)
    assert out.exists()

    import pdfplumber

    with pdfplumber.open(str(out)) as ocr_pdf:
        extracted = ocr_pdf.pages[0].extract_text() or ""

    assert "&" in extracted
    assert "&amp;" not in extracted
    # All three words on one line, space-separated
    assert "Fish & Chips" in extracted


def test_on_error_validation(practice_page, tmp_path):
    with pytest.raises(ValueError, match="on_error"):
        create_searchable_pdf(practice_page, str(tmp_path / "x.pdf"), on_error="ignore")
    with pytest.raises(ValueError, match="dpi"):
        create_searchable_pdf(practice_page, str(tmp_path / "x.pdf"), dpi=0)


def test_failing_page_raises_export_error(monkeypatch, practice_page, tmp_path):
    """A page that fails to render must raise ExportError, not silently skip."""
    monkeypatch.setattr(
        type(practice_page),
        "render",
        lambda self, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(ExportError, match="page"):
        create_searchable_pdf(practice_page, str(tmp_path / "x.pdf"))
    assert not (tmp_path / "x.pdf").exists()
