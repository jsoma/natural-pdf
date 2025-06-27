import pytest

from natural_pdf import PDF


def test_highlight_detection_types_of_type():
    pdf = PDF("pdfs/types-of-type.pdf")
    page = pdf.pages[0]
    hl_words = [w for w in page.words if getattr(w, "highlight", False)]
    assert hl_words, "Expected highlighted words"
    assert any("highlighted" in w.text.lower() for w in hl_words)
