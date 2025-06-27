import pytest

from natural_pdf import PDF


def test_underline_detection_types_of_type():
    pdf = PDF("pdfs/types-of-type.pdf")
    page = pdf.pages[0]

    underlined_words = [w for w in page.words if getattr(w, "underline", False)]
    assert underlined_words, "Expected at least one underlined word"

    texts = " ".join(w.text for w in underlined_words).lower()
    assert "underlined" in texts, "Word 'Underlined' should be flagged underline"
    # ensure 'but' not underlined
    assert not any(
        w.text.lower().startswith("but") and w.underline for w in page.words
    ), "Word 'but' should not be underlined"
