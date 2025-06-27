import pytest

from natural_pdf import PDF


def test_strikethrough_detection_on_types_of_type():
    """Ensure strikethrough is detected in sample PDF."""
    pdf = PDF("pdfs/types-of-type.pdf")
    page = pdf.pages[0]

    # Char-level – at least one character must be marked strike
    struck_chars = [ch for ch in page.chars if getattr(ch, "strike", False)]
    assert struck_chars, "Expected at least one character with strike=True"

    # Word-level – detect the word that is fully struck out ("strikeout")
    struck_words = [w for w in page.words if getattr(w, "strike", False)]
    assert struck_words, "Expected at least one word with strike=True"
    assert any(
        "strikeout" in w.text.lower() for w in struck_words
    ), "Word 'strikeout' should be struck"
