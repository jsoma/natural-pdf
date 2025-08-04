import pytest

from natural_pdf import PDF


def test_highlight_detection_types_of_type():
    pdf = PDF("pdfs/types-of-type.pdf")
    page = pdf.pages[0]

    # Get all words and check highlight status
    all_words = page.words
    highlighted_words = [w for w in all_words if w.is_highlighted]
    non_highlighted_words = [w for w in all_words if not w.is_highlighted]

    # Verify we have both highlighted and non-highlighted text
    assert highlighted_words, "Expected highlighted words"
    assert non_highlighted_words, "Expected non-highlighted words"

    # Verify specific highlighted text
    assert any(
        "highlighted" in w.text.lower() for w in highlighted_words
    ), "Expected 'Highlighted text' to be detected as highlighted"

    # Verify specific non-highlighted text
    normal_text_words = [w for w in all_words if "normal" in w.text.lower()]
    assert normal_text_words, "Expected to find 'Normal text' in the document"
    assert all(
        not w.is_highlighted for w in normal_text_words
    ), "'Normal text' should not be detected as highlighted"

    # Verify other non-highlighted text types
    bold_text_words = [w for w in all_words if "bold" in w.text.lower()]
    assert bold_text_words, "Expected to find 'Bold text' in the document"
    assert all(
        not w.is_highlighted for w in bold_text_words
    ), "'Bold text' should not be detected as highlighted"

    # Check that most text is not highlighted (reasonable assumption)
    highlight_ratio = len(highlighted_words) / len(all_words)
    assert highlight_ratio < 0.5, f"Too many words detected as highlighted: {highlight_ratio:.1%}"
