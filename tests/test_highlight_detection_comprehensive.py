import pytest

from natural_pdf import PDF


def test_highlight_detection_comprehensive():
    """Test that highlight detection correctly identifies highlighted and non-highlighted text."""
    pdf = PDF("pdfs/types-of-type.pdf")
    page = pdf.pages[0]

    # Test at different element levels
    # 1. Test with words
    all_words = page.words
    highlighted_words = [w for w in all_words if w.is_highlighted]
    non_highlighted_words = [w for w in all_words if not w.is_highlighted]

    assert highlighted_words, "Expected some highlighted words"
    assert non_highlighted_words, "Expected some non-highlighted words"

    # 2. Test with find_all at text level
    all_text_elements = page.find_all("text")
    highlighted_text = [t for t in all_text_elements if t.is_highlighted]
    non_highlighted_text = [t for t in all_text_elements if not t.is_highlighted]

    assert highlighted_text, "Expected some highlighted text elements"
    assert non_highlighted_text, "Expected some non-highlighted text elements"

    # 3. Verify specific content
    # "Highlighted text" should be highlighted
    highlighted_content = page.find_all('text:contains("Highlighted text")')
    assert highlighted_content, "Should find 'Highlighted text' in document"
    for elem in highlighted_content:
        assert elem.is_highlighted, f"'{elem.text}' should be highlighted"

    # "Normal text" should NOT be highlighted
    normal_content = page.find_all('text:contains("Normal text")')
    assert normal_content, "Should find 'Normal text' in document"
    for elem in normal_content:
        assert not elem.is_highlighted, f"'{elem.text}' should NOT be highlighted"

    # "Bold text" should NOT be highlighted (bold is not the same as highlight)
    bold_content = page.find_all('text:contains("Bold text")')
    assert bold_content, "Should find 'Bold text' in document"
    for elem in bold_content:
        assert not elem.is_highlighted, f"'{elem.text}' should NOT be highlighted"
        assert elem.bold, f"'{elem.text}' should be bold"

    # "talic text" should NOT be highlighted (the 'i' is separate)
    italic_content = page.find_all('text:contains("talic text")')
    assert italic_content, "Should find 'talic text' in document"
    for elem in italic_content:
        assert not elem.is_highlighted, f"'{elem.text}' should NOT be highlighted"
        assert elem.italic, f"'{elem.text}' should be italic"

    # 4. Test that we DON'T accidentally get the highlight() method
    for elem in all_text_elements:
        # Ensure is_highlighted is a boolean, not a method
        assert isinstance(
            elem.is_highlighted, bool
        ), f"is_highlighted should be bool, not {type(elem.is_highlighted)}"

        # Ensure highlight is a method (for visual highlighting)
        assert callable(elem.highlight), "highlight should be a method"

    # 5. Verify highlight_color property works for highlighted text
    for elem in highlighted_text:
        # Highlighted text might have a color
        color = elem.highlight_color
        if color is not None:
            # Due to pdfminer.six bug, colors can be float (grayscale) or tuple (RGB)
            assert isinstance(
                color, (tuple, list, float)
            ), f"highlight_color should be tuple/list/float, got {type(color)}"


def test_highlight_selector():
    """Test that the :highlighted pseudo-selector works correctly."""
    pdf = PDF("pdfs/types-of-type.pdf")
    page = pdf.pages[0]

    # Find all highlighted text using the selector
    highlighted = page.find_all("text:highlighted")
    assert highlighted, "Should find highlighted text with :highlighted selector"

    # Verify all found elements are actually highlighted
    for elem in highlighted:
        assert elem.is_highlighted, f"Element found with :highlighted should be highlighted"

    # Verify we can combine selectors
    highlighted_with_text = page.find_all('text:highlighted:contains("text")')
    assert highlighted_with_text, "Should find highlighted elements containing 'text'"

    for elem in highlighted_with_text:
        assert elem.is_highlighted, "Should be highlighted"
        assert "text" in elem.text.lower(), "Should contain 'text'"
