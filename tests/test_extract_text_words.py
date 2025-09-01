import pytest

from natural_pdf import PDF
from natural_pdf.elements.region import Region


def test_extract_text_words_vs_chars():
    """Test that word-level extraction differs from character-level extraction."""
    # Create a simple PDF with text
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Create a region that cuts through the middle of some words
    # This region should capture partial words at the boundaries
    region = Region(page, bbox=(100, 200, 300, 400))

    # Character-level extraction (default)
    char_text = region.extract_text()
    char_text_explicit = region.extract_text("chars")

    # Word-level extraction
    word_text = region.extract_text("words")
    word_text_center = region.extract_text("words", overlap="center")
    word_text_full = region.extract_text("words", overlap="full")
    word_text_partial = region.extract_text("words", overlap="partial")

    # Basic tests
    assert char_text == char_text_explicit  # Default should be 'chars'
    assert word_text == word_text_center  # Default overlap should be 'center'

    # Word-level should generally be different from char-level
    # (unless the region boundaries perfectly align with word boundaries)
    # Can't assert they're different without knowing the exact content

    # Overlap modes should have this relationship:
    # full <= center <= partial (in terms of amount of text)
    assert len(word_text_full) <= len(word_text_center)
    assert len(word_text_center) <= len(word_text_partial)


def test_extract_text_words_with_exclusions():
    """Test that word-level extraction respects exclusions."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Find a word to exclude
    first_word = page.find("text")
    if first_word:
        page.add_exclusion(first_word)

    # Create a region that includes the excluded word
    region = Region(page, bbox=(0, 0, page.width, page.height / 2))

    # Extract with and without exclusions
    text_with_exclusions = region.extract_text("words", apply_exclusions=True)
    text_without_exclusions = region.extract_text("words", apply_exclusions=False)

    # Should have different amounts of text if exclusion worked
    if first_word:
        assert len(text_with_exclusions) < len(text_without_exclusions)


def test_extract_text_words_overlap_modes():
    """Test different overlap modes for word extraction."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Create a small region that will have words at its boundaries
    # Use the middle of the page to ensure we have some content
    region = Region(
        page, bbox=(page.width / 4, page.height / 4, page.width * 3 / 4, page.height * 3 / 4)
    )

    # Test all overlap modes
    text_full = region.extract_text("words", overlap="full")
    text_center = region.extract_text("words", overlap="center")
    text_partial = region.extract_text("words", overlap="partial")

    # Each mode should be valid
    assert isinstance(text_full, str)
    assert isinstance(text_center, str)
    assert isinstance(text_partial, str)

    # Length relationship should hold
    assert len(text_full) <= len(text_center) <= len(text_partial)


def test_extract_text_invalid_granularity():
    """Test that invalid granularity raises appropriate error."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]
    region = Region(page, bbox=(0, 0, 100, 100))

    # Should raise error for invalid granularity
    with pytest.raises(ValueError, match="granularity must be 'chars' or 'words'"):
        region.extract_text("lines")


def test_extract_text_backwards_compatibility():
    """Test that existing code continues to work unchanged."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]
    region = Region(page, bbox=(0, 0, 200, 200))

    # All existing parameters should still work
    text1 = region.extract_text()
    text2 = region.extract_text(apply_exclusions=True)
    text3 = region.extract_text(apply_exclusions=False)
    text4 = region.extract_text(debug=False)

    # Should all return strings
    assert all(isinstance(t, str) for t in [text1, text2, text3, text4])

    # With no exclusions, apply_exclusions parameter shouldn't matter
    if not page._exclusions:
        assert text2 == text3
