import pytest

from natural_pdf.elements.region import Region


def _selected_word_text(region, overlap):
    words = region.find_all("text", overlap=overlap)
    return words.extract_text(separator=" ")


def test_region_word_extraction_is_an_explicit_selection(practice_pdf):
    page = practice_pdf.pages[0]
    region = Region(page, bbox=(100, 200, 300, 400))

    text_center = _selected_word_text(region, "center")
    text_full = _selected_word_text(region, "full")
    text_partial = _selected_word_text(region, "partial")

    assert all(isinstance(text, str) for text in (text_center, text_full, text_partial))
    assert len(text_full) <= len(text_center) <= len(text_partial)


def test_explicit_word_selection_respects_exclusions(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    first_word = page.find("text")
    if first_word:
        page.add_exclusion(first_word)
    region = Region(page, bbox=(0, 0, page.width, page.height / 2))

    with_exclusions = region.find_all("text", overlap="center", apply_exclusions=True).extract_text(
        separator=" "
    )
    without_exclusions = region.find_all(
        "text", overlap="center", apply_exclusions=False
    ).extract_text(separator=" ")

    if first_word:
        assert len(with_exclusions) < len(without_exclusions)


def test_region_rejects_removed_word_mode(practice_pdf):
    page = practice_pdf.pages[0]
    region = Region(page, bbox=(0, 0, 100, 100))

    with pytest.raises(TypeError):
        region.extract_text(granularity="words")
    with pytest.raises(TypeError):
        region.extract_text("words")
