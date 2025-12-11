import pytest

from natural_pdf import PDF

PDF_PATH = "pdfs/en_68f5d334eb339_Vol_52_no_36-english.pdf"


def _get_table_region(page):
    return page.find("text:contains(Table 1)").below(
        until="text:contains(Page 3)", include_endpoint=False
    )


def test_page_rotate_is_view_and_preserves_original():
    pdf = PDF(PDF_PATH)
    page = pdf.pages[2]
    original_rotation = page._page.rotation
    original_size = (page.width, page.height)

    rotated = page.rotate(90)

    # Original page untouched
    assert page._page.rotation == original_rotation
    assert (page.width, page.height) == original_size

    # Rotated view swaps dimensions
    assert rotated.width == pytest.approx(original_size[1])
    assert rotated.height == pytest.approx(original_size[0])

    # Text is reprocessed in upright orientation
    text = rotated.extract_text()
    assert "Dengue Fever" in text


def test_region_rotate_reprocesses_text_and_tables_without_side_effects():
    pdf = PDF(PDF_PATH)
    page = pdf.pages[2]
    region = _get_table_region(page)

    original_text = region.extract_text()
    rotated_region = region.rotate(90)

    # Original region/page untouched
    assert region.extract_text() == original_text
    assert page._page.rotation == 0

    rotated_text = rotated_region.extract_text()
    assert "Dengue Fever" in rotated_text
    assert "Colombo" in rotated_text

    table = rotated_region.extract_table()
    assert table  # table extraction succeeds
    flattened = [" ".join(cell for cell in row if cell) for row in table]
    assert any("Colombo" in line for line in flattened)
