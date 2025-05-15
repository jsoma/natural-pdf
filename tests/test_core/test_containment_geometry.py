import pytest

def test_containment_geometry_on_page_and_element(geometry_pdf):
    """
    Tests the behavior of find_all with different `contains`
    options on both a Page and an Element, using geometry.pdf.
    """
    page = geometry_pdf.pages[0]
    rect = page.find('rect')

    # Test on the page
    all_text_on_page = page.find_all('text')
    assert len(all_text_on_page) == 4, f"Expected 4 text elements on page, got {len(all_text_on_page)}"

    # Test on the rect element
    # Default (contains='all')
    text_fully_in_rect = rect.find_all('text')
    assert len(text_fully_in_rect) == 1, f"Expected 1 text element fully in rect, got {len(text_fully_in_rect)}"

    # contains='any'
    text_any_overlap_rect = rect.find_all('text', contains='any')
    assert len(text_any_overlap_rect) == 3, f"Expected 3 text elements with any overlap in rect, got {len(text_any_overlap_rect)}"

    # contains='center'
    text_center_in_rect = rect.find_all('text', contains='center')
    assert len(text_center_in_rect) == 2, f"Expected 2 text elements with center in rect, got {len(text_center_in_rect)}" 