"""Test that PDF-level exclusions work correctly in find, find_all, and get_elements methods."""

from unittest.mock import MagicMock, Mock

import pytest

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.text import TextElement


def test_find_with_pdf_exclusions():
    """Test that find() applies PDF-level exclusions even when page has no exclusions."""
    # Create mock PDF with exclusions
    mock_pdf = Mock()
    mock_pdf._exclusions = [
        (lambda page: page.find("text:contains(HEADER)").expand(), "header_exclusion")
    ]

    # Create mock page with NO page-level exclusions
    mock_page = Mock()
    mock_page.index = 0
    mock_page._exclusions = []  # Empty page exclusions
    mock_page._parent = mock_pdf

    # Create mock elements
    header_element = Mock(spec=TextElement)
    header_element.text = "HEADER"
    header_element.bbox = (100, 0, 200, 50)
    header_element.x0 = 100
    header_element.x1 = 200
    header_element.top = 0
    header_element.bottom = 50

    content_element = Mock(spec=TextElement)
    content_element.text = "Content"
    content_element.bbox = (100, 100, 200, 150)
    content_element.x0 = 100
    content_element.x1 = 200
    content_element.top = 100
    content_element.bottom = 150

    # Mock the find method to return header element when searching for HEADER
    def mock_find(selector, apply_exclusions=True):
        if "HEADER" in selector:
            return header_element
        return None

    # Mock region expansion
    mock_region = Mock()
    mock_region.bbox = (100, 0, 200, 50)
    mock_region.label = "header_exclusion"
    header_element.expand = Mock(return_value=mock_region)

    # Setup page methods
    mock_page.find = mock_find

    # Import the real Page class to test the actual methods
    from natural_pdf.core.page import Page

    # Create a real page instance but with our mocked attributes
    page = Page.__new__(Page)
    page._index = 0  # Set the internal _index attribute
    page._exclusions = []
    page._parent = mock_pdf
    page._page_obj = Mock()
    page._computing_exclusions = False  # Add the new flag
    page._computing_exclusions = False  # Add the new flag

    # Mock the internal methods we need
    page._get_exclusion_regions = Mock(return_value=[mock_region])
    page._filter_elements_by_exclusions = Mock(return_value=[content_element])

    # Mock _apply_selector to return both elements initially
    mock_collection = Mock(spec=ElementCollection)
    mock_collection.elements = [header_element, content_element]
    mock_collection.first = header_element
    page._apply_selector = Mock(return_value=mock_collection)

    # Test find() with apply_exclusions=True
    result = page.find("text", apply_exclusions=True)

    # Should have called _filter_elements_by_exclusions even though page._exclusions is empty
    page._filter_elements_by_exclusions.assert_called_once()
    assert result == content_element  # Header should be excluded


def test_find_all_with_pdf_exclusions():
    """Test that find_all() applies PDF-level exclusions even when page has no exclusions."""
    # Similar setup as above
    mock_pdf = Mock()
    mock_pdf._exclusions = [
        (lambda page: page.find("text:contains(FOOTER)").expand(), "footer_exclusion")
    ]

    mock_page = Mock()
    mock_page.index = 0
    mock_page._exclusions = []  # Empty page exclusions
    mock_page._parent = mock_pdf

    # Create mock elements
    content1 = Mock(spec=TextElement)
    content1.text = "Content 1"
    content1.bbox = (100, 100, 200, 150)

    footer = Mock(spec=TextElement)
    footer.text = "FOOTER"
    footer.bbox = (100, 500, 200, 550)

    content2 = Mock(spec=TextElement)
    content2.text = "Content 2"
    content2.bbox = (100, 200, 200, 250)

    # Import the real Page class
    from natural_pdf.core.page import Page

    # Create a real page instance
    page = Page.__new__(Page)
    page._index = 0  # Set the internal _index attribute
    page._exclusions = []
    page._parent = mock_pdf
    page._page_obj = Mock()
    page._computing_exclusions = False  # Add the new flag

    # Mock the methods
    mock_collection = Mock(spec=ElementCollection)
    mock_collection.elements = [content1, footer, content2]
    page._apply_selector = Mock(return_value=mock_collection)
    page._filter_elements_by_exclusions = Mock(return_value=[content1, content2])

    # Test find_all() with apply_exclusions=True
    result = page.find_all("text", apply_exclusions=True)

    # Should have called _filter_elements_by_exclusions
    page._filter_elements_by_exclusions.assert_called_once()
    assert isinstance(result, ElementCollection)


def test_get_elements_with_pdf_exclusions():
    """Test that get_elements() applies PDF-level exclusions even when page has no exclusions."""
    mock_pdf = Mock()
    mock_pdf._exclusions = [(Mock(), "some_exclusion")]

    # Import the real Page class
    from natural_pdf.core.page import Page

    # Create a real page instance
    page = Page.__new__(Page)
    page._index = 0  # Set the internal _index attribute
    page._exclusions = []  # Empty page exclusions
    page._parent = mock_pdf
    page._page_obj = Mock()
    page._computing_exclusions = False  # Add the new flag

    # Mock elements
    all_elements = [Mock(), Mock(), Mock()]
    filtered_elements = [all_elements[0], all_elements[2]]  # Exclude middle one

    # Mock the element manager
    mock_element_mgr = Mock()
    mock_element_mgr.get_all_elements = Mock(return_value=all_elements)
    page._element_mgr = mock_element_mgr

    # Mock the filter method
    page._filter_elements_by_exclusions = Mock(return_value=filtered_elements)

    # Test get_elements() with apply_exclusions=True
    result = page.get_elements(apply_exclusions=True)

    # Should have called _filter_elements_by_exclusions
    page._filter_elements_by_exclusions.assert_called_once_with(
        all_elements, debug_exclusions=False
    )
    assert result == filtered_elements


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
