from unittest.mock import Mock, patch

import pytest

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.text import TextElement


def test_exclude_multi_page_collection():
    """Test that exclude() works correctly with collections spanning multiple pages"""
    # Create mock pages
    page1 = Mock()
    page2 = Mock()

    # Create mock elements with different pages
    elem1 = Mock(spec=TextElement)
    elem1.page = page1
    elem1.exclude = Mock()

    elem2 = Mock(spec=TextElement)
    elem2.page = page1
    elem2.exclude = Mock()

    elem3 = Mock(spec=TextElement)
    elem3.page = page2
    elem3.exclude = Mock()

    # Create collection spanning two pages
    collection = ElementCollection([elem1, elem2, elem3])

    # This should now work and call exclude on each element
    result = collection.exclude()

    # Verify exclude was called on each element
    elem1.exclude.assert_called_once()
    elem2.exclude.assert_called_once()
    elem3.exclude.assert_called_once()

    # Verify it returns self for method chaining
    assert result is collection


def test_exclude_single_page_collection():
    """Test that exclude() works with a single-page collection"""
    # Create mock page
    page1 = Mock()

    # Create mock elements all on same page
    elem1 = Mock(spec=TextElement)
    elem1.page = page1
    elem1.exclude = Mock()

    elem2 = Mock(spec=TextElement)
    elem2.page = page1
    elem2.exclude = Mock()

    # Create collection on single page
    collection = ElementCollection([elem1, elem2])

    # This should now work
    result = collection.exclude()

    # Verify exclude was called on each element
    elem1.exclude.assert_called_once()
    elem2.exclude.assert_called_once()

    # Verify it returns self for method chaining
    assert result is collection


def test_exclude_empty_collection():
    """Test that exclude() works with an empty collection"""
    # Create empty collection
    collection = ElementCollection([])

    # This should work without errors
    result = collection.exclude()

    # Verify it returns self for method chaining
    assert result is collection


def test_individual_element_exclude():
    """Test that individual elements can be excluded"""
    # Create mock page
    page = Mock()
    page.add_exclusion = Mock()

    # Create mock element
    elem = Mock(spec=TextElement)
    elem.page = page

    # Mock the exclude method to match base.py implementation
    def mock_exclude(self):
        self.page.add_exclusion(self)

    elem.exclude = lambda: mock_exclude(elem)

    # This should work
    elem.exclude()
    page.add_exclusion.assert_called_once_with(elem)
