#!/usr/bin/env python3
"""Test PageCollection.groupby functionality."""

from unittest.mock import MagicMock, Mock

import pytest

from natural_pdf.core.page_collection import PageCollection
from natural_pdf.core.page_groupby import PageGroupBy


def test_groupby_with_selector():
    """Test groupby with CSS selector string."""
    # Create mock pages
    page1 = Mock()
    page2 = Mock()
    page3 = Mock()

    # Mock find results for selector 'text[size=16]'
    # Page 1: finds element with text "Chapter 1"
    element1 = Mock()
    element1.extract_text.return_value = "Chapter 1"
    page1.find.return_value = element1

    # Page 2: finds element with text "Chapter 1" (same group)
    element2 = Mock()
    element2.extract_text.return_value = "Chapter 1"
    page2.find.return_value = element2

    # Page 3: finds element with text "Chapter 2"
    element3 = Mock()
    element3.extract_text.return_value = "Chapter 2"
    page3.find.return_value = element3

    # Create page collection
    pages = PageCollection([page1, page2, page3])

    # Group by selector
    grouped = pages.groupby("text[size=16]")

    # Test that it returns a PageGroupBy object
    assert isinstance(grouped, PageGroupBy)

    # Test iteration
    groups_dict = dict(grouped)
    assert "Chapter 1" in groups_dict
    assert "Chapter 2" in groups_dict
    assert len(groups_dict["Chapter 1"]) == 2  # pages 1 and 2
    assert len(groups_dict["Chapter 2"]) == 1  # page 3

    # Verify find was called correctly on each page
    page1.find.assert_called_with("text[size=16]")
    page2.find.assert_called_with("text[size=16]")
    page3.find.assert_called_with("text[size=16]")


def test_groupby_with_callable():
    """Test groupby with callable function."""
    # Create mock pages
    page1 = Mock()
    page2 = Mock()
    page3 = Mock()

    # Create page collection
    pages = PageCollection([page1, page2, page3])

    # Mock callable that returns different values for different pages
    def extract_chapter(page):
        if page is page1 or page is page2:
            return "Chapter A"
        else:
            return "Chapter B"

    # Group by callable
    grouped = pages.groupby(extract_chapter)

    # Test iteration
    groups_dict = dict(grouped)
    assert "Chapter A" in groups_dict
    assert "Chapter B" in groups_dict
    assert len(groups_dict["Chapter A"]) == 2
    assert len(groups_dict["Chapter B"]) == 1


def test_groupby_with_none_results():
    """Test groupby when selector finds no elements (returns None)."""
    # Create mock pages
    page1 = Mock()
    page2 = Mock()

    # Mock find results - page1 finds element, page2 finds nothing
    element1 = Mock()
    element1.extract_text.return_value = "Found Text"
    page1.find.return_value = element1
    page2.find.return_value = None  # No matching element

    # Create page collection
    pages = PageCollection([page1, page2])

    # Group by selector
    grouped = pages.groupby("text[size=16]")

    # Test results
    groups_dict = dict(grouped)
    assert "Found Text" in groups_dict
    assert None in groups_dict  # Pages with no match group under None
    assert len(groups_dict["Found Text"]) == 1
    assert len(groups_dict[None]) == 1


def test_groupby_dict_like_access():
    """Test dict-like access methods."""
    # Create mock pages
    page1 = Mock()
    page2 = Mock()

    # Mock find results
    element1 = Mock()
    element1.extract_text.return_value = "Group A"
    page1.find.return_value = element1

    element2 = Mock()
    element2.extract_text.return_value = "Group B"
    page2.find.return_value = element2

    # Create page collection
    pages = PageCollection([page1, page2])
    grouped = pages.groupby("text[size=16]")

    # Test .get() method
    group_a = grouped.get("Group A")
    assert group_a is not None
    assert len(group_a) == 1

    # Test .get() with default
    missing_group = grouped.get("Missing Group", "default")
    assert missing_group == "default"

    # Test .get_group() method
    group_b = grouped.get_group("Group B")
    assert len(group_b) == 1

    # Test .get_group() with missing key raises KeyError
    with pytest.raises(KeyError, match="Group key 'Missing' not found"):
        grouped.get_group("Missing")

    # Test .keys() method
    keys = grouped.keys()
    assert "Group A" in keys
    assert "Group B" in keys
    assert len(keys) == 2


def test_groupby_apply():
    """Test apply method for batch operations."""
    # Create mock pages
    page1 = Mock()
    page2 = Mock()
    page3 = Mock()

    # Mock find results - two groups
    element1 = Mock()
    element1.extract_text.return_value = "Section 1"
    page1.find.return_value = element1

    element2 = Mock()
    element2.extract_text.return_value = "Section 1"
    page2.find.return_value = element2

    element3 = Mock()
    element3.extract_text.return_value = "Section 2"
    page3.find.return_value = element3

    # Create page collection
    pages = PageCollection([page1, page2, page3])
    grouped = pages.groupby("text[size=16]")

    # Apply function that returns page count
    results = grouped.apply(lambda page_collection: len(page_collection))

    assert results["Section 1"] == 2
    assert results["Section 2"] == 1


def test_groupby_lazy_evaluation():
    """Test that groups are computed lazily."""
    # Create mock pages
    page1 = Mock()

    # Create page collection
    pages = PageCollection([page1])
    grouped = pages.groupby("text[size=16]")

    # Groups should not be computed yet
    assert grouped._groups is None

    # Access should trigger computation
    list(grouped)  # Force computation
    assert grouped._groups is not None


def test_groupby_caching():
    """Test that groups are cached after first computation."""
    # Create mock pages
    page1 = Mock()
    element1 = Mock()
    element1.extract_text.return_value = "Test"
    page1.find.return_value = element1

    # Create page collection
    pages = PageCollection([page1])
    grouped = pages.groupby("text[size=16]")

    # First access
    groups1 = grouped._compute_groups()

    # Second access should return cached result
    groups2 = grouped._compute_groups()
    assert groups1 is groups2  # Same object reference


def test_groupby_indexing():
    """Test index-based access."""
    # Create mock pages
    page1 = Mock()
    page2 = Mock()
    page3 = Mock()

    # Mock different groups (2 pages in first group, 1 in second)
    element1 = Mock()
    element1.extract_text.return_value = "Group A"
    page1.find.return_value = element1

    element2 = Mock()
    element2.extract_text.return_value = "Group A"
    page2.find.return_value = element2

    element3 = Mock()
    element3.extract_text.return_value = "Group B"
    page3.find.return_value = element3

    # Create page collection
    pages = PageCollection([page1, page2, page3])
    grouped = pages.groupby("text[size=16]")

    # Test positive indexing
    first_group = grouped[0]
    assert len(first_group) == 2  # Group A has 2 pages

    second_group = grouped[1]
    assert len(second_group) == 1  # Group B has 1 page

    # Test negative indexing
    last_group = grouped[-1]
    assert len(last_group) == 1  # Last group is Group B

    second_last = grouped[-2]
    assert len(second_last) == 2  # Second-to-last group is Group A

    # Test access by key
    group_a = grouped["Group A"]
    assert len(group_a) == 2

    group_b = grouped["Group B"]
    assert len(group_b) == 1

    # Test error cases
    with pytest.raises(IndexError, match="Group index 5 out of range"):
        grouped[5]

    with pytest.raises(IndexError, match="Group index -5 out of range"):
        grouped[-5]

    with pytest.raises(KeyError, match="Group key 'Missing' not found"):
        grouped["Missing"]


def test_groupby_repr_and_len():
    """Test string representation and length."""
    # Create mock pages
    page1 = Mock()
    page2 = Mock()

    # Mock different groups
    element1 = Mock()
    element1.extract_text.return_value = "Group 1"
    page1.find.return_value = element1

    element2 = Mock()
    element2.extract_text.return_value = "Group 2"
    page2.find.return_value = element2

    # Create page collection
    pages = PageCollection([page1, page2])
    grouped = pages.groupby("text[size=16]")

    # Test length
    assert len(grouped) == 2

    # Test repr
    repr_str = repr(grouped)
    assert "PageGroupBy" in repr_str
    assert "groups=2" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
