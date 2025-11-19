#!/usr/bin/env python3
"""Test Page.add_exclusion() with lists and tuples of regions/elements."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from natural_pdf.core.page import Page
from natural_pdf.elements.region import Region
from natural_pdf.services.exclusion_service import ExclusionService


def _make_page_stub():
    """Return a mock Page wired up with the real exclusion helpers."""
    page = Mock(spec=Page)
    page.index = 0
    page.width = 612
    page.height = 792
    page._exclusions = []
    page._element_mgr = Mock()
    page._element_mgr.invalidate_cache = Mock()

    # Bind the concrete helper implementations we rely on
    bound_methods = [
        "add_exclusion",
        "_element_to_region",
        "_invalidate_exclusion_cache",
    ]
    for name in bound_methods:
        setattr(page, name, getattr(Page, name).__get__(page, Page))

    # Most tests never hit find_all, but wire a safe default anyway
    page.find_all = Mock(return_value=[])

    # Inject real ExclusionService
    page.services = Mock()
    page.services.exclusion = ExclusionService(context=Mock())

    return page


def test_add_exclusion_list_of_elements():
    """Test add_exclusion with a list of elements (region method)."""
    page = _make_page_stub()

    # Create mock elements
    element1 = SimpleNamespace(bbox=(100, 100, 200, 150))
    element2 = SimpleNamespace(bbox=(300, 200, 400, 250))

    elements = [element1, element2]

    # Should work without error
    result = page.add_exclusion(elements, label="test_list")

    # Should return self for chaining
    assert result is page

    # Should have added 2 exclusions
    assert len(page._exclusions) == 2

    # Each should be a region exclusion
    for exclusion_data in page._exclusions:
        region, label, method = exclusion_data
        assert isinstance(region, Region)
        assert label == "test_list"
        assert method == "region"


def test_add_exclusion_list_of_elements_element_method():
    """Test add_exclusion with a list of elements (element method)."""
    page = _make_page_stub()

    # Create mock elements
    element1 = SimpleNamespace(bbox=(100, 100, 200, 150))
    element2 = SimpleNamespace(bbox=(300, 200, 400, 250))

    elements = [element1, element2]

    # Should work with element method
    result = page.add_exclusion(elements, label="test_elements", method="element")

    assert result is page
    assert len(page._exclusions) == 2

    # Elements should be stored directly (not converted to regions)
    for i, exclusion_data in enumerate(page._exclusions):
        element, label, method = exclusion_data
        assert element is elements[i]  # Same object reference
        assert label == "test_elements"
        assert method == "element"


def test_add_exclusion_list_of_regions():
    """Test add_exclusion with a list of Region objects."""
    page = _make_page_stub()

    # Create Region objects
    region1 = Region(page, (100, 100, 200, 150))
    region2 = Region(page, (300, 200, 400, 250))

    regions = [region1, region2]

    # Should work with list of regions
    result = page.add_exclusion(regions, label="test_regions")

    assert result is page
    assert len(page._exclusions) == 2

    # Regions should be stored directly
    for i, exclusion_data in enumerate(page._exclusions):
        region, label, method = exclusion_data
        assert region is regions[i]
        assert label == "test_regions"
        assert method == "region"


def test_add_exclusion_tuple():
    """Test add_exclusion with a tuple of elements."""
    page = _make_page_stub()

    # Create tuple of mock elements
    element1 = SimpleNamespace(bbox=(100, 100, 200, 150))
    element2 = SimpleNamespace(bbox=(300, 200, 400, 250))

    elements = (element1, element2)  # Tuple instead of list

    # Should work with tuple
    result = page.add_exclusion(elements, label="test_tuple")

    assert result is page
    assert len(page._exclusions) == 2


def test_add_exclusion_empty_list():
    """Test add_exclusion with an empty list."""
    page = _make_page_stub()

    # Should handle empty list gracefully
    result = page.add_exclusion([], label="empty")

    assert result is page
    assert len(page._exclusions) == 0  # Nothing should be added


def test_add_exclusion_list_with_invalid_items():
    """Test add_exclusion with list containing items without bbox."""
    page = _make_page_stub()

    # Create mix of valid and invalid elements
    valid_element = SimpleNamespace(bbox=(100, 100, 200, 150))
    invalid_element = object()  # No bbox attribute

    elements = [valid_element, invalid_element]

    # Should error when an item lacks bbox
    with pytest.raises(TypeError):
        page.add_exclusion(elements, label="mixed")


def test_add_exclusion_mixed_regions_and_elements():
    """Test add_exclusion with list containing both regions and elements."""
    page = _make_page_stub()

    # Create mix of region and element
    region = Region(page, (100, 100, 200, 150))

    element = SimpleNamespace(bbox=(300, 200, 400, 250))

    mixed_list = [region, element]

    # Should work with mixed types
    result = page.add_exclusion(mixed_list, label="mixed_types")

    assert result is page
    assert len(page._exclusions) == 2

    # First should be the original region, second should be converted to region
    region_stored, label1, method1 = page._exclusions[0]
    assert region_stored is region

    region_converted, label2, method2 = page._exclusions[1]
    assert isinstance(region_converted, Region)
    assert region_converted is not element  # Different object, converted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
