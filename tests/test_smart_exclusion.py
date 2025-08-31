"""Test smart exclusion that uses appropriate method based on element type"""

from unittest.mock import Mock, call, patch

import pytest

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement


def test_text_element_uses_element_exclusion():
    """Test that TextElement.exclude() uses 'element' method"""
    # Create mock page
    page = Mock()
    page.add_exclusion = Mock()

    # Create text element
    text_elem = TextElement(
        {"text": "watermark", "x0": 10, "top": 20, "x1": 30, "bottom": 40}, page
    )

    # Exclude the text element
    text_elem.exclude()

    # Verify it was called with 'element' method
    page.add_exclusion.assert_called_once_with(text_elem, method="element")


def test_region_uses_region_exclusion():
    """Test that Region.exclude() uses 'region' method"""
    # Create mock page
    page = Mock()
    page.add_exclusion = Mock()

    # Create region
    region = Region(page, (10, 20, 30, 40))

    # Exclude the region
    region.exclude()

    # Verify it was called with 'region' method
    page.add_exclusion.assert_called_once_with(region, method="region")


def test_mixed_collection_uses_appropriate_methods():
    """Test that a collection with mixed element types uses appropriate exclusion methods"""
    # Create mock pages
    page1 = Mock()
    page1.add_exclusion = Mock()
    page2 = Mock()
    page2.add_exclusion = Mock()

    # Create mixed elements
    text1 = TextElement({"text": "watermark1", "x0": 10, "top": 20, "x1": 30, "bottom": 40}, page1)

    text2 = TextElement({"text": "watermark2", "x0": 50, "top": 60, "x1": 70, "bottom": 80}, page1)

    region1 = Region(page2, (100, 110, 120, 130))

    # Create collection with mixed types
    collection = ElementCollection([text1, text2, region1])

    # Exclude the collection
    collection.exclude()

    # Verify each was called with appropriate method
    assert page1.add_exclusion.call_count == 2
    page1.add_exclusion.assert_any_call(text1, method="element")
    page1.add_exclusion.assert_any_call(text2, method="element")

    page2.add_exclusion.assert_called_once_with(region1, method="region")


def test_watermark_scenario():
    """Test a realistic watermark exclusion scenario"""
    # This tests the scenario where:
    # 1. We have a watermark text overlaying other content
    # 2. We exclude the watermark (should use element method)
    # 3. We can still access text in the same area

    # Create mock page with mock elements
    page = Mock()
    page.add_exclusion = Mock()
    page._exclusions = []

    # Create watermark text
    watermark = TextElement(
        {
            "text": "CONFIDENTIAL",
            "x0": 100,
            "top": 100,
            "x1": 200,
            "bottom": 120,
            "fontname": "Helvetica-Bold",
            "size": 36,
        },
        page,
    )

    # Create regular text that overlaps with watermark
    regular_text = TextElement(
        {
            "text": "Important document content",
            "x0": 90,
            "top": 105,
            "x1": 210,
            "bottom": 115,
            "fontname": "Times",
            "size": 12,
        },
        page,
    )

    # Exclude the watermark
    watermark.exclude()

    # Verify it used element method (not region)
    page.add_exclusion.assert_called_once_with(watermark, method="element")

    # The regular text in the same area should NOT be excluded
    # (This would need to be tested with actual page filtering logic)
