#!/usr/bin/env python3
"""Test the new .attr() method for extracting attribute values."""

import statistics
from pathlib import Path

import pytest

from natural_pdf import PDF
from natural_pdf.elements.element_collection import ElementCollection


def test_attr_basic():
    """Test basic attr extraction."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get text elements
    elements = page.find_all("text")[:10]

    # Extract text content
    texts = elements.attr("text")

    # Should return a list of strings
    assert isinstance(texts, list)
    assert all(isinstance(t, str) for t in texts)
    assert len(texts) <= len(elements)  # May skip None values


def test_attr_numeric_values():
    """Test extracting numeric attributes."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get text elements
    elements = page.find_all("text")[:10]

    # Extract sizes
    sizes = elements.attr("size")

    # Should return a list of numbers
    assert isinstance(sizes, list)
    assert all(isinstance(s, (int, float)) for s in sizes)

    # Test with standard library functions
    if sizes:
        assert max(sizes) >= min(sizes)
        avg_size = statistics.mean(sizes)
        assert isinstance(avg_size, float)


def test_attr_skip_empty():
    """Test skip_empty parameter."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Create a mixed collection (some elements might not have certain attributes)
    elements = page.find_all("*")[:20]

    # With skip_empty=True (default)
    with_skip = elements.attr("text", skip_empty=True)

    # With skip_empty=False
    without_skip = elements.attr("text", skip_empty=False)

    # without_skip should have more or equal items
    assert len(without_skip) >= len(with_skip)

    # with_skip should have no None values
    assert None not in with_skip

    # without_skip might have None values
    # (for elements that don't have 'text' attribute)


def test_attr_bbox_components():
    """Test extracting bbox components."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get some elements with bboxes
    elements = page.find_all("text")[:5]

    # Extract bbox (should be tuples)
    bboxes = elements.attr("bbox")

    assert isinstance(bboxes, list)
    assert all(isinstance(b, tuple) and len(b) == 4 for b in bboxes)

    # Extract individual bbox components if they exist as properties
    # (This might not work depending on element implementation)
    try:
        widths = elements.attr("width")
        if widths:
            assert all(isinstance(w, (int, float)) for w in widths)
    except AttributeError:
        pass  # Not all elements have width as a direct attribute


def test_attr_with_statistics():
    """Test using attr with statistics module."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get text elements
    elements = page.find_all("text")[:20]

    # Extract sizes
    sizes = elements.attr("size")

    if len(sizes) > 1:
        # Calculate statistics
        mean_size = statistics.mean(sizes)
        median_size = statistics.median(sizes)

        # Filter elements larger than average
        large_elements = elements.filter(lambda e: getattr(e, "size", 0) > mean_size)

        # Should have some but not all elements
        assert 0 < len(large_elements) < len(elements)


def test_attr_empty_collection():
    """Test attr on empty collection."""
    empty = ElementCollection([])

    # Extract from empty collection
    result = empty.attr("text")

    assert isinstance(result, list)
    assert len(result) == 0

    # With skip_empty=False
    result_no_skip = empty.attr("text", skip_empty=False)
    assert isinstance(result_no_skip, list)
    assert len(result_no_skip) == 0


def test_attr_on_single_element():
    """Test .attr() works on individual elements."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get a single text element
    element = page.find("text")

    # Use .attr() on single element
    text = element.attr("text")
    size = element.attr("size")
    bbox = element.attr("bbox")

    # Should return the actual values, not lists
    assert isinstance(text, (str, type(None)))
    assert isinstance(size, (int, float, type(None)))
    assert isinstance(bbox, (tuple, type(None)))

    # Should be same as direct access
    assert text == element.text
    assert size == element.size
    assert bbox == element.bbox

    # Non-existent attribute should return None
    assert element.attr("nonexistent") is None


def test_attr_on_region():
    """Test .attr() works on regions."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Create a region
    region = page.find("text").expand(10)

    # Use .attr() on region
    width = region.attr("width")
    height = region.attr("height")
    bbox = region.attr("bbox")

    # Should return the actual values
    assert isinstance(width, (int, float))
    assert isinstance(height, (int, float))
    assert isinstance(bbox, tuple)

    # Should be same as direct access
    assert width == region.width
    assert height == region.height
    assert bbox == region.bbox


def test_attr_consistent_api():
    """Test .attr() provides consistent API for elements and collections."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get both single element and collection
    single = page.find("text")
    collection = page.find_all("text")[:5]

    # Both should have .attr() method
    assert hasattr(single, "attr")
    assert hasattr(collection, "attr")

    # Single element returns single value
    single_size = single.attr("size")
    assert not isinstance(single_size, list)

    # Collection returns list
    collection_sizes = collection.attr("size")
    assert isinstance(collection_sizes, list)

    # The first element in collection should match single (if it's the same element)
    if collection[0] == single:
        assert collection_sizes[0] == single_size


def test_attr_nonexistent_attribute():
    """Test extracting non-existent attribute."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get text elements
    elements = page.find_all("text")[:5]

    # Try to extract non-existent attribute
    result = elements.attr("nonexistent_attribute")

    # Should return empty list (all None values are skipped by default)
    assert isinstance(result, list)
    assert len(result) == 0

    # With skip_empty=False, should get list of Nones
    result_with_none = elements.attr("nonexistent_attribute", skip_empty=False)
    assert len(result_with_none) == len(elements)
    assert all(v is None for v in result_with_none)
