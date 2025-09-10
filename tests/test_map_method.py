#!/usr/bin/env python3
"""Test the new .map() method with skip_empty parameter."""

from pathlib import Path

import pytest

from natural_pdf import PDF
from natural_pdf.elements.element_collection import ElementCollection


def test_map_basic_transformation():
    """Test basic map functionality."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get some text elements
    elements = page.find_all("text")[:5]

    # Map to extract text
    texts = elements.map(lambda e: e.extract_text())

    # Should return a list of strings
    assert isinstance(texts, list)
    assert len(texts) == len(elements)
    assert all(isinstance(t, (str, type(None))) for t in texts)


def test_map_with_skip_empty():
    """Test map with skip_empty parameter."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get all text elements
    elements = page.find_all("text")[:10]

    # Map to extract text, some might be empty
    all_texts = elements.map(lambda e: e.extract_text())
    filtered_texts = elements.map(lambda e: e.extract_text(), skip_empty=True)

    # Filtered should have no None or empty strings
    assert None not in filtered_texts
    assert "" not in filtered_texts
    assert all(t for t in filtered_texts)  # All values are truthy

    # Filtered should be <= original length
    assert len(filtered_texts) <= len(all_texts)


def test_map_element_transformation():
    """Test map when transforming elements (should return ElementCollection)."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get some text elements
    elements = page.find_all("text")[:5]

    # Map to expand elements
    expanded = elements.map(lambda e: e.expand(5))

    # Should return an ElementCollection since we're returning elements
    assert isinstance(expanded, ElementCollection)
    assert len(expanded) == len(elements)


def test_map_with_additional_args():
    """Test map with additional arguments passed to the function."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get some text elements
    elements = page.find_all("text")[:5]

    # Define a function that takes additional arguments
    def process_element(element, prefix="", suffix=""):
        text = element.extract_text() or ""
        return f"{prefix}{text}{suffix}"

    # Map with additional arguments
    results = elements.map(process_element, prefix="[", suffix="]")

    assert isinstance(results, list)
    assert all("[" in r and "]" in r for r in results if r)


def test_map_with_none_results():
    """Test map handles None results correctly."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get some elements
    elements = page.find_all("text")[:5]

    # Map function that sometimes returns None
    def maybe_extract(element):
        text = element.extract_text()
        # Return None for short text
        return text if text and len(text) > 10 else None

    # Without skip_empty
    with_nones = elements.map(maybe_extract)
    assert isinstance(with_nones, list)
    # May contain None values

    # With skip_empty
    without_nones = elements.map(maybe_extract, skip_empty=True)
    assert isinstance(without_nones, list)
    assert None not in without_nones
    assert all(t for t in without_nones)  # All values are truthy


def test_map_preserves_collection_type():
    """Test that map preserves collection type when returning elements."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get text elements
    elements = page.find_all("text")[:5]

    # Map that returns elements should preserve ElementCollection
    def transform_element(e):
        return e.expand(2) if e.extract_text() else None

    # Without skip_empty - should be ElementCollection even with Nones
    result = elements.map(transform_element)
    assert isinstance(result, ElementCollection)

    # With skip_empty - should still be ElementCollection
    result_filtered = elements.map(transform_element, skip_empty=True)
    assert isinstance(result_filtered, ElementCollection)
    # Should have no None values
    assert all(e is not None for e in result_filtered)


def test_map_empty_collection():
    """Test map on empty collection."""
    empty = ElementCollection([])

    # Map on empty collection
    result = empty.map(lambda e: e.extract_text())
    assert isinstance(result, list)
    assert len(result) == 0

    # With skip_empty
    result_filtered = empty.map(lambda e: e.extract_text(), skip_empty=True)
    assert isinstance(result_filtered, list)
    assert len(result_filtered) == 0
