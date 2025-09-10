#!/usr/bin/env python3
"""Test the new .unique() method for removing duplicates."""

from pathlib import Path

import pytest

from natural_pdf import PDF
from natural_pdf.elements.element_collection import ElementCollection


def test_unique_basic():
    """Test basic unique functionality with duplicate elements."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get some text elements
    elements = page.find_all("text")[:5]

    # Create a collection with duplicates
    duplicated = ElementCollection(list(elements) + list(elements[:2]))
    assert len(duplicated) == 7  # 5 + 2 duplicates

    # Remove duplicates
    unique = duplicated.unique()
    assert len(unique) == 5  # Should have only unique elements
    assert isinstance(unique, ElementCollection)


def test_unique_with_key_function():
    """Test unique with a key function for custom comparison."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get text elements
    elements = page.find_all("text")[:10]

    # Get unique elements based on their text content
    unique_by_text = elements.unique(key=lambda e: e.extract_text())

    # Should have removed any elements with duplicate text
    texts = [e.extract_text() for e in unique_by_text]
    assert len(texts) == len(set(texts))  # All texts should be unique


def test_unique_preserves_order():
    """Test that unique preserves the order of first occurrence."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get some elements
    elements = page.find_all("text")[:3]

    # Create specific order with duplicates
    # [0, 1, 2, 1, 0] should become [0, 1, 2]
    ordered = ElementCollection([elements[0], elements[1], elements[2], elements[1], elements[0]])

    unique = ordered.unique()
    assert len(unique) == 3
    # Check order is preserved (first occurrence)
    assert unique[0] == elements[0]
    assert unique[1] == elements[1]
    assert unique[2] == elements[2]


def test_unique_with_none_values():
    """Test unique handles None values in key function."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get elements
    elements = page.find_all("text")[:5]

    # Key function that returns None for some elements
    def key_with_none(e):
        text = e.extract_text()
        return text if text and len(text) > 10 else None

    # Should handle None keys properly
    unique = elements.unique(key=key_with_none)
    assert isinstance(unique, ElementCollection)
    # Multiple elements with None key should be deduplicated to one


def test_unique_empty_collection():
    """Test unique on empty collection."""
    empty = ElementCollection([])

    unique = empty.unique()
    assert isinstance(unique, ElementCollection)
    assert len(unique) == 0


def test_unique_with_position_key():
    """Test unique using position as key."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get elements
    elements = page.find_all("text")[:10]

    # Some elements might have the same y-coordinate (same line)
    # Deduplicate by y-position (keep one element per line)
    unique_by_line = elements.unique(key=lambda e: round(e.bbox[1]))

    # Check that all remaining elements have unique y-positions
    y_positions = [round(e.bbox[1]) for e in unique_by_line]
    assert len(y_positions) == len(set(y_positions))


def test_unique_with_unhashable_keys():
    """Test unique with unhashable key results (like lists)."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get elements
    elements = page.find_all("text")[:5]

    # Key function that returns a list (unhashable)
    def unhashable_key(e):
        return [e.bbox[0], e.bbox[1]]  # Lists are not hashable

    # Should still work by converting to string representation
    unique = elements.unique(key=unhashable_key)
    assert isinstance(unique, ElementCollection)
    # Should have handled unhashable types gracefully
