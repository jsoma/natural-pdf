#!/usr/bin/env python3
"""Test arithmetic expressions in selectors like [width>max()*0.9]."""

from pathlib import Path

import pytest

from natural_pdf import PDF


def test_selector_max_multiply():
    """Test selector with max() * multiplier."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get all rectangles
    all_rects = page.find_all("rect")
    if not all_rects:
        pytest.skip("No rectangles found in test PDF")

    # Get maximum width manually
    widths = [r.width for r in all_rects if hasattr(r, "width")]
    if not widths:
        pytest.skip("No rectangles with width found")
    max_width = max(widths)

    # Find rectangles with width > 90% of max
    large_rects = page.find_all("rect[width>max()*0.9]")

    # Verify results
    for rect in large_rects:
        assert rect.width > max_width * 0.9

    # Should have at least one (the max itself)
    assert len(large_rects) >= 1

    # All large rects should be in original set
    for rect in large_rects:
        assert rect in all_rects


def test_selector_min_plus():
    """Test selector with min() + addition."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get all text elements
    all_texts = page.find_all("text")
    if not all_texts:
        pytest.skip("No text elements found")

    # Get minimum size manually
    sizes = [t.size for t in all_texts if hasattr(t, "size")]
    if not sizes:
        pytest.skip("No text with size found")
    min_size = min(sizes)

    # Find text with size > min + 2
    larger_texts = page.find_all("text[size>min()+2]")

    # Verify results
    for text in larger_texts:
        assert text.size > min_size + 2

    # Should exclude the minimum sized texts
    min_sized_texts = [t for t in all_texts if t.size == min_size]
    for text in min_sized_texts:
        assert text not in larger_texts


def test_selector_avg_minus():
    """Test selector with avg() - subtraction."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get all text elements
    all_texts = page.find_all("text")
    if not all_texts:
        pytest.skip("No text elements found")

    # Calculate average size manually
    sizes = [t.size for t in all_texts if hasattr(t, "size")]
    if not sizes:
        pytest.skip("No text with size found")
    avg_size = sum(sizes) / len(sizes)

    # Find text with size < avg - 1
    small_texts = page.find_all("text[size<avg()-1]")

    # Verify results
    for text in small_texts:
        assert text.size < avg_size - 1


def test_selector_median_divide():
    """Test selector with median() / division."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get all text elements
    all_texts = page.find_all("text")
    if not all_texts:
        pytest.skip("No text elements found")

    # Calculate median size manually
    sizes = [t.size for t in all_texts if hasattr(t, "size")]
    if not sizes:
        pytest.skip("No text with size found")
    sorted_sizes = sorted(sizes)
    n = len(sorted_sizes)
    if n % 2 == 0:
        median_size = (sorted_sizes[n // 2 - 1] + sorted_sizes[n // 2]) / 2
    else:
        median_size = sorted_sizes[n // 2]

    # Find text with size < median / 2 (very small text)
    tiny_texts = page.find_all("text[size<median()/2]")

    # Verify results
    for text in tiny_texts:
        assert text.size < median_size / 2


def test_selector_max_multiply_exact():
    """Test selector with max() for exact match."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get all text elements
    all_texts = page.find_all("text")
    if not all_texts:
        pytest.skip("No text elements found")

    # Get maximum size
    sizes = [t.size for t in all_texts if hasattr(t, "size")]
    if not sizes:
        pytest.skip("No text with size found")
    max_size = max(sizes)

    # Find text with exact max size (max()*1 or just max())
    max_texts = page.find_all("text[size=max()]")

    # Verify all have max size
    for text in max_texts:
        assert text.size == max_size

    # Should have at least one
    assert len(max_texts) >= 1


def test_selector_combined_expressions():
    """Test combining multiple aggregate expressions."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get all text elements
    all_texts = page.find_all("text")
    if not all_texts:
        pytest.skip("No text elements found")

    # Calculate sizes
    sizes = [t.size for t in all_texts if hasattr(t, "size")]
    if not sizes:
        pytest.skip("No text with size found")

    min_size = min(sizes)
    max_size = max(sizes)

    # Find text in middle range: > min*1.5 and < max*0.8
    middle_texts = page.find_all("text[size>min()*1.5][size<max()*0.8]")

    # Verify results
    for text in middle_texts:
        assert text.size > min_size * 1.5
        assert text.size < max_size * 0.8


def test_selector_expression_with_comparison_operators():
    """Test expressions work with all comparison operators."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Get all text elements
    all_texts = page.find_all("text")
    if not all_texts:
        pytest.skip("No text elements found")

    sizes = [t.size for t in all_texts if hasattr(t, "size")]
    if not sizes:
        pytest.skip("No text with size found")
    avg_size = sum(sizes) / len(sizes)

    # Test >= operator
    texts_gte = page.find_all("text[size>=avg()*0.95]")
    for text in texts_gte:
        assert text.size >= avg_size * 0.95

    # Test <= operator
    texts_lte = page.find_all("text[size<=avg()*1.05]")
    for text in texts_lte:
        assert text.size <= avg_size * 1.05

    # Test != operator (though less common with aggregates)
    texts_ne = page.find_all("text[size!=max()]")
    max_size = max(sizes)
    for text in texts_ne:
        assert text.size != max_size


def test_selector_expression_no_elements():
    """Test aggregate expressions when no elements match base selector."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Try to find non-existent elements with aggregate
    # This should not raise an error, just return empty collection
    result = page.find_all("nonexistent[width>max()*0.9]")
    assert len(result) == 0

    # Try with a real element type but impossible condition
    result = page.find_all("text[size>max()*2]")  # Nothing can be > 2x the max
    assert len(result) == 0


def test_selector_expression_edge_cases():
    """Test edge cases in arithmetic expressions."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "01-practice.pdf"
    pdf = PDF(pdf_path)
    page = pdf[0]

    all_texts = page.find_all("text")
    if not all_texts:
        pytest.skip("No text elements found")

    # Test with 0 multiplier (should find nothing for > comparison)
    zero_mult = page.find_all("text[size>max()*0]")
    # All sizes should be > 0, so this should return all texts with size
    texts_with_size = [t for t in all_texts if hasattr(t, "size") and t.size is not None]
    assert len(zero_mult) == len(texts_with_size)

    # Test with very large multiplier
    large_mult = page.find_all("text[size<max()*1000]")
    # All texts should match this
    assert len(large_mult) == len(texts_with_size)

    # Test addition with 0
    add_zero = page.find_all("text[size=min()+0]")
    min_size = min(t.size for t in texts_with_size)
    for text in add_zero:
        assert text.size == min_size
