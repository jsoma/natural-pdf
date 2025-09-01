"""Test the text:closest() selector for fuzzy text matching."""

import pytest

from natural_pdf import PDF


def test_text_closest_basic():
    """Test basic fuzzy text matching with real PDF content."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Test finding text that might be OCR'd poorly
    # Look for "Durham" which appears in the PDF
    results = page.find_all('text:closest("Durham")')
    # With threshold 0.0, it should match all text elements
    all_text = page.find_all("text")
    assert len(results) == len(all_text)

    # Test with a reasonable threshold to get meaningful matches
    results = page.find_all('text:closest("Durham@0.4")')
    assert len(results) >= 1  # Should find at least "Durham's Meatpacking"
    texts = [r.text for r in results]
    assert any("Durham" in text for text in texts)

    # Test fuzzy matching for "Chicago" with OCR errors
    results = page.find_all('text:closest("Chicgo@0.6")')  # Missing 'a'
    texts = [r.text for r in results]
    # Should still find "Chicago"
    assert any("Chicago" in text for text in texts)

    # Test very low threshold - should match everything
    results_low = page.find_all('text:closest("xyz@0.0")')
    assert len(results_low) == len(all_text)


def test_text_closest_with_threshold():
    """Test fuzzy matching with different similarity thresholds."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Test with "Violation" and varying thresholds
    search_term = "Violation"

    # High threshold - only very close matches
    results_high = page.find_all(f'text:closest("{search_term}@0.9")')

    # Medium threshold - more matches
    results_medium = page.find_all(f'text:closest("{search_term}@0.6")')

    # Low threshold - many matches
    results_low = page.find_all(f'text:closest("{search_term}@0.3")')

    # Results should increase as threshold decreases
    assert len(results_high) <= len(results_medium) <= len(results_low)

    # High threshold should find exact matches
    assert len(results_high) >= 1
    texts = [r.text for r in results_high]
    assert any("Violation" in text for text in texts)


def test_text_closest_case_sensitivity():
    """Test case-sensitive vs case-insensitive fuzzy matching."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Look for "chicago" - the PDF has "Chicago"
    results_insensitive = page.find_all('text:closest("chicago@0.6")')

    # Case sensitive search
    results_sensitive = page.find_all('text:closest("chicago@0.8")', case=True)

    # Case insensitive should find "Chicago, Ill.", case sensitive should not
    assert len(results_insensitive) >= 1
    assert len(results_sensitive) == 0

    # Verify case insensitive found "Chicago"
    texts = [r.text for r in results_insensitive]
    assert any("Chicago" in text for text in texts)


def test_text_closest_with_other_selectors():
    """Test combining :closest with other selectors."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Find fuzzy match with size constraint
    # Look for text similar to "Date" with size > 10
    results = page.find_all('text:closest("Date@0.7")[size>10]')
    if results:
        assert all(r.size > 10 for r in results)

    # Combine with position constraint
    results = page.find_all('text:closest("Violation@0.8")[top>100]')
    if results:
        assert all(r.top > 100 for r in results)


def test_text_closest_empty_and_whitespace():
    """Test fuzzy matching with empty strings and whitespace."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Empty string should not match anything
    results = page.find_all('text:closest("")')
    assert len(results) == 0

    # Whitespace should be trimmed
    results1 = page.find_all('text:closest("  Durham  @0.8")')
    results2 = page.find_all('text:closest("Durham@0.8")')
    assert len(results1) == len(results2)


def test_text_closest_special_characters():
    """Test fuzzy matching with special characters including @."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Test searching for text with punctuation
    results = page.find_all('text:closest("Date:@0.8")')
    texts = [r.text for r in results]
    # Should find "Date:" with colon
    assert any("Date:" in text for text in texts)

    # Test that @ separator is only the last one
    # This should search for "test@test" with threshold 0.8
    results = page.find_all('text:closest("test@test@0.8")')
    # Should return an ElementCollection (not crash on parsing)
    assert hasattr(results, "__iter__")  # Should be iterable


def test_text_closest_ocr_simulation():
    """Test fuzzy matching for common OCR errors."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Common OCR errors:
    # - l/I confusion: "Chicago, Ill." might be read as "Chicago, lll."
    # - rn -> m: "Durham" might be read as "Durharn"
    # - Missing punctuation: "Date:" -> "Date"

    # Should find "Durham" even with OCR error (rn -> m)
    results = page.find_all('text:closest("Durharn@0.4")')  # rn instead of m
    texts = [r.text for r in results]
    assert any("Durham" in text for text in texts)

    # Should handle missing punctuation
    results = page.find_all('text:closest("Date@0.8")')  # Missing colon
    texts = [r.text for r in results]
    assert any("Date:" in text for text in texts)

    # Should handle partial matches
    results = page.find_all('text:closest("Viol@0.7")')  # Partial word
    texts = [r.text for r in results]
    assert any("Violation" in text for text in texts)


def test_text_closest_practical_use_cases():
    """Test practical use cases for OCR'd documents."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Use case 1: Finding labels that might be OCR'd poorly
    # Looking for "Summary:" which might be read as "Surnmary:" or "Summary;"
    results = page.find_all('text:closest("Surnmary:@0.8")')  # rn instead of mm
    texts = [r.text for r in results]
    assert any("Summary:" in text for text in texts)

    # Use case 2: Finding dates with format variations
    # "February 3, 1905" might be OCR'd with various errors
    results = page.find_all('text:closest("February 3 1905@0.7")')  # Missing comma
    texts = [r.text for r in results]
    assert any("1905" in text for text in texts)

    # Use case 3: Finding section headers
    results = page.find_all('text:closest("Violatons@0.8")')  # Common OCR error: i->a
    texts = [r.text for r in results]
    assert any("Violations" in text for text in texts)
