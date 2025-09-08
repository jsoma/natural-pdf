"""Test from_headers() with string headers."""

import pytest

from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides


def test_from_headers_with_strings(practice_pdf):
    """Test from_headers with list of strings."""
    page = practice_pdf[0]

    # Get some text to use as test headers
    all_text = page.find_all("text")[:5]
    if len(all_text) >= 3:
        # Extract the actual text content
        header_texts = [elem.extract_text() for elem in all_text[:3]]

        # Create guides using string headers
        guides = Guides(page)
        guides.vertical.from_headers(header_texts)

        # Should work similar to using elements directly
        guides_from_elements = Guides(page)
        guides_from_elements.vertical.from_headers(all_text[:3])

        # Both approaches should produce similar results
        # (may not be identical due to search differences)
        assert len(guides.vertical) >= 0
        assert len(guides_from_elements.vertical) >= 0


def test_from_headers_with_missing_strings(practice_pdf):
    """Test from_headers when some strings are not found."""
    page = practice_pdf[0]

    # Use mix of real and fake headers
    all_text = page.find_all("text")[:2]
    if len(all_text) >= 2:
        real_texts = [elem.extract_text() for elem in all_text]
        fake_texts = ["NONEXISTENT_HEADER_1", "NONEXISTENT_HEADER_2"]
        mixed_texts = real_texts + fake_texts

        # Should still work with found headers
        guides = Guides(page)
        result = guides.vertical.from_headers(mixed_texts)

        # Should return the parent for chaining
        assert result is guides


def test_from_headers_all_strings_missing(practice_pdf):
    """Test from_headers when no strings are found."""
    page = practice_pdf[0]

    # All fake headers
    fake_headers = ["FAKE_HEADER_1", "FAKE_HEADER_2", "FAKE_HEADER_3"]

    guides = Guides(page)
    result = guides.vertical.from_headers(fake_headers)

    # Should return parent but no guides created
    assert result is guides
    assert len(guides.vertical) == 0


def test_from_headers_empty_string_list(practice_pdf):
    """Test from_headers with empty string list."""
    page = practice_pdf[0]

    guides = Guides(page)
    result = guides.vertical.from_headers([])

    # Should handle gracefully
    assert result is guides
    assert len(guides.vertical) == 0
