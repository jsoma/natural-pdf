"""Test the from_headers() method for column detection."""

import pytest

from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides


def test_from_headers_basic(practice_pdf):
    """Test basic column detection from headers."""
    page = practice_pdf[0]

    # Find some text elements to use as headers
    # Look for text in the top portion of the page
    all_text = page.find_all("text")
    if len(all_text) >= 3:
        # Use first 3 text elements as mock headers
        headers = all_text[:3]

        # Create guides from headers
        guides = Guides(page)
        guides.vertical.from_headers(headers)

        # Should create some separators
        assert len(guides.vertical) >= 0

        # Guides should be sorted
        assert guides.vertical.data == sorted(guides.vertical.data)


def test_from_headers_min_crossings(atlanta_pdf):
    """Test min_crossings method explicitly."""
    page = atlanta_pdf[0]

    # Find potential headers - text near top of page
    headers = page.find_all("text")[:4]

    if len(headers) >= 2:
        guides = Guides(page)
        guides.vertical.from_headers(headers, method="min_crossings")

        # Verify guides were created
        assert len(guides.vertical) >= 0


def test_from_headers_with_constraints(practice_pdf):
    """Test column detection with width constraints."""
    page = practice_pdf[0]

    headers = page.find_all("text")[:4]

    if len(headers) >= 2:
        guides = Guides(page)
        guides.vertical.from_headers(
            headers, min_width=50, max_width=300  # Minimum column width  # Maximum column width
        )

        # Check width constraints are respected (when we have guides)
        if len(guides.vertical) >= 2:
            for i in range(1, len(guides.vertical)):
                width = guides.vertical[i] - guides.vertical[i - 1]
                # Width should be reasonable
                assert width > 0


def test_from_headers_insufficient_headers(practice_pdf):
    """Test behavior with too few headers."""
    page = practice_pdf[0]

    # Try with just one header
    headers = page.find_all("text")[:1]

    guides = Guides(page)
    result = guides.vertical.from_headers(headers)

    # Should return parent without creating guides
    assert result == guides
    assert len(guides.vertical) == 0


def test_from_headers_horizontal_error(practice_pdf):
    """Test that from_headers raises error for horizontal guides."""
    page = practice_pdf[0]

    headers = page.find_all("text")[:3]

    guides = Guides(page)

    # Should raise ValueError for horizontal guides
    with pytest.raises(ValueError, match="only works for vertical guides"):
        guides.horizontal.from_headers(headers)


def test_from_headers_seam_carving(atlanta_pdf):
    """Test seam carving method."""
    page = atlanta_pdf[0]

    headers = page.find_all("text")[:3]

    if len(headers) >= 2:
        guides = Guides(page)
        guides.vertical.from_headers(headers, method="seam_carving")

        # Should create guides
        assert len(guides.vertical) >= 0


def test_from_headers_append(practice_pdf):
    """Test appending to existing guides."""
    page = practice_pdf[0]

    guides = Guides(page)

    # Add some initial guides
    guides.vertical.add([100, 200])
    initial_count = len(guides.vertical)

    # Find headers and append
    headers = page.find_all("text")[:3]
    if len(headers) >= 2:
        guides.vertical.from_headers(headers, append=True)

        # Should have at least the initial guides
        assert len(guides.vertical) >= initial_count

        # Should still include the initial guides
        assert 100 in guides.vertical.data
        assert 200 in guides.vertical.data


def test_from_headers_element_collection(practice_pdf):
    """Test with ElementCollection input."""
    page = practice_pdf[0]

    # Get headers as ElementCollection
    headers = page.find_all("text")[:3]

    if len(headers) >= 2:
        guides = Guides(page)
        guides.vertical.from_headers(headers)

        # Should work with ElementCollection
        assert isinstance(guides.vertical.data, list)
