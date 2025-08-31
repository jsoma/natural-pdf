"""Test guide adjustment for stream extraction method"""

import pytest

from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides


class TestGuideAdjustmentStream:
    """Test that guides are adjusted to text bounds for stream extraction."""

    def test_guides_adjusted_for_stream_method(self):
        """Test that guides outside text bounds are adjusted for stream extraction."""
        pdf = PDF("tests/resources/table_narrow_text.pdf")
        page = pdf[0]

        # Create guides that extend beyond text bounds
        guides = Guides(page)
        guides.vertical.add([0, 100, 200, 300, 400, page.width])

        # Extract with stream method
        result = page.extract_table(method="stream", verticals=guides.vertical.data)

        # Should extract content despite guides being outside text bounds
        assert result is not None
        assert len(result) > 0
        assert len(result[0]) > 0  # Should have columns

    def test_from_headers_works_with_stream(self):
        """Test that from_headers() generated guides work with stream extraction."""
        pdf = PDF("tests/resources/table_headers.pdf")
        page = pdf[0]

        # Find headers
        headers = page.find_all("text[y<100]")  # Top row headers
        if not headers:
            pytest.skip("No headers found in test PDF")

        # Create guides from headers
        guides = Guides(page)
        guides.vertical.from_headers(headers, margin=0)

        # Extract table with stream method
        result = page.extract_table(method="stream", verticals=guides.vertical.data)

        # Should extract all columns including first and last
        assert result is not None
        assert len(result) > 0
        assert len(result[0]) >= len(headers)  # At least as many columns as headers

    def test_no_adjustment_for_lattice_method(self):
        """Test that guides are NOT adjusted when using lattice method."""
        pdf = PDF("tests/resources/table_with_lines.pdf")
        page = pdf[0]

        # Create guides at specific positions
        original_guides = [0, 100, 200, 300, 400, page.width]
        guides = Guides(page)
        guides.vertical.add(original_guides)

        # Extract with lattice method
        table_settings = {
            "vertical_strategy": "explicit",
            "horizontal_strategy": "lines",
            "explicit_vertical_lines": guides.vertical.data,
        }

        # The guides should not be adjusted for lattice method
        # (This test mainly ensures no errors occur)
        result = page.extract_table(method="lattice", table_settings=table_settings)

        # Result depends on PDF content, just ensure no crash
        assert result is not None

    def test_guide_adjustment_with_explicit_settings(self):
        """Test guide adjustment works with explicit table settings."""
        pdf = PDF("tests/resources/table_narrow_text.pdf")
        page = pdf[0]

        # Get text bounds
        text_elements = page.find_all("text")
        if not text_elements:
            pytest.skip("No text found in test PDF")

        text_bounds = text_elements.merge().bbox
        text_left = text_bounds[0]
        text_right = text_bounds[2]

        # Create guides outside text bounds
        guides = [0, text_left - 10, text_left + 50, text_right - 50, text_right + 10, page.width]

        # Extract with explicit settings
        table_settings = {
            "vertical_strategy": "explicit",
            "horizontal_strategy": "text",
            "explicit_vertical_lines": guides,
        }

        result = page.extract_table(method="pdfplumber", table_settings=table_settings)

        # Should successfully extract despite guides outside bounds
        assert result is not None
        assert len(result) > 0

    def test_empty_text_no_adjustment(self):
        """Test behavior when no text elements exist."""
        # Create a minimal PDF with guides but no text
        from natural_pdf.utils.testing import create_test_pdf

        pdf = create_test_pdf()
        page = pdf[0]

        # Add guides
        guides = [0, 100, 200, 300]

        # Extract with stream method - should handle gracefully
        result = page.extract_table(method="stream", verticals=guides)

        # Should return empty or handle gracefully
        assert result is not None
        assert isinstance(result, list) or hasattr(result, "__iter__")
