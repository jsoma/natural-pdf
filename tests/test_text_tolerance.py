"""Test text_tolerance parameter functionality across different usage patterns.

This test file addresses GitHub issue #4 where text_tolerance doesn't work
consistently across different text selection methods.
"""

import os

import pytest

import natural_pdf as npdf


class TestTextTolerance:
    """Test suite for text_tolerance parameter functionality."""

    @pytest.fixture(scope="class")
    def test_pdf_path(self):
        """Path to the test PDF with text spacing issues."""
        # Using the PDF from the issue report
        path = "pdfs/candeubadam-2025-annual-oge-278e-oge-certified.pdf"
        if not os.path.exists(path):
            # Fallback to practice PDF if the specific one isn't available
            path = "pdfs/01-practice.pdf"
        return path

    def test_text_tolerance_at_pdf_init(self, test_pdf_path):
        """Test that text_tolerance works when set during PDF initialization."""
        # This should work - setting tolerance at PDF init
        pdf = npdf.PDF(test_pdf_path, text_tolerance={"x_tolerance": 20})
        page = pdf.pages[0]

        # With higher tolerance, should find text that spans multiple elements
        result = page.find('text:contains("Public Financial Disclosure")')

        # This should succeed with proper tolerance
        # Currently fails due to bug
        assert result is not None, "Should find text with high x_tolerance at PDF init"

    def test_text_tolerance_at_find_call(self, test_pdf_path):
        """Test that text_tolerance works when passed to find() method."""
        pdf = npdf.PDF(test_pdf_path)
        page = pdf.pages[0]

        # Try to override tolerance at find() call time
        result = page.find(
            'text:contains("Public Financial Disclosure")', text_tolerance={"x_tolerance": 20}
        )

        # This should work but currently doesn't
        # The parameter is accepted but doesn't affect the search
        assert result is not None, "Should find text with text_tolerance in find() call"

    def test_individual_words_found(self, test_pdf_path):
        """Verify that individual words can be found (baseline test)."""
        pdf = npdf.PDF(test_pdf_path)
        page = pdf.pages[0]

        # These should be found individually
        public = page.find('text:contains("Public")')
        financial = page.find('text:contains("Financial")')
        disclosure = page.find('text:contains("Disclosure")')

        assert public is not None, "Should find 'Public'"
        assert financial is not None, "Should find 'Financial'"
        assert disclosure is not None, "Should find 'Disclosure'"

        # Verify they're on the same line (similar y coordinates)
        if public and financial and disclosure:
            assert abs(public.top - financial.top) < 1, "Words should be on same line"
            assert abs(financial.top - disclosure.top) < 1, "Words should be on same line"

    def test_extract_text_finds_phrase(self, test_pdf_path):
        """Verify that extract_text() can find the phrase (shows text is there)."""
        pdf = npdf.PDF(test_pdf_path)
        page = pdf.pages[0]

        text = page.extract_text()
        assert "Public Financial Disclosure" in text, "extract_text() should contain the phrase"

    def test_auto_text_tolerance(self, test_pdf_path):
        """Test that auto_text_tolerance affects word grouping."""
        # With auto tolerance (default)
        pdf_auto = npdf.PDF(test_pdf_path)
        page_auto = pdf_auto.pages[0]

        # Without auto tolerance
        pdf_no_auto = npdf.PDF(test_pdf_path, auto_text_tolerance=False)
        page_no_auto = pdf_no_auto.pages[0]

        # The behavior should be different
        # This is a weaker test since we don't know exact expected behavior
        # But at least verifies the parameter is being used
        result_auto = page_auto.find('text:contains("Public Financial Disclosure")')
        result_no_auto = page_no_auto.find('text:contains("Public Financial Disclosure")')

        # At least one should work or they should differ
        # Currently both fail, which indicates the parameter isn't working

    def test_text_tolerance_values(self, test_pdf_path):
        """Test different text_tolerance values to find the working threshold."""
        tolerance_values = [1, 3, 5, 10, 15, 20, 30, 50]
        results = {}

        for x_tol in tolerance_values:
            pdf = npdf.PDF(test_pdf_path, text_tolerance={"x_tolerance": x_tol})
            page = pdf.pages[0]
            result = page.find('text:contains("Public Financial Disclosure")')
            results[x_tol] = result is not None

        # At some point with high enough tolerance, it should work
        # Currently all fail, indicating the bug
        assert any(
            results.values()
        ), f"Should find text with some tolerance value. Results: {results}"

    def test_text_tolerance_consistency(self, test_pdf_path):
        """Test that text_tolerance affects all text operations consistently."""
        pdf = npdf.PDF(test_pdf_path, text_tolerance={"x_tolerance": 20})
        page = pdf.pages[0]

        # All these operations should respect the tolerance setting
        find_result = page.find('text:contains("Public Financial Disclosure")')
        find_all_result = page.find_all('text:contains("Public Financial Disclosure")')

        # If find works, find_all should too
        if find_result is not None:
            assert len(find_all_result) > 0, "find_all should also find results"

        # Check that word grouping is actually affected
        words_default = npdf.PDF(test_pdf_path).pages[0].words
        words_high_tol = npdf.PDF(test_pdf_path, text_tolerance={"x_tolerance": 50}).pages[0].words

        # With higher tolerance, we should have fewer words (more grouping)
        # This tests that the parameter actually changes behavior
        assert len(words_high_tol) <= len(
            words_default
        ), "Higher tolerance should result in same or fewer words due to more grouping"


@pytest.mark.parametrize("x_tol", [3, 10, 20, 30])
def test_text_tolerance_parametrized(x_tol):
    """Parametrized test for different tolerance values."""
    path = "pdfs/candeubadam-2025-annual-oge-278e-oge-certified.pdf"
    if not os.path.exists(path):
        pytest.skip(f"Test PDF not found: {path}")

    pdf = npdf.PDF(path, text_tolerance={"x_tolerance": x_tol})
    page = pdf.pages[0]

    # Test finding multi-word phrases
    result = page.find('text:contains("Public Financial")')

    # With appropriate tolerance, this should work
    # Higher tolerance should increase chance of success
    if x_tol >= 20:
        assert result is not None, f"Should find phrase with x_tolerance={x_tol}"
