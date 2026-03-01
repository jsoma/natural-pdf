"""Tests for space injection in word extraction.

PDFs without embedded space characters (like OGE Form 278e) produce word
elements like "Form" and "278e" as separate elements.  Space injection
detects inter-character gaps that represent word boundaries within a single
merged word element and inserts spaces so that selectors like
`page.find('text:contains("Form 278")')` work correctly.
"""

import pytest

from natural_pdf import PDF

OGE_PDF = "pdfs/starks-geoffrey-278e.pdf"
PRACTICE_PDF = "pdfs/01-practice.pdf"


# ---------------------------------------------------------------------------
# OGE Form 278e — PDF without explicit space characters
# ---------------------------------------------------------------------------


class TestOGEFormSpaceInjection:
    """Verify space injection works on a real PDF that lacks space chars."""

    def test_find_form_278_succeeds(self):
        """The selector should find 'Form 278' after space injection."""
        pdf = PDF(OGE_PDF)
        try:
            page = pdf.pages[0]
            result = page.find('text:contains("Form 278")')
            assert result is not None, (
                "Expected to find a word element containing 'Form 278' " "but find() returned None"
            )
        finally:
            pdf.close()

    def test_title_text_correct(self):
        """The title should be a single element with correct text."""
        pdf = PDF(OGE_PDF)
        try:
            page = pdf.pages[0]
            result = page.find('text:contains("Public Financial Disclosure Report")')
            assert result is not None, "Should find a single element for the title"
            assert "Public Financial Disclosure Report (OGE Form 278e)" in result.text
        finally:
            pdf.close()

    def test_disabled_with_zero_ratio(self):
        """Setting space_gap_ratio=0 should disable injection for small text."""
        pdf = PDF(OGE_PDF, text_tolerance={"space_gap_ratio": 0})
        try:
            page = pdf.pages[0]
            # The 9pt header line merges via x_tolerance but relies on
            # space injection to insert spaces. With injection disabled,
            # the merged text should lack spaces (e.g. "OGEForm278e...").
            result = page.find('text:contains("OGEForm278e")')
            assert result is not None, (
                "With space_gap_ratio=0, the small-text OGE header should "
                "remain as 'OGEForm278e' without injected spaces"
            )
        finally:
            pdf.close()

    def test_configurable_via_text_tolerance(self):
        """space_gap_ratio should be passable via text_tolerance dict."""
        pdf = PDF(OGE_PDF, text_tolerance={"space_gap_ratio": 0.2})
        try:
            page = pdf.pages[0]
            # With a higher ratio, fewer spaces are injected but the core
            # ones (like between "Form" and "278e") should still appear
            result = page.find('text:contains("Form 278")')
            assert result is not None, "Expected 'Form 278' to be findable with space_gap_ratio=0.2"
        finally:
            pdf.close()


# ---------------------------------------------------------------------------
# Practice PDF — already has embedded spaces
# ---------------------------------------------------------------------------


class TestPracticePDFNoRegression:
    """Ensure space injection doesn't break PDFs with normal spacing."""

    def test_practice_pdf_text_intact(self):
        """Text extraction should still work correctly on normal PDFs."""
        pdf = PDF(PRACTICE_PDF)
        try:
            page = pdf.pages[0]
            text = page.extract_text()
            assert text is not None and len(text) > 0, "Should extract text"
        finally:
            pdf.close()

    def test_practice_pdf_no_double_spaces(self):
        """Space injection should not add extra spaces to normal PDFs."""
        pdf = PDF(PRACTICE_PDF)
        try:
            page = pdf.pages[0]
            words = page.find_all("text")
            for w in words:
                t = w.text or ""
                assert "  " not in t, f"Double space found in word element: '{t}'"
        finally:
            pdf.close()
