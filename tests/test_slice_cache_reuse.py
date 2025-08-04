"""Test that sliced PageCollections correctly reuse cached pages from parent PDF.

This tests the fix for the issue where exclusions weren't applied to pages
accessed through sliced PageCollections if the pages were cached before the
exclusions were added.
"""

import tempfile
from pathlib import Path

import pytest

import natural_pdf as npdf
from natural_pdf.core.pdf import _LazyPageList


def create_test_pdf_content():
    """Create minimal PDF content for testing."""
    # Minimal valid PDF structure
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>
endobj
4 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
72 720 Td
(Page 1 Test) Tj
ET
endstream
endobj
6 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
72 720 Td
(Page 2 Test) Tj
ET
endstream
endobj
xref
0 7
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000264 00000 n
0000000413 00000 n
0000000507 00000 n
trailer
<< /Size 7 /Root 1 0 R >>
startxref
601
%%EOF"""


def test_slice_reuses_cached_pages_with_exclusions():
    """Test that slices reuse cached pages from parent PDF, preserving exclusions."""
    # Create a test PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(create_test_pdf_content())
        pdf_path = f.name

    try:
        # Load PDF
        pdf = npdf.PDF(pdf_path)

        # Step 1: Create a slice and ACCESS pages BEFORE adding exclusions
        early_slice = pdf.pages[:2]
        # IMPORTANT: Access the pages NOW, before adding exclusions
        early_page0 = early_slice[0]
        early_page1 = early_slice[1]

        # Verify pages are cached and have no exclusions
        assert early_page0 is not None
        assert early_page1 is not None
        initial_exclusions = len(early_page0._exclusions)
        assert (
            initial_exclusions == 0
        ), f"Expected no exclusions initially, got {initial_exclusions}"

        # Step 2: NOW add exclusion to PDF (after pages are already cached)
        def test_exclusion(page):
            # Simple exclusion that returns a region
            return page.region(0, 0, 100, 100)

        pdf.add_exclusion(test_exclusion, label="test_exclusion")

        # Step 3: Access pages from the early slice again
        # These should be the same cached pages (without the new exclusion)
        slice_page0 = early_slice[0]
        slice_page1 = early_slice[1]

        # They should be the same objects (reused from cache)
        assert slice_page0 is early_page0
        assert slice_page1 is early_page1

        # Step 4: Direct access should return the same cached pages
        direct_page0 = pdf.pages[0]
        direct_page1 = pdf.pages[1]

        # Should return the same cached pages
        assert direct_page0 is early_page0
        assert direct_page1 is early_page1

        # The page's _exclusions list doesn't change, but _get_exclusion_regions
        # dynamically includes PDF-level exclusions
        assert len(direct_page0._exclusions) == initial_exclusions
        assert len(direct_page1._exclusions) == initial_exclusions

        # But when we get the actual exclusion regions, it includes PDF-level exclusions
        regions0 = direct_page0._get_exclusion_regions(include_callable=True)
        regions1 = direct_page1._get_exclusion_regions(include_callable=True)
        assert len(regions0) > initial_exclusions
        assert len(regions1) > initial_exclusions

        # Step 5: Create a new slice - should reuse cached pages
        new_slice = pdf.pages[:2]
        new_page0 = new_slice[0]
        new_page1 = new_slice[1]

        # Should be the same cached objects
        assert new_page0 is direct_page0
        assert new_page1 is direct_page1

        # And they also have access to PDF-level exclusions dynamically
        new_regions0 = new_page0._get_exclusion_regions(include_callable=True)
        new_regions1 = new_page1._get_exclusion_regions(include_callable=True)
        assert len(new_regions0) > initial_exclusions
        assert len(new_regions1) > initial_exclusions

    finally:
        # Clean up
        Path(pdf_path).unlink(missing_ok=True)


def test_lazy_page_list_checks_parent_cache():
    """Test that _LazyPageList._create_page checks parent cache first."""
    # Create a test PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(create_test_pdf_content())
        pdf_path = f.name

    try:
        # Load PDF
        pdf = npdf.PDF(pdf_path)

        # Access a page to cache it
        main_page = pdf.pages[0]
        assert main_page is not None

        # Add a marker to the cached page
        main_page._test_marker = "cached_from_main"

        # Create a slice - it should reuse the cached page
        slice_pages = pdf.pages[:1]
        slice_page = slice_pages[0]

        # Verify it's the same object with our marker
        assert slice_page is main_page
        assert hasattr(slice_page, "_test_marker")
        assert slice_page._test_marker == "cached_from_main"

    finally:
        # Clean up
        Path(pdf_path).unlink(missing_ok=True)


def test_exclusions_applied_to_new_pages_in_slice():
    """Test that exclusions are applied to pages created after exclusion is added."""
    # Create a test PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(create_test_pdf_content())
        pdf_path = f.name

    try:
        # Load PDF
        pdf = npdf.PDF(pdf_path)

        # Add exclusion BEFORE accessing any pages
        def test_exclusion(page):
            return page.region(0, 0, 100, 100)

        pdf.add_exclusion(test_exclusion, label="early_exclusion")

        # Create a slice and access pages
        slice_pages = pdf.pages[:2]
        page0 = slice_pages[0]
        page1 = slice_pages[1]

        # Pages should have the exclusion
        assert len(page0._exclusions) > 0
        assert len(page1._exclusions) > 0

        # Verify the exclusion label
        assert any(exc[1] == "early_exclusion" for exc in page0._exclusions)
        assert any(exc[1] == "early_exclusion" for exc in page1._exclusions)

    finally:
        # Clean up
        Path(pdf_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
