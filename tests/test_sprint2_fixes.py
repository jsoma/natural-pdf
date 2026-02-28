"""Sprint 2 regression tests for code review fixes."""

import pytest

import natural_pdf as npdf
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region


@pytest.fixture(scope="module")
def pdf():
    """Load test PDF once for all tests."""
    p = npdf.PDF("pdfs/01-practice.pdf")
    yield p
    p.close()


class TestFix7WithinConstraintNone:
    """Fix 7: Non-overlapping within constraint should return None."""

    def test_non_overlapping_within_returns_none(self, pdf):
        """below() with a within region that doesn't overlap should return None."""
        page = pdf.pages[0]
        # Find an element that's not at the very top of the page
        elements = page.find_all("text")
        elem = None
        for e in elements:
            if e.top > 100:
                elem = e
                break
        if not elem:
            pytest.skip("No text element far enough from top")

        # Create a constraint region entirely above the element
        # below() starts from elem.bottom, but constraint ends before that → no overlap
        far_above = page.create_region(0, 0, page.width, elem.top - 50)

        result = elem.below(within=far_above)
        assert result is None

    def test_overlapping_within_returns_region(self, pdf):
        """below() with a properly overlapping within should return a Region."""
        page = pdf.pages[0]
        elem = page.find("text")
        if not elem:
            pytest.skip("No text found")

        # Create a large constraint region that overlaps
        big_region = page.create_region(0, 0, page.width, page.height)
        result = elem.below(within=big_region)
        assert result is not None


class TestFix9DoubleClose:
    """Fix 9: Double close should not error."""

    def test_double_close_no_error(self):
        """Calling close() twice should not raise."""
        p = npdf.PDF("pdfs/01-practice.pdf")
        p.close()
        p.close()  # Should not raise

    def test_context_manager_close(self):
        """Context manager exit should close cleanly."""
        with npdf.PDF("pdfs/01-practice.pdf") as p:
            _ = p.pages[0]
        # After __exit__, close has been called
        # Calling again should be safe
        p.close()


class TestFix10CrossPageMerge:
    """Fix 10: Cross-page merge should return FlowRegion."""

    def test_same_page_merge_returns_region(self, pdf):
        """Merge of same-page elements should return Region."""
        page = pdf.pages[0]
        elements = page.find_all("text")[:3]
        if len(elements) < 2:
            pytest.skip("Need at least 2 elements")

        result = elements.merge()
        assert isinstance(result, Region)

    def test_cross_page_merge_returns_flow_region(self):
        """Merge of elements from multiple pages should return FlowRegion."""
        from natural_pdf.flows.region import FlowRegion

        p = npdf.PDF("pdfs/Atlanta_Public_Schools_GA_sample.pdf")
        try:
            # Get elements from different pages
            elem1 = p.pages[0].find("text")
            elem2 = p.pages[1].find("text")
            if not elem1 or not elem2:
                pytest.skip("Need text on both pages")

            collection = ElementCollection([elem1, elem2])
            result = collection.merge()

            assert isinstance(result, FlowRegion)
            # Should have constituent regions from both pages
            assert len(result.constituent_regions) == 2
            assert result.constituent_regions[0].page == p.pages[0]
            assert result.constituent_regions[1].page == p.pages[1]
        finally:
            p.close()

    def test_cross_page_merge_supports_extract_text(self):
        """FlowRegion from cross-page merge should support extract_text."""
        p = npdf.PDF("pdfs/Atlanta_Public_Schools_GA_sample.pdf")
        try:
            elem1 = p.pages[0].find("text")
            elem2 = p.pages[1].find("text")
            if not elem1 or not elem2:
                pytest.skip("Need text on both pages")

            collection = ElementCollection([elem1, elem2])
            result = collection.merge()
            text = result.extract_text()
            assert isinstance(text, str)
        finally:
            p.close()

    def test_empty_merge_raises(self):
        """Empty collection merge should raise ValueError."""
        empty = ElementCollection([])
        with pytest.raises(ValueError, match="empty"):
            empty.merge()
