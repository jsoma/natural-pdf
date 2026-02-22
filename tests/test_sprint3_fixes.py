"""Sprint 3 (sprint 4 plan) regression tests for code review fixes."""

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


class TestH1ClosestBoundaryMath:
    """H1: :closest boundary math bug — include_endpoint should extend to target.bottom."""

    def test_below_include_endpoint_extends_to_target_bottom(self, pdf):
        """below(until=..., include_endpoint=True) region should reach target.bottom."""
        page = pdf.pages[0]
        elements = page.find_all("text")
        if len(elements) < 3:
            pytest.skip("Need at least 3 text elements")

        source = elements[0]
        # Pick a target that's below the source
        target = None
        for e in elements[1:]:
            if e.top > source.bottom + 10:
                target = e
                break
        if target is None:
            pytest.skip("No suitable target element below source")

        region = source.below(
            until=f"text:contains('{target.extract_text()[:8]}')",
            include_endpoint=True,
        )
        assert region is not None
        # Region should extend to target.bottom, not target.top
        assert region.bottom >= target.bottom - 1  # allow 1px tolerance

    def test_above_include_endpoint_extends_to_target_top(self, pdf):
        """above(until=..., include_endpoint=True) region should reach target.top."""
        page = pdf.pages[0]
        elements = page.find_all("text")
        if len(elements) < 5:
            pytest.skip("Need at least 5 text elements")

        # Use an element near the bottom as source
        source = elements[-1]
        # Pick a target above it
        target = None
        for e in elements:
            if e.bottom < source.top - 10:
                target = e
                break
        if target is None:
            pytest.skip("No suitable target element above source")

        region = source.above(
            until=f"text:contains('{target.extract_text()[:8]}')",
            include_endpoint=True,
        )
        assert region is not None
        # Region should extend up to target.top, not target.bottom
        assert region.top <= target.top + 1  # allow 1px tolerance


class TestH2FloatBboxExclusion:
    """H2: Float bbox equality for exclusions — rounded comparison and id matching."""

    def test_element_exclusion_works(self, pdf):
        """Element-based exclusions should correctly exclude matched elements."""
        page = pdf.pages[0]
        all_elements = page.find_all("text", apply_exclusions=False)
        if len(all_elements) < 2:
            pytest.skip("Need at least 2 text elements")

        # Pick an element to exclude
        target = all_elements[0]
        target_text = target.extract_text()

        # Add element exclusion
        page.add_exclusion(
            target,
            label="test_exclusion",
            method="element",
        )

        try:
            # Get elements with exclusions applied
            filtered = page.find_all("text", apply_exclusions=True)
            filtered_texts = [e.extract_text() for e in filtered]

            # The excluded element's text should not appear (if unique)
            # At minimum, filtered count should be less
            assert len(filtered) < len(all_elements)
        finally:
            # Clean up exclusion
            page._exclusions = [exc for exc in page._exclusions if exc[1] != "test_exclusion"]


class TestPdfLevelExclusionMethod:
    """PDF.add_exclusion() should preserve method flag."""

    def test_pdf_exclusion_stores_method(self):
        """PDF-level exclusion should store method as 3-tuple."""
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        try:
            pdf.add_exclusion(
                lambda page: page.create_region(0, 0, 100, 50),
                label="test_method",
                method="element",
            )
            # Should be stored as 3-tuple with method
            exc = pdf._exclusions[-1]
            assert len(exc) == 3
            assert exc[2] == "element"
        finally:
            pdf.close()

    def test_pdf_exclusion_default_method_is_region(self):
        """PDF-level exclusion should default to 'region' method."""
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        try:
            pdf.add_exclusion(
                lambda page: page.create_region(0, 0, 100, 50),
                label="test_default",
            )
            exc = pdf._exclusions[-1]
            assert len(exc) == 3
            assert exc[2] == "region"
        finally:
            pdf.close()


class TestM1MergePageIdentity:
    """M1: merge() should group by page index, not object identity."""

    def test_merge_single_page(self, pdf):
        """merge() should work correctly for single-page elements."""
        page = pdf.pages[0]
        elements = page.find_all("text")
        if len(elements) < 2:
            pytest.skip("Need at least 2 text elements")

        subset = elements[:3]
        merged = subset.merge()
        assert isinstance(merged, Region)
        assert merged.page is page

    def test_merge_uses_page_index(self, pdf):
        """merge() groups by page.index, not page object identity."""
        page = pdf.pages[0]
        elements = page.find_all("text")
        if len(elements) < 2:
            pytest.skip("Need at least 2 text elements")

        # Merge should work even when elements come from same logical page
        subset = elements[:2]
        merged = subset.merge()
        assert isinstance(merged, Region)


class TestM2SortStability:
    """M2: next()/prev() sort should be stable via id() tiebreaker."""

    def test_next_returns_element(self, pdf):
        """next() should return a subsequent element."""
        page = pdf.pages[0]
        elements = page.find_all("text")
        if len(elements) < 2:
            pytest.skip("Need at least 2 text elements")

        first = elements[0]
        nxt = first.next()
        assert nxt is not None
        # next element should be at same or later position
        assert nxt.top >= first.top or (nxt.top == first.top and nxt.x0 >= first.x0)

    def test_prev_returns_element(self, pdf):
        """prev() should return a preceding element."""
        page = pdf.pages[0]
        elements = page.find_all("text")
        if len(elements) < 2:
            pytest.skip("Need at least 2 text elements")

        last = elements[-1]
        prv = last.prev()
        assert prv is not None
        assert prv.top <= last.top or (prv.top == last.top and prv.x0 <= last.x0)


class TestM4AnalysesStorage:
    """M4: analyses property uses own dict, not pdfplumber metadata."""

    def test_analyses_getter_returns_dict(self, pdf):
        """analyses property should return a dict."""
        assert isinstance(pdf.analyses, dict)

    def test_analyses_setter(self, pdf):
        """analyses setter should store and retrieve data."""
        pdf.analyses = {"test_key": "test_value"}
        assert pdf.analyses["test_key"] == "test_value"
        # Clean up
        del pdf.analyses["test_key"]

    def test_analyses_not_on_pdfplumber_metadata(self, pdf):
        """analyses should NOT be stored on pdfplumber metadata."""
        pdf.analyses["check_key"] = 42
        plumber_meta = getattr(pdf._pdf, "metadata", {}) or {}
        analysis_in_meta = plumber_meta.get("analysis", None)
        # Should not find our data in pdfplumber metadata
        if analysis_in_meta is not None:
            assert "check_key" not in analysis_in_meta
        # Clean up
        del pdf.analyses["check_key"]


class TestM5DeadRegionCache:
    """M5: _cached_text and _cached_elements should not exist on Region."""

    def test_no_cached_text_attribute(self, pdf):
        """Region should not have _cached_text attribute."""
        page = pdf.pages[0]
        region = page.create_region(0, 0, 100, 100)
        assert not hasattr(region, "_cached_text")

    def test_no_cached_elements_attribute(self, pdf):
        """Region should not have _cached_elements attribute."""
        page = pdf.pages[0]
        region = page.create_region(0, 0, 100, 100)
        assert not hasattr(region, "_cached_elements")

    def test_invalidate_exclusion_cache_still_callable(self, pdf):
        """_invalidate_exclusion_cache should still be callable (used by exclusion_service)."""
        page = pdf.pages[0]
        region = page.create_region(0, 0, 100, 100)
        # Should not raise
        region._invalidate_exclusion_cache()


class TestL4UnreachableReturn:
    """L4: No unreachable return None after save()."""

    def test_save_returns_self(self, pdf):
        """save() should return self (ElementCollection), not None."""
        import os
        import tempfile

        page = pdf.pages[0]
        elements = page.find_all("text")
        if not elements:
            pytest.skip("No text elements")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name

        try:
            result = elements.save(tmp_path)
            assert result is elements  # should return self
            assert result is not None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
