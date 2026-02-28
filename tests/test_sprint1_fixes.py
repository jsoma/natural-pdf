"""Sprint 1 regression tests for code review fixes."""

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


class TestFix1EndpointProperty:
    """Fix 1: Duplicate endpoint property — second definition deleted."""

    def test_endpoint_returns_end_element(self, pdf):
        """endpoint should always alias end_element, not boundary_element."""
        page = pdf.pages[0]
        elements = page.find_all("text")
        if len(elements) < 2:
            pytest.skip("Need at least 2 text elements")

        first = elements[0]
        region = first.below(until="text")

        # endpoint should be identical to end_element
        assert region.endpoint is region.end_element

    def test_endpoint_not_none_for_region_until(self, pdf):
        """endpoint should not be None when target was found, even if target is region-type."""
        page = pdf.pages[0]
        elements = page.find_all("text")
        if len(elements) < 2:
            pytest.skip("Need at least 2 text elements")

        first = elements[0]
        region = first.below(until="text")

        # end_element is always set when target is found
        assert region.endpoint is not None

    def test_endpoint_none_when_no_until(self, pdf):
        """endpoint should be None when no until was used."""
        page = pdf.pages[0]
        elements = page.find_all("text")
        if not elements:
            pytest.skip("Need at least 1 text element")

        region = elements[0].below()
        assert region.endpoint is None


class TestFix2HasPolygon:
    """Fix 2: has_polygon should be False for normal text elements."""

    def test_text_element_has_polygon_false(self, pdf):
        """Normal text elements should not have polygon set."""
        page = pdf.pages[0]
        elem = page.find("text:bold")
        if not elem:
            pytest.skip("No bold text found")

        assert not elem.has_polygon

    def test_polygon_true_when_set(self, pdf):
        """has_polygon should be True when _polygon is explicitly set."""
        page = pdf.pages[0]
        elem = page.find("text")
        if not elem:
            pytest.skip("No text found")

        # Manually set a polygon
        elem._polygon = [(0, 0), (10, 0), (10, 10)]
        assert elem.has_polygon

        # Cleanup
        elem._polygon = None

    def test_polygon_false_for_empty_list(self, pdf):
        """has_polygon should be False for empty or too-small polygon."""
        page = pdf.pages[0]
        elem = page.find("text")
        if not elem:
            pytest.skip("No text found")

        elem._polygon = []
        assert not elem.has_polygon

        elem._polygon = [(0, 0), (1, 1)]  # Only 2 points
        assert not elem.has_polygon

        # Cleanup
        elem._polygon = None


class TestFix3ClipPreservesElements:
    """Fix 3: clip() per-element path should preserve non-clippable elements."""

    def test_clip_preserves_non_clippable(self, pdf):
        """Elements without clip method should be preserved in per-element path."""
        page = pdf.pages[0]
        elements = page.find_all("text")[:3]
        if len(elements) < 2:
            pytest.skip("Need at least 2 elements")

        # Create mock clip objects (same count as elements)
        clip_objs = [None] * len(elements)

        # Even if clip_objs are None, non-clippable elements should be preserved
        # This tests the else branch — elements without .clip() method
        result = elements.clip(obj=clip_objs)
        assert len(result) == len(elements)


class TestFix4PageIndexFallback:
    """Fix 4: _direction_multipage should use Page.index for O(1) lookup."""

    def test_page_has_index_attribute(self, pdf):
        """Pages should have a 0-based index attribute."""
        for i, page in enumerate(pdf.pages):
            assert hasattr(page, "index")
            assert page.index == i


class TestFix6GetDescendantsDeque:
    """Fix 6: get_descendants should use deque for O(1) popleft."""

    def test_get_descendants_works(self, pdf):
        """get_descendants should still return correct results after deque change."""
        page = pdf.pages[0]
        # Analyze layout to get regions with children
        try:
            regions = page.analyze_layout(engine="tatr")
        except Exception:
            pytest.skip("TATR engine not available")

        # If we have any regions, test get_descendants
        for region in regions:
            descendants = region.get_descendants()
            # Should be a list
            assert isinstance(descendants, list)
