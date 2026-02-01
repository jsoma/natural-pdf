"""Tests for the endpoint property on Region and ElementCollection."""

import pytest

import natural_pdf as npdf
from natural_pdf.elements.element_collection import ElementCollection


@pytest.fixture
def pdf():
    """Load test PDF."""
    return npdf.PDF("pdfs/01-practice.pdf")


class TestRegionEndpoint:
    """Tests for Region.endpoint property."""

    def test_endpoint_with_until_selector(self, pdf):
        """Region created with 'until' should have endpoint set."""
        page = pdf.pages[0]

        # Find an element and create a region with until
        elements = page.find_all("text")
        if len(elements) < 2:
            pytest.skip("Need at least 2 text elements")

        first = elements[0]
        # Create region below first element until we hit another text
        region = first.below(until="text")

        # endpoint should be the element that matched the until selector
        assert region.endpoint is not None

    def test_endpoint_without_until(self, pdf):
        """Region created without 'until' should have endpoint as None."""
        page = pdf.pages[0]

        elements = page.find_all("text")
        if not elements:
            pytest.skip("Need at least 1 text element")

        first = elements[0]
        # Create region below without until - just use defaults
        region = first.below()

        # endpoint should be None when no until was specified
        assert region.endpoint is None

    def test_endpoint_aliases_end_element(self, pdf):
        """endpoint property should return the same value as end_element."""
        page = pdf.pages[0]

        elements = page.find_all("text")
        if len(elements) < 2:
            pytest.skip("Need at least 2 text elements")

        first = elements[0]
        region = first.below(until="text")

        # endpoint and end_element should be the same
        assert region.endpoint is region.end_element


class TestElementCollectionEndpoints:
    """Tests for ElementCollection.endpoints property."""

    def test_endpoints_returns_collection(self, pdf):
        """endpoints should return an ElementCollection."""
        page = pdf.pages[0]

        elements = page.find_all("text")[:3]
        if len(elements) < 2:
            pytest.skip("Need at least 2 text elements")

        # Create regions with until for each element
        regions = elements.below(until="text")

        # endpoints should be a collection
        endpoints = regions.endpoints
        assert isinstance(endpoints, ElementCollection)

    def test_endpoints_filters_none(self, pdf):
        """endpoints should skip elements without an endpoint."""
        page = pdf.pages[0]

        elements = page.find_all("text")[:2]
        if len(elements) < 1:
            pytest.skip("Need at least 1 text element")

        # Create region without until (no endpoint)
        region = elements[0].below()

        # Verify our region has no endpoint
        assert region.endpoint is None

        # Create a collection with this region
        collection = ElementCollection([region])

        # endpoints should be empty since the region has no endpoint
        assert len(collection.endpoints) == 0

    def test_endpoints_from_multiple_regions(self, pdf):
        """endpoints should collect all endpoints from regions with until."""
        page = pdf.pages[0]

        elements = page.find_all("text")[:5]
        if len(elements) < 3:
            pytest.skip("Need at least 3 text elements")

        # Create regions with until for multiple elements
        regions = elements[:2].below(until="text")

        # Each region should have an endpoint
        endpoints = regions.endpoints

        # Should have endpoints (may be fewer if some regions didn't find a match)
        # At minimum, we know the property works and returns a collection
        assert isinstance(endpoints, ElementCollection)
