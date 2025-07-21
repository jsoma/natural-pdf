#!/usr/bin/env python3
"""Test the limit parameter for show() method on PDFs and page collections."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from natural_pdf import PDF
from natural_pdf.core.page_collection import PageCollection
from natural_pdf.core.render_spec import RenderSpec, Visualizable


def test_pdf_show_limit():
    """Test that pdf.show(limit=N) only shows N pages."""
    # Create a mock PDF with multiple pages
    pdf = Mock(spec=PDF)
    pdf.__class__ = PDF

    # Create mock pages
    mock_pages = []
    for i in range(50):  # Create 50 pages
        page = Mock()
        page.index = i
        page._get_render_specs = Mock(return_value=[RenderSpec(page=page)])
        mock_pages.append(page)

    # Create a mock page collection
    page_collection = Mock(spec=PageCollection)
    page_collection.pages = mock_pages
    page_collection.__iter__ = lambda self: iter(self.pages)
    page_collection.__len__ = lambda self: len(self.pages)

    # Mock the _get_render_specs method
    def mock_get_render_specs(mode="show", **kwargs):
        max_pages = kwargs.get("max_pages")
        pages_to_render = mock_pages[:max_pages] if max_pages else mock_pages

        specs = []
        for page in pages_to_render:
            spec = RenderSpec(page=page)
            specs.append(spec)
        return specs

    page_collection._get_render_specs = mock_get_render_specs
    pdf.pages = page_collection

    # Mock the highlighter
    mock_highlighter = Mock()
    mock_highlighter.unified_render = Mock(return_value=Mock())  # Return a mock image

    pdf._highlighter = mock_highlighter
    pdf._get_highlighter = Mock(return_value=mock_highlighter)

    # Mock _get_render_specs for PDF to delegate to pages
    pdf._get_render_specs = lambda mode="show", **kwargs: page_collection._get_render_specs(
        mode=mode, **kwargs
    )

    # Mix in the Visualizable methods
    for attr_name in dir(Visualizable):
        if not attr_name.startswith("_") or attr_name == "_get_highlighter":
            attr = getattr(Visualizable, attr_name)
            if callable(attr):
                setattr(pdf, attr_name, attr.__get__(pdf, pdf.__class__))

    # Test default limit (30)
    pdf.show()

    # Check that unified_render was called with 30 specs
    assert mock_highlighter.unified_render.called
    call_args = mock_highlighter.unified_render.call_args
    specs = call_args[1]["specs"]
    assert len(specs) == 30, f"Expected 30 specs with default limit, got {len(specs)}"

    # Test custom limit
    mock_highlighter.unified_render.reset_mock()
    pdf.show(limit=10)

    assert mock_highlighter.unified_render.called
    call_args = mock_highlighter.unified_render.call_args
    specs = call_args[1]["specs"]
    assert len(specs) == 10, f"Expected 10 specs with limit=10, got {len(specs)}"

    # Test limit=None (show all)
    mock_highlighter.unified_render.reset_mock()
    pdf.show(limit=None)

    assert mock_highlighter.unified_render.called
    call_args = mock_highlighter.unified_render.call_args
    specs = call_args[1]["specs"]
    assert len(specs) == 50, f"Expected 50 specs with limit=None, got {len(specs)}"


def test_page_collection_show_limit():
    """Test that page_collection.show(limit=N) only shows N pages."""
    # Create mock pages
    mock_pages = []
    for i in range(20):
        page = Mock()
        page.index = i
        page._get_render_specs = Mock(return_value=[RenderSpec(page=page)])
        mock_pages.append(page)

    # Create page collection
    page_collection = PageCollection(mock_pages)

    # Mock the highlighter
    mock_highlighter = Mock()
    mock_highlighter.unified_render = Mock(return_value=Mock())

    # Mock _get_highlighter
    page_collection._get_highlighter = Mock(return_value=mock_highlighter)

    # Test with limit
    page_collection.show(limit=5)

    assert mock_highlighter.unified_render.called
    call_args = mock_highlighter.unified_render.call_args
    specs = call_args[1]["specs"]
    assert len(specs) == 5, f"Expected 5 specs with limit=5, got {len(specs)}"

    # Test without limit (should use default of 30, but we only have 20 pages)
    mock_highlighter.unified_render.reset_mock()
    page_collection.show()

    assert mock_highlighter.unified_render.called
    call_args = mock_highlighter.unified_render.call_args
    specs = call_args[1]["specs"]
    assert len(specs) == 20, f"Expected 20 specs (all pages), got {len(specs)}"


def test_single_page_show_not_affected():
    """Test that limit parameter doesn't affect single page show()."""
    # Create a mock page
    page = Mock()
    page.index = 0

    # Create a proper mock that inherits from Visualizable
    class MockPage(Visualizable):
        def __init__(self):
            self.index = 0
            self._highlighter = Mock()
            self._highlighter.unified_render = Mock(return_value=Mock())

        def _get_render_specs(self, mode="show", **kwargs):
            return [RenderSpec(page=self)]

        def _get_highlighter(self):
            return self._highlighter

    page = MockPage()

    # Test that limit is passed but doesn't break single pages
    result = page.show(limit=5)

    # Should render successfully
    assert page._highlighter.unified_render.called
    call_args = page._highlighter.unified_render.call_args
    specs = call_args[1]["specs"]
    assert len(specs) == 1, f"Expected 1 spec for single page, got {len(specs)}"


if __name__ == "__main__":
    test_pdf_show_limit()
    print("✓ PDF show limit test passed")

    test_page_collection_show_limit()
    print("✓ PageCollection show limit test passed")

    test_single_page_show_not_affected()
    print("✓ Single page show test passed")

    print("\nAll tests passed!")
