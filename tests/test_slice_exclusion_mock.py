"""Test exclusion handling with slices using mocks."""

from unittest.mock import MagicMock, Mock, patch

import pytest

import natural_pdf as npdf
from natural_pdf.core.pdf import _LazyPageList


def test_lazy_page_list_reuses_cached_pages():
    """Test that _LazyPageList reuses cached pages from parent PDF."""
    # Create mock objects
    mock_pdf = Mock()
    mock_pdf._exclusions = [("test_exclusion", "test_label")]
    mock_pdf._regions = []
    mock_pdf._pages = None  # Will be set later

    # Create a mock plumber PDF
    mock_plumber_pdf = Mock()
    mock_plumber_pdf.pages = [Mock() for _ in range(3)]  # 3 pages

    # Create the main _LazyPageList (simulating pdf.pages)
    main_pages = _LazyPageList(
        parent_pdf=mock_pdf, plumber_pdf=mock_plumber_pdf, font_attrs=None, load_text=True
    )

    # Set up the reference so slices can find the main page list
    mock_pdf._pages = main_pages

    # Mock the Page class
    with patch("natural_pdf.core.page.Page") as MockPage:
        # Create mock page instances
        mock_page_0 = Mock()
        mock_page_0.number = 1
        mock_page_0.index = 0
        mock_page_0._exclusions = [("test_exclusion", "test_label")]

        mock_page_1 = Mock()
        mock_page_1.number = 2
        mock_page_1.index = 1
        mock_page_1._exclusions = [("test_exclusion", "test_label")]

        # Configure MockPage to return different instances
        MockPage.side_effect = [mock_page_0, mock_page_1]

        # Access pages to cache them
        page0 = main_pages[0]
        page1 = main_pages[1]

        # Verify pages are cached
        assert main_pages._cache[0] is mock_page_0
        assert main_pages._cache[1] is mock_page_1

        # Now create a slice (simulating pdf.pages[:2])
        sliced_pages = main_pages[:2]

        # Access pages from the slice
        slice_page0 = sliced_pages[0]
        slice_page1 = sliced_pages[1]

        # The key assertion: sliced pages should reuse the cached pages
        assert slice_page0 is mock_page_0, "Slice should reuse cached page 0"
        assert slice_page1 is mock_page_1, "Slice should reuse cached page 1"

        # Verify Page constructor was only called twice (not 4 times)
        assert MockPage.call_count == 2, "Page should only be created once per page"


def test_exclusions_persist_across_slices():
    """Test that exclusions added after caching are visible in slices."""
    # Create a more realistic mock setup
    mock_pdf = Mock()
    mock_pdf._exclusions = []
    mock_pdf._regions = []

    mock_plumber_pdf = Mock()
    mock_plumber_pdf.pages = [Mock() for _ in range(2)]

    # Create main page list
    main_pages = _LazyPageList(
        parent_pdf=mock_pdf, plumber_pdf=mock_plumber_pdf, font_attrs=None, load_text=True
    )

    # Set up the reference so pages can check parent cache
    mock_pdf._pages = main_pages

    with patch("natural_pdf.core.page.Page") as MockPage:
        # Create a mock page with exclusion tracking
        mock_page = Mock()
        mock_page.number = 1
        mock_page.index = 0
        mock_page._exclusions = []
        mock_page.add_exclusion = Mock(
            side_effect=lambda exc, label: mock_page._exclusions.append((exc, label))
        )

        MockPage.return_value = mock_page

        # Access page before adding exclusion
        page_before = main_pages[0]
        assert len(page_before._exclusions) == 0, "Page should start with no exclusions"

        # Add exclusion to PDF
        mock_pdf._exclusions.append(("new_exclusion", "new_label"))

        # Create slice and access page
        slice_pages = main_pages[:1]
        page_from_slice = slice_pages[0]

        # Should be the same object
        assert page_from_slice is page_before, "Should reuse the same page object"

        # The page still has the exclusions it had when created
        # (This demonstrates why we need to apply PDF exclusions when pages are created)
        assert (
            len(page_from_slice._exclusions) == 0
        ), "Cached page doesn't get new exclusions retroactively"


def test_new_pages_get_all_exclusions():
    """Test that pages created after exclusions are added get all exclusions."""
    mock_pdf = Mock()
    mock_pdf._exclusions = [("exclusion1", "label1"), ("exclusion2", "label2")]
    mock_pdf._regions = []

    mock_plumber_pdf = Mock()
    mock_plumber_pdf.pages = [Mock()]

    main_pages = _LazyPageList(
        parent_pdf=mock_pdf, plumber_pdf=mock_plumber_pdf, font_attrs=None, load_text=True
    )

    # Set up reference
    mock_pdf._pages = main_pages

    with patch("natural_pdf.core.page.Page") as MockPage:
        mock_page = Mock()
        mock_page.number = 1
        mock_page.index = 0
        mock_page._exclusions = []
        exclusions_added = []
        mock_page.add_exclusion = Mock(
            side_effect=lambda exc, label: exclusions_added.append((exc, label))
        )

        MockPage.return_value = mock_page

        # Access page (will be created with exclusions)
        page = main_pages[0]

        # Verify all exclusions were applied
        assert mock_page.add_exclusion.call_count == 2, "Both exclusions should be applied"
        assert exclusions_added == [("exclusion1", "label1"), ("exclusion2", "label2")]


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
