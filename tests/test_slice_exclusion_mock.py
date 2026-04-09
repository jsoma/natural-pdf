"""Test exclusion handling with slices using mocks."""

from unittest.mock import Mock, patch

import pytest

from natural_pdf.core.pdf import _LazyPageList


class FakePage:
    created = 0

    def __init__(
        self,
        plumber_page,
        *,
        parent,
        index,
        font_attrs=None,
        load_text=True,
        context=None,
    ):
        type(self).created += 1
        self.plumber_page = plumber_page
        self._parent = parent
        self.index = index
        self.number = index + 1
        self._exclusions = []
        self.font_attrs = font_attrs
        self.load_text = load_text
        self.context = context

    def effective_exclusions(self):
        exclusions = list(self._exclusions)
        existing_labels = {label for _, label, _ in exclusions if label}
        for pdf_exclusion in getattr(self._parent, "_exclusions", []):
            label = pdf_exclusion[1] if len(pdf_exclusion) >= 2 else None
            if label and label in existing_labels:
                continue
            if len(pdf_exclusion) == 2:
                exclusions.append((pdf_exclusion[0], pdf_exclusion[1], "region"))
            else:
                exclusions.append(pdf_exclusion)
        return exclusions


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
        FakePage.created = 0
        # Create mock page instances
        mock_page_0 = FakePage(Mock(), parent=mock_pdf, index=0)
        mock_page_1 = FakePage(Mock(), parent=mock_pdf, index=1)

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

    with patch("natural_pdf.core.page.Page", FakePage):
        FakePage.created = 0

        # Access page before adding exclusion
        page_before = main_pages[0]
        assert page_before.effective_exclusions() == [], "Page should start with no exclusions"

        # Add exclusion to PDF
        mock_pdf._exclusions.append(("new_exclusion", "new_label"))

        # Create slice and access page
        slice_pages = main_pages[:1]
        page_from_slice = slice_pages[0]

        # Should be the same object
        assert page_from_slice is page_before, "Should reuse the same page object"

        # Cached pages should reflect PDF-level exclusions dynamically through their parent.
        assert page_from_slice.effective_exclusions() == [("new_exclusion", "new_label", "region")]


def test_new_pages_get_all_exclusions():
    """Test that newly created pages see all PDF exclusions without eager seeding."""
    mock_pdf = Mock()
    mock_pdf._exclusions = [("exclusion1", "label1", "region"), ("exclusion2", "label2", "region")]
    mock_pdf._regions = []

    mock_plumber_pdf = Mock()
    mock_plumber_pdf.pages = [Mock()]

    main_pages = _LazyPageList(
        parent_pdf=mock_pdf, plumber_pdf=mock_plumber_pdf, font_attrs=None, load_text=True
    )

    # Set up reference
    mock_pdf._pages = main_pages

    with patch("natural_pdf.core.page.Page", FakePage):
        FakePage.created = 0

        # Access page (will be created with exclusions)
        page = main_pages[0]

        # PDF exclusions remain dynamic via the parent rather than being copied into page state.
        assert page._exclusions == []
        assert page.effective_exclusions() == [
            ("exclusion1", "label1", "region"),
            ("exclusion2", "label2", "region"),
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
