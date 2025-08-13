"""Test horizontal guides alignment with top/bottom options."""

from unittest.mock import Mock

import pytest

from natural_pdf.analyzers.guides import Guides


def test_horizontal_guides_top_bottom_alignment():
    """Test that top/bottom alignment options work for horizontal guides."""

    # Create mock elements at different positions
    class MockElement:
        def __init__(self, text, x0, x1, y0, y1):
            self.text = text
            self.x0 = x0
            self.x1 = x1
            self.top = y0
            self.bottom = y1
            self.bbox = (x0, y0, x1, y1)

    # Create elements to use as markers
    elem1 = MockElement("Header 1", 100, 200, 50, 70)
    elem2 = MockElement("Header 2", 300, 400, 150, 170)
    elem3 = MockElement("Header 3", 100, 200, 250, 270)

    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 500, 400)

        def find(self, selector, apply_exclusions=True):
            # Map text to elements
            elements = {"Header 1": elem1, "Header 2": elem2, "Header 3": elem3}
            for text, elem in elements.items():
                if f'"{text}"' in selector:
                    return elem
            return None

    mock_page = MockPage()

    # Test 1: align='top' for horizontal guides
    guides_top = Guides.from_content(
        mock_page,
        axis="horizontal",
        markers=["Header 1", "Header 2", "Header 3"],
        align="top",
        outer=False,
    )

    # Should create guides at the top edge of each element
    assert 50 in guides_top.horizontal
    assert 150 in guides_top.horizontal
    assert 250 in guides_top.horizontal
    assert len(guides_top.horizontal) == 3

    # Test 2: align='bottom' for horizontal guides
    guides_bottom = Guides.from_content(
        mock_page,
        axis="horizontal",
        markers=["Header 1", "Header 2", "Header 3"],
        align="bottom",
        outer=False,
    )

    # Should create guides at the bottom edge of each element
    assert 70 in guides_bottom.horizontal
    assert 170 in guides_bottom.horizontal
    assert 270 in guides_bottom.horizontal
    assert len(guides_bottom.horizontal) == 3

    # Test 3: With outer guides
    guides_top_outer = Guides.from_content(
        mock_page,
        axis="horizontal",
        markers=["Header 1", "Header 2", "Header 3"],
        align="top",
        outer=True,
    )

    # Should have page boundaries plus element tops
    assert 0 in guides_top_outer.horizontal  # page top
    assert 50 in guides_top_outer.horizontal
    assert 150 in guides_top_outer.horizontal
    assert 250 in guides_top_outer.horizontal
    assert 400 in guides_top_outer.horizontal  # page bottom
    assert len(guides_top_outer.horizontal) == 5


def test_horizontal_guides_with_element_collection():
    """Test horizontal guides with ElementCollection using top/bottom alignment."""

    class MockElement:
        def __init__(self, text, x0, x1, y0, y1):
            self.text = text
            self.x0 = x0
            self.x1 = x1
            self.top = y0
            self.bottom = y1
            self.bbox = (x0, y0, x1, y1)

    class MockElementCollection:
        def __init__(self, elements):
            self.elements = elements
            self._elements = elements

        def __iter__(self):
            return iter(self.elements)

        def __len__(self):
            return len(self.elements)

        def extract_each_text(self):
            return [elem.text for elem in self.elements]

    elem1 = MockElement("Row 1", 50, 450, 100, 120)
    elem2 = MockElement("Row 2", 50, 450, 200, 220)
    elem3 = MockElement("Row 3", 50, 450, 300, 320)

    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 500, 500)

        def find(self, selector, apply_exclusions=True):
            # Should not be called when using ElementCollection
            raise AssertionError("find() should not be called with ElementCollection")

    mock_page = MockPage()

    # Create ElementCollection
    rows = MockElementCollection([elem1, elem2, elem3])

    # Test with align='top'
    guides_top = Guides.from_content(
        mock_page, axis="horizontal", markers=rows, align="top", outer=False
    )

    assert 100 in guides_top.horizontal
    assert 200 in guides_top.horizontal
    assert 300 in guides_top.horizontal
    assert len(guides_top.horizontal) == 3

    # Test with align='bottom'
    guides_bottom = Guides.from_content(
        mock_page, axis="horizontal", markers=rows, align="bottom", outer=False
    )

    assert 120 in guides_bottom.horizontal
    assert 220 in guides_bottom.horizontal
    assert 320 in guides_bottom.horizontal
    assert len(guides_bottom.horizontal) == 3


def test_guides_list_horizontal_alignment():
    """Test GuidesList.from_content with horizontal alignment options."""

    class MockElement:
        def __init__(self, text, x0, x1, y0, y1):
            self.text = text
            self.x0 = x0
            self.x1 = x1
            self.top = y0
            self.bottom = y1
            self.bbox = (x0, y0, x1, y1)

    elem1 = MockElement("Section A", 100, 400, 80, 100)
    elem2 = MockElement("Section B", 100, 400, 180, 200)

    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 500, 300)

        def find(self, selector, apply_exclusions=True):
            elements = {"Section A": elem1, "Section B": elem2}
            for text, elem in elements.items():
                if f'"{text}"' in selector:
                    return elem
            return None

    mock_page = MockPage()
    guides = Guides(context=mock_page)

    # Use GuidesList.from_content with align='top'
    guides.horizontal.from_content(markers=["Section A", "Section B"], align="top", outer=True)

    # Should have outer guides plus tops
    assert 0 in guides.horizontal
    assert 80 in guides.horizontal
    assert 180 in guides.horizontal
    assert 300 in guides.horizontal
    assert len(guides.horizontal) == 4

    # Reset and test with align='bottom'
    guides2 = Guides(context=mock_page)
    guides2.horizontal.from_content(markers=["Section A", "Section B"], align="bottom", outer=False)

    assert 100 in guides2.horizontal
    assert 200 in guides2.horizontal
    assert len(guides2.horizontal) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
