"""Test guides.from_stripes() functionality."""

from unittest.mock import Mock

import pytest

from natural_pdf.analyzers.guides import Guides


def test_from_stripes_manual_selection():
    """Test from_stripes with manually provided stripe elements."""

    # Create mock stripe elements (rectangles)
    class MockRect:
        def __init__(self, fill, x0, x1, top, bottom):
            self.fill = fill
            self.x0 = x0
            self.x1 = x1
            self.top = top
            self.bottom = bottom

    # Create stripes at different row positions
    stripe1 = MockRect("#00ffff", 0, 500, 100, 120)
    stripe2 = MockRect("#00ffff", 0, 500, 140, 160)
    stripe3 = MockRect("#00ffff", 0, 500, 180, 200)

    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 500, 300)

    mock_page = MockPage()
    guides = Guides(context=mock_page)

    # Test horizontal guides from stripes
    guides.horizontal.from_stripes([stripe1, stripe2, stripe3])

    # Should have guides at top and bottom of each stripe
    assert 100 in guides.horizontal
    assert 120 in guides.horizontal
    assert 140 in guides.horizontal
    assert 160 in guides.horizontal
    assert 180 in guides.horizontal
    assert 200 in guides.horizontal
    assert len(guides.horizontal) == 6


def test_from_stripes_with_color():
    """Test from_stripes with specific color."""

    class MockRect:
        def __init__(self, fill, x0, x1, top, bottom):
            self.fill = fill
            self.x0 = x0
            self.x1 = x1
            self.top = top
            self.bottom = bottom

    # Create mix of colored rectangles
    cyan1 = MockRect("#00ffff", 0, 500, 100, 120)
    white1 = MockRect("#ffffff", 0, 500, 120, 140)
    cyan2 = MockRect("#00ffff", 0, 500, 140, 160)

    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 500, 300)
            self.rects = [cyan1, white1, cyan2]

        def find_all(self, selector):
            if selector == "rect[fill=#00ffff]":
                return [r for r in self.rects if r.fill == "#00ffff"]
            return []

    mock_page = MockPage()
    guides = Guides(context=mock_page)

    # Test with specific color
    guides.horizontal.from_stripes(color="#00ffff")

    # Should only have guides from cyan stripes
    assert 100 in guides.horizontal
    assert 120 in guides.horizontal
    assert 140 in guides.horizontal
    assert 160 in guides.horizontal
    assert len(guides.horizontal) == 4


def test_from_stripes_auto_detect():
    """Test from_stripes with automatic color detection."""

    class MockRect:
        def __init__(self, fill, x0, x1, top, bottom):
            self.fill = fill
            self.x0 = x0
            self.x1 = x1
            self.top = top
            self.bottom = bottom

    # Create rectangles with different colors
    # Cyan appears most frequently (excluding white)
    rects = [
        MockRect("#00ffff", 0, 500, 100, 120),
        MockRect("#ffffff", 0, 500, 120, 140),
        MockRect("#00ffff", 0, 500, 140, 160),
        MockRect("#ffffff", 0, 500, 160, 180),
        MockRect("#00ffff", 0, 500, 180, 200),
        MockRect("#ff0000", 0, 500, 200, 220),  # Red - less common
    ]

    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 500, 300)
            self.rects = rects

        def find_all(self, selector):
            if selector == "rect[fill]":
                return self.rects
            return []

    mock_page = MockPage()
    guides = Guides(context=mock_page)

    # Test auto-detection
    guides.horizontal.from_stripes()

    # Should detect cyan as most common non-white color
    assert 100 in guides.horizontal
    assert 120 in guides.horizontal
    assert 140 in guides.horizontal
    assert 160 in guides.horizontal
    assert 180 in guides.horizontal
    assert 200 in guides.horizontal
    assert len(guides.horizontal) == 6


def test_from_stripes_vertical():
    """Test from_stripes for vertical guides (column stripes)."""

    class MockRect:
        def __init__(self, fill, x0, x1, top, bottom):
            self.fill = fill
            self.x0 = x0
            self.x1 = x1
            self.top = top
            self.bottom = bottom

    # Create vertical stripes (columns)
    stripe1 = MockRect("#e0e0e0", 100, 150, 0, 400)
    stripe2 = MockRect("#e0e0e0", 200, 250, 0, 400)
    stripe3 = MockRect("#e0e0e0", 300, 350, 0, 400)

    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 500, 400)

    mock_page = MockPage()
    guides = Guides(context=mock_page)

    # Test vertical guides from stripes
    guides.vertical.from_stripes([stripe1, stripe2, stripe3])

    # Should have guides at left and right of each stripe
    assert 100 in guides.vertical
    assert 150 in guides.vertical
    assert 200 in guides.vertical
    assert 250 in guides.vertical
    assert 300 in guides.vertical
    assert 350 in guides.vertical
    assert len(guides.vertical) == 6


def test_from_stripes_no_stripes_found():
    """Test from_stripes when no stripes are found."""

    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 500, 300)

        def find_all(self, selector):
            return []  # No rectangles found

    mock_page = MockPage()
    guides = Guides(context=mock_page)

    # Test with no stripes found
    original_count = len(guides.horizontal)
    guides.horizontal.from_stripes()

    # Should not add any guides
    assert len(guides.horizontal) == original_count


def test_from_stripes_duplicates_removed():
    """Test that duplicate edges are removed."""

    class MockRect:
        def __init__(self, fill, x0, x1, top, bottom):
            self.fill = fill
            self.x0 = x0
            self.x1 = x1
            self.top = top
            self.bottom = bottom

    # Create adjacent stripes that share edges
    stripe1 = MockRect("#00ffff", 0, 500, 100, 120)
    stripe2 = MockRect("#00ffff", 0, 500, 120, 140)  # Top at 120 (duplicate)

    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 500, 300)

    mock_page = MockPage()
    guides = Guides(context=mock_page)

    guides.horizontal.from_stripes([stripe1, stripe2])

    # Should have only unique guides
    assert 100 in guides.horizontal
    assert 120 in guides.horizontal
    assert 140 in guides.horizontal
    assert len(guides.horizontal) == 3  # Not 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
