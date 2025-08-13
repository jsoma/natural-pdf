"""Test that markers in from_content are processed in spatial order."""

from unittest.mock import Mock

import pytest

from natural_pdf.analyzers.guides import Guides


def test_from_content_markers_sorted_spatially():
    """Test that markers are processed in spatial order regardless of input order."""

    # Create mock elements at different positions
    class MockElement:
        def __init__(self, text, x0, x1, y0, y1):
            self.text = text
            self.x0 = x0
            self.x1 = x1
            self.top = y0
            self.bottom = y1

    # Elements positioned out of order
    elements = {
        "Right": MockElement("Right", 300, 350, 100, 120),
        "Middle": MockElement("Middle", 150, 200, 100, 120),
        "Left": MockElement("Left", 50, 100, 100, 120),
    }

    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 400, 600)

        def find(self, selector, apply_exclusions=True):
            # Extract the text from selector like 'text:contains("Right")'
            for text, elem in elements.items():
                if f'"{text}"' in selector:
                    return elem
            return None

        def find_all(self, selector):
            # For the test of using 'text' selector
            if selector == "text":
                return list(elements.values())
            return []

    mock_page = MockPage()

    # Test 1: Vertical guides with markers in wrong order
    guides = Guides.from_content(
        mock_page,
        axis="vertical",
        markers=["Right", "Middle", "Left"],  # Wrong order
        align="left",
        outer=True,
    )

    # The guides should be sorted left to right
    vertical_coords = sorted(guides.vertical)
    assert vertical_coords == [
        0,
        50,
        150,
        300,
        400,
    ], f"Expected [0, 50, 150, 300, 400] but got {vertical_coords}"

    # Test 2: Without outer guides
    guides_no_outer = Guides.from_content(
        mock_page, axis="vertical", markers=["Right", "Middle", "Left"], align="left", outer=False
    )

    vertical_coords_no_outer = sorted(guides_no_outer.vertical)
    assert vertical_coords_no_outer == [
        50,
        150,
        300,
    ], f"Expected [50, 150, 300] but got {vertical_coords_no_outer}"

    # Test 3: Horizontal guides with markers in wrong order
    elements_horiz = {
        "Bottom": MockElement("Bottom", 100, 150, 300, 320),
        "Middle": MockElement("Middle", 100, 150, 150, 170),
        "Top": MockElement("Top", 100, 150, 50, 70),
    }

    class MockPageHoriz:
        def __init__(self):
            self.bbox = (0, 0, 400, 400)

        def find(self, selector, apply_exclusions=True):
            for text, elem in elements_horiz.items():
                if f'"{text}"' in selector:
                    return elem
            return None

    mock_page_horiz = MockPageHoriz()

    guides_horiz = Guides.from_content(
        mock_page_horiz,
        axis="horizontal",
        markers=["Bottom", "Middle", "Top"],  # Wrong order
        align="left",  # top for horizontal
        outer=True,
    )

    # The guides should be sorted top to bottom
    horiz_coords = sorted(guides_horiz.horizontal)
    assert horiz_coords == [
        0,
        50,
        150,
        300,
        400,
    ], f"Expected [0, 50, 150, 300, 400] but got {horiz_coords}"

    # Test 4: Test the 'between' alignment with out-of-order markers
    guides_between = Guides.from_content(
        mock_page,
        axis="vertical",
        markers=["Right", "Left"],  # Wrong order
        align="between",
        outer=True,
    )

    # Between Left (50-100) and Right (300-350) should be at (100+300)/2 = 200
    vertical_between = sorted(guides_between.vertical)
    # With outer=True: [0, 200, 400]
    assert 200 in vertical_between, f"Expected midpoint at 200, got {vertical_between}"


def test_guides_list_from_content_markers_sorted():
    """Test that GuidesList.from_content also sorts markers spatially."""

    # Create mock elements
    class MockElement:
        def __init__(self, text, x0, x1, y0, y1):
            self.text = text
            self.x0 = x0
            self.x1 = x1
            self.top = y0
            self.bottom = y1

    elements = {
        "C": MockElement("C", 250, 300, 100, 120),
        "A": MockElement("A", 50, 100, 100, 120),
        "B": MockElement("B", 150, 200, 100, 120),
    }

    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 400, 600)

        def find(self, selector, apply_exclusions=True):
            for text, elem in elements.items():
                if f'"{text}"' in selector:
                    return elem
            return None

    mock_page = MockPage()
    guides = Guides(context=mock_page)

    # Use GuidesList.from_content with out-of-order markers
    guides.vertical.from_content(markers=["C", "A", "B"], align="left", outer=True)

    # Should be sorted: [0, 50, 150, 250, 400]
    vertical_coords = sorted(guides.vertical)
    assert vertical_coords == [
        0,
        50,
        150,
        250,
        400,
    ], f"Expected [0, 50, 150, 250, 400] but got {vertical_coords}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
