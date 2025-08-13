"""Test that ElementCollection works correctly with guides.from_content."""

from unittest.mock import Mock

import pytest

from natural_pdf.analyzers.guides import Guides


def test_element_collection_uses_actual_elements():
    """Test that ElementCollection elements are used directly, not searched by text."""

    # Create mock elements with specific positions and text
    class MockElement:
        def __init__(self, text, x0, x1, y0, y1):
            self.text = text
            self.x0 = x0
            self.x1 = x1
            self.top = y0
            self.bottom = y1
            self.bbox = (x0, y0, x1, y1)

    # Create elements including one with "ST" that should be found
    # and another element containing "ST" that shouldn't interfere
    target_st = MockElement("ST", 539.63316, 550.52172, 70.8018, 79.3218)
    wrong_st = MockElement("ALPHABETIC LISTING BY TYPE", 332.88096, 459.77784, 21.35424, 29.87424)

    class MockElementCollection:
        def __init__(self, elements):
            self.elements = elements
            self._elements = elements  # Some code might use _elements

        def __iter__(self):
            return iter(self.elements)

        def __len__(self):
            return len(self.elements)

        def extract_each_text(self):
            return [elem.text for elem in self.elements]

    # Create a mock page that would return the wrong element when searching for "ST"
    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 792, 612)
            self.all_elements = [wrong_st, target_st]  # wrong_st comes first

        def find(self, selector, apply_exclusions=True):
            # This simulates the bug - searching for "ST" returns the wrong element
            if 'contains("ST")' in selector:
                return wrong_st  # Returns "ALPHABETIC LISTING BY TYPE"
            return None

        def find_all(self, selector):
            if selector == "text":
                return self.all_elements
            return []

    mock_page = MockPage()

    # Create guides using an ElementCollection
    elements = MockElementCollection([target_st])
    guides = Guides.from_content(
        mock_page, axis="vertical", markers=elements, align="left", outer=False
    )

    # The guide should be at the actual element's position (539.63316)
    # NOT at the position of the wrong element (332.88096)
    assert 539.63316 in guides.vertical, f"Expected guide at 539.63316, got {guides.vertical}"
    assert (
        332.88096 not in guides.vertical
    ), f"Should not have guide at 332.88096, got {guides.vertical}"

    # Test with multiple elements
    elem1 = MockElement("ADDRESS", 328.32012, 369.50580, 70.8018, 79.3218)
    elem2 = MockElement("ST", 539.63316, 550.52172, 70.8018, 79.3218)

    elements = MockElementCollection([elem1, elem2])
    guides = Guides.from_content(
        mock_page, axis="vertical", markers=elements, align="left", outer=False
    )

    # Should have guides at both actual positions
    assert 328.32012 in guides.vertical
    assert 539.63316 in guides.vertical
    assert len(guides.vertical) == 2


def test_guides_list_element_collection():
    """Test GuidesList.from_content with ElementCollection."""

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

    elem1 = MockElement("Header 1", 100, 150, 50, 70)
    elem2 = MockElement("Header 2", 300, 350, 50, 70)

    class MockPage:
        def __init__(self):
            self.bbox = (0, 0, 500, 700)

        def find(self, selector, apply_exclusions=True):
            # Should not be called when using ElementCollection directly
            raise AssertionError("find() should not be called with ElementCollection")

    mock_page = MockPage()
    guides = Guides(context=mock_page)

    # Use ElementCollection with GuidesList
    elements = MockElementCollection([elem1, elem2])
    guides.vertical.from_content(markers=elements, align="left", outer=False)

    # Should have guides at element positions
    assert 100 in guides.vertical
    assert 300 in guides.vertical
    assert len(guides.vertical) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
