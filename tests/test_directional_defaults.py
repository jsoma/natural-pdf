#!/usr/bin/env python3
"""Test the new default height/width values for directional methods."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from natural_pdf.elements.region import Region


def test_left_defaults_to_element_height():
    """Test that left() uses height='element' by default."""
    # Create a mock element with the necessary attributes
    element = Mock()
    element.page = Mock()
    element.page.width = 612
    element.page.height = 792
    element.bbox = (100, 100, 200, 150)
    element.x0, element.y0, element.x1, element.y1 = element.bbox
    element.width = 100
    element.height = 50
    element.top = 100
    element.bottom = 150
    element.left_x = 100
    element.right_x = 200

    # Mock the _direction method to capture its arguments
    direction_calls = []

    def mock_direction(direction, size, cross_size, **kwargs):
        direction_calls.append(
            {"direction": direction, "size": size, "cross_size": cross_size, "kwargs": kwargs}
        )
        return Mock(spec=Region)

    element._direction = mock_direction

    # Import and bind the left method to our mock
    from natural_pdf.elements.base import Element

    element.left = Element.left.__get__(element, type(element))

    # Call left() without specifying height
    result = element.left()

    # Verify height='element' was used
    assert len(direction_calls) == 1
    assert direction_calls[0]["cross_size"] == "element"
    assert direction_calls[0]["direction"] == "left"


def test_right_defaults_to_element_height():
    """Test that right() uses height='element' by default."""
    # Create a mock element
    element = Mock()
    element.page = Mock()
    element.page.width = 612
    element.page.height = 792
    element.bbox = (100, 100, 200, 150)
    element.x0, element.y0, element.x1, element.y1 = element.bbox
    element.width = 100
    element.height = 50

    # Mock the _direction method
    direction_calls = []

    def mock_direction(direction, size, cross_size, **kwargs):
        direction_calls.append(
            {"direction": direction, "size": size, "cross_size": cross_size, "kwargs": kwargs}
        )
        return Mock(spec=Region)

    element._direction = mock_direction

    # Import and bind the right method
    from natural_pdf.elements.base import Element

    element.right = Element.right.__get__(element, type(element))

    # Call right() without specifying height
    result = element.right()

    # Verify height='element' was used
    assert len(direction_calls) == 1
    assert direction_calls[0]["cross_size"] == "element"
    assert direction_calls[0]["direction"] == "right"


def test_above_defaults_to_full_width():
    """Test that above() uses width='full' by default."""
    # Create a mock element
    element = Mock()
    element.page = Mock()
    element.page.width = 612
    element.page.height = 792
    element.bbox = (100, 100, 200, 150)

    # Mock the _direction method
    direction_calls = []

    def mock_direction(direction, size, cross_size, **kwargs):
        direction_calls.append(
            {"direction": direction, "size": size, "cross_size": cross_size, "kwargs": kwargs}
        )
        return Mock(spec=Region)

    element._direction = mock_direction

    # Import and bind the above method
    from natural_pdf.elements.base import Element

    element.above = Element.above.__get__(element, type(element))

    # Call above() without specifying width
    result = element.above()

    # Verify width='full' was used
    assert len(direction_calls) == 1
    assert direction_calls[0]["cross_size"] == "full"
    assert direction_calls[0]["direction"] == "above"


def test_below_defaults_to_full_width():
    """Test that below() uses width='full' by default."""
    # Create a mock element
    element = Mock()
    element.page = Mock()
    element.page.width = 612
    element.page.height = 792
    element.bbox = (100, 100, 200, 150)

    # Mock the _direction method
    direction_calls = []

    def mock_direction(direction, size, cross_size, **kwargs):
        direction_calls.append(
            {"direction": direction, "size": size, "cross_size": cross_size, "kwargs": kwargs}
        )
        return Mock(spec=Region)

    element._direction = mock_direction

    # Import and bind the below method
    from natural_pdf.elements.base import Element

    element.below = Element.below.__get__(element, type(element))

    # Call below() without specifying width
    result = element.below()

    # Verify width='full' was used
    assert len(direction_calls) == 1
    assert direction_calls[0]["cross_size"] == "full"
    assert direction_calls[0]["direction"] == "below"


def test_explicit_parameters_override_defaults():
    """Test that explicit parameters override the defaults."""
    # Create a mock element
    element = Mock()
    element.page = Mock()
    element.page.width = 612
    element.page.height = 792
    element.bbox = (100, 100, 200, 150)
    element.x0, element.y0, element.x1, element.y1 = element.bbox
    element.width = 100
    element.height = 50

    # Mock the _direction method
    direction_calls = []

    def mock_direction(direction, size, cross_size, **kwargs):
        direction_calls.append(
            {"direction": direction, "size": size, "cross_size": cross_size, "kwargs": kwargs}
        )
        return Mock(spec=Region)

    element._direction = mock_direction

    # Import and bind all methods
    from natural_pdf.elements.base import Element

    element.left = Element.left.__get__(element, type(element))
    element.right = Element.right.__get__(element, type(element))
    element.above = Element.above.__get__(element, type(element))
    element.below = Element.below.__get__(element, type(element))

    # Test left with explicit height='full'
    element.left(height="full")
    assert direction_calls[-1]["cross_size"] == "full"

    # Test right with explicit height='full'
    element.right(height="full")
    assert direction_calls[-1]["cross_size"] == "full"

    # Test above with explicit width='element'
    element.above(width="element")
    assert direction_calls[-1]["cross_size"] == "element"

    # Test below with explicit width='element'
    element.below(width="element")
    assert direction_calls[-1]["cross_size"] == "element"


def test_numeric_height_width_values():
    """Test that numeric height/width values are passed through correctly."""
    # Create a mock element
    element = Mock()
    element.page = Mock()
    element.page.width = 612
    element.page.height = 792
    element.bbox = (100, 100, 200, 150)

    # Mock the _direction method
    direction_calls = []

    def mock_direction(direction, size, cross_size, **kwargs):
        direction_calls.append(
            {"direction": direction, "size": size, "cross_size": cross_size, "kwargs": kwargs}
        )
        return Mock(spec=Region)

    element._direction = mock_direction

    # Import and bind all methods
    from natural_pdf.elements.base import Element

    element.left = Element.left.__get__(element, type(element))
    element.right = Element.right.__get__(element, type(element))
    element.above = Element.above.__get__(element, type(element))
    element.below = Element.below.__get__(element, type(element))

    # Test with numeric values
    element.left(height=100)
    assert direction_calls[-1]["cross_size"] == 100

    element.right(height=200.5)
    assert direction_calls[-1]["cross_size"] == 200.5

    element.above(width=150)
    assert direction_calls[-1]["cross_size"] == 150

    element.below(width=75.25)
    assert direction_calls[-1]["cross_size"] == 75.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
