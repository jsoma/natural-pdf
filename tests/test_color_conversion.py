#!/usr/bin/env python3
"""Test color conversion from pdfplumber format to natural-pdf format."""

from unittest.mock import Mock

import pytest

from natural_pdf.elements.text import TextElement


def test_grayscale_single_value_tuple():
    """Test that single-value grayscale tuples from pdfplumber are handled correctly."""
    # This fixes the bug where (0.4,) was being converted to (0, 0, 0) instead of (0.4, 0.4, 0.4)

    # Create mock text element with grayscale color (0.4,) - like from pdfplumber
    mock_obj = {
        "non_stroking_color": (0.4,),  # pdfplumber format
        "size": 11,
        "fontname": "BCDEEE+Aptos",
    }

    text_element = Mock()
    text_element._obj = mock_obj
    text_element.color = TextElement.color.__get__(text_element, TextElement)

    # Should convert to RGB grayscale tuple
    result = text_element.color
    expected = (0.4, 0.4, 0.4)
    assert result == expected, f"Expected {expected}, got {result}"


def test_grayscale_black_single_tuple():
    """Test black grayscale (0,) converts correctly."""
    mock_obj = {
        "non_stroking_color": (0,),  # pdfplumber format for black
        "size": 11,
        "fontname": "BCDEEE+Aptos",
    }

    text_element = Mock()
    text_element._obj = mock_obj
    text_element.color = TextElement.color.__get__(text_element, TextElement)

    result = text_element.color
    expected = (0, 0, 0)
    assert result == expected, f"Expected {expected}, got {result}"


def test_grayscale_white_single_tuple():
    """Test white grayscale (1,) converts correctly."""
    mock_obj = {
        "non_stroking_color": (1,),  # pdfplumber format for white
        "size": 11,
        "fontname": "BCDEEE+Aptos",
    }

    text_element = Mock()
    text_element._obj = mock_obj
    text_element.color = TextElement.color.__get__(text_element, TextElement)

    result = text_element.color
    expected = (1, 1, 1)
    assert result == expected, f"Expected {expected}, got {result}"


def test_existing_rgb_format_still_works():
    """Test that existing RGB tuple format continues to work."""
    mock_obj = {
        "non_stroking_color": (0.2, 0.5, 0.8),  # RGB format
        "size": 11,
        "fontname": "BCDEEE+Aptos",
    }

    text_element = Mock()
    text_element._obj = mock_obj
    text_element.color = TextElement.color.__get__(text_element, TextElement)

    result = text_element.color
    expected = (0.2, 0.5, 0.8)
    assert result == expected, f"Expected {expected}, got {result}"


def test_existing_single_float_still_works():
    """Test that existing single float format continues to work."""
    mock_obj = {
        "non_stroking_color": 0.7,  # Single float format
        "size": 11,
        "fontname": "BCDEEE+Aptos",
    }

    text_element = Mock()
    text_element._obj = mock_obj
    text_element.color = TextElement.color.__get__(text_element, TextElement)

    result = text_element.color
    expected = (0.7, 0.7, 0.7)
    assert result == expected, f"Expected {expected}, got {result}"


def test_existing_cmyk_still_works():
    """Test that CMYK format still converts to approximate RGB."""
    mock_obj = {
        "non_stroking_color": (0.1, 0.2, 0.3, 0.4),  # CMYK format
        "size": 11,
        "fontname": "BCDEEE+Aptos",
    }

    text_element = Mock()
    text_element._obj = mock_obj
    text_element.color = TextElement.color.__get__(text_element, TextElement)

    result = text_element.color
    # Should be converted from CMYK to RGB
    assert len(result) == 3, f"Should return RGB tuple, got {result}"
    assert all(isinstance(v, float) for v in result), f"Should be float values, got {result}"


def test_fallback_to_black_for_unknown_format():
    """Test that unknown formats still fall back to black."""
    mock_obj = {
        "non_stroking_color": "unknown",  # Unknown format
        "size": 11,
        "fontname": "BCDEEE+Aptos",
    }

    text_element = Mock()
    text_element._obj = mock_obj
    text_element.color = TextElement.color.__get__(text_element, TextElement)

    result = text_element.color
    expected = (0, 0, 0)  # Should fall back to black
    assert result == expected, f"Expected {expected}, got {result}"


def test_missing_color_defaults_to_black():
    """Test that missing color information defaults to black."""
    mock_obj = {
        # No 'non_stroking_color' key
        "size": 11,
        "fontname": "BCDEEE+Aptos",
    }

    text_element = Mock()
    text_element._obj = mock_obj
    text_element.color = TextElement.color.__get__(text_element, TextElement)

    result = text_element.color
    expected = (0, 0, 0)  # Should default to black
    assert result == expected, f"Expected {expected}, got {result}"


def simulate_hex_conversion(rgb_tuple):
    """Simulate the RGB to hex conversion that happens in describe/base.py."""
    if isinstance(rgb_tuple, (tuple, list)) and len(rgb_tuple) >= 3:
        try:
            if all(isinstance(v, (int, float)) for v in rgb_tuple[:3]):
                r, g, b = [int(v * 255) if v <= 1 else int(v) for v in rgb_tuple[:3]]
                return f"#{r:02x}{g:02x}{b:02x}"
        except:
            pass
    return str(rgb_tuple)


def test_full_color_pipeline():
    """Test the full color pipeline from pdfplumber to hex."""

    # Test the specific case from the user's issue
    test_cases = [
        ((0.4,), "#666666"),  # Gray 40% -> should be gray, not black
        ((0,), "#000000"),  # Black -> should be black
        ((1,), "#ffffff"),  # White -> should be white
    ]

    for pdfplumber_color, expected_hex in test_cases:
        # Step 1: TextElement.color converts pdfplumber format to RGB
        mock_obj = {"non_stroking_color": pdfplumber_color, "size": 11, "fontname": "BCDEEE+Aptos"}

        text_element = Mock()
        text_element._obj = mock_obj
        text_element.color = TextElement.color.__get__(text_element, TextElement)

        rgb_result = text_element.color

        # Step 2: describe/base.py converts RGB to hex
        hex_result = simulate_hex_conversion(rgb_result)

        assert (
            hex_result == expected_hex
        ), f"Pipeline {pdfplumber_color} -> {rgb_result} -> {hex_result}, expected {expected_hex}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
