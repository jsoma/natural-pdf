"""Test demonstrating the exact pdfminer.six color stack bug."""

import pytest

from natural_pdf import PDF


def test_pdfminer_color_stack_bug_explanation():
    """
    Demonstrate exactly how pdfminer.six misparses colors.

    When the PDF contains: 1 1 0 sc

    What should happen (RGB interpretation):
    1. Push 1, 1, 0 onto stack -> stack = [1, 1, 0]
    2. sc command with DeviceRGB (3 components)
    3. Pop 3 values -> [1, 1, 0]
    4. Result: RGB(1, 1, 0) = yellow ✓

    What actually happens (pdfminer bug):
    1. Push 1, 1, 0 onto stack -> stack = [1, 1, 0]
    2. sc command with DeviceGray (1 component) - wrong default!
    3. Pop 1 value from end -> [0]
    4. Result: Gray(0) = black ✗

    The bug has two parts:
    1. Wrong default color space (DeviceGray instead of DeviceRGB)
    2. Stack is LIFO, so pop(1) gets the last value (0) not first (1)
    """
    pdf = PDF("pdfs/types-of-type.pdf")
    page = pdf.pages[0]

    # Find the yellow highlight
    highlighted = page.find('text:contains("Highlighted text")')
    assert highlighted is not None
    assert highlighted.is_highlighted is True

    # Check the color
    color = highlighted.highlight_color

    # Check if patch is active
    from natural_pdf.utils.pdfminer_patches import _patches_applied

    if _patches_applied:
        # With our patch, it correctly shows yellow
        assert color == (1.0, 1.0, 0.0)
    else:
        # Without patch, bug causes it to show black
        assert color == 0.0

    # Document other rectangles for comparison
    all_rects = page._page.rects

    # Count how many have tuple colors (correct) vs float (bug)
    tuple_colors = [r for r in all_rects if isinstance(r.get("non_stroking_color"), tuple)]
    float_colors = [r for r in all_rects if isinstance(r.get("non_stroking_color"), float)]

    # Most rectangles use explicit color space and work correctly
    assert len(tuple_colors) > 0  # These work

    if _patches_applied:
        # With patch, all colors should be tuples (no more float bug)
        assert len(float_colors) == 0
    else:
        # Without patch, some would have the float bug
        assert len(float_colors) > 0

    # The bug only affects colors set without explicit color space
    # '/Cs1 cs 0 0 0 sc' -> (0.0, 0.0, 0.0) ✓
    # '1 1 0 sc' -> 0.0 ✗
