"""
Utility functions for color formatting and conversion.
"""

from typing import Any, List, Optional, Tuple, Union

# List of known color attribute names in natural-pdf
COLOR_ATTRIBUTES = [
    "color",
    "fill",
    "stroke",
    "non_stroking_color",
    "stroking_color",
    "text_color",
    "background_color",
    "highlight_color",
    "border_color",
]


def rgb_to_hex(color: Union[Tuple[float, ...], List[float]]) -> str:
    """
    Convert an RGB/RGBA color tuple to hex string.

    Args:
        color: RGB tuple with values either in [0,1] or [0,255] range
               Can be RGB (3 values) or RGBA (4 values)

    Returns:
        Hex color string (e.g., '#ff0000' for red)
    """
    if not isinstance(color, (tuple, list)) or len(color) < 3:
        raise ValueError(f"Invalid color format: {color}")

    # Take first 3 values (RGB), ignore alpha if present
    r, g, b = color[:3]

    # Determine if values are in [0,1] or [0,255] range
    # If any positive value is > 1, assume [0,255] range
    max_val = max(abs(r), abs(g), abs(b))

    if max_val > 1:
        # Values are in 0-255 range
        r_int = int(min(255, max(0, r)))
        g_int = int(min(255, max(0, g)))
        b_int = int(min(255, max(0, b)))
    else:
        # Values are in 0-1 range, convert to 0-255
        r_int = int(min(255, max(0, r * 255)))
        g_int = int(min(255, max(0, g * 255)))
        b_int = int(min(255, max(0, b * 255)))

    return f"#{r_int:02x}{g_int:02x}{b_int:02x}"


def is_color_attribute(attr_name: str) -> bool:
    """
    Check if an attribute name is a known color attribute.

    Args:
        attr_name: The attribute name to check

    Returns:
        True if this is a known color attribute
    """
    return attr_name.lower() in [attr.lower() for attr in COLOR_ATTRIBUTES]


def format_color_value(value: Any, attr_name: Optional[str] = None) -> str:
    """
    Format a color value for display, converting tuples to hex when appropriate.

    Args:
        value: The value to format
        attr_name: Optional attribute name to help determine if this is a color

    Returns:
        Formatted string representation
    """
    # If attr_name is provided and it's not a color attribute, return as-is
    if attr_name and not is_color_attribute(attr_name):
        return str(value)

    # Check if value looks like an RGB color tuple
    if isinstance(value, (tuple, list)):
        # Must have 3 or 4 values (RGB or RGBA)
        if len(value) in (3, 4):
            # Check if all values are numeric
            if all(isinstance(v, (int, float)) for v in value):
                # Additional validation: values should be in reasonable ranges
                # Either all in [0,1] or all in [0,255]
                if all(0 <= v <= 1 for v in value[:3]) or all(0 <= v <= 255 for v in value[:3]):
                    try:
                        return rgb_to_hex(value)
                    except Exception:
                        # If conversion fails, fall back to string representation
                        pass

    # Default: convert to string
    return str(value)
