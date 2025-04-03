"""
Visualization utilities for natural-pdf.
"""
from typing import List, Dict, Tuple, Optional, Union, Any, Set
import io
import math
import random
import itertools # Added for cycling
from PIL import Image, ImageDraw, ImageFont

# Define a base list of visually distinct colors for highlighting
# Format: (R, G, B)
_BASE_HIGHLIGHT_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Navy
    (255, 215, 0),    # Gold
    (75, 0, 130),     # Indigo
    (240, 128, 128),  # Light Coral
    (32, 178, 170),   # Light Sea Green
    (138, 43, 226),   # Blue Violet
    (160, 82, 45),    # Sienna
]

# Default Alpha for highlight fills
DEFAULT_FILL_ALPHA = 100

class ColorManager:
    """
    Manages color assignment for highlights, ensuring consistency for labels.
    """
    def __init__(self, alpha: int = DEFAULT_FILL_ALPHA):
        """
        Initializes the ColorManager.

        Args:
            alpha (int): The default alpha transparency (0-255) for highlight fills.
        """
        self._alpha = alpha
        # Shuffle the base colors to avoid the same sequence every time
        self._available_colors = random.sample(_BASE_HIGHLIGHT_COLORS, len(_BASE_HIGHLIGHT_COLORS))
        self._color_cycle = itertools.cycle(self._available_colors)
        self._labels_colors: Dict[str, Tuple[int, int, int, int]] = {}

    def _get_rgba_color(self, rgb: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Applies the instance's alpha to an RGB tuple."""
        return (*rgb, self._alpha)

    def get_color(self, label: Optional[str] = None, 
                  force_cycle: bool = False) -> Tuple[int, int, int, int]:
        """
        Gets an RGBA color tuple.

        If a label is provided, it returns a consistent color for that label.
        If no label is provided, it cycles through the available colors (unless force_cycle=False).
        If force_cycle is True, it always returns the next color in the cycle, ignoring the label.

        Args:
            label (Optional[str]): The label associated with the highlight.
            force_cycle (bool): If True, ignore the label and always get the next cycle color.

        Returns:
            Tuple[int, int, int, int]: An RGBA color tuple (0-255).
        """
        if force_cycle:
            # Always get the next color, don't store by label
            rgb = next(self._color_cycle)
            return self._get_rgba_color(rgb)
            
        if label is not None:
            if label in self._labels_colors:
                # Return existing color for this label
                return self._labels_colors[label]
            else:
                # New label, get next color and store it
                rgb = next(self._color_cycle)
                rgba = self._get_rgba_color(rgb)
                self._labels_colors[label] = rgba
                return rgba
        else:
            # No label and not forced cycle - get next color from cycle
            rgb = next(self._color_cycle)
            return self._get_rgba_color(rgb)

    def get_label_colors(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Returns the current mapping of labels to colors."""
        return self._labels_colors.copy()

    def reset(self) -> None:
        """Resets the color cycle and clears the label-to-color mapping."""
        # Re-shuffle and reset the cycle
        self._available_colors = random.sample(_BASE_HIGHLIGHT_COLORS, len(_BASE_HIGHLIGHT_COLORS))
        self._color_cycle = itertools.cycle(self._available_colors)
        self._labels_colors = {}

# --- Global color state and functions removed --- 
# HIGHLIGHT_COLORS, _color_cycle, _current_labels_colors, _used_colors_iterator
# get_next_highlight_color(), reset_highlight_colors()

def create_legend(labels_colors: Dict[str, Tuple[int, int, int, int]], 
                 width: int = 200, 
                 item_height: int = 30) -> Image.Image:
    """
    Create a legend image for the highlighted elements.
    
    Args:
        labels_colors: Dictionary mapping labels to colors
        width: Width of the legend image
        item_height: Height of each legend item
        
    Returns:
        PIL Image with the legend
    """
    # Calculate the height based on the number of labels
    height = len(labels_colors) * item_height + 10  # 10px padding
    
    # Create a white image
    legend = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(legend)
    
    # Try to load a font, use default if not available
    try:
        # Use a commonly available font, adjust size
        font = ImageFont.truetype("DejaVuSans.ttf", 14) 
    except IOError:
        try:
             font = ImageFont.truetype("Arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()
    
    # Draw each legend item
    y = 5  # Start with 5px padding
    for label, color in labels_colors.items():
        # Get the color components
        # Handle potential case where alpha isn't provided (use default 255)
        if len(color) == 3:
            r, g, b = color
            alpha = 255 # Assume opaque if alpha is missing
        else:
            r, g, b, alpha = color
        
        # Calculate the apparent color when drawn on white background
        # Alpha blending formula: result = (source * alpha) + (dest * (1-alpha))
        # Where alpha is normalized to 0-1 range
        alpha_norm = alpha / 255.0
        apparent_r = int(r * alpha_norm + 255 * (1 - alpha_norm))
        apparent_g = int(g * alpha_norm + 255 * (1 - alpha_norm))
        apparent_b = int(b * alpha_norm + 255 * (1 - alpha_norm))
        
        # Use solid color that matches the apparent color of the semi-transparent highlight
        legend_color = (apparent_r, apparent_g, apparent_b, 255)
        
        # Draw the color box
        draw.rectangle([(10, y), (30, y + item_height - 5)], fill=legend_color)
        
        # Draw the label text
        draw.text((40, y + (item_height // 2) - 6), label, fill=(0, 0, 0, 255), font=font)
        
        # Move to the next position
        y += item_height
    
    return legend

def merge_images_with_legend(image: Image.Image, 
                            legend: Image.Image, 
                            position: str = 'right') -> Image.Image:
    """
    Merge an image with a legend.
    
    Args:
        image: Main image
        legend: Legend image
        position: Position of the legend ('right', 'bottom', 'top', 'left')
        
    Returns:
        Merged image
    """
    if not legend:
        return image # Return original image if legend is None or empty
        
    # Determine background color from top-left pixel (safer than assuming white)
    bg_color = image.getpixel((0,0)) if image.mode == 'RGBA' else (255, 255, 255, 255)

    if position == 'right':
        # Create a new image with extra width for the legend
        merged_width = image.width + legend.width
        merged_height = max(image.height, legend.height)
        merged = Image.new('RGBA', (merged_width, merged_height), bg_color)
        merged.paste(image, (0, 0))
        merged.paste(legend, (image.width, 0), legend if legend.mode == 'RGBA' else None) # Handle transparency
    elif position == 'bottom':
        # Create a new image with extra height for the legend
        merged_width = max(image.width, legend.width)
        merged_height = image.height + legend.height
        merged = Image.new('RGBA', (merged_width, merged_height), bg_color)
        merged.paste(image, (0, 0))
        merged.paste(legend, (0, image.height), legend if legend.mode == 'RGBA' else None)
    elif position == 'top':
        # Create a new image with extra height for the legend
        merged_width = max(image.width, legend.width)
        merged_height = image.height + legend.height
        merged = Image.new('RGBA', (merged_width, merged_height), bg_color)
        merged.paste(legend, (0, 0), legend if legend.mode == 'RGBA' else None)
        merged.paste(image, (0, legend.height))
    elif position == 'left':
        # Create a new image with extra width for the legend
        merged_width = image.width + legend.width
        merged_height = max(image.height, legend.height)
        merged = Image.new('RGBA', (merged_width, merged_height), bg_color)
        merged.paste(legend, (0, 0), legend if legend.mode == 'RGBA' else None)
        merged.paste(image, (legend.width, 0))
    else:
        # Invalid position, return the original image
        print(f"Warning: Invalid legend position '{position}'. Returning original image.")
        merged = image
    
    return merged