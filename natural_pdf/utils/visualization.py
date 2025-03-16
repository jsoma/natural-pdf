"""
Visualization utilities for natural-pdf.
"""
from typing import List, Dict, Tuple, Optional, Union, Any
import io
import math
import random
from PIL import Image, ImageDraw, ImageFont

# Define a list of visually distinct colors for highlighting
# Format: (R, G, B, alpha)
HIGHLIGHT_COLORS = [
    (255, 255, 0, 100),    # Yellow (semi-transparent)
    (255, 0, 0, 100),      # Red (semi-transparent)
    (0, 255, 0, 100),      # Green (semi-transparent)
    (0, 0, 255, 100),      # Blue (semi-transparent)
    (255, 0, 255, 100),    # Magenta (semi-transparent)
    (0, 255, 255, 100),    # Cyan (semi-transparent)
    (255, 165, 0, 100),    # Orange (semi-transparent)
    (128, 0, 128, 100),    # Purple (semi-transparent)
    (0, 128, 0, 100),      # Dark Green (semi-transparent)
    (0, 0, 128, 100),      # Navy (semi-transparent)
]

# Keep track of the next color to use
_next_color_index = 0

def get_next_highlight_color() -> Tuple[int, int, int, int]:
    """
    Get the next highlight color in the cycle.
    
    Returns:
        Tuple of (R, G, B, alpha) values
    """
    global _next_color_index
    color = HIGHLIGHT_COLORS[_next_color_index % len(HIGHLIGHT_COLORS)]
    _next_color_index += 1
    return color

def reset_highlight_colors():
    """Reset the highlight color cycle."""
    global _next_color_index
    _next_color_index = 0

def get_random_highlight_color() -> Tuple[int, int, int, int]:
    """
    Get a random highlight color.
    
    Returns:
        Tuple of (R, G, B, alpha) values
    """
    return random.choice(HIGHLIGHT_COLORS)

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
        font = ImageFont.truetype("Arial", 12)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw each legend item
    y = 5  # Start with 5px padding
    for label, color in labels_colors.items():
        # Get the color components
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
        draw.text((40, y + item_height // 4), label, fill=(0, 0, 0, 255), font=font)
        
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
    if position == 'right':
        # Create a new image with extra width for the legend
        merged = Image.new('RGBA', (image.width + legend.width, max(image.height, legend.height)), 
                          (255, 255, 255, 255))
        merged.paste(image, (0, 0))
        merged.paste(legend, (image.width, 0))
    elif position == 'bottom':
        # Create a new image with extra height for the legend
        merged = Image.new('RGBA', (max(image.width, legend.width), image.height + legend.height), 
                          (255, 255, 255, 255))
        merged.paste(image, (0, 0))
        merged.paste(legend, (0, image.height))
    elif position == 'top':
        # Create a new image with extra height for the legend
        merged = Image.new('RGBA', (max(image.width, legend.width), image.height + legend.height),
                          (255, 255, 255, 255))
        merged.paste(legend, (0, 0))
        merged.paste(image, (0, legend.height))
    elif position == 'left':
        # Create a new image with extra width for the legend
        merged = Image.new('RGBA', (image.width + legend.width, max(image.height, legend.height)),
                          (255, 255, 255, 255))
        merged.paste(legend, (0, 0))
        merged.paste(image, (legend.width, 0))
    else:
        # Invalid position, return the original image
        merged = image
    
    return merged