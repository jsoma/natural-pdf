"""
Visualization utilities for natural-pdf.
"""

import io
import itertools  # Added for cycling
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pypdfium2
from PIL import Image, ImageDraw, ImageFont

# Define a base list of visually distinct colors for highlighting
# Format: (R, G, B)
_BASE_HIGHLIGHT_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Navy
    (255, 215, 0),  # Gold
    (75, 0, 130),  # Indigo
    (240, 128, 128),  # Light Coral
    (32, 178, 170),  # Light Sea Green
    (138, 43, 226),  # Blue Violet
    (160, 82, 45),  # Sienna
]

# Default Alpha for highlight fills
DEFAULT_FILL_ALPHA = 100

# Add quantitative color mapping functionality
import matplotlib.cm as cm
import matplotlib.pyplot as plt


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

    def get_color(
        self, label: Optional[str] = None, force_cycle: bool = False
    ) -> Tuple[int, int, int, int]:
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


def create_legend(
    labels_colors: Dict[str, Tuple[int, int, int, int]], width: int = 200, item_height: int = 30
) -> Image.Image:
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
    legend = Image.new("RGBA", (width, height), (255, 255, 255, 255))
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
            alpha = 255  # Assume opaque if alpha is missing
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


def create_colorbar(
    values: List[float],
    colormap: str = "viridis",
    bins: Optional[Union[int, List[float]]] = None,
    width: int = 80,
    height: int = 20,
    orientation: str = "horizontal",
) -> Image.Image:
    """
    Create a color bar for quantitative data visualization.

    Args:
        values: List of numeric values to create color bar for
        colormap: Name of the matplotlib colormap to use
        bins: Optional binning specification (int for equal bins, list for custom bins)
        width: Width of the color bar
        height: Height of the color bar
        orientation: 'horizontal' or 'vertical'

    Returns:
        PIL Image with the color bar
    """
    import numpy as np

    # Get value range
    vmin = min(values)
    vmax = max(values)

    if vmin == vmax:
        # Handle edge case where all values are the same
        vmax = vmin + 1

    # Create the colorbar image
    if orientation == "horizontal":
        bar_width = width - 40  # Leave space for labels (reduced from 60)
        bar_height = height
        total_width = width
        total_height = height + 40  # Extra space for labels
    else:
        bar_width = width
        bar_height = max(height, 120)  # Ensure minimum height for vertical colorbar
        total_width = width + 80  # Extra space for labels (increased for larger text)
        total_height = bar_height + 60  # Extra space for labels

    # Create base image
    img = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to load a font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except IOError:
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except IOError:
            # Load default font but try to get a larger size
            try:
                font = ImageFont.load_default(size=16)
            except:
                font = ImageFont.load_default()

    # Draw the color blocks (5 discrete blocks)
    if orientation == "horizontal":
        # Create 5 discrete color blocks
        num_blocks = 5
        block_width = bar_width // num_blocks

        for i in range(num_blocks):
            # Calculate value for this block (center of block)
            block_start = i / num_blocks
            block_center = (i + 0.5) / num_blocks
            value = vmin + block_center * (vmax - vmin)

            # Get color for this block
            rgb = get_colormap_color(colormap, value, vmin, vmax)
            color = (*rgb, 255)

            # Calculate block position
            x_start = 20 + i * block_width
            x_end = 20 + (i + 1) * block_width

            # Draw filled rectangle for this block
            draw.rectangle(
                [(x_start, 10), (x_end, 10 + bar_height)],
                fill=color,
                outline=(0, 0, 0, 255),
                width=1,
            )

        # Add value labels
        if bins is not None:
            # Show bin boundaries
            if isinstance(bins, int):
                # Equal-width bins
                step = (vmax - vmin) / bins
                tick_values = [vmin + i * step for i in range(bins + 1)]
            else:
                # Custom bins
                tick_values = bins

            for tick_val in tick_values:
                if vmin <= tick_val <= vmax:
                    x_pos = int(20 + (tick_val - vmin) / (vmax - vmin) * bar_width)
                    # Draw tick mark
                    draw.line(
                        [(x_pos, 10 + bar_height), (x_pos, 10 + bar_height + 5)],
                        fill=(0, 0, 0, 255),
                        width=1,
                    )
                    # Draw label
                    label_text = f"{tick_val:.2f}".rstrip("0").rstrip(".")
                    text_bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    draw.text(
                        (x_pos - text_width // 2, 10 + bar_height + 8),
                        label_text,
                        fill=(0, 0, 0, 255),
                        font=font,
                    )
        else:
            # Show min and max values
            # Min value
            min_text = f"{vmin:.2f}".rstrip("0").rstrip(".")
            draw.text((20, 10 + bar_height + 8), min_text, fill=(0, 0, 0, 255), font=font)

            # Max value
            max_text = f"{vmax:.2f}".rstrip("0").rstrip(".")
            text_bbox = draw.textbbox((0, 0), max_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text(
                (20 + bar_width - text_width, 10 + bar_height + 8),
                max_text,
                fill=(0, 0, 0, 255),
                font=font,
            )

    else:  # vertical orientation
        # Create 5 discrete color blocks
        num_blocks = 5
        block_height = bar_height // num_blocks

        for i in range(num_blocks):
            # Calculate value for this block (center of block, top = max, bottom = min)
            block_center = (i + 0.5) / num_blocks
            value = vmax - block_center * (vmax - vmin)

            # Get color for this block
            rgb = get_colormap_color(colormap, value, vmin, vmax)
            color = (*rgb, 255)

            # Calculate block position
            y_start = 30 + i * block_height
            y_end = 30 + (i + 1) * block_height

            # Draw filled rectangle for this block
            draw.rectangle(
                [(10, y_start), (10 + bar_width, y_end)],
                fill=color,
                outline=(0, 0, 0, 255),
                width=1,
            )

        # Add value labels
        if bins is not None:
            # Show bin boundaries
            if isinstance(bins, int):
                # Equal-width bins
                step = (vmax - vmin) / bins
                tick_values = [vmin + i * step for i in range(bins + 1)]
            else:
                # Custom bins
                tick_values = bins

            for tick_val in tick_values:
                if vmin <= tick_val <= vmax:
                    y_pos = int(30 + (vmax - tick_val) / (vmax - vmin) * bar_height)
                    # Draw tick mark
                    draw.line(
                        [(10 + bar_width, y_pos), (10 + bar_width + 5, y_pos)],
                        fill=(0, 0, 0, 255),
                        width=1,
                    )
                    # Draw label
                    label_text = f"{tick_val:.2f}".rstrip("0").rstrip(".")
                    draw.text(
                        (10 + bar_width + 8, y_pos - 6), label_text, fill=(0, 0, 0, 255), font=font
                    )
        else:
            # Show min and max values
            # Max value (top)
            max_text = f"{vmax:.2f}".rstrip("0").rstrip(".")
            draw.text((10 + bar_width + 8, 30 - 6), max_text, fill=(0, 0, 0, 255), font=font)

            # Min value (bottom)
            min_text = f"{vmin:.2f}".rstrip("0").rstrip(".")
            draw.text(
                (10 + bar_width + 8, 30 + bar_height - 6), min_text, fill=(0, 0, 0, 255), font=font
            )

    return img


def merge_images_with_legend(
    image: Image.Image, legend: Image.Image, position: str = "right"
) -> Image.Image:
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
        return image  # Return original image if legend is None or empty

    bg_color = (255, 255, 255, 255)  # Always use white for the merged background
    bg_color = (255, 255, 255, 255)  # Always use white for the merged background

    if position == "right":
        # Create a new image with extra width for the legend
        merged_width = image.width + legend.width
        merged_height = max(image.height, legend.height)
        merged = Image.new("RGBA", (merged_width, merged_height), bg_color)
        merged.paste(image, (0, 0))
        merged.paste(
            legend, (image.width, 0), legend if legend.mode == "RGBA" else None
        )  # Handle transparency
    elif position == "bottom":
        # Create a new image with extra height for the legend
        merged_width = max(image.width, legend.width)
        merged_height = image.height + legend.height
        merged = Image.new("RGBA", (merged_width, merged_height), bg_color)
        merged.paste(image, (0, 0))
        merged.paste(legend, (0, image.height), legend if legend.mode == "RGBA" else None)
    elif position == "top":
        # Create a new image with extra height for the legend
        merged_width = max(image.width, legend.width)
        merged_height = image.height + legend.height
        merged = Image.new("RGBA", (merged_width, merged_height), bg_color)
        merged.paste(legend, (0, 0), legend if legend.mode == "RGBA" else None)
        merged.paste(image, (0, legend.height))
    elif position == "left":
        # Create a new image with extra width for the legend
        merged_width = image.width + legend.width
        merged_height = max(image.height, legend.height)
        merged = Image.new("RGBA", (merged_width, merged_height), bg_color)
        merged.paste(legend, (0, 0), legend if legend.mode == "RGBA" else None)
        merged.paste(image, (legend.width, 0))
    else:
        # Invalid position, return the original image
        print(f"Warning: Invalid legend position '{position}'. Returning original image.")
        merged = image

    return merged


def render_plain_page(page, resolution):
    """
    Render a page to PIL Image using the specified resolution.

    Args:
        page: Page object to render
        resolution: DPI resolution for rendering

    Returns:
        PIL Image of the rendered page
    """
    doc = pypdfium2.PdfDocument(page._page.pdf.stream)

    pdf_page = doc[page.index]

    # Convert resolution (DPI) to scale factor for pypdfium2
    # PDF standard is 72 DPI, so scale = resolution / 72
    scale_factor = resolution / 72.0

    bitmap = pdf_page.render(
        scale=scale_factor,
    )
    image = bitmap.to_pil().convert("RGB")

    pdf_page.close()
    doc.close()

    return image


def detect_quantitative_data(values: List[Any]) -> bool:
    """
    Detect if a list of values represents quantitative data suitable for gradient coloring.

    Args:
        values: List of attribute values from elements

    Returns:
        True if data appears to be quantitative, False otherwise
    """
    # Filter out None values
    numeric_values = []
    for v in values:
        if v is not None:
            try:
                # Try to convert to float
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                # Not numeric, likely categorical
                pass

    # If we have fewer than 2 numeric values, treat as categorical
    if len(numeric_values) < 2:
        return False

    # If more than 80% of values are numeric and we have >8 unique values, treat as quantitative
    numeric_ratio = len(numeric_values) / len(values)
    unique_values = len(set(numeric_values))

    return numeric_ratio > 0.8 and unique_values > 8


def get_colormap_color(
    colormap_name: str, value: float, vmin: float, vmax: float
) -> Tuple[int, int, int]:
    """
    Get a color from a matplotlib colormap based on a normalized value.

    Args:
        colormap_name: Name of the colormap ('viridis', 'plasma', etc.)
        value: The value to map to a color
        vmin: Minimum value in the data range
        vmax: Maximum value in the data range

    Returns:
        RGB color tuple (0-255)
    """
    # Try to get the colormap from matplotlib
    try:
        cmap = cm.get_cmap(colormap_name)
    except (ValueError, KeyError):
        # Fallback to viridis if colormap doesn't exist
        cmap = cm.get_cmap("viridis")

    # Normalize value to [0, 1]
    if vmax == vmin:
        t = 0.0
    else:
        t = (value - vmin) / (vmax - vmin)

    # Clamp to [0, 1]
    t = max(0.0, min(1.0, t))

    # Get RGBA color from matplotlib (values are 0-1)
    rgba = cmap(t)

    # Convert to 0-255 RGB
    r = int(rgba[0] * 255)
    g = int(rgba[1] * 255)
    b = int(rgba[2] * 255)

    return (r, g, b)


def apply_bins_to_values(
    values: List[float], bins: Union[int, List[float]]
) -> Tuple[List[str], List[float]]:
    """
    Apply binning to quantitative values.

    Args:
        values: List of numeric values
        bins: Either number of bins (int) or list of bin edges (List[float])

    Returns:
        Tuple of (bin_labels, bin_values) where bin_values are the centers of bins
    """
    if isinstance(bins, int):
        # Equal-width bins
        min_val = min(values)
        max_val = max(values)
        bin_edges = [min_val + i * (max_val - min_val) / bins for i in range(bins + 1)]
    else:
        # Custom bin edges
        bin_edges = sorted(bins)

    # Create bin labels and centers
    bin_labels = []
    bin_centers = []
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        bin_labels.append(f"{start:.2f}-{end:.2f}")
        bin_centers.append((start + end) / 2)

    return bin_labels, bin_centers


def create_quantitative_color_mapping(
    values: List[Any], colormap: str = "viridis", bins: Optional[Union[int, List[float]]] = None
) -> Dict[Any, Tuple[int, int, int, int]]:
    """
    Create a color mapping for quantitative data using matplotlib colormaps.

    Args:
        values: List of values to map to colors
        colormap: Name of any matplotlib colormap (e.g., 'viridis', 'plasma', 'inferno',
                 'magma', 'coolwarm', 'RdBu', 'tab10', etc.). See matplotlib.cm for full list.
        bins: Optional binning specification (int for equal-width bins, list for custom bins)

    Returns:
        Dictionary mapping values to RGBA colors
    """
    # Convert to numeric values, filtering out None/non-numeric
    numeric_values = []
    value_to_numeric = {}

    for v in values:
        if v is not None:
            try:
                numeric_val = float(v)
                numeric_values.append(numeric_val)
                value_to_numeric[v] = numeric_val
            except (ValueError, TypeError):
                pass

    if not numeric_values:
        # Fallback to categorical if no numeric values
        return {}

    # Determine min/max for normalization
    vmin = min(numeric_values)
    vmax = max(numeric_values)

    # Apply binning if specified
    if bins is not None:
        bin_labels, bin_centers = apply_bins_to_values(numeric_values, bins)
        # Create mapping from original values to bin centers
        result = {}
        for orig_val, numeric_val in value_to_numeric.items():
            # Find which bin this value belongs to
            if isinstance(bins, int):
                bin_width = (vmax - vmin) / bins
                bin_idx = min(int((numeric_val - vmin) / bin_width), bins - 1)
            else:
                bin_idx = 0
                for i, edge in enumerate(bins[1:], 1):
                    if numeric_val <= edge:
                        bin_idx = i - 1
                        break
                else:
                    bin_idx = len(bins) - 2

            # Get color for this bin center
            bin_center = bin_centers[bin_idx]
            rgb = get_colormap_color(colormap, bin_center, vmin, vmax)
            result[orig_val] = (*rgb, DEFAULT_FILL_ALPHA)

        return result
    else:
        # Continuous gradient mapping
        result = {}
        for orig_val, numeric_val in value_to_numeric.items():
            rgb = get_colormap_color(colormap, numeric_val, vmin, vmax)
            result[orig_val] = (*rgb, DEFAULT_FILL_ALPHA)

        return result
