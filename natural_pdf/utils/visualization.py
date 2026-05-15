"""
Visualization utilities for natural-pdf.
"""

import itertools  # Added for cycling
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

logger = logging.getLogger(__name__)

try:
    import pypdfium2  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    pypdfium2 = None  # type: ignore[assignment]
from PIL import Image, ImageDraw, ImageFont

from natural_pdf.utils.locks import pdf_render_lock


class DirectCropRenderUnsupportedError(RuntimeError):
    """Raised when a page cannot use the direct cropped-render fast path."""


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

LEGEND_FONT_SIZE = 14
LEGEND_MIN_WIDTH = 180
LEGEND_MAX_WIDTH = 340
LEGEND_SIDE_WIDTH_RATIO = 0.35

# Quantitative color mapping (matplotlib imported lazily in get_colormap_color)


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
        self._available_colors = list(_BASE_HIGHLIGHT_COLORS)
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
        self._available_colors = list(_BASE_HIGHLIGHT_COLORS)
        self._color_cycle = itertools.cycle(self._available_colors)
        self._labels_colors = {}


# --- Global color state and functions removed ---
# HIGHLIGHT_COLORS, _color_cycle, _current_labels_colors, _used_colors_iterator
# get_next_highlight_color(), reset_highlight_colors()


def create_legend(
    labels_colors: Mapping[str, Sequence[int]],
    width: int = 250,
    *,
    max_height: Optional[int] = None,
    font_size: int = LEGEND_FONT_SIZE,
) -> Image.Image:
    """
    Create a legend image for the highlighted elements.

    Supports multi-line labels (labels containing ``\\n``).  Each item's
    height is computed dynamically from its text content.

    Args:
        labels_colors: Dictionary mapping labels to colors
        width: Width of the legend image

    Returns:
        PIL Image with the legend
    """
    return create_wrapped_legend(
        labels_colors, width=width, max_height=max_height, font_size=font_size
    )


def _load_legend_font(size: int = LEGEND_FONT_SIZE) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf", "FreeSans.ttf"):
        try:
            return cast(ImageFont.ImageFont, ImageFont.truetype(name, size))
        except (IOError, OSError):
            continue
    return cast(ImageFont.ImageFont, ImageFont.load_default())


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _split_long_token(
    token: str,
    *,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: int,
) -> List[str]:
    if _text_width(draw, token, font) <= max_width:
        return [token]

    parts: List[str] = []
    current = ""
    for char in token:
        candidate = current + char
        if current and _text_width(draw, candidate, font) > max_width:
            parts.append(current)
            current = char
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts or [token]


def _wrap_label_text(
    label: str,
    *,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    lines: List[str] = []
    for paragraph in str(label).splitlines() or [""]:
        prefix = paragraph[: len(paragraph) - len(paragraph.lstrip())]
        words = paragraph.strip().split()
        if not words:
            lines.append("")
            continue

        current = prefix
        for word in words:
            pieces = _split_long_token(word, draw=draw, font=font, max_width=max_width)
            for piece in pieces:
                separator = "" if current == prefix else " "
                candidate = f"{current}{separator}{piece}"
                if current != prefix and _text_width(draw, candidate, font) > max_width:
                    lines.append(current)
                    current = f"{prefix}{piece}"
                else:
                    current = candidate
        lines.append(current)
    return "\n".join(lines)


def _legend_swatch_color(color: Sequence[int]) -> Tuple[int, int, int, int]:
    if len(color) == 3:
        r, g, b = cast(Tuple[int, int, int], tuple(color))  # type: ignore[misc]
        alpha = 255
    elif len(color) >= 4:
        r, g, b, alpha = cast(Tuple[int, int, int, int], tuple(color[:4]))  # type: ignore[misc]
    else:
        raise ValueError("Color sequences must have at least three components.")

    alpha_norm = alpha / 255.0
    apparent_r = int(r * alpha_norm + 255 * (1 - alpha_norm))
    apparent_g = int(g * alpha_norm + 255 * (1 - alpha_norm))
    apparent_b = int(b * alpha_norm + 255 * (1 - alpha_norm))
    return (apparent_r, apparent_g, apparent_b, 255)


def pack_legend_columns(
    item_heights: Sequence[Union[int, float]],
    *,
    max_height: Optional[Union[int, float]],
    item_gap: Union[int, float],
    padding_top: Union[int, float] = 0,
    padding_bottom: Union[int, float] = 0,
) -> List[List[int]]:
    """Pack legend row indexes into columns constrained by height."""
    if not item_heights:
        return []
    if max_height is None:
        return [list(range(len(item_heights)))]

    available_height = max(0.0, float(max_height) - float(padding_top) - float(padding_bottom))
    gap = float(item_gap)
    columns: List[List[int]] = []
    current: List[int] = []
    used_height = 0.0

    for idx, raw_height in enumerate(item_heights):
        item_height = max(0.0, float(raw_height))
        needed_height = item_height if not current else gap + item_height
        if current and used_height + needed_height > available_height:
            columns.append(current)
            current = [idx]
            used_height = item_height
        else:
            current.append(idx)
            used_height += needed_height

    if current:
        columns.append(current)
    return columns


def create_wrapped_legend(
    labels_colors: Mapping[str, Sequence[int]],
    width: int = 250,
    *,
    max_height: Optional[int] = None,
    font_size: int = LEGEND_FONT_SIZE,
) -> Image.Image:
    """Create a measured legend with wrapped labels and height-aware columns."""
    width = max(1, int(width))
    font = _load_legend_font(font_size)

    padding_top = 8
    padding_bottom = 8
    padding_right = 10
    item_gap = 8  # vertical gap between items
    swatch_size = 15
    swatch_x = 10
    text_x = 36  # x position for text (after swatch)
    line_spacing = 4  # spacing between lines in multiline text
    text_width = max(1, width - text_x - padding_right)

    _tmp = Image.new("RGBA", (1, 1))
    _tmp_draw = ImageDraw.Draw(_tmp)

    measured_items: List[Tuple[str, Sequence[int], int]] = []
    for label, color in labels_colors.items():
        wrapped_label = _wrap_label_text(
            str(label), draw=_tmp_draw, font=font, max_width=text_width
        )
        bbox = _tmp_draw.multiline_textbbox((0, 0), wrapped_label, font=font, spacing=line_spacing)
        text_h = bbox[3] - bbox[1]
        measured_items.append((wrapped_label, color, max(text_h, swatch_size)))

    columns = pack_legend_columns(
        [item[2] for item in measured_items],
        max_height=max_height,
        item_gap=item_gap,
        padding_top=padding_top,
        padding_bottom=padding_bottom,
    )
    column_count = max(1, len(columns))

    def column_height(indexes: List[int]) -> int:
        return (
            padding_top
            + sum(measured_items[idx][2] for idx in indexes)
            + item_gap * max(len(indexes) - 1, 0)
            + padding_bottom
        )

    total_height = max(
        [column_height(column) for column in columns] or [padding_top + padding_bottom]
    )
    if max_height is not None:
        total_height = min(int(max_height), max(1, total_height))

    legend = Image.new("RGBA", (width * column_count, total_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(legend)
    for column_idx, column in enumerate(columns or [[]]):
        x_offset = column_idx * width
        if total_height > 1:
            draw.line(
                [(x_offset, 0), (x_offset, total_height)],
                fill=(230, 230, 230, 255),
                width=1,
            )

        y = padding_top
        for item_idx in column:
            label, color, item_h = measured_items[item_idx]
            legend_color = _legend_swatch_color(color)

            draw.rectangle(
                [
                    (x_offset + swatch_x, y),
                    (x_offset + swatch_x + swatch_size, y + swatch_size),
                ],
                fill=legend_color,
                outline=(120, 120, 120, 255),
            )

            draw.multiline_text(
                (x_offset + text_x, y),
                label,
                fill=(0, 0, 0, 255),
                font=font,
                spacing=line_spacing,
            )

            y += item_h + item_gap

    return legend


def legend_width_for_image(image_width: int, position: str = "right") -> int:
    """Choose a capped legend width appropriate for a rendered image."""
    position = (position or "right").lower()
    if position in {"top", "bottom"}:
        return max(1, int(image_width))
    return max(
        LEGEND_MIN_WIDTH,
        min(LEGEND_MAX_WIDTH, int(max(1, image_width) * LEGEND_SIDE_WIDTH_RATIO)),
    )


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
            except (TypeError, OSError):
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

    position = (position or "right").lower()
    bg_color = (255, 255, 255, 255)  # Always use white for the merged background

    if position == "right":
        # Create a new image with extra width for the legend
        merged_width = image.width + legend.width
        merged_height = max(image.height, legend.height)
        merged = Image.new("RGBA", (merged_width, merged_height), bg_color)
        image_y = (merged_height - image.height) // 2
        legend_y = (merged_height - legend.height) // 2
        merged.paste(image, (0, image_y))
        merged.paste(
            legend, (image.width, legend_y), legend if legend.mode == "RGBA" else None
        )  # Handle transparency
    elif position == "bottom":
        # Create a new image with extra height for the legend
        merged_width = max(image.width, legend.width)
        merged_height = image.height + legend.height
        merged = Image.new("RGBA", (merged_width, merged_height), bg_color)
        image_x = (merged_width - image.width) // 2
        legend_x = (merged_width - legend.width) // 2
        merged.paste(image, (image_x, 0))
        merged.paste(legend, (legend_x, image.height), legend if legend.mode == "RGBA" else None)
    elif position == "top":
        # Create a new image with extra height for the legend
        merged_width = max(image.width, legend.width)
        merged_height = image.height + legend.height
        merged = Image.new("RGBA", (merged_width, merged_height), bg_color)
        image_x = (merged_width - image.width) // 2
        legend_x = (merged_width - legend.width) // 2
        merged.paste(legend, (legend_x, 0), legend if legend.mode == "RGBA" else None)
        merged.paste(image, (image_x, legend.height))
    elif position == "left":
        # Create a new image with extra width for the legend
        merged_width = image.width + legend.width
        merged_height = max(image.height, legend.height)
        merged = Image.new("RGBA", (merged_width, merged_height), bg_color)
        image_y = (merged_height - image.height) // 2
        legend_y = (merged_height - legend.height) // 2
        merged.paste(legend, (0, legend_y), legend if legend.mode == "RGBA" else None)
        merged.paste(image, (legend.width, image_y))
    else:
        # Invalid position, return the original image
        logger.warning("Invalid legend position '%s'. Returning original image.", position)
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
    # Prefer pdfplumber's renderer (honors rotations/overrides) if available.
    try:
        if hasattr(page, "_page") and hasattr(page._page, "to_image"):
            img_obj = page._page.to_image(resolution=resolution)
            if hasattr(img_obj, "annotated"):
                return img_obj.annotated.convert("RGB")
            if hasattr(img_obj, "original"):
                return img_obj.original.convert("RGB")
    except Exception as exc:  # pragma: no cover - fall back to pdfium rendering
        logger.debug("render_plain_page fallback to pdfium due to %s", exc, exc_info=True)

    if pypdfium2 is None:
        raise RuntimeError(
            "pypdfium2 is required to render pages. Install with `pip install pypdfium2`."
        )

    with pdf_render_lock:
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


def _pdfium_crop_units_from_pixel_crop(pixel_amount: int, scale_factor: float) -> float:
    """Convert a desired integer crop amount to a pypdfium crop unit.

    pypdfium2 applies ``ceil(crop_amount * scale)`` internally. Subtracting a
    tiny epsilon preserves the same integer pixel crop used by the existing
    full-page-render-then-PIL-crop path.
    """
    if pixel_amount <= 0:
        return 0.0
    return max(0.0, (pixel_amount - 1e-6) / scale_factor)


def render_cropped_page(page, resolution, crop_bbox):
    """
    Render a page crop directly with pypdfium while preserving existing crop pixels.

    The existing render path rasterizes the full page, then crops with
    ``int(coord * scale)`` pixel coordinates. pypdfium's native crop API instead
    rounds page and crop amounts with ``ceil``. This helper computes the current
    pixel crop rectangle first, then converts those integer crop amounts back to
    pypdfium crop units so the resulting bitmap has the same size and pixels as
    the old path.
    """
    if pypdfium2 is None:
        raise DirectCropRenderUnsupportedError(
            "pypdfium2 is required to render pages. Install with `pip install pypdfium2`."
        )
    if not hasattr(page, "_page") or not hasattr(page._page, "pdf"):
        raise DirectCropRenderUnsupportedError(
            "Page does not expose a pdfplumber page for direct crop rendering."
        )

    scale_factor = resolution / 72.0
    x0, top, x1, bottom = crop_bbox

    with pdf_render_lock:
        pdf = page._page.pdf
        if getattr(pdf, "path", None):
            src = pdf.path
        else:
            pdf.stream.seek(0)
            src = pdf.stream

        doc = pypdfium2.PdfDocument(src, password=getattr(pdf, "password", None))
        pdf_page = doc.get_page(page._page.page_number - 1)
        try:
            src_width = math.ceil(pdf_page.get_width() * scale_factor)
            src_height = math.ceil(pdf_page.get_height() * scale_factor)

            left_px = int(x0 * scale_factor)
            top_px = int(top * scale_factor)
            right_edge_px = int(x1 * scale_factor)
            bottom_edge_px = int(bottom * scale_factor)

            left_px = max(0, min(left_px, src_width))
            top_px = max(0, min(top_px, src_height))
            right_edge_px = max(0, min(right_edge_px, src_width))
            bottom_edge_px = max(0, min(bottom_edge_px, src_height))

            if right_edge_px <= left_px or bottom_edge_px <= top_px:
                raise ValueError(f"Invalid crop bounds: {crop_bbox}")

            right_px = src_width - right_edge_px
            bottom_px = src_height - bottom_edge_px
            crop = (
                _pdfium_crop_units_from_pixel_crop(left_px, scale_factor),
                _pdfium_crop_units_from_pixel_crop(bottom_px, scale_factor),
                _pdfium_crop_units_from_pixel_crop(right_px, scale_factor),
                _pdfium_crop_units_from_pixel_crop(top_px, scale_factor),
            )

            bitmap = pdf_page.render(
                scale=scale_factor,
                crop=crop,
                no_smoothtext=True,
                no_smoothpath=True,
                no_smoothimage=True,
                prefer_bgrx=True,
            )
            return bitmap.to_pil().convert("RGB")
        finally:
            pdf_page.close()
            doc.close()


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
    try:
        import matplotlib  # lazy import – heavy dependency
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for colormap-based coloring. "
            "Install with: pip install matplotlib"
        ) from e

    # Try to get the colormap (colormaps[] preferred since matplotlib 3.7)
    _get = getattr(matplotlib, "colormaps", {}).get
    try:
        cmap = _get(colormap_name) or matplotlib.cm.get_cmap(colormap_name)
    except (ValueError, KeyError):
        cmap = _get("viridis") or matplotlib.cm.get_cmap("viridis")

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
