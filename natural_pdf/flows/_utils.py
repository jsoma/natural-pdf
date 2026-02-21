"""Shared utilities for the flows subsystem."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw


def stack_images(
    images: Sequence[Image.Image],
    direction: str = "vertical",
    gap: int = 5,
    background: Tuple[int, int, int] = (255, 255, 255),
    separator_color: Optional[Tuple[int, int, int]] = None,
    separator_thickness: int = 0,
) -> Optional[Image.Image]:
    """Stack a sequence of PIL images into one composite image.

    Args:
        images: Non-empty sequence of PIL images to stack.
        direction: ``"vertical"`` or ``"horizontal"``.
        gap: Pixel gap between images (applied *after* separators).
        background: RGB background colour for the composite canvas.
        separator_color: If given, draw a solid-colour bar between images.
        separator_thickness: Thickness (px) of each separator bar.

    Returns:
        A new PIL ``Image`` combining all inputs, or ``None`` when *images*
        is empty.
    """
    if not images:
        return None

    if len(images) == 1:
        return images[0]

    num_gaps = len(images) - 1
    sep_total = num_gaps * separator_thickness if separator_color else 0
    gap_total = num_gaps * gap

    if direction == "vertical":
        final_width = max(img.width for img in images)
        final_height = sum(img.height for img in images) + gap_total + sep_total

        composite = Image.new("RGB", (final_width, final_height), background)
        draw = ImageDraw.Draw(composite) if separator_color else None

        cursor = 0
        for i, img in enumerate(images):
            if i > 0:
                if draw and separator_thickness > 0:
                    draw.rectangle(
                        [(0, cursor), (final_width, cursor + separator_thickness)],
                        fill=separator_color,
                    )
                    cursor += separator_thickness
                cursor += gap

            paste_x = (final_width - img.width) // 2
            composite.paste(img, (paste_x, cursor))
            cursor += img.height

        return composite

    elif direction == "horizontal":
        final_height = max(img.height for img in images)
        final_width = sum(img.width for img in images) + gap_total + sep_total

        composite = Image.new("RGB", (final_width, final_height), background)
        draw = ImageDraw.Draw(composite) if separator_color else None

        cursor = 0
        for i, img in enumerate(images):
            if i > 0:
                if draw and separator_thickness > 0:
                    draw.rectangle(
                        [(cursor, 0), (cursor + separator_thickness, final_height)],
                        fill=separator_color,
                    )
                    cursor += separator_thickness
                cursor += gap

            paste_y = (final_height - img.height) // 2
            composite.paste(img, (cursor, paste_y))
            cursor += img.width

        return composite

    else:
        raise ValueError(f"Invalid direction '{direction}'. Must be 'vertical' or 'horizontal'.")
