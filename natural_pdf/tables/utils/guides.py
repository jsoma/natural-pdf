"""Guide-related helpers for table extraction."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def adjust_explicit_vertical_guides(
    region,
    table_settings: Dict[str, Any],
    *,
    apply_exclusions: bool = True,
) -> None:
    """Clamp explicit vertical guides to detected text bounds for text strategies."""

    if (
        table_settings.get("horizontal_strategy") != "text"
        or table_settings.get("vertical_strategy") != "explicit"
        or "explicit_vertical_lines" not in table_settings
    ):
        return

    text_elements = region.find_all("text", apply_exclusions=apply_exclusions)
    if not text_elements:
        return

    text_bounds = text_elements.merge().bbox
    text_left = text_bounds[0]
    text_right = text_bounds[2]

    adjusted_verticals: List[float] = []
    for guide in table_settings["explicit_vertical_lines"]:
        if guide < text_left:
            adjusted_verticals.append(text_left)
            logger.debug(
                "Region %s: Adjusted left guide from %.1f to %.1f",
                getattr(region, "bbox", None),
                guide,
                text_left,
            )
        elif guide > text_right:
            adjusted_verticals.append(text_right)
            logger.debug(
                "Region %s: Adjusted right guide from %.1f to %.1f",
                getattr(region, "bbox", None),
                guide,
                text_right,
            )
        else:
            adjusted_verticals.append(guide)

    table_settings["explicit_vertical_lines"] = adjusted_verticals
