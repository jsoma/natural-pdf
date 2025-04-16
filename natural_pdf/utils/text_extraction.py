# natural_pdf/utils/text_extraction.py
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pdfplumber.utils.geometry import get_bbox_overlap, merge_bboxes, objects_to_bbox
from pdfplumber.utils.text import TEXTMAP_KWARGS, WORD_EXTRACTOR_KWARGS, chars_to_textmap

if TYPE_CHECKING:
    from natural_pdf.elements.region import Region  # Use type hint

logger = logging.getLogger(__name__)


def filter_chars_spatially(
    char_dicts: List[Dict[str, Any]],
    exclusion_regions: List["Region"],
    target_region: Optional["Region"] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Filters a list of character dictionaries spatially based on exclusions
    and an optional target region.

    Args:
        char_dicts: List of character dictionaries to filter.
        exclusion_regions: List of Region objects to exclude characters from.
        target_region: Optional Region object. If provided, only characters within
                       this region (respecting polygons) are kept.
        debug: Enable debug logging.

    Returns:
        Filtered list of character dictionaries.
    """
    if not char_dicts:
        return []

    initial_count = len(char_dicts)
    filtered_chars = char_dicts

    # 1. Filter by Target Region (if provided)
    if target_region:
        target_bbox = target_region.bbox
        target_is_polygon = target_region.has_polygon  # Check once
        region_filtered_chars = []
        for char_dict in filtered_chars:
            # Ensure basic geometry keys exist before processing
            if not all(k in char_dict for k in ["x0", "top", "x1", "bottom"]):
                if debug:
                    logger.warning(
                        f"Skipping char due to missing geometry: {char_dict.get('text', '?')}"
                    )
                continue
            char_bbox = (char_dict["x0"], char_dict["top"], char_dict["x1"], char_dict["bottom"])
            # BBox pre-filter first
            if get_bbox_overlap(char_bbox, target_bbox) is None:
                continue
            # Precise check if needed
            char_center_x = (char_dict["x0"] + char_dict["x1"]) / 2
            char_center_y = (char_dict["top"] + char_dict["bottom"]) / 2
            if target_is_polygon:
                if target_region.is_point_inside(char_center_x, char_center_y):
                    region_filtered_chars.append(char_dict)
                # else: # Optionally log discarded by polygon
                #     if debug: logger.debug(...)
            else:  # Rectangular region, bbox overlap was sufficient
                region_filtered_chars.append(char_dict)
        filtered_chars = region_filtered_chars
        if debug:
            logger.debug(
                f"filter_chars_spatially: {len(filtered_chars)}/{initial_count} chars remaining after target region filter."
            )
        if not filtered_chars:
            return []

    # 2. Filter by Exclusions (if any)
    if exclusion_regions:
        final_chars = []
        # Only calculate union_bbox if there are exclusions AND chars remaining
        union_bbox = merge_bboxes(excl.bbox for excl in exclusion_regions)
        for char_dict in filtered_chars:  # Process only chars within target
            # Ensure basic geometry keys exist before processing
            if not all(k in char_dict for k in ["x0", "top", "x1", "bottom"]):
                # Already warned in target region filter if applicable
                continue
            char_bbox = (char_dict["x0"], char_dict["top"], char_dict["x1"], char_dict["bottom"])
            # BBox pre-filter vs exclusion union
            if get_bbox_overlap(char_bbox, union_bbox) is None:
                final_chars.append(char_dict)  # Cannot be excluded
                continue
            # Precise check against individual overlapping exclusions
            is_excluded = False
            char_center_x = (char_dict["x0"] + char_dict["x1"]) / 2
            char_center_y = (char_dict["top"] + char_dict["bottom"]) / 2
            for exclusion in exclusion_regions:
                # Optional: Add bbox overlap check here too before point_inside
                if get_bbox_overlap(char_bbox, exclusion.bbox) is not None:
                    if exclusion.is_point_inside(char_center_x, char_center_y):
                        is_excluded = True
                        if debug:
                            char_text = char_dict.get("text", "?")
                            log_msg = f"  - Excluding char '{char_text}' at {char_bbox} due to overlap with exclusion {exclusion.bbox}"
                            logger.debug(log_msg)
                        break
            if not is_excluded:
                final_chars.append(char_dict)
        filtered_chars = final_chars
        if debug:
            logger.debug(
                f"filter_chars_spatially: {len(filtered_chars)}/{initial_count} chars remaining after exclusion filter."
            )
        if not filtered_chars:
            return []

    return filtered_chars


def generate_text_layout(
    char_dicts: List[Dict[str, Any]],
    layout_context_bbox: Tuple[float, float, float, float],
    user_kwargs: Dict[str, Any],
) -> str:
    """
    Takes a list of filtered character dictionaries and generates
    text output using pdfplumber's layout engine.

    Args:
        char_dicts: The final list of character dictionaries to include.
        layout_context_bbox: The bounding box (x0, top, x1, bottom) to use for
                             calculating default layout width/height/shifts.
        user_kwargs: Dictionary of user-provided keyword arguments.

    Returns:
        The formatted text string.
    """
    if not char_dicts:
        logger.debug("generate_text_layout: No characters provided.")
        return ""

    # Prepare layout kwargs, prioritizing user input
    layout_kwargs = {}
    allowed_keys = set(WORD_EXTRACTOR_KWARGS) | set(TEXTMAP_KWARGS)
    for key, value in user_kwargs.items():
        if key in allowed_keys:
            layout_kwargs[key] = value

    # Default to layout=True unless explicitly False
    use_layout = layout_kwargs.get("layout", True)  # Default to layout if called
    layout_kwargs["layout"] = use_layout

    if use_layout:
        ctx_x0, ctx_top, ctx_x1, ctx_bottom = layout_context_bbox
        ctx_width = ctx_x1 - ctx_x0
        ctx_height = ctx_bottom - ctx_top

        # Set layout defaults based on context_bbox if not overridden by user
        if "layout_bbox" not in layout_kwargs:
            layout_kwargs["layout_bbox"] = layout_context_bbox
        # Only set default layout_width if neither width specifier is present
        if "layout_width_chars" not in layout_kwargs and "layout_width" not in layout_kwargs:
            layout_kwargs["layout_width"] = ctx_width
        if "layout_height" not in layout_kwargs:
            layout_kwargs["layout_height"] = ctx_height
        # Adjust shift based on context's top-left corner
        if "x_shift" not in layout_kwargs:
            layout_kwargs["x_shift"] = ctx_x0
        if "y_shift" not in layout_kwargs:
            layout_kwargs["y_shift"] = ctx_top

        logger.debug(
            f"generate_text_layout: Calling chars_to_textmap with {len(char_dicts)} chars and kwargs: {layout_kwargs}"
        )
        try:
            # Sort final list by reading order before passing to textmap
            # TODO: Make sorting key dynamic based on layout_kwargs directions?
            char_dicts.sort(key=lambda c: (c.get("top", 0), c.get("x0", 0)))
            textmap = chars_to_textmap(char_dicts, **layout_kwargs)
            result = textmap.as_string
        except Exception as e:
            logger.error(
                f"generate_text_layout: Error calling chars_to_textmap: {e}", exc_info=True
            )
            logger.warning(
                "generate_text_layout: Falling back to simple character join due to layout error."
            )
            # Ensure chars are sorted before fallback join
            fallback_chars = sorted(char_dicts, key=lambda c: (c.get("top", 0), c.get("x0", 0)))
            result = "".join(c.get("text", "") for c in fallback_chars)
    else:
        # Simple join if layout=False
        logger.debug("generate_text_layout: Using simple join (layout=False).")
        # Sort by document order for simple join as well
        char_dicts.sort(key=lambda c: (c.get("page_number", 0), c.get("top", 0), c.get("x0", 0)))
        result = "".join(c.get("text", "") for c in char_dicts)

    return result
