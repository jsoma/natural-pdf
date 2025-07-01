# natural_pdf/utils/text_extraction.py
import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from pdfplumber.utils.geometry import (
    cluster_objects,
    get_bbox_overlap,
    merge_bboxes,
    objects_to_bbox,
)
from pdfplumber.utils.text import TEXTMAP_KWARGS, WORD_EXTRACTOR_KWARGS, chars_to_textmap

if TYPE_CHECKING:
    from natural_pdf.elements.region import Region  # Use type hint

logger = logging.getLogger(__name__)


def _get_layout_kwargs(
    layout_context_bbox: Optional[Tuple[float, float, float, float]] = None,
    user_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Prepares the keyword arguments for pdfplumber's chars_to_textmap based
    on defaults, context bbox, and allowed user overrides.
    """
    # 1. Start with an empty dict for layout kwargs
    layout_kwargs = {}

    # Build allowed keys set without trying to copy the constants
    allowed_keys = set(TEXTMAP_KWARGS) | set(WORD_EXTRACTOR_KWARGS)

    # Add common, well-known default values
    layout_kwargs.update(
        {
            "x_tolerance": 5,
            "y_tolerance": 5,
            "x_density": 7.25,
            "y_density": 13,
            "mode": "box",
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
        }
    )

    # 2. Apply context if provided
    if layout_context_bbox:
        ctx_x0, ctx_top, ctx_x1, ctx_bottom = layout_context_bbox
        layout_kwargs["layout_width"] = ctx_x1 - ctx_x0
        layout_kwargs["layout_height"] = ctx_bottom - ctx_top
        layout_kwargs["x_shift"] = ctx_x0
        layout_kwargs["y_shift"] = ctx_top
        # Add layout_bbox itself
        layout_kwargs["layout_bbox"] = layout_context_bbox

    # 3. Apply user overrides (only for allowed keys)
    if user_kwargs:
        for key, value in user_kwargs.items():
            if key in allowed_keys:
                layout_kwargs[key] = value
            elif key == "layout":  # Always allow layout flag
                layout_kwargs[key] = value
            else:
                logger.warning(f"Ignoring unsupported layout keyword argument: '{key}'")

    # 4. Ensure layout flag is present, defaulting to False (caller can override)
    if "layout" not in layout_kwargs:
        layout_kwargs["layout"] = False

    return layout_kwargs


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


def _apply_content_filter(
    char_dicts: List[Dict[str, Any]], content_filter: Union[str, Callable[[str], bool], List[str]]
) -> List[Dict[str, Any]]:
    """
    Applies content filtering to character dictionaries based on their text content.

    Args:
        char_dicts: List of character dictionaries to filter.
        content_filter: Can be:
            - A regex pattern string (characters matching the pattern are EXCLUDED)
            - A callable that takes text and returns True to KEEP the character
            - A list of regex patterns (characters matching ANY pattern are EXCLUDED)

    Returns:
        Filtered list of character dictionaries.
    """
    if not char_dicts or content_filter is None:
        return char_dicts

    initial_count = len(char_dicts)
    filtered_chars = []

    # Handle different filter types
    if isinstance(content_filter, str):
        # Single regex pattern - exclude matching characters
        try:
            pattern = re.compile(content_filter)
            for char_dict in char_dicts:
                text = char_dict.get("text", "")
                if not pattern.search(text):
                    filtered_chars.append(char_dict)
        except re.error as e:
            logger.warning(
                f"Invalid regex pattern '{content_filter}': {e}. Skipping content filtering."
            )
            return char_dicts

    elif isinstance(content_filter, list):
        # List of regex patterns - exclude characters matching ANY pattern
        try:
            patterns = [re.compile(p) for p in content_filter]
            for char_dict in char_dicts:
                text = char_dict.get("text", "")
                if not any(pattern.search(text) for pattern in patterns):
                    filtered_chars.append(char_dict)
        except re.error as e:
            logger.warning(f"Invalid regex pattern in list: {e}. Skipping content filtering.")
            return char_dicts

    elif callable(content_filter):
        # Callable filter - keep characters where function returns True
        try:
            for char_dict in char_dicts:
                text = char_dict.get("text", "")
                if content_filter(text):
                    filtered_chars.append(char_dict)
        except Exception as e:
            logger.warning(f"Error in content filter function: {e}. Skipping content filtering.")
            return char_dicts
    else:
        logger.warning(
            f"Unsupported content_filter type: {type(content_filter)}. Skipping content filtering."
        )
        return char_dicts

    filtered_count = initial_count - len(filtered_chars)
    if filtered_count > 0:
        logger.debug(f"Content filter removed {filtered_count} characters.")

    return filtered_chars


def generate_text_layout(
    char_dicts: List[Dict[str, Any]],
    layout_context_bbox: Optional[Tuple[float, float, float, float]] = None,
    user_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generates a string representation of text from character dictionaries,
    attempting to reconstruct layout using pdfplumber's utilities.

    Args:
        char_dicts: List of character dictionary objects.
        layout_context_bbox: Optional bounding box for layout context.
        user_kwargs: User-provided kwargs, potentially overriding defaults.

    Returns:
        String representation of the text.
    """
    # --- Filter out invalid char dicts early ---
    initial_count = len(char_dicts)
    valid_char_dicts = [c for c in char_dicts if isinstance(c.get("text"), str)]
    filtered_count = initial_count - len(valid_char_dicts)
    if filtered_count > 0:
        logger.debug(
            f"generate_text_layout: Filtered out {filtered_count} char dicts with non-string/None text."
        )

    if not valid_char_dicts:  # Return empty if no valid chars remain
        logger.debug("generate_text_layout: No valid character dicts found after filtering.")
        return ""

    # Make a working copy of user_kwargs so we can safely pop custom keys
    incoming_kwargs = user_kwargs.copy() if user_kwargs else {}

    # --- Apply content filtering if specified ---
    content_filter = incoming_kwargs.pop("content_filter", None)
    if content_filter is not None:
        valid_char_dicts = _apply_content_filter(valid_char_dicts, content_filter)

    # --- Handle custom 'strip' option ------------------------------------
    # * strip=True  – post-process the final string to remove leading/trailing
    #                 whitespace (typically used when layout=False)
    # * strip=False – preserve whitespace exactly as produced.
    # Default behaviour depends on the layout flag (see below).
    explicit_strip_flag = incoming_kwargs.pop("strip", None)  # May be None

    # Prepare layout arguments now that we've removed the non-pdfplumber key
    layout_kwargs = _get_layout_kwargs(layout_context_bbox, incoming_kwargs)
    use_layout = layout_kwargs.get("layout", False)

    # Determine final strip behaviour: if caller specified override, honour it;
    # otherwise default to !use_layout (True when layout=False, False when
    # layout=True) per user request.
    strip_result = explicit_strip_flag if explicit_strip_flag is not None else (not use_layout)

    try:
        # Sort chars primarily by top, then x0 before layout analysis – required by
        # pdfplumber so that grouping into lines works deterministically.
        valid_char_dicts.sort(key=lambda c: (c.get("top", 0), c.get("x0", 0)))

        # Build the text map. `layout_kwargs` still contains the caller-specified or
        # default "layout" flag, which chars_to_textmap will respect.
        textmap = chars_to_textmap(valid_char_dicts, **layout_kwargs)
        result = textmap.as_string

        # ----------------------------------------------------------------
        # Optional post-processing strip
        # ----------------------------------------------------------------
        if strip_result and isinstance(result, str):
            # Remove trailing spaces on each line then trim leading/trailing
            # blank lines for a cleaner output while keeping internal newlines.
            result = "\n".join(line.rstrip() for line in result.splitlines()).strip()
    except Exception as e:
        # Fallback to simple join on error
        logger.error(f"generate_text_layout: Error calling chars_to_textmap: {e}", exc_info=False)
        logger.warning(
            "generate_text_layout: Falling back to simple character join due to layout error."
        )
        # Fallback already has sorted characters if layout was attempted
        # Need to use the valid_char_dicts here too
        result = "".join(c.get("text", "") for c in valid_char_dicts)
        if strip_result:
            result = result.strip()

    return result
