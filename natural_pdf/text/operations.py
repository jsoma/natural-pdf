# natural_pdf/utils/text_extraction.py
import logging
import re
import unicodedata
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

from pdfplumber.utils.geometry import get_bbox_overlap, merge_bboxes

from natural_pdf.text.contracts import ContentFilter
from natural_pdf.text.pipeline import filter_text
from natural_pdf.utils.bidi_mirror import mirror_brackets

if TYPE_CHECKING:
    from natural_pdf.elements.region import Region  # Use type hint

logger = logging.getLogger(__name__)

_MULTI_SPACE = re.compile(r"[^\S\n]+")  # runs of whitespace that aren't newlines
_MULTI_NEWLINE = re.compile(r"\n{3,}")  # 3+ consecutive newlines


def normalize_whitespace(text: str) -> str:
    """Collapse redundant whitespace for feeding into NLI / classification models.

    - Runs of spaces/tabs (but not newlines) → single space
    - 3+ consecutive newlines → double newline
    - Strip leading/trailing whitespace
    """
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


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
            else:  # Rectangular region - check center is inside bbox
                # Use strict inequality on boundaries to exclude characters that merely touch
                # the region edge (their center would be outside)
                if (
                    target_bbox[0] <= char_center_x < target_bbox[2]
                    and target_bbox[1] <= char_center_y < target_bbox[3]
                ):
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


def apply_content_filter_to_text(
    text: str,
    content_filter: Optional[ContentFilter],
) -> str:
    """Apply the public content-filter contract to a string.

    Regex filters remove matching spans. Callable filters receive one character
    at a time and return truthy values for characters that should be kept.
    Invalid filters raise instead of returning potentially sensitive,
    unfiltered text.
    """
    return filter_text(text, content_filter)


def _create_alt_text_char_dict(region, source_label: str = "alt_text") -> Dict[str, Any]:
    """Create a char_dict from a region's alt_text and bbox.

    Produces a single character dictionary suitable for the canonical spatial
    text pipeline. The format mirrors what ``Region.to_text_element`` builds.
    """
    page = region.page
    initial_doctop = getattr(getattr(page, "_page", None), "initial_doctop", 0)
    return {
        "text": region.alt_text,
        "x0": region.x0,
        "top": region.top,
        "x1": region.x1,
        "bottom": region.bottom,
        "doctop": region.top + initial_doctop,
        "width": region.x1 - region.x0,
        "height": region.bottom - region.top,
        "object_type": "char",
        "page_number": getattr(page, "page_number", 0),
        "fontname": "AltText",
        "size": 10.0,
        "upright": True,
        "direction": 1,
        "adv": region.x1 - region.x0,
        "source": source_label,
        # Keep the object that introduced synthetic text available to the
        # provenance resolver.  ``source`` above is intentionally a stable
        # public label; this private marker retains the concrete Region for
        # citations without changing the public char schema.
        "_natural_pdf_text_source": region,
        "confidence": 1.0,
        "stroking_color": (0, 0, 0),
        "non_stroking_color": (0, 0, 0),
    }


def _synthetic_space_char(
    prev_char: Dict[str, Any],
    next_char: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a positioned space char between two native character dicts."""
    px1 = float(prev_char.get("x1", prev_char.get("x0", 0)) or 0)
    nx0 = float(next_char.get("x0", next_char.get("x1", px1)) or px1)
    x = (px1 + nx0) / 2
    epsilon = 1e-6

    top = min(float(prev_char.get("top", 0) or 0), float(next_char.get("top", 0) or 0))
    bottom = max(
        float(prev_char.get("bottom", top) or top),
        float(next_char.get("bottom", top) or top),
    )

    space = dict(prev_char)
    space.update(
        {
            "text": " ",
            "x0": x - epsilon,
            "x1": x + epsilon,
            "top": top,
            "bottom": bottom,
            "width": 2 * epsilon,
            "height": bottom - top,
            "adv": 0,
            "object_type": "char",
        }
    )
    if "doctop" in prev_char:
        space["doctop"] = (
            float(prev_char.get("doctop", top) or top)
            - float(prev_char.get("top", top) or top)
            + top
        )
    return space


def _expand_word_chars_for_injected_spaces(
    char_dicts: List[Dict[str, Any]], word_text: str
) -> Optional[List[Dict[str, Any]]]:
    """Return chars plus synthetic spaces when word_text only adds spaces."""
    raw_text = "".join(c.get("text", "") for c in char_dicts)
    if not raw_text or word_text == raw_text:
        return char_dicts
    if word_text.replace(" ", "") != raw_text:
        return None

    expanded: List[Dict[str, Any]] = []
    raw_index = 0
    for char in word_text:
        if char == " ":
            if raw_index <= 0 or raw_index >= len(char_dicts):
                return None
            expanded.append(_synthetic_space_char(char_dicts[raw_index - 1], char_dicts[raw_index]))
            continue

        if raw_index >= len(char_dicts) or char_dicts[raw_index].get("text", "") != char:
            return None
        expanded.append(char_dicts[raw_index])
        raw_index += 1

    if raw_index != len(char_dicts):
        return None
    return expanded


def word_elements_to_textmap_char_dicts(word_elements: Iterable[Any]) -> List[Dict[str, Any]]:
    """Collect word chars while preserving word-engine-injected spaces.

    The word engine can merge tightly spaced glyphs and inject spaces into the
    word text. Feeding only the original glyph chars back into chars_to_textmap
    loses those inferred spaces, so we reinsert them as lightweight synthetic
    space chars when the word text otherwise matches the raw glyph sequence.
    """
    all_char_dicts: List[Dict[str, Any]] = []
    for word in word_elements:
        char_dicts = list(getattr(word, "_char_dicts", []) or [])
        if not char_dicts:
            continue

        word_obj = getattr(word, "_obj", {})
        word_text = word_obj.get("text") if isinstance(word_obj, dict) else None
        if not isinstance(word_text, str):
            word_text = getattr(word, "text", None)
        if isinstance(word_text, str):
            expanded = _expand_word_chars_for_injected_spaces(char_dicts, word_text)
            if expanded is not None:
                all_char_dicts.extend(expanded)
                continue

        all_char_dicts.extend(char_dicts)
    return all_char_dicts


def apply_bidi_processing(text: str) -> str:
    """Convert visual-order RTL text into logical order when needed."""

    if not text or not text.strip():
        return text

    def _contains_rtl(s: str) -> bool:
        return any(unicodedata.bidirectional(ch) in ("R", "AL", "AN") for ch in s)

    if not _contains_rtl(text):
        return text

    from bidi.algorithm import get_display  # type: ignore

    processed_lines = []
    for line in text.split("\n"):
        if line.strip():
            base_dir = "R" if _contains_rtl(line) else "L"
            logical_line = get_display(line, base_dir=base_dir)
            if isinstance(logical_line, bytes):
                logical_line = logical_line.decode("utf-8", errors="ignore")
            processed_lines.append(mirror_brackets(logical_line))
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


__all__ = [
    "filter_chars_spatially",
    "word_elements_to_textmap_char_dicts",
    "apply_content_filter_to_text",
    "apply_bidi_processing",
]
