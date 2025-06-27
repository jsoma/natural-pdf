"""
Element-specific describe functions.
"""

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from natural_pdf.elements.base import Element

logger = logging.getLogger(__name__)


def describe_text_elements(elements: List["Element"]) -> Dict[str, Any]:
    """
    Describe text elements with typography and OCR analysis.

    Args:
        elements: List of text elements

    Returns:
        Dictionary with text analysis sections
    """
    if not elements:
        return {"message": "No text elements found"}

    result = {}

    # Source breakdown
    sources = Counter()
    ocr_elements = []

    for element in elements:
        source = getattr(element, "source", "unknown")
        sources[source] += 1
        if source == "ocr":
            ocr_elements.append(element)

    if len(sources) > 1:
        result["sources"] = dict(sources)

    # Typography analysis
    typography = _analyze_typography(elements)
    if typography:
        result["typography"] = typography

    # OCR quality analysis
    if ocr_elements:
        ocr_quality = _analyze_ocr_quality(ocr_elements)
        if ocr_quality:
            result["ocr_quality"] = ocr_quality

    return result


def describe_rect_elements(elements: List["Element"]) -> Dict[str, Any]:
    """
    Describe rectangle elements with size and style analysis.

    Args:
        elements: List of rectangle elements

    Returns:
        Dictionary with rectangle analysis
    """
    if not elements:
        return {"message": "No rectangle elements found"}

    result = {}

    # Size analysis
    sizes = []
    stroke_count = 0
    fill_count = 0
    colors = Counter()
    stroke_widths = []

    for element in elements:
        # Size
        width = getattr(element, "width", 0)
        height = getattr(element, "height", 0)
        if width and height:
            sizes.append((width, height))

        # Style properties - use RectangleElement properties
        stroke = getattr(element, "stroke", None)
        if stroke and stroke != (0, 0, 0):  # Check if stroke color exists and isn't black
            stroke_count += 1
        fill = getattr(element, "fill", None)
        if fill and fill != (0, 0, 0):  # Check if fill color exists and isn't black
            fill_count += 1

        # Stroke width
        stroke_width = getattr(element, "stroke_width", 0)
        if stroke_width > 0:
            stroke_widths.append(stroke_width)

        # Color - use the element's stroke/fill properties
        color = stroke or fill
        if color:
            if isinstance(color, (tuple, list)):
                if color == (0, 0, 0) or color == (0.0, 0.0, 0.0):
                    colors["black"] += 1
                elif color == (1, 1, 1) or color == (1.0, 1.0, 1.0):
                    colors["white"] += 1
                else:
                    colors[str(color)] += 1
            else:
                colors[str(color)] += 1

    # Size statistics
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        result["size_stats"] = {
            "width_range": f"{min(widths):.0f}-{max(widths):.0f}",
            "height_range": f"{min(heights):.0f}-{max(heights):.0f}",
            "avg_area": f"{sum(w*h for w,h in sizes)/len(sizes):.0f} sq pts",
        }

    # Style breakdown
    style_info = {}
    if stroke_count:
        style_info["stroke"] = stroke_count
    if fill_count:
        style_info["fill"] = fill_count
    if stroke_widths:
        stroke_width_counts = Counter(stroke_widths)
        # Convert float keys to strings to avoid formatting issues
        stroke_width_dict = {str(k): v for k, v in stroke_width_counts.most_common()}
        style_info["stroke_widths"] = stroke_width_dict
    if colors:
        style_info["colors"] = dict(colors.most_common(5))

    if style_info:
        result["styles"] = style_info

    return result


def describe_line_elements(elements: List["Element"]) -> Dict[str, Any]:
    """
    Describe line elements with length and style analysis.

    Args:
        elements: List of line elements

    Returns:
        Dictionary with line analysis
    """
    if not elements:
        return {"message": "No line elements found"}

    result = {}

    lengths = []
    widths = []
    colors = Counter()

    for element in elements:
        # Calculate length
        x0 = getattr(element, "x0", 0)
        y0 = getattr(element, "top", 0)
        x1 = getattr(element, "x1", 0)
        y1 = getattr(element, "bottom", 0)

        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if length > 0:
            lengths.append(length)

        # Line width - use the element's width property
        width = getattr(element, "width", 0)  # LineElement has a width property
        if width:
            widths.append(width)

        # Color - use the element's color property
        color = getattr(element, "color", None)  # LineElement has a color property
        if color:
            if isinstance(color, (tuple, list)):
                if color == (0, 0, 0) or color == (0.0, 0.0, 0.0):
                    colors["black"] += 1
                else:
                    colors[str(color)] += 1
            else:
                colors[str(color)] += 1

    # Length statistics
    if lengths:
        result["length_stats"] = {
            "min": f"{min(lengths):.0f}",
            "max": f"{max(lengths):.0f}",
            "avg": f"{sum(lengths)/len(lengths):.0f}",
        }

    # Width statistics
    if widths:
        width_counts = Counter(widths)
        # Convert float keys to strings to avoid formatting issues
        result["line_widths"] = {str(k): v for k, v in width_counts.most_common()}

    # Orientation analysis
    horizontal_count = sum(1 for el in elements if getattr(el, "is_horizontal", False))
    vertical_count = sum(1 for el in elements if getattr(el, "is_vertical", False))
    diagonal_count = len(elements) - horizontal_count - vertical_count

    if horizontal_count or vertical_count or diagonal_count:
        orientation_info = {}
        if horizontal_count:
            orientation_info["horizontal"] = horizontal_count
        if vertical_count:
            orientation_info["vertical"] = vertical_count
        if diagonal_count:
            orientation_info["diagonal"] = diagonal_count
        result["orientations"] = orientation_info

    # Colors
    if colors:
        result["colors"] = dict(colors.most_common())

    return result


def describe_region_elements(elements: List["Element"]) -> Dict[str, Any]:
    """
    Describe region elements with type and metadata analysis.

    Args:
        elements: List of region elements

    Returns:
        Dictionary with region analysis
    """
    if not elements:
        return {"message": "No region elements found"}

    result = {}

    # Region types
    types = Counter()
    sizes = []
    metadata_keys = set()

    for element in elements:
        # Type
        region_type = getattr(element, "type", "unknown")
        types[region_type] += 1

        # Size
        width = getattr(element, "width", 0)
        height = getattr(element, "height", 0)
        if width and height:
            sizes.append(width * height)

        # Metadata keys
        if hasattr(element, "metadata") and element.metadata:
            metadata_keys.update(element.metadata.keys())

    # Type breakdown
    if types:
        result["types"] = dict(types.most_common())

    # Size statistics
    if sizes:
        result["size_stats"] = {
            "min_area": f"{min(sizes):.0f} sq pts",
            "max_area": f"{max(sizes):.0f} sq pts",
            "avg_area": f"{sum(sizes)/len(sizes):.0f} sq pts",
        }

    # Metadata
    if metadata_keys:
        result["metadata_keys"] = sorted(list(metadata_keys))

    return result


def _analyze_typography(elements: List["Element"]) -> Dict[str, Any]:
    """Analyze typography patterns in text elements."""
    fonts = Counter()
    sizes = Counter()
    styles = {"bold": 0, "italic": 0, "strikeout": 0, "underline": 0, "highlight": 0}
    colors = Counter()

    for element in elements:
        # Font family - use TextElement's font_family property for cleaner names
        font_family = getattr(element, "font_family", None)
        fontname = getattr(element, "fontname", "Unknown")
        display_font = font_family if font_family and font_family != fontname else fontname
        if display_font:
            fonts[display_font] += 1

        # Size
        size = getattr(element, "size", None)
        if size:
            # Round to nearest 0.5
            rounded_size = round(size * 2) / 2
            sizes[f"{rounded_size}pt"] += 1

        # Styles
        if getattr(element, "bold", False):
            styles["bold"] += 1
        if getattr(element, "italic", False):
            styles["italic"] += 1
        if getattr(element, "strikeout", False):
            styles["strikeout"] += 1
        if getattr(element, "underline", False):
            styles["underline"] += 1
        if getattr(element, "highlight", False):
            styles["highlight"] += 1

        # Color - use TextElement's color property
        color = getattr(element, "color", None)
        if color:
            if isinstance(color, (tuple, list)):
                if color == (0, 0, 0) or color == (0.0, 0.0, 0.0):
                    colors["black"] += 1
                elif color == (1, 1, 1) or color == (1.0, 1.0, 1.0):
                    colors["white"] += 1
                else:
                    colors["other"] += 1
            else:
                colors[str(color)] += 1

    result = {}

    # Fonts
    if fonts:
        result["fonts"] = dict(fonts.most_common(10))

    # Sizes (as horizontal table)
    if sizes:
        result["sizes"] = dict(sizes.most_common())

    # Styles
    style_list = []
    for style, count in styles.items():
        if count > 0:
            style_list.append(f"{count} {style}")
    if style_list:
        result["styles"] = ", ".join(style_list)

    # Colors
    if colors and len(colors) > 1:  # Only show if there are multiple colors
        result["colors"] = dict(colors.most_common())

    return result


def _analyze_ocr_quality(elements: List["Element"]) -> Dict[str, Any]:
    """Analyze OCR quality metrics."""
    confidences = []

    for element in elements:
        confidence = getattr(element, "confidence", None)
        if confidence is not None:
            confidences.append(confidence)

    if not confidences:
        return {}

    result = {}

    # Basic stats
    result["confidence_stats"] = {
        "mean": f"{sum(confidences)/len(confidences):.2f}",
        "min": f"{min(confidences):.2f}",
        "max": f"{max(confidences):.2f}",
    }

    # Threshold analysis with ASCII bars
    thresholds = [
        ("99%+", 0.99),
        ("95%+", 0.95),
        ("90%+", 0.90),
    ]

    element_count = len(elements)
    threshold_bars = {}

    for label, threshold in thresholds:
        count = sum(1 for c in confidences if c >= threshold)
        percentage = count / element_count

        # Create ASCII bar (40 characters wide)
        filled_chars = int(percentage * 40)
        empty_chars = 40 - filled_chars
        bar = "█" * filled_chars + "░" * empty_chars

        # Format: "95%+ (32/43) 74%: `████████████████████████████████░░░░░░░░`"
        threshold_bars[f"{label} ({count}/{element_count}) {percentage:.0%}"] = f"`{bar}`"

    result["quality_distribution"] = threshold_bars

    # Show lowest quality items
    element_confidences = []
    for element in elements:
        confidence = getattr(element, "confidence", None)
        if confidence is not None:
            # Get text content for display
            text = getattr(element, "text", "").strip()
            if text:
                # Truncate long text
                display_text = text[:60] + "..." if len(text) > 60 else text
                element_confidences.append((confidence, display_text))

    if element_confidences:
        # Sort by confidence (lowest first) and take bottom 10
        lowest_quality = sorted(element_confidences, key=lambda x: x[0])[:10]
        if lowest_quality:
            lowest_items = {}
            for i, (confidence, text) in enumerate(lowest_quality, 1):
                lowest_items[f"#{i}"] = f"**{confidence:.2f}**: {text}"
            result["lowest_scoring"] = lowest_items

    return result
