"""
Main describe functions for pages, collections, and regions.
"""

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .elements import (
    describe_line_elements,
    describe_rect_elements,
    describe_region_elements,
    describe_text_elements,
)
from .summary import ElementSummary, InspectionSummary

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region

logger = logging.getLogger(__name__)


def describe_page(page: "Page") -> ElementSummary:
    """
    Describe what's on a page with high-level summary.

    Args:
        page: Page to describe

    Returns:
        ElementSummary with page overview
    """
    data = {}

    # Get all elements
    all_elements = page.get_elements()

    if not all_elements:
        data["message"] = "No elements found on page"
        return ElementSummary(data, f"Page {page.number} Summary")

    # Element counts by type (exclude chars - too granular)
    type_counts = Counter()
    for element in all_elements:
        element_type = getattr(element, "type", "unknown")
        if element_type != "char":  # Skip character elements
            type_counts[element_type] += 1

    # Format element counts as dictionary for proper list formatting
    element_summary = {}
    for element_type, count in type_counts.most_common():
        type_display = element_type.replace("_", " ").title()
        if element_type == "word":
            # Add source breakdown for text
            text_elements = [e for e in all_elements if getattr(e, "type", "") == "word"]
            sources = Counter()
            for elem in text_elements:
                source = getattr(elem, "source", "unknown")
                sources[source] += 1

            if len(sources) > 1:
                source_parts = []
                for source, source_count in sources.most_common():
                    source_parts.append(f"{source_count} {source}")
                element_summary["text"] = f"{count} elements ({', '.join(source_parts)})"
            else:
                element_summary["text"] = f"{count} elements"
        else:
            element_summary[element_type] = f"{count} elements"

    data["elements"] = element_summary

    # Text analysis if we have text elements (exclude chars - too granular)
    text_elements = [e for e in all_elements if getattr(e, "type", "") == "word"]
    if text_elements:
        text_analysis = describe_text_elements(text_elements)
        if text_analysis and "message" not in text_analysis:
            data["text_analysis"] = text_analysis

    return ElementSummary(data, f"Page {page.number} Summary")


def describe_collection(collection: "ElementCollection") -> ElementSummary:
    """
    Describe an element collection with type-specific analysis.

    Args:
        collection: ElementCollection to describe

    Returns:
        ElementSummary with collection analysis
    """
    elements = list(collection)

    if not elements:
        data = {"message": "Empty collection"}
        return ElementSummary(data, "Collection Summary")

    data = {}

    # Group elements by type
    by_type = {}
    for element in elements:
        element_type = getattr(element, "type", "unknown")
        by_type.setdefault(element_type, []).append(element)

    # Overall summary for mixed collections (exclude chars from overview)
    if len(by_type) > 1:
        type_counts = {k: len(v) for k, v in by_type.items() if k != "char"}
        total = sum(type_counts.values())

        summary_parts = []
        for element_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            type_display = element_type.replace("_", " ").title()
            summary_parts.append(f"**{type_display}**: {count}")

        if summary_parts:  # Only add overview if we have non-char elements
            data["overview"] = {"total_elements": total, "type_breakdown": summary_parts}

    # Type-specific analysis (exclude chars - too granular)
    for element_type, type_elements in by_type.items():
        if element_type == "char":
            # Skip character elements - too granular for useful analysis
            continue
        elif element_type == "word":
            analysis = describe_text_elements(type_elements)
        elif element_type == "rect":
            analysis = describe_rect_elements(type_elements)
        elif element_type == "line":
            analysis = describe_line_elements(type_elements)
        elif element_type == "region":
            analysis = describe_region_elements(type_elements)
        else:
            analysis = {"count": len(type_elements)}

        if analysis and "message" not in analysis:
            section_name = element_type.replace("_", " ").title()
            if len(by_type) == 1:
                # Single type collection - flatten the structure
                data.update(analysis)
            else:
                # Mixed collection - keep sections separate
                data[section_name] = analysis

    # Count non-char elements for title
    non_char_count = len([e for e in elements if getattr(e, "type", "unknown") != "char"])
    title = f"Collection Summary ({non_char_count} elements)"
    return ElementSummary(data, title)


def describe_region(region: "Region") -> ElementSummary:
    """
    Describe a region with its properties and contents.

    Args:
        region: Region to describe

    Returns:
        ElementSummary with region analysis
    """
    data = {}

    # Region info
    region_info = {
        "page": region.page.number,
        "dimensions": f"{region.width:.0f}×{region.height:.0f} pts",
        "area": f"{region.width * region.height:.0f} sq pts",
        "position": f"({region.x0:.0f}, {region.top:.0f}) to ({region.x1:.0f}, {region.bottom:.0f})",
    }

    # Add metadata if available
    if hasattr(region, "metadata") and region.metadata:
        region_info["metadata"] = region.metadata

    data["region_info"] = region_info

    # Content analysis
    content_elements = region.find_all("*")
    if content_elements:
        content_analysis = describe_collection(content_elements)
        # Extract the data and add as "content" section
        data["content"] = content_analysis.to_dict()
    else:
        data["content"] = {"message": "No elements found in region"}

    return ElementSummary(data, "Region Summary")


def inspect_collection(collection: "ElementCollection", limit: int = 30) -> InspectionSummary:
    """
    Inspect elements in a collection with detailed tabular view.

    Args:
        collection: ElementCollection to inspect
        limit: Maximum elements per type to show (default: 30)

    Returns:
        InspectionSummary with element tables
    """
    elements = list(collection)

    if not elements:
        data = {"message": "Empty collection"}
        return InspectionSummary(data, "Collection Inspection")

    data = {}

    # Check if multi-page
    pages = set()
    for element in elements:
        if hasattr(element, "page") and hasattr(element.page, "number"):
            pages.add(element.page.number)
    show_page_column = len(pages) > 1

    # Group by type
    by_type = {}
    for element in elements:
        element_type = getattr(element, "type", "unknown")
        by_type.setdefault(element_type, []).append(element)

    # Create tables for each type (exclude chars - too granular)
    for element_type, type_elements in by_type.items():
        if element_type == "char":
            # Skip character elements - too granular for useful inspection
            continue

        # Limit elements shown
        display_elements = type_elements[:limit]

        # Get appropriate columns for this type
        columns = _get_columns_for_type(element_type, show_page_column)

        # Extract data for each element
        element_data = []
        for element in display_elements:
            row = {}
            for col in columns:
                value = _extract_element_value(element, col)
                row[col] = value
            element_data.append(row)

        # Create section
        section_name = f"{element_type}_elements"
        section_data = {"elements": element_data, "columns": columns}

        # Add note if truncated
        if len(type_elements) > limit:
            section_data["note"] = (
                f"Showing {limit} of {len(type_elements)} elements (pass limit= to see more)"
            )

        data[section_name] = section_data

    # Count non-char elements for title
    non_char_count = len([e for e in elements if getattr(e, "type", "unknown") != "char"])
    title = f"Collection Inspection ({non_char_count} elements)"
    return InspectionSummary(data, title)


def _get_columns_for_type(element_type: str, show_page_column: bool) -> List[str]:
    """Get appropriate columns for element type."""
    base_columns = ["x0", "top", "x1", "bottom"]

    if element_type == "word":
        columns = (
            ["text"]
            + base_columns
            + [
                "font_family",
                "font_variant",
                "size",
                "bold",
                "italic",
                "strike",
                "underline",
                "highlight",
                "source",
                "confidence",
            ]
        )
        # Add foreground text colour too
        columns.append("color")
    elif element_type == "rect":
        columns = base_columns + ["width", "height", "stroke", "fill", "stroke_width"]
    elif element_type == "line":
        columns = base_columns + ["width", "is_horizontal", "is_vertical"]  # LineElement properties
    elif element_type == "region":
        columns = base_columns + ["width", "height", "type", "color"]
    elif element_type == "blob":
        columns = base_columns + ["width", "height", "color"]
    else:
        columns = base_columns + ["type"]

    if show_page_column:
        columns.append("page")

    return columns


def _extract_element_value(element: "Element", column: str) -> Any:
    """Extract value for a column from an element."""
    try:
        if column == "text":
            text = getattr(element, "text", "")
            if text and len(text) > 60:
                return text[:60] + "..."
            return text or ""

        elif column == "page":
            if hasattr(element, "page") and hasattr(element.page, "number"):
                return element.page.number
            return ""

        elif column == "confidence":
            confidence = getattr(element, "confidence", None)
            if confidence is not None and isinstance(confidence, (int, float)):
                return f"{confidence:.2f}"
            return ""

        elif column == "font_family":
            # Use the cleaner font_family property from TextElement
            font_family = getattr(element, "font_family", None)
            if font_family:
                return font_family
            # Fallback to fontname
            return getattr(element, "fontname", "")

        elif column == "font_variant":
            variant = getattr(element, "font_variant", None)
            if variant:
                return variant
            # Fallback – try to derive from fontname if property missing
            fontname = getattr(element, "fontname", "")
            if "+" in fontname:
                return fontname.split("+", 1)[0]
            return ""

        elif column in ["bold", "italic", "strike", "underline"]:
            value = getattr(element, column, False)
            return value if isinstance(value, bool) else False

        elif column == "highlight":
            # If element is highlighted, return its colour; otherwise blank
            if getattr(element, "highlight", False):
                col_val = getattr(element, "highlight_color", None)
                if col_val is None:
                    return "True"  # fallback if colour missing
                # Convert tuple to hex
                if isinstance(col_val, (tuple, list)) and len(col_val) >= 3:
                    try:
                        r, g, b = [int(v * 255) if v <= 1 else int(v) for v in col_val[:3]]
                        return f"#{r:02x}{g:02x}{b:02x}"
                    except Exception:
                        return str(col_val)
                return str(col_val)
            return ""

        elif column in ["stroke", "fill", "color"]:
            value = getattr(element, column, None)
            # If already a string (e.g. '#ff00aa' or 'red') return as is
            if isinstance(value, str):
                return value
            # If tuple/list convert to hex
            if value and isinstance(value, (tuple, list)) and len(value) >= 3:
                try:
                    r, g, b = [int(v * 255) if v <= 1 else int(v) for v in value[:3]]
                    return f"#{r:02x}{g:02x}{b:02x}"
                except Exception:
                    return str(value)
            return ""

        elif column in ["x0", "top", "x1", "bottom", "width", "height", "size", "stroke_width"]:
            value = getattr(element, column, 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(round(value))
            return 0

        elif column in ["is_horizontal", "is_vertical"]:
            value = getattr(element, column, False)
            return value if isinstance(value, bool) else False

        else:
            # Generic attribute access
            value = getattr(element, column, "")
            if value is None:
                return ""
            return str(value)

    except Exception as e:
        # Fallback for any unexpected errors
        logger.warning(f"Error extracting {column} from element: {e}")
        return ""


def describe_element(element: "Element") -> "ElementSummary":
    """
    Describe an individual element with its properties and attributes.

    Args:
        element: The element to describe

    Returns:
        ElementSummary with formatted element properties
    """
    from natural_pdf.describe.summary import ElementSummary

    # Get basic element info
    element_type = getattr(element, "type", element.__class__.__name__)

    # Build the description data - use dict structure for proper list formatting
    data = {
        "info": {
            "object_type": "element",
            "element_type": element_type,
            "class_name": element.__class__.__name__,
        }
    }

    # Add geometric properties - use dict structure for proper list formatting
    if hasattr(element, "bbox"):
        data["geometry"] = {
            "position": f"({round(element.x0, 1)}, {round(element.top, 1)}, {round(element.x1, 1)}, {round(element.bottom, 1)})",
            "size": f"({round(element.width, 1)}, {round(element.height, 1)})",
        }

    # Add text content if available - use dict structure for proper list formatting
    if hasattr(element, "text") and element.text:
        text = str(element.text).strip()
        display_text = text[:50] + "..." if len(text) > 50 else text
        data["content"] = {"text": f"'{display_text}'", "length": f"{len(text)} chars"}

    # Add common text properties - use dict structure for proper list formatting
    text_props = {}
    for prop in [
        "font_family",
        "size",
        "bold",
        "italic",
        "strike",
        "underline",
        "highlight",
        "source",
        "confidence",
    ]:
        if hasattr(element, prop):
            value = getattr(element, prop)
            if value is not None:
                if prop == "confidence" and isinstance(value, (int, float)):
                    text_props[prop] = round(value, 3)
                elif prop == "size" and isinstance(value, (int, float)):
                    text_props[prop] = round(value, 1)
                elif prop in ["bold", "italic", "strike", "underline"]:
                    text_props[prop] = value
                else:
                    text_props[prop] = value

    if text_props:
        data["properties"] = text_props

    # Add color information - use dict structure for proper list formatting
    color_info = {}
    for prop in ["color", "fill", "stroke"]:
        if hasattr(element, prop):
            value = getattr(element, prop)
            if value is not None:
                if isinstance(value, (tuple, list)) and len(value) >= 3:
                    # Convert RGB to hex if it's a color tuple
                    try:
                        if all(isinstance(v, (int, float)) for v in value[:3]):
                            r, g, b = [int(v * 255) if v <= 1 else int(v) for v in value[:3]]
                            color_info[prop] = f"#{r:02x}{g:02x}{b:02x}"
                        else:
                            color_info[prop] = str(value)
                    except:
                        color_info[prop] = str(value)
                else:
                    color_info[prop] = str(value)

    if color_info:
        data["colors"] = color_info

    # Add page information - use dict structure for proper list formatting
    if hasattr(element, "page") and element.page:
        page_num = getattr(element.page, "number", None)
        if page_num is not None:
            data["page"] = {"number": page_num}

    # Add polygon information if available - use dict structure for proper list formatting
    if hasattr(element, "has_polygon") and element.has_polygon:
        if hasattr(element, "polygon"):
            polygon = element.polygon
            if polygon and len(polygon) > 0:
                data["shape"] = {"polygon_points": len(polygon)}

    # Create title
    title = f"{element_type.title()} Element"
    if hasattr(element, "text") and element.text:
        preview = str(element.text).strip()[:30]
        if preview:
            title += f": '{preview}'"

    return ElementSummary(data, title)
