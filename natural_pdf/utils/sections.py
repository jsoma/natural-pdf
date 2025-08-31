"""Centralized utilities for section extraction to avoid code duplication.

This module provides the core logic for get_sections() and get_section_between()
functionality that's used across Page, PDF, Region, and Flow classes.
"""

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.region import Region

logger = logging.getLogger(__name__)


def calculate_section_bounds(
    start_element: "Element",
    end_element: "Element",
    include_boundaries: str,
    orientation: str,
    parent_bounds: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Calculate the bounding box for a section between two elements.

    This centralizes the logic for determining section boundaries based on
    the include_boundaries parameter and orientation.

    Args:
        start_element: Element marking the start of the section
        end_element: Element marking the end of the section
        include_boundaries: How to include boundary elements: 'start', 'end', 'both', or 'none'
        orientation: 'vertical' or 'horizontal' - determines section direction
        parent_bounds: The bounding box (x0, top, x1, bottom) of the parent region

    Returns:
        Tuple of (x0, top, x1, bottom) representing the section bounds
    """
    parent_x0, parent_top, parent_x1, parent_bottom = parent_bounds

    if orientation == "vertical":
        # Use full width of the parent region for vertical sections
        x0 = parent_x0
        x1 = parent_x1

        # Determine vertical boundaries based on include_boundaries
        if include_boundaries == "both":
            # Include both boundary elements
            top = start_element.top
            bottom = end_element.bottom
        elif include_boundaries == "start":
            # Include start element, exclude end element
            top = start_element.top
            bottom = end_element.top  # Stop at the top of end element
        elif include_boundaries == "end":
            # Exclude start element, include end element
            top = start_element.bottom  # Start at the bottom of start element
            bottom = end_element.bottom
        else:  # "none"
            # Exclude both boundary elements
            top = start_element.bottom  # Start at the bottom of start element
            bottom = end_element.top  # Stop at the top of end element

    else:  # horizontal
        # Use full height of the parent region for horizontal sections
        top = parent_top
        bottom = parent_bottom

        # Determine horizontal boundaries based on include_boundaries
        if include_boundaries == "both":
            # Include both boundary elements
            x0 = start_element.x0
            x1 = end_element.x1
        elif include_boundaries == "start":
            # Include start element, exclude end element
            x0 = start_element.x0
            x1 = end_element.x0  # Stop at the left of end element
        elif include_boundaries == "end":
            # Exclude start element, include end element
            x0 = start_element.x1  # Start at the right of start element
            x1 = end_element.x1
        else:  # "none"
            # Exclude both boundary elements
            x0 = start_element.x1  # Start at the right of start element
            x1 = end_element.x0  # Stop at the left of end element

    return (x0, top, x1, bottom)


def validate_section_bounds(bounds: Tuple[float, float, float, float], orientation: str) -> bool:
    """
    Validate that section bounds are valid (not inverted).

    Args:
        bounds: The bounding box (x0, top, x1, bottom) to validate
        orientation: 'vertical' or 'horizontal' - determines which dimension to check

    Returns:
        True if bounds are valid, False otherwise
    """
    x0, top, x1, bottom = bounds

    if orientation == "vertical":
        if top >= bottom:
            logger.debug(f"Invalid vertical section boundaries: top={top} >= bottom={bottom}")
            return False
    else:  # horizontal
        if x0 >= x1:
            logger.debug(f"Invalid horizontal section boundaries: x0={x0} >= x1={x1}")
            return False

    return True


def pair_boundary_elements(
    start_elements: List["Element"],
    end_elements: Optional[List["Element"]],
    orientation: str = "vertical",
) -> List[Tuple["Element", Optional["Element"]]]:
    """
    Pair up start and end boundary elements for section extraction.

    This implements the logic for matching start elements with their corresponding
    end elements, handling cases where end_elements is None or has different length.

    Args:
        start_elements: List of elements marking section starts
        end_elements: Optional list of elements marking section ends
        orientation: 'vertical' or 'horizontal' - affects element ordering

    Returns:
        List of (start_element, end_element) tuples
    """
    if not start_elements:
        return []

    # Sort elements by position
    if orientation == "vertical":
        start_elements = sorted(start_elements, key=lambda e: (e.top, e.x0))
        if end_elements:
            end_elements = sorted(end_elements, key=lambda e: (e.top, e.x0))
    else:
        start_elements = sorted(start_elements, key=lambda e: (e.x0, e.top))
        if end_elements:
            end_elements = sorted(end_elements, key=lambda e: (e.x0, e.top))

    pairs = []

    if not end_elements:
        # No end elements - pair each start with the next start
        for i in range(len(start_elements) - 1):
            pairs.append((start_elements[i], start_elements[i + 1]))
        # Last element has no pair unless we want to go to end of container
        pairs.append((start_elements[-1], None))
    else:
        # Match each start with the next end that comes after it
        used_ends = set()

        for start in start_elements:
            # Find the first unused end element that comes after this start
            matching_end = None

            for end in end_elements:
                if end in used_ends:
                    continue

                # Check if end comes after start
                if orientation == "vertical":
                    if end.top > start.bottom or (end.top == start.bottom and end.x0 >= start.x0):
                        matching_end = end
                        break
                else:  # horizontal
                    if end.x0 > start.x1 or (end.x0 == start.x1 and end.top >= start.top):
                        matching_end = end
                        break

            if matching_end:
                pairs.append((start, matching_end))
                used_ends.add(matching_end)
            else:
                # No matching end found
                pairs.append((start, None))

    return pairs


def process_selector_to_elements(
    selector_or_elements: Union[str, List["Element"], "Element", None],
    search_context: Any,  # Can be Page, Region, Flow, etc.
    find_method_name: str = "find_all",
) -> List["Element"]:
    """
    Process a selector string or element list into a normalized list of elements.

    Args:
        selector_or_elements: Selector string, element, list of elements, or None
        search_context: Object with find_all method (Page, Region, etc.)
        find_method_name: Name of the method to call for searching (default: "find_all")

    Returns:
        List of elements (empty list if None or no matches)
    """
    if selector_or_elements is None:
        return []

    if isinstance(selector_or_elements, str):
        # It's a selector string - search for matching elements
        if hasattr(search_context, find_method_name):
            result = getattr(search_context, find_method_name)(selector_or_elements)
            if hasattr(result, "elements"):
                return result.elements
            elif isinstance(result, list):
                return result
            else:
                return []
        else:
            logger.warning(f"Search context {type(search_context)} lacks {find_method_name} method")
            return []

    # Handle single element
    if hasattr(selector_or_elements, "bbox"):  # Duck typing for Element
        return [selector_or_elements]

    # Handle ElementCollection or similar
    if hasattr(selector_or_elements, "elements"):
        return selector_or_elements.elements

    # Handle list/iterable
    if hasattr(selector_or_elements, "__iter__"):
        return list(selector_or_elements)

    return []


def extract_sections_from_region(
    region: "Region",
    start_elements: Union[str, List["Element"], None],
    end_elements: Union[str, List["Element"], None] = None,
    include_boundaries: str = "both",
    orientation: str = "vertical",
    get_section_between_func: Optional[Any] = None,
) -> List["Region"]:
    """
    Core implementation of get_sections() that can be reused across classes.

    This implements the full logic for extracting multiple sections from a region
    based on start/end boundary elements.

    Args:
        region: The region to extract sections from
        start_elements: Elements or selector marking section starts
        end_elements: Optional elements or selector marking section ends
        include_boundaries: How to include boundary elements
        orientation: Section orientation ('vertical' or 'horizontal')
        get_section_between_func: Optional custom function to create sections

    Returns:
        List of Region objects representing the sections
    """
    # Process selectors to get element lists
    start_elements = process_selector_to_elements(start_elements, region)
    end_elements = process_selector_to_elements(end_elements, region) if end_elements else []

    # Validate inputs
    if not start_elements:
        logger.debug("No start elements found for section extraction")
        return []

    # Get all elements in the region and sort by position
    all_elements = region.get_elements()
    if not all_elements:
        return []

    # Sort elements based on orientation
    if orientation == "vertical":
        all_elements.sort(key=lambda e: (e.top, e.x0))
    else:
        all_elements.sort(key=lambda e: (e.x0, e.top))

    # Create element index map
    element_to_index = {el: i for i, el in enumerate(all_elements)}

    # Build boundary list with indices
    boundaries = []

    # Add start boundaries
    for elem in start_elements:
        idx = element_to_index.get(elem)
        if idx is not None:
            boundaries.append({"index": idx, "element": elem, "type": "start"})

    # Add end boundaries
    for elem in end_elements:
        idx = element_to_index.get(elem)
        if idx is not None:
            boundaries.append({"index": idx, "element": elem, "type": "end"})

    # Sort boundaries by document order
    boundaries.sort(key=lambda x: x["index"])

    # Generate sections
    sections = []
    current_start = None
    section_func = get_section_between_func or region.get_section_between

    for boundary in boundaries:
        if boundary["type"] == "start":
            if current_start is None:
                # Start a new section
                current_start = boundary
            elif not end_elements:
                # No end elements specified - use starts as both start and end
                # Create section from previous start to this start (which acts as end)
                start_elem = current_start["element"]
                end_elem = boundary["element"]  # Use the actual boundary element as end

                section = section_func(start_elem, end_elem, include_boundaries, orientation)
                sections.append(section)

                # This boundary becomes the new start
                current_start = boundary

        elif boundary["type"] == "end" and current_start:
            # Create section from current start to this end
            section = section_func(
                current_start["element"], boundary["element"], include_boundaries, orientation
            )
            sections.append(section)
            current_start = None

    # Handle final section if we have an unclosed start
    if current_start:
        start_elem = current_start["element"]
        # For the final section, we need to go to the end of the region
        # Create a dummy end element at the region boundary
        if orientation == "vertical":
            # Create section to bottom of region
            section = section_func(start_elem, None, include_boundaries, orientation)
        else:
            # Create section to right edge of region
            section = section_func(start_elem, None, include_boundaries, orientation)
        sections.append(section)

    return sections
