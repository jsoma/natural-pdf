"""Spatial utilities for consistent element-region relationships.

This module centralizes the logic for determining whether elements belong to regions,
ensuring consistent behavior across Region, Page, and Flow components.

The default strategy is 'center' - an element belongs to a region if its center
point falls within that region. This prevents double-counting of elements at
boundaries and provides predictable behavior for operations like get_sections()
with include_boundaries='none'.

Example:
    from natural_pdf.utils.spatial import is_element_in_region

    # Check if element is in region using center-based logic (default)
    if is_element_in_region(element, region):
        print("Element is in region")

    # Use different strategies
    if is_element_in_region(element, region, strategy="intersects"):
        print("Element overlaps with region")
"""

import logging
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.region import Region

logger = logging.getLogger(__name__)

# Element inclusion strategies
InclusionStrategy = Literal["center", "intersects", "contains"]


def is_element_in_region(
    element: "Element",
    region: "Region",
    strategy: InclusionStrategy = "center",
    check_page: bool = True,
) -> bool:
    """
    Unified function to check if an element is inside a region.

    This centralizes the logic used across Region, Page, and Flow to ensure
    consistent behavior throughout the library.

    Args:
        element: The element to check
        region: The region to check against
        strategy: The inclusion strategy to use:
            - "center": Element belongs if its center point is inside (default)
            - "intersects": Element belongs if any part overlaps
            - "contains": Element belongs only if fully contained
        check_page: Whether to verify element and region are on the same page

    Returns:
        bool: True if element is in region according to the strategy
    """
    # Validate inputs
    if not hasattr(element, "bbox") or not element.bbox:
        logger.debug(f"Element lacks bbox attributes: {element}")
        return False

    if not hasattr(region, "bbox") or not region.bbox:
        logger.debug(f"Region lacks bbox attributes: {region}")
        return False

    # Check page membership if requested
    if check_page:
        if not hasattr(element, "page") or not hasattr(region, "page"):
            return False
        if element.page != region.page:
            return False

    # Apply the appropriate strategy
    if strategy == "center":
        # Use existing region method if available
        if hasattr(region, "is_element_center_inside"):
            return region.is_element_center_inside(element)
        else:
            # Fallback calculation
            elem_center_x = (element.x0 + element.x1) / 2
            elem_center_y = (element.top + element.bottom) / 2

            # Use region's is_point_inside if available
            if hasattr(region, "is_point_inside"):
                return region.is_point_inside(elem_center_x, elem_center_y)
            else:
                # Simple bounds check
                return (
                    region.x0 <= elem_center_x <= region.x1
                    and region.top <= elem_center_y <= region.bottom
                )

    elif strategy == "intersects":
        # Use existing region method if available
        if hasattr(region, "intersects"):
            return region.intersects(element)
        else:
            # Simple bbox overlap check
            return not (
                element.x1 < region.x0
                or element.x0 > region.x1
                or element.bottom < region.top
                or element.top > region.bottom
            )

    elif strategy == "contains":
        # Use existing region method if available
        if hasattr(region, "contains"):
            return region.contains(element)
        else:
            # Simple full containment check
            return (
                region.x0 <= element.x0
                and element.x1 <= region.x1
                and region.top <= element.top
                and element.bottom <= region.bottom
            )

    else:
        raise ValueError(f"Unknown inclusion strategy: {strategy}")


def get_inclusion_strategy() -> InclusionStrategy:
    """
    Get the current global inclusion strategy.

    This could be made configurable via environment variable or settings.
    For now, returns the default strategy.

    Returns:
        The current inclusion strategy (default: "center")
    """
    # Could read from settings or environment
    # return os.environ.get("NATURAL_PDF_INCLUSION_STRATEGY", "center")
    return "center"


def calculate_element_overlap_percentage(element: "Element", region: "Region") -> float:
    """
    Calculate what percentage of an element overlaps with a region.

    Args:
        element: The element to check
        region: The region to check against

    Returns:
        float: Percentage of element area that overlaps with region (0.0 to 1.0)
    """
    if not hasattr(element, "bbox") or not hasattr(region, "bbox"):
        return 0.0

    # Calculate intersection bounds
    intersect_x0 = max(element.x0, region.x0)
    intersect_y0 = max(element.top, region.top)
    intersect_x1 = min(element.x1, region.x1)
    intersect_y1 = min(element.bottom, region.bottom)

    # Check if there's an intersection
    if intersect_x1 <= intersect_x0 or intersect_y1 <= intersect_y0:
        return 0.0

    # Calculate areas
    element_area = (element.x1 - element.x0) * (element.bottom - element.top)
    if element_area == 0:
        return 0.0

    intersect_area = (intersect_x1 - intersect_x0) * (intersect_y1 - intersect_y0)

    return intersect_area / element_area
