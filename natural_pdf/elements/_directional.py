"""Module-level implementations of directional navigation.

These functions hold the bodies of ``DirectionalMixin._direction`` and
``DirectionalMixin._direction_multipage`` (see ``natural_pdf/elements/base.py``).
The mixin methods are thin wrappers that call these functions with ``self`` as
the ``source`` argument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

from natural_pdf.core.interfaces import SupportsGeometry

if TYPE_CHECKING:
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.region import Region
    from natural_pdf.flows.region import FlowRegion


def compute_direction(
    source: Any,
    direction: str,
    size: Optional[float] = None,
    cross_size: str = "full",
    include_source: bool = False,
    until: Optional[str] = None,
    include_endpoint: bool = True,
    offset: float = 0.0,
    apply_exclusions: bool = True,
    multipage: Optional[bool] = None,
    within: Optional["Region"] = None,
    anchor: str = "start",
    **kwargs,
) -> Optional[Union["Region", "FlowRegion"]]:
    """
    Create a region in a specified direction relative to ``source``.

    Returns None if a 'within' constraint produces a zero/negative-area intersection.

    Args:
        source: The element/region the direction is relative to
        direction: 'left', 'right', 'above', or 'below'
        size: Size in the primary direction (width for horizontal, height for vertical)
        cross_size: Size in the cross direction ('full' or 'element')
        include_source: Whether to include this element/region's area in the result
        until: Optional selector string to specify a boundary element
        include_endpoint: Whether to include the boundary element found by 'until'
        offset: Pixel offset when excluding source/endpoint (default: None, uses natural_pdf.options.layout.directional_offset)
        apply_exclusions: Whether to respect exclusions when using 'until' selector (default: True)
        multipage: If True, allows the region to span multiple pages
        within: Optional region to constrain the result to (default: None)
        anchor: Reference point - 'start', 'center', 'end', or explicit edges like 'top', 'bottom', 'left', 'right'
        **kwargs: Additional parameters for the 'until' selector search

    Returns:
        Region object
    """
    from natural_pdf.elements.base import (
        _DirectionalHost,
        _get_directional_option,
        _get_directional_within,
    )

    is_horizontal = direction in ("left", "right")

    # The cross-direction parameter is a mode string. A number here used to be
    # silently treated like "element" — the single most-reported silent-wrong
    # behavior from both users and LLM agents — so reject it loudly.
    if cross_size not in ("full", "element"):
        cross_param = "height" if is_horizontal else "width"
        extent_param = "width" if is_horizontal else "height"
        raise TypeError(
            f"{direction}() got {cross_param}={cross_size!r}: {cross_param} selects the "
            f"cross-direction mode and must be 'full' or 'element'. "
            f"Use {extent_param}=<number> to control how far the region extends "
            f"{direction}; for a custom cross-size, use "
            f"{direction}({cross_param}='element').expand(...) or page.region(...)."
        )
    is_positive = direction in ("right", "below")  # right/below are positive directions
    host = cast(_DirectionalHost, source)
    pixel_offset = offset  # Use provided offset for excluding elements/endpoints

    # initialise optional coordinate holders to satisfy static checkers
    x0_initial = x1_initial = host.x0
    y0_initial = y1_initial = host.top
    x0_final = x1_final = host.x0
    y0_final = y1_final = host.top
    y0 = host.top
    y1 = host.bottom
    x0 = host.x0
    x1 = host.x1

    # Normalize anchor parameter
    def normalize_anchor(anchor_value: str, dir: str) -> str:
        """Convert start/end/center to explicit edges based on direction."""
        if anchor_value == "center":
            return "center"
        if anchor_value == "start":
            # Start means the edge where the directional region begins
            # (the boundary between source and the new region)
            if dir == "below":
                return "bottom"
            if dir == "above":
                return "top"
            if dir == "right":
                return "right"
            if dir == "left":
                return "left"
        elif anchor_value == "end":
            # End means the opposite edge (allows finding elements that overlap with source)
            if dir == "below":
                return "top"
            if dir == "above":
                return "bottom"
            if dir == "right":
                return "left"
            if dir == "left":
                return "right"
        # Already explicit (top/bottom/left/right) or unhandled direction fallback
        return anchor_value

    normalized_anchor = normalize_anchor(anchor, direction)

    # 1. Determine initial boundaries based on direction and include_source
    if is_horizontal:
        # Initial cross-boundaries (vertical)
        y0 = 0 if cross_size == "full" else host.top
        y1 = host.page.height if cross_size == "full" else host.bottom

        # Initial primary boundaries (horizontal)
        if is_positive:  # right
            x0_initial = host.x0 if include_source else host.x1 + pixel_offset
            x1_initial = host.x1  # This edge moves
        else:  # left
            x0_initial = host.x0  # This edge moves
            x1_initial = host.x1 if include_source else host.x0 - pixel_offset
    else:  # Vertical
        # Initial cross-boundaries (horizontal)
        x0 = 0 if cross_size == "full" else host.x0
        x1 = host.page.width if cross_size == "full" else host.x1

        # Initial primary boundaries (vertical)
        if is_positive:  # below
            y0_initial = host.top if include_source else host.bottom + pixel_offset
            y1_initial = host.bottom  # This edge moves
        else:  # above
            y0_initial = host.top  # This edge moves
            y1_initial = host.bottom if include_source else host.top - pixel_offset

    # 2. Calculate the final primary boundary, considering 'size' or page limits
    if is_horizontal:
        if is_positive:  # right
            x1_final = min(
                host.page.width,
                x1_initial + (size if size is not None else (host.page.width - x1_initial)),
            )
            x0_final = x0_initial
        else:  # left
            x0_final = max(0, x0_initial - (size if size is not None else x0_initial))
            x1_final = x1_initial
    else:  # Vertical
        if is_positive:  # below
            y1_final = min(
                host.page.height,
                y1_initial + (size if size is not None else (host.page.height - y1_initial)),
            )
            y0_final = y0_initial
        else:  # above
            y0_final = max(0, y0_initial - (size if size is not None else y0_initial))
            y1_final = y1_initial

    # 3. Handle 'until' selector if provided
    target = None
    if until:
        from natural_pdf.elements.element_collection import ElementCollection

        # An explicit argument takes precedence over task-local/contextual
        # defaults.  ``Region.within()`` is task-local rather than global.
        constraint_region = within or _get_directional_within(host)

        # Check if until uses :closest selector (preserve ordering)
        preserve_order = isinstance(until, str) and ":closest" in until

        # If until is an elementcollection, just use it
        if isinstance(until, ElementCollection):
            # Only take ones on the same page
            all_matches = [
                cast(SupportsGeometry, m)
                for m in until
                if hasattr(m, "page") and getattr(m, "page") == host.page
            ]
        else:
            # If we have a constraint region, search within it instead of the whole page
            from natural_pdf.elements.region import Region

            if isinstance(constraint_region, Region) and constraint_region.page == host.page:
                all_matches = constraint_region.find_all(
                    until, apply_exclusions=apply_exclusions, **kwargs
                )
            else:
                all_matches = host.page.find_all(until, apply_exclusions=apply_exclusions, **kwargs)
        matches_in_direction = []

        # Filter and sort matches based on direction and anchor parameter
        # Also filter by cross-direction bounds when cross_size='element'

        # IMPORTANT: Exclude self from matches to prevent finding ourselves
        all_matches = [m for m in all_matches if m is not source]

        # Filter to objects with the required geometric interface
        geometric_matches: List[SupportsGeometry] = []
        for candidate in all_matches:
            if all(hasattr(candidate, attr) for attr in ("x0", "x1", "top", "bottom", "page")):
                geometric_matches.append(cast(SupportsGeometry, candidate))

        all_matches = geometric_matches

        # Determine reference point based on normalized_anchor
        # Note: We use <= or >= consistently to include adjacent/touching elements,
        # since we already exclude self from matches at line 348.
        if direction == "above":
            if normalized_anchor == "top":
                ref_y = host.top
                comparator = lambda m: m.bottom <= ref_y
            elif normalized_anchor == "center":
                ref_y = (host.top + host.bottom) / 2
                comparator = lambda m: m.bottom <= ref_y
            else:  # 'bottom'
                ref_y = host.bottom
                comparator = lambda m: m.bottom <= ref_y

            matches_in_direction = [m for m in all_matches if comparator(m)]
            # Filter by horizontal bounds if cross_size='element'
            if cross_size == "element":
                matches_in_direction = [
                    m for m in matches_in_direction if m.x0 < host.x1 and m.x1 > host.x0
                ]
            # Only sort by position if not using :closest (which is already sorted by quality)
            if not preserve_order:
                matches_in_direction.sort(key=lambda e: e.bottom, reverse=True)

        elif direction == "below":
            if normalized_anchor == "top":
                ref_y = host.top
                comparator = lambda m: m.top >= ref_y
            elif normalized_anchor == "center":
                ref_y = (host.top + host.bottom) / 2
                comparator = lambda m: m.top >= ref_y
            else:  # 'bottom'
                ref_y = host.bottom
                comparator = lambda m: m.top >= ref_y

            matches_in_direction = [m for m in all_matches if comparator(m)]
            # Filter by horizontal bounds if cross_size='element'
            if cross_size == "element":
                matches_in_direction = [
                    m for m in matches_in_direction if m.x0 < host.x1 and m.x1 > host.x0
                ]
            # Only sort by position if not using :closest (which is already sorted by quality)
            if not preserve_order:
                matches_in_direction.sort(key=lambda e: e.top)

        elif direction == "left":
            if normalized_anchor == "left":
                ref_x = host.x0
                comparator = lambda m: m.x1 <= ref_x
            elif normalized_anchor == "center":
                ref_x = (host.x0 + host.x1) / 2
                comparator = lambda m: m.x1 <= ref_x
            else:  # 'right'
                ref_x = host.x1
                comparator = lambda m: m.x1 <= ref_x

            matches_in_direction = [m for m in all_matches if comparator(m)]
            # Filter by vertical bounds if cross_size='element'
            if cross_size == "element":
                matches_in_direction = [
                    m for m in matches_in_direction if m.top < host.bottom and m.bottom > host.top
                ]
            # Only sort by position if not using :closest (which is already sorted by quality)
            if not preserve_order:
                matches_in_direction.sort(key=lambda e: e.x1, reverse=True)

        elif direction == "right":
            if normalized_anchor == "left":
                ref_x = host.x0
                comparator = lambda m: m.x0 >= ref_x
            elif normalized_anchor == "center":
                ref_x = (host.x0 + host.x1) / 2
                comparator = lambda m: m.x0 >= ref_x
            else:  # 'right'
                ref_x = host.x1
                comparator = lambda m: m.x0 >= ref_x

            matches_in_direction = [m for m in all_matches if comparator(m)]
            # Filter by vertical bounds if cross_size='element'
            if cross_size == "element":
                matches_in_direction = [
                    m for m in matches_in_direction if m.top < host.bottom and m.bottom > host.top
                ]
            # Only sort by position if not using :closest (which is already sorted by quality)
            if not preserve_order:
                matches_in_direction.sort(key=lambda e: e.x0)

        if matches_in_direction:
            target = matches_in_direction[0]

            # Adjust the primary boundary based on the target
            if is_horizontal:
                if is_positive:  # right
                    x1_final = target.x1 if include_endpoint else target.x0 - pixel_offset
                else:  # left
                    x0_final = target.x0 if include_endpoint else target.x1 + pixel_offset
            else:  # Vertical
                if is_positive:  # below
                    if include_endpoint:
                        y1_final = target.bottom
                    else:
                        y1_final = target.top - pixel_offset
                else:  # above
                    if include_endpoint:
                        y0_final = target.top
                    else:
                        y0_final = target.bottom + pixel_offset

            # Adjust cross boundaries if cross_size is 'element'
            if cross_size == "element":
                if is_horizontal:  # Adjust y0, y1
                    y0 = min(y0, host.top)
                    y1 = max(y1, host.bottom)
                else:  # Adjust x0, x1
                    x0 = min(x0, host.x0)
                    x1 = max(x1, host.x1)

    # 4. Finalize bbox coordinates
    if is_horizontal:
        bbox = (x0_final, y0, x1_final, y1)
    else:
        bbox = (x0, y0_final, x1, y1_final)

    # Ensure valid coordinates (x0 <= x1, y0 <= y1)
    final_x0 = min(bbox[0], bbox[2])
    final_y0 = min(bbox[1], bbox[3])
    final_x1 = max(bbox[0], bbox[2])
    final_y1 = max(bbox[1], bbox[3])
    final_bbox = (final_x0, final_y0, final_x1, final_y1)

    # 4.5. Apply an explicit or resolved within constraint.
    constraint_region = within or _get_directional_within(host)
    if constraint_region:
        # Ensure constraint is on same page
        if hasattr(constraint_region, "page") and constraint_region.page != host.page:
            raise ValueError("within constraint must be on the same page as the source element")

        # Apply constraint by intersecting with the constraint region's bounds
        final_x0 = max(final_x0, constraint_region.x0)
        final_y0 = max(final_y0, constraint_region.top)
        final_x1 = min(final_x1, constraint_region.x1)
        final_y1 = min(final_y1, constraint_region.bottom)

        # If constraint produces zero or negative area, return None
        if final_x1 <= final_x0 or final_y1 <= final_y0:
            return None

        # Update final_bbox with constrained values
        final_bbox = (final_x0, final_y0, final_x1, final_y1)

    # 5. Check if multipage is needed
    # Use the PDFContext default if available, otherwise global options.
    if multipage is None:
        use_multipage = _get_directional_option(host, "auto_multipage")
    else:
        use_multipage = multipage

    # Multipage is not supported with within constraint
    if use_multipage and constraint_region:
        raise ValueError("multipage navigation is not supported with within constraint")

    # Prevent recursion: if called with internal flag, don't use multipage
    if kwargs.get("_from_flow", False):
        use_multipage = False

    if use_multipage:
        # Check if we need to cross page boundaries
        needs_multipage = False

        # Case 1: until was specified but target not found on current page
        if until and not target:
            needs_multipage = True

        # Case 2: size extends beyond page boundaries
        if not until:
            if direction == "below" and final_bbox[3] >= host.page.height:
                needs_multipage = True
            elif direction == "above" and final_bbox[1] <= 0:
                needs_multipage = True
            elif direction == "right" and final_bbox[2] >= host.page.width:
                needs_multipage = True
            elif direction == "left" and final_bbox[0] <= 0:
                needs_multipage = True

        if needs_multipage:
            # Use multipage implementation
            return source._direction_multipage(
                direction=direction,
                size=size,
                cross_size=cross_size,
                include_source=include_source,
                until=until,
                include_endpoint=include_endpoint,
                offset=offset,
                apply_exclusions=apply_exclusions,
                **kwargs,
            )

    # 6. Create and return appropriate object based on source type
    from natural_pdf.elements.region import Region

    result = Region(host.page, final_bbox)
    result.source_element = cast("Element | Region", source)
    result.includes_source = include_source
    # Optionally store the boundary element if found
    if target:
        target_type = getattr(target, "type", None) or getattr(target, "object_type", None)
        if target_type != "region":
            result.boundary_element = cast("Element", target)
        setattr(result, "end_element", target)

    return result


def compute_direction_multipage(
    source: Any,
    direction: str,
    size: Optional[float] = None,
    cross_size: str = "full",
    include_source: bool = False,
    until: Optional[str] = None,
    include_endpoint: bool = True,
    offset: float = 0.0,
    apply_exclusions: bool = True,
    **kwargs,
) -> Union["Region", "FlowRegion"]:
    """
    Handle multipage directional navigation by creating a Flow.

    Returns FlowRegion if result spans multiple pages, Region if on single page.
    """
    from natural_pdf.elements.base import Element, _DirectionalHost

    host = cast(_DirectionalHost, source)
    # Get access to the PDF to create a Flow
    pdf = host.page.pdf
    # Find the index of the current page
    current_page_idx = getattr(host.page, "index", None)
    if current_page_idx is None:
        for idx, page in enumerate(pdf.pages):
            if page == host.page:
                current_page_idx = idx
                break

    if current_page_idx is None:
        # Fallback - just use current page
        from natural_pdf.flows.flow import Flow

        flow = Flow(segments=[host.page], arrangement="vertical")
        from natural_pdf.flows.element import FlowElement

        flow_element = FlowElement(physical_object=cast("Element | Region", source), flow=flow)
        return getattr(flow_element, direction)(**kwargs)

    # Determine which pages to include in the Flow based on direction
    if direction in ("below", "right"):
        # Include current page and all following pages
        flow_pages = pdf.pages[current_page_idx:]
    else:  # above, left
        # Include all pages up to and including current page
        flow_pages = pdf.pages[: current_page_idx + 1]

    # Create a temporary Flow
    from natural_pdf.core.page_collection import PageCollection
    from natural_pdf.flows.flow import Flow

    if isinstance(flow_pages, PageCollection):
        segments_source = flow_pages
    else:
        segments_source = list(flow_pages)

    flow = Flow(segments=segments_source, arrangement="vertical")

    # Find the element in the flow
    # We need to create a FlowElement that corresponds to self
    from natural_pdf.flows.element import FlowElement

    flow_element = FlowElement(physical_object=cast("Element | Region", source), flow=flow)

    # Call the directional method on the FlowElement
    # Remove parameters that FlowElement methods don't expect
    flow_kwargs = kwargs.copy()
    flow_kwargs.pop("multipage", None)  # Remove multipage parameter
    flow_kwargs.pop("apply_exclusions", None)  # FlowElement might not have this
    flow_kwargs.pop("offset", None)  # FlowElement doesn't have offset
    flow_kwargs.pop("cross_alignment", None)  # Remove to avoid duplicate

    # Map cross_size to appropriate FlowElement parameter
    if direction in ["below", "above"]:
        # For vertical directions, cross_size maps to width parameters
        if cross_size == "full":
            width_absolute = None  # Let FlowElement use its defaults
        elif cross_size == "element":
            width_absolute = host.width
        elif isinstance(cross_size, (int, float)):
            width_absolute = cross_size
        else:
            width_absolute = None

        result = (
            flow_element.below(
                height=size,
                width_absolute=width_absolute,
                include_source=include_source,
                until=until,
                include_endpoint=include_endpoint,
                **flow_kwargs,
            )
            if direction == "below"
            else flow_element.above(
                height=size,
                width_absolute=width_absolute,
                include_source=include_source,
                until=until,
                include_endpoint=include_endpoint,
                **flow_kwargs,
            )
        )
    else:  # left, right
        # For horizontal directions, cross_size maps to height parameters
        if cross_size == "full":
            height_absolute = None  # Let FlowElement use its defaults
        elif cross_size == "element":
            height_absolute = host.height
        elif isinstance(cross_size, (int, float)):
            height_absolute = cross_size
        else:
            height_absolute = None

        result = (
            flow_element.left(
                width=size,
                height_absolute=height_absolute,
                include_source=include_source,
                until=until,
                include_endpoint=include_endpoint,
                **flow_kwargs,
            )
            if direction == "left"
            else flow_element.right(
                width=size,
                height_absolute=height_absolute,
                include_source=include_source,
                until=until,
                include_endpoint=include_endpoint,
                **flow_kwargs,
            )
        )

    # If the result is a FlowRegion with only one constituent region,
    # return that Region instead
    from natural_pdf.flows.region import FlowRegion

    if isinstance(result, FlowRegion) and len(result.constituent_regions) == 1:
        single_region = result.constituent_regions[0]
        # Copy over any metadata
        if hasattr(result, "boundary_element_found"):
            boundary_candidate = result.boundary_element_found
            if isinstance(boundary_candidate, Element):
                single_region.boundary_element = boundary_candidate
        return single_region

    return result
