import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, cast

if TYPE_CHECKING:
    from natural_pdf.core.page import Page as PhysicalPage  # For type checking physical_object.page
    from natural_pdf.elements.base import Element as PhysicalElement
    from natural_pdf.elements.region import Region as PhysicalRegion

    from .flow import Flow
    from .region import FlowRegion

logger = logging.getLogger(__name__)


class FlowElement:
    """
    Represents a physical PDF Element or Region that is anchored within a Flow.
    This class provides methods for flow-aware directional navigation (e.g., below, above)
    that operate across the segments defined in its associated Flow.
    """

    def __init__(self, physical_object: Union["PhysicalElement", "PhysicalRegion"], flow: "Flow"):
        """
        Initializes a FlowElement.

        Args:
            physical_object: The actual natural_pdf.elements.base.Element or
                             natural_pdf.elements.region.Region object.
            flow: The Flow instance this element is part of.
        """
        if not (hasattr(physical_object, "bbox") and hasattr(physical_object, "page")):
            raise TypeError(
                f"physical_object must be a valid PDF element-like object with 'bbox' and 'page' attributes. Got {type(physical_object)}"
            )
        self.physical_object: Union["PhysicalElement", "PhysicalRegion"] = physical_object
        self.flow: "Flow" = flow

    # --- Properties to delegate to the physical_object ---
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self.physical_object.bbox

    @property
    def x0(self) -> float:
        return self.physical_object.x0

    @property
    def top(self) -> float:
        return self.physical_object.top

    @property
    def x1(self) -> float:
        return self.physical_object.x1

    @property
    def bottom(self) -> float:
        return self.physical_object.bottom

    @property
    def width(self) -> float:
        return self.physical_object.width

    @property
    def height(self) -> float:
        return self.physical_object.height

    @property
    def text(self) -> Optional[str]:
        return getattr(self.physical_object, "text", None)

    @property
    def page(self) -> Optional["PhysicalPage"]:
        """Returns the physical page of the underlying element."""
        return getattr(self.physical_object, "page", None)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate unknown attribute access to the physical_object.

        This ensures that attributes like 'type', 'region_type', 'source', 'model', etc.
        from the physical element are accessible on the FlowElement wrapper.

        Args:
            name: The attribute name being accessed

        Returns:
            The attribute value from physical_object

        Raises:
            AttributeError: If the attribute doesn't exist on physical_object either
        """
        try:
            return getattr(self.physical_object, name)
        except AttributeError:
            # Provide a helpful error message that mentions both FlowElement and physical_object
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' "
                f"(also not found on underlying {type(self.physical_object).__name__})"
            )

    def _flow_direction(
        self,
        direction: str,  # "above", "below", "left", "right"
        size: Optional[float] = None,
        cross_size_ratio: Optional[float] = None,  # Default to None for full flow width
        cross_size_absolute: Optional[float] = None,
        cross_alignment: str = "center",  # "start", "center", "end"
        until: Optional[str] = None,
        include_source: bool = False,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegion":
        # Ensure correct import for creating new PhysicalRegion instances if needed
        from natural_pdf.elements.region import Region as PhysicalRegion_Class  # Runtime import

        from .region import (  # Runtime import for return if not stringized, but stringizing is safer
            FlowRegion,
        )

        collected_constituent_regions: List[PhysicalRegion_Class] = (
            []
        )  # PhysicalRegion_Class is runtime
        boundary_element_hit: Optional["PhysicalElement"] = None  # Stringized
        # Ensure remaining_size is float, even if size is int.
        remaining_size = float(size) if size is not None else float("inf")

        # 1. Identify Starting Segment and its index
        start_segment_index = -1
        for i, segment_in_flow in enumerate(self.flow.segments):
            if self.physical_object.page != segment_in_flow.page:
                continue

            obj_center_x = (self.physical_object.x0 + self.physical_object.x1) / 2
            obj_center_y = (self.physical_object.top + self.physical_object.bottom) / 2

            if segment_in_flow.is_point_inside(obj_center_x, obj_center_y):
                start_segment_index = i
                break
            obj_bbox = self.physical_object.bbox
            seg_bbox = segment_in_flow.bbox
            if not (
                obj_bbox[2] < seg_bbox[0]
                or obj_bbox[0] > seg_bbox[2]
                or obj_bbox[3] < seg_bbox[1]
                or obj_bbox[1] > seg_bbox[3]
            ):
                if start_segment_index == -1:
                    start_segment_index = i

        if start_segment_index == -1:
            page_num_str = (
                str(self.physical_object.page.page_number) if self.physical_object.page else "N/A"
            )
            logger.warning(
                f"FlowElement's physical object {self.physical_object.bbox} on page {page_num_str} "
                f"not found within any flow segment. Cannot perform directional operation '{direction}'."
            )
            # Need FlowRegion for the return type, ensure it's available or stringized
            from .region import FlowRegion as RuntimeFlowRegion

            return RuntimeFlowRegion(
                flow=self.flow,
                constituent_regions=[],
                source_flow_element=self,
                boundary_element_found=None,
            )

        is_primary_vertical = self.flow.arrangement == "vertical"
        segment_iterator: range

        if direction == "below":
            if not is_primary_vertical:
                raise NotImplementedError("'below' is for vertical flows.")
            is_forward = True
            segment_iterator = range(start_segment_index, len(self.flow.segments))
        elif direction == "above":
            if not is_primary_vertical:
                raise NotImplementedError("'above' is for vertical flows.")
            is_forward = False
            segment_iterator = range(start_segment_index, -1, -1)
        elif direction == "right":
            is_forward = True
            segment_iterator = range(start_segment_index, len(self.flow.segments))
        elif direction == "left":
            is_forward = False
            segment_iterator = range(start_segment_index, -1, -1)
        else:
            raise ValueError(
                f"Internal error: Invalid direction '{direction}' for _flow_direction."
            )

        for current_segment_idx in segment_iterator:
            if remaining_size <= 0 and size is not None:
                break
            if boundary_element_hit:
                break

            current_segment: PhysicalRegion_Class = self.flow.segments[current_segment_idx]
            segment_contribution: Optional[PhysicalRegion_Class] = None

            op_source: Union["PhysicalElement", PhysicalRegion_Class]  # Stringized PhysicalElement
            op_direction_params: dict = {
                "direction": direction,
                "until": until,
                "include_endpoint": include_endpoint,
                "include_source": include_source,
                **kwargs,
            }

            # --- Cross-size logic: Default based on direction ---
            cross_size_for_op: Union[str, float]
            if cross_size_absolute is not None:
                cross_size_for_op = cross_size_absolute
            elif cross_size_ratio is not None:  # User explicitly provided a ratio
                # Cross dimension depends on direction, not flow arrangement
                base_cross_dim = (
                    self.physical_object.width
                    if direction in ["above", "below"]
                    else self.physical_object.height
                )
                cross_size_for_op = base_cross_dim * cross_size_ratio
            else:  # Default case: neither absolute nor ratio provided
                # Default to element size for left/right, full for above/below
                if direction in ["left", "right"]:
                    cross_size_for_op = self.physical_object.height
                else:
                    cross_size_for_op = "full"
            op_direction_params["cross_size"] = cross_size_for_op

            if current_segment_idx == start_segment_index:
                op_source = self.physical_object
                op_direction_params["size"] = remaining_size if size is not None else None
                op_direction_params["include_source"] = include_source

                source_for_op_call = op_source
                if not isinstance(source_for_op_call, PhysicalRegion_Class):
                    if hasattr(source_for_op_call, "to_region"):
                        source_for_op_call = source_for_op_call.to_region()
                    else:
                        logger.error(
                            f"FlowElement: Cannot convert op_source {type(op_source)} to region."
                        )
                        continue

                # 1. Perform directional operation *without* 'until' initially to get basic shape.
                initial_op_params = {
                    "direction": direction,
                    "size": remaining_size if size is not None else None,
                    "cross_size": cross_size_for_op,
                    "cross_alignment": cross_alignment,  # Pass alignment
                    "include_source": include_source,
                    "_from_flow": True,  # Prevent multipage recursion
                    # Pass other relevant kwargs if Region._direction uses them (e.g. strict_type)
                    **{k: v for k, v in kwargs.items() if k in ["strict_type", "first_match_only"]},
                }
                initial_region_from_op = source_for_op_call._direction(**initial_op_params)

                # 2. Clip this initial region to the current flow segment's boundaries.
                clipped_search_area = current_segment.clip(initial_region_from_op)
                segment_contribution = clipped_search_area  # Default contribution

                # 3. If 'until' is specified, search for it *within* the clipped_search_area.
                if (
                    until
                    and clipped_search_area
                    and clipped_search_area.width > 0
                    and clipped_search_area.height > 0
                ):
                    # kwargs for find_all are the general kwargs passed to _flow_direction
                    until_matches = clipped_search_area.find_all(until, **kwargs)

                    if until_matches:
                        potential_hit: Optional["PhysicalElement"] = None
                        if direction == "below":
                            potential_hit = until_matches.sort(key=lambda m: m.top).first
                        elif direction == "above":
                            potential_hit = until_matches.sort(
                                key=lambda m: m.bottom, reverse=True
                            ).first
                        elif direction == "right":
                            potential_hit = until_matches.sort(key=lambda m: m.x0).first
                        elif direction == "left":
                            potential_hit = until_matches.sort(
                                key=lambda m: m.x1, reverse=True
                            ).first

                        if potential_hit:
                            boundary_element_hit = potential_hit  # Set the overall boundary flag
                            # Adjust segment_contribution to stop at this boundary_element_hit.
                            if direction in ["below", "above"]:
                                if direction == "below":
                                    edge = (
                                        boundary_element_hit.bottom
                                        if include_endpoint
                                        else (boundary_element_hit.top - 1)
                                    )
                                else:  # direction == "above"
                                    edge = (
                                        boundary_element_hit.top
                                        if include_endpoint
                                        else (boundary_element_hit.bottom + 1)
                                    )
                                segment_contribution = segment_contribution.clip(
                                    bottom=edge if direction == "below" else None,
                                    top=edge if direction == "above" else None,
                                )
                            else:  # direction in ["right", "left"]
                                if direction == "right":
                                    edge = (
                                        boundary_element_hit.x1
                                        if include_endpoint
                                        else (boundary_element_hit.x0 - 1)
                                    )
                                else:  # direction == "left"
                                    edge = (
                                        boundary_element_hit.x0
                                        if include_endpoint
                                        else (boundary_element_hit.x1 + 1)
                                    )
                                segment_contribution = segment_contribution.clip(
                                    right=edge if direction == "right" else None,
                                    left=edge if direction == "left" else None,
                                )
            else:
                candidate_region_in_segment = current_segment
                if until and not boundary_element_hit:
                    until_matches = candidate_region_in_segment.find_all(until, **kwargs)
                    if until_matches:
                        potential_hit = None
                        if direction == "below":
                            potential_hit = until_matches.sort(key=lambda m: m.top).first
                        elif direction == "above":
                            potential_hit = until_matches.sort(
                                key=lambda m: m.bottom, reverse=True
                            ).first
                        elif direction == "right":
                            potential_hit = until_matches.sort(key=lambda m: m.x0).first
                        elif direction == "left":
                            potential_hit = until_matches.sort(
                                key=lambda m: m.x1, reverse=True
                            ).first

                        if potential_hit:
                            boundary_element_hit = potential_hit
                            if direction in ["below", "above"]:
                                if direction == "below":
                                    edge = (
                                        boundary_element_hit.bottom
                                        if include_endpoint
                                        else (boundary_element_hit.top - 1)
                                    )
                                else:  # direction == "above"
                                    edge = (
                                        boundary_element_hit.top
                                        if include_endpoint
                                        else (boundary_element_hit.bottom + 1)
                                    )
                                candidate_region_in_segment = candidate_region_in_segment.clip(
                                    bottom=edge if direction == "below" else None,
                                    top=edge if direction == "above" else None,
                                )
                            else:  # direction in ["right", "left"]
                                if direction == "right":
                                    edge = (
                                        boundary_element_hit.x1
                                        if include_endpoint
                                        else (boundary_element_hit.x0 - 1)
                                    )
                                else:  # direction == "left"
                                    edge = (
                                        boundary_element_hit.x0
                                        if include_endpoint
                                        else (boundary_element_hit.x1 + 1)
                                    )
                                candidate_region_in_segment = candidate_region_in_segment.clip(
                                    right=edge if direction == "right" else None,
                                    left=edge if direction == "left" else None,
                                )
                segment_contribution = candidate_region_in_segment

            if (
                segment_contribution
                and segment_contribution.width > 0
                and segment_contribution.height > 0
                and size is not None
            ):
                current_part_consumed_size = 0.0
                if direction in ["below", "above"]:
                    current_part_consumed_size = segment_contribution.height
                    if current_part_consumed_size > remaining_size:
                        new_edge = (
                            (segment_contribution.top + remaining_size)
                            if is_forward
                            else (segment_contribution.bottom - remaining_size)
                        )
                        segment_contribution = segment_contribution.clip(
                            bottom=new_edge if is_forward else None,
                            top=new_edge if not is_forward else None,
                        )
                        current_part_consumed_size = remaining_size
                else:  # direction in ["left", "right"]
                    current_part_consumed_size = segment_contribution.width
                    if current_part_consumed_size > remaining_size:
                        new_edge = (
                            (segment_contribution.x0 + remaining_size)
                            if is_forward
                            else (segment_contribution.x1 - remaining_size)
                        )
                        segment_contribution = segment_contribution.clip(
                            right=new_edge if is_forward else None,
                            left=new_edge if not is_forward else None,
                        )
                        current_part_consumed_size = remaining_size
                remaining_size -= current_part_consumed_size

            if (
                segment_contribution
                and segment_contribution.width > 0
                and segment_contribution.height > 0
            ):
                collected_constituent_regions.append(segment_contribution)

            # If boundary was hit in this segment, and we are not on the start segment (where we might still collect part of it)
            # or if we are on the start segment AND the contribution became zero (e.g. until was immediate)
            if boundary_element_hit and (
                current_segment_idx != start_segment_index
                or not segment_contribution
                or (segment_contribution.width <= 0 or segment_contribution.height <= 0)
            ):
                break  # Stop iterating through more segments

            is_logically_last_segment = (
                is_forward and current_segment_idx == len(self.flow.segments) - 1
            ) or (not is_forward and current_segment_idx == 0)
            if not is_logically_last_segment and self.flow.segment_gap > 0 and size is not None:
                if remaining_size > 0:
                    remaining_size -= self.flow.segment_gap

        from .region import FlowRegion as RuntimeFlowRegion  # Ensure it's available for return

        return RuntimeFlowRegion(
            flow=self.flow,
            constituent_regions=collected_constituent_regions,
            source_flow_element=self,
            boundary_element_found=boundary_element_hit,
        )

    # --- Public Directional Methods ---
    # These will largely mirror DirectionalMixin but call _flow_direction.

    def above(
        self,
        height: Optional[float] = None,
        width_ratio: Optional[float] = None,
        width_absolute: Optional[float] = None,
        width_alignment: str = "center",
        until: Optional[str] = None,
        include_source: bool = False,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegion":  # Stringized
        if self.flow.arrangement == "vertical":
            return self._flow_direction(
                direction="above",
                size=height,
                cross_size_ratio=width_ratio,
                cross_size_absolute=width_absolute,
                cross_alignment=width_alignment,
                until=until,
                include_source=include_source,
                include_endpoint=include_endpoint,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "'above' in a horizontal flow is ambiguous with current 1D flow logic and not yet implemented."
            )

    def below(
        self,
        height: Optional[float] = None,
        width_ratio: Optional[float] = None,
        width_absolute: Optional[float] = None,
        width_alignment: str = "center",
        until: Optional[str] = None,
        include_source: bool = False,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegion":  # Stringized
        if self.flow.arrangement == "vertical":
            return self._flow_direction(
                direction="below",
                size=height,
                cross_size_ratio=width_ratio,
                cross_size_absolute=width_absolute,
                cross_alignment=width_alignment,
                until=until,
                include_source=include_source,
                include_endpoint=include_endpoint,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "'below' in a horizontal flow is ambiguous with current 1D flow logic and not yet implemented."
            )

    def left(
        self,
        width: Optional[float] = None,
        height_ratio: Optional[float] = None,
        height_absolute: Optional[float] = None,
        height_alignment: str = "center",
        until: Optional[str] = None,
        include_source: bool = False,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegion":  # Stringized
        return self._flow_direction(
            direction="left",
            size=width,
            cross_size_ratio=height_ratio,
            cross_size_absolute=height_absolute,
            cross_alignment=height_alignment,
            until=until,
            include_source=include_source,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def right(
        self,
        width: Optional[float] = None,
        height_ratio: Optional[float] = None,
        height_absolute: Optional[float] = None,
        height_alignment: str = "center",
        until: Optional[str] = None,
        include_source: bool = False,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegion":  # Stringized
        return self._flow_direction(
            direction="right",
            size=width,
            cross_size_ratio=height_ratio,
            cross_size_absolute=height_absolute,
            cross_alignment=height_alignment,
            until=until,
            include_source=include_source,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"<FlowElement for {self.physical_object.__class__.__name__} {self.bbox} in {self.flow}>"
