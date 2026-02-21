import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from natural_pdf.core.page import Page as PhysicalPage  # For type checking physical_object.page
    from natural_pdf.elements.base import Element as PhysicalElement
    from natural_pdf.elements.region import Region as PhysicalRegion

    from .flow import Flow
    from .region import FlowRegion

logger = logging.getLogger(__name__)

from natural_pdf.selectors.host_mixin import SelectorHostMixin


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

    def _clip_region_until(
        self,
        region: Optional["PhysicalRegion"],
        *,
        direction: str,
        until: Optional[str],
        include_endpoint: bool,
        search_kwargs: Dict[str, Any],
    ) -> Tuple[Optional["PhysicalRegion"], Optional["PhysicalElement"]]:
        """Apply an :param:`until` selector to a candidate region and return the clipped result."""
        if not until or region is None or region.width <= 0 or region.height <= 0:
            return region, None

        until_matches = region.find_all(until, **search_kwargs)
        if not until_matches:
            return region, None

        hit: Optional["PhysicalElement"] = None
        if direction == "below":
            hit = until_matches.sort(key=lambda match: match.top).first
        elif direction == "above":
            hit = until_matches.sort(key=lambda match: match.bottom, reverse=True).first
        elif direction == "right":
            hit = until_matches.sort(key=lambda match: match.x0).first
        elif direction == "left":
            hit = until_matches.sort(key=lambda match: match.x1, reverse=True).first

        if not hit:
            return region, None

        clip_kwargs: Dict[str, float] = {}
        if direction == "below":
            clip_kwargs["bottom"] = hit.bottom if include_endpoint else hit.top - 1
        elif direction == "above":
            clip_kwargs["top"] = hit.top if include_endpoint else hit.bottom + 1
        elif direction == "right":
            clip_kwargs["right"] = hit.x1 if include_endpoint else hit.x0 - 1
        else:  # direction == "left"
            clip_kwargs["left"] = hit.x0 if include_endpoint else hit.x1 + 1

        try:
            clipped_region = region.clip(**clip_kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to clip region using until=%s: %s", until, exc)
            return region, hit

        return clipped_region, hit

    # ------------------------------------------------------------------
    # Helpers extracted from _flow_direction for readability / testability
    # ------------------------------------------------------------------

    def _find_start_segment(self) -> int:
        """Return the index of the flow segment containing this element, or -1."""
        start_segment_index = -1
        for i, segment_in_flow in enumerate(self.flow.segments):
            if self.physical_object.page != segment_in_flow.page:
                continue

            obj_center_x = (self.physical_object.x0 + self.physical_object.x1) / 2
            obj_center_y = (self.physical_object.top + self.physical_object.bottom) / 2

            if segment_in_flow.is_point_inside(obj_center_x, obj_center_y):
                return i

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

        return start_segment_index

    def _resolve_cross_size(
        self,
        direction: str,
        cross_size_ratio: Optional[float],
        cross_size_absolute: Optional[float],
    ) -> Union[str, float]:
        """Determine the cross-axis size value for a directional operation."""
        if cross_size_absolute is not None:
            return cross_size_absolute
        if cross_size_ratio is not None:
            base_cross_dim = (
                self.physical_object.width
                if direction in ("above", "below")
                else self.physical_object.height
            )
            return base_cross_dim * cross_size_ratio
        # Default: element size for left/right, full for above/below
        if direction in ("left", "right"):
            return self.physical_object.height
        return "full"

    @staticmethod
    def _build_segment_iterator(
        direction: str,
        start_index: int,
        num_segments: int,
    ) -> Tuple[range, bool]:
        """Return ``(range, is_forward)`` for iterating segments in *direction*."""
        if direction == "below":
            return range(start_index, num_segments), True
        elif direction == "above":
            return range(start_index, -1, -1), False
        elif direction == "right":
            return range(start_index, num_segments), True
        elif direction == "left":
            return range(start_index, -1, -1), False
        else:
            raise ValueError(
                f"Internal error: Invalid direction '{direction}' for _flow_direction."
            )

    def _shape_start_segment(
        self,
        current_segment: "PhysicalRegion",
        direction: str,
        cross_size_for_op: Union[str, float],
        cross_alignment: str,
        remaining_size: float,
        size: Optional[float],
        include_source: bool,
        until: Optional[str],
        include_endpoint: bool,
        **kwargs,
    ) -> Tuple[Optional["PhysicalRegion"], Optional["PhysicalElement"]]:
        """Build the contribution from the *start* segment.

        Returns ``(clipped_region, boundary_hit_or_None)``.
        """
        from natural_pdf.elements.region import Region as PhysicalRegion_Class

        source_for_op = self.physical_object
        if not isinstance(source_for_op, PhysicalRegion_Class):
            if hasattr(source_for_op, "to_region"):
                source_for_op = source_for_op.to_region()
            else:
                logger.error(
                    "FlowElement: Cannot convert op_source %s to region.",
                    type(self.physical_object),
                )
                return None, None

        initial_op_params: Dict[str, Any] = {
            "direction": direction,
            "size": remaining_size if size is not None else None,
            "cross_size": cross_size_for_op,
            "cross_alignment": cross_alignment,
            "include_source": include_source,
            "_from_flow": True,
            **{k: v for k, v in kwargs.items() if k in ("strict_type", "first_match_only")},
        }
        initial_region = source_for_op._direction(**initial_op_params)

        clipped = current_segment.clip(initial_region)

        contribution, hit = self._clip_region_until(
            clipped,
            direction=direction,
            until=until,
            include_endpoint=include_endpoint,
            search_kwargs=kwargs,
        )
        return contribution, hit

    @staticmethod
    def _clip_to_budget(
        contribution: "PhysicalRegion",
        remaining: float,
        direction: str,
        is_forward: bool,
    ) -> Tuple["PhysicalRegion", float]:
        """Clip *contribution* to fit within *remaining* points.

        Returns ``(possibly_clipped_region, new_remaining)``.
        """
        if direction in ("below", "above"):
            consumed = contribution.height
            if consumed > remaining:
                edge = (
                    (contribution.top + remaining)
                    if is_forward
                    else (contribution.bottom - remaining)
                )
                contribution = contribution.clip(
                    bottom=edge if is_forward else None,
                    top=edge if not is_forward else None,
                )
                consumed = remaining
        else:  # left / right
            consumed = contribution.width
            if consumed > remaining:
                edge = (
                    (contribution.x0 + remaining) if is_forward else (contribution.x1 - remaining)
                )
                contribution = contribution.clip(
                    right=edge if is_forward else None,
                    left=edge if not is_forward else None,
                )
                consumed = remaining

        return contribution, remaining - consumed

    # ------------------------------------------------------------------
    # Main directional engine
    # ------------------------------------------------------------------

    def _flow_direction(
        self,
        direction: str,  # "above", "below", "left", "right"
        size: Optional[float] = None,
        cross_size_ratio: Optional[float] = None,
        cross_size_absolute: Optional[float] = None,
        cross_alignment: str = "center",
        until: Optional[str] = None,
        include_source: bool = False,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegion":
        from natural_pdf.elements.region import Region as PhysicalRegion_Class

        from .region import FlowRegion as RuntimeFlowRegion

        # Validate direction vs arrangement
        is_primary_vertical = self.flow.arrangement == "vertical"
        if direction in ("below", "above") and not is_primary_vertical:
            raise NotImplementedError(f"'{direction}' is for vertical flows.")

        # 1. Find starting segment
        start_idx = self._find_start_segment()
        if start_idx == -1:
            page_num_str = (
                str(self.physical_object.page.page_number) if self.physical_object.page else "N/A"
            )
            logger.warning(
                "FlowElement's physical object %s on page %s "
                "not found within any flow segment. Cannot perform directional operation '%s'.",
                self.physical_object.bbox,
                page_num_str,
                direction,
            )
            return RuntimeFlowRegion(
                flow=self.flow,
                constituent_regions=[],
                source_flow_element=self,
                boundary_element_found=None,
            )

        # 2. Resolve cross-size and build iterator
        cross_size_for_op = self._resolve_cross_size(
            direction, cross_size_ratio, cross_size_absolute
        )
        segment_iterator, is_forward = self._build_segment_iterator(
            direction, start_idx, len(self.flow.segments)
        )

        # 3. Iterate segments collecting contributions
        parts: List[PhysicalRegion_Class] = []
        boundary_hit: Optional["PhysicalElement"] = None
        remaining = float(size) if size is not None else float("inf")

        for seg_idx in segment_iterator:
            if (size is not None and remaining <= 0) or boundary_hit:
                break

            current_segment: PhysicalRegion_Class = self.flow.segments[seg_idx]

            if seg_idx == start_idx:
                contribution, hit = self._shape_start_segment(
                    current_segment,
                    direction=direction,
                    cross_size_for_op=cross_size_for_op,
                    cross_alignment=cross_alignment,
                    remaining_size=remaining,
                    size=size,
                    include_source=include_source,
                    until=until,
                    include_endpoint=include_endpoint,
                    **kwargs,
                )
                if hit:
                    boundary_hit = hit
            else:
                contribution = current_segment
                if not boundary_hit:
                    contribution, hit = self._clip_region_until(
                        contribution,
                        direction=direction,
                        until=until,
                        include_endpoint=include_endpoint,
                        search_kwargs=kwargs,
                    )
                    if hit:
                        boundary_hit = hit

            # Clip to size budget
            if (
                contribution
                and contribution.width > 0
                and contribution.height > 0
                and size is not None
            ):
                contribution, remaining = self._clip_to_budget(
                    contribution, remaining, direction, is_forward
                )

            # Collect valid contribution
            if contribution and contribution.width > 0 and contribution.height > 0:
                parts.append(contribution)

            # Stop after boundary hit (unless we still collected a valid start contribution)
            if boundary_hit and (
                seg_idx != start_idx
                or not contribution
                or contribution.width <= 0
                or contribution.height <= 0
            ):
                break

            # Deduct segment gap from budget
            is_last = (is_forward and seg_idx == len(self.flow.segments) - 1) or (
                not is_forward and seg_idx == 0
            )
            if not is_last and self.flow.segment_gap > 0 and size is not None and remaining > 0:
                remaining -= self.flow.segment_gap

        return RuntimeFlowRegion(
            flow=self.flow,
            constituent_regions=parts,
            source_flow_element=self,
            boundary_element_found=boundary_hit,
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
