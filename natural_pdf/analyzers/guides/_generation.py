"""Shared guide-generation helpers for Guides and GuidesList."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.guides.guides_provider import run_guides_detect

from .helpers import GuidesContext, _constituent_regions, _is_flow_region

Axis = Literal["vertical", "horizontal"]
OuterBoundaryMode = Union[bool, Literal["first", "last"]]


@dataclass
class AxisGenerationResult:
    """Container for one axis worth of generated guide coordinates."""

    axis: Axis
    coordinates: List[float] = field(default_factory=list)
    region_coordinates: Optional[Dict[Any, List[float]]] = None


def resolve_generation_context(
    obj: Optional[GuidesContext],
    fallback_context: Optional[GuidesContext] = None,
) -> GuidesContext:
    target_obj = obj or fallback_context
    if target_obj is None:
        raise ValueError("No object provided and no context available")
    return target_obj


def normalize_content_align(
    axis: Axis,
    align: Union[
        Literal["left", "right", "center", "between"],
        Literal["top", "bottom"],
    ],
) -> Literal["left", "right", "center", "between"]:
    if axis == "horizontal":
        if align == "top":
            return "left"
        if align == "bottom":
            return "right"
    return align


def build_content_options(
    axis: Axis,
    markers: Union[str, List[str], ElementCollection, None],
    align: Union[
        Literal["left", "right", "center", "between"],
        Literal["top", "bottom"],
    ],
    outer: OuterBoundaryMode,
    tolerance: float,
    apply_exclusions: bool,
) -> Dict[str, Any]:
    return {
        "markers": markers,
        "align": normalize_content_align(axis, align),
        "outer": outer,
        "tolerance": tolerance,
        "apply_exclusions": apply_exclusions,
    }


def build_line_options(
    axis: Axis,
    *,
    threshold: Union[float, str],
    source_label: Optional[str],
    max_lines_h: Optional[int],
    max_lines_v: Optional[int],
    outer: bool,
    detection_method: str,
    resolution: int,
    detect_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "threshold": threshold,
        "source_label": source_label,
        "max_lines_h": max_lines_h if axis == "horizontal" else None,
        "max_lines_v": max_lines_v if axis == "vertical" else None,
        "outer": outer,
        "detection_method": detection_method,
        "resolution": resolution,
        **detect_kwargs,
    }


def build_whitespace_options(min_gap: float) -> Dict[str, Any]:
    return {"min_gap": min_gap}


def build_headers_options(
    headers: Union[ElementCollection, Sequence[Any], None],
    *,
    method: Literal["min_crossings", "seam_carving"],
    min_width: Optional[float],
    max_width: Optional[float],
    margin: float,
    row_stabilization: bool,
    num_samples: int,
) -> Dict[str, Any]:
    return {
        "headers": headers,
        "method": method,
        "min_width": min_width,
        "max_width": max_width,
        "margin": margin,
        "row_stabilization": row_stabilization,
        "num_samples": num_samples,
    }


def build_stripes_options(
    stripes: Optional[Union[ElementCollection, Sequence[Any]]],
    *,
    color: Optional[str],
) -> Dict[str, Any]:
    return {"stripes": stripes, "color": color}


def generate_axis_coordinates(
    *,
    axis: Axis,
    method: str,
    context: GuidesContext,
    options: Dict[str, Any],
) -> AxisGenerationResult:
    if _is_flow_region(context):
        region_coordinates: Dict[Any, List[float]] = {}
        all_coordinates: List[float] = []

        for region in _constituent_regions(context):
            result = run_guides_detect(
                axis=axis,
                method=method,
                context=region,
                options=options,
            )
            coords = [float(value) for value in result.coordinates]
            region_coordinates[region] = coords
            all_coordinates.extend(coords)

        return AxisGenerationResult(
            axis=axis,
            coordinates=sorted(set(all_coordinates)),
            region_coordinates=region_coordinates,
        )

    result = run_guides_detect(
        axis=axis,
        method=method,
        context=context,
        options=options,
    )
    return AxisGenerationResult(
        axis=axis,
        coordinates=[float(value) for value in result.coordinates],
    )
