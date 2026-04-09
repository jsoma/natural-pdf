"""Shared axis mutation helpers for guides."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence

from ._generation import AxisGenerationResult

if TYPE_CHECKING:
    from .base import Guides, GuidesList

Axis = Literal["vertical", "horizontal"]


def normalize_axis_values(values: Iterable[float]) -> list[float]:
    return sorted({float(value) for value in values})


def _axis_list(guides: "Guides", axis: Axis) -> "GuidesList":
    return getattr(guides, f"_{axis}")


def set_axis_coordinates(guides: "Guides", axis: Axis, values: Iterable[float]) -> list[float]:
    normalized = normalize_axis_values(values)
    if getattr(guides, "is_flow_region", False):
        from .flow_adapter import FlowGuideAdapter

        FlowGuideAdapter(guides).set_axis_coordinates(axis, normalized)
        return normalized

    _axis_list(guides, axis)._set_data_direct(normalized)
    return normalized


def update_axis_coordinates(
    guides: "Guides",
    axis: Axis,
    values: Iterable[float],
    *,
    append: bool,
) -> list[float]:
    combined = list(getattr(guides, axis)) if append else []
    combined.extend(float(value) for value in values)
    return set_axis_coordinates(guides, axis, combined)


def apply_region_axis_values(
    guides: "Guides",
    axis: Axis,
    region_values: Mapping[Any, Sequence[float]],
    *,
    append: bool,
) -> None:
    if getattr(guides, "is_flow_region", False):
        from .flow_adapter import FlowGuideAdapter

        FlowGuideAdapter(guides).update_axis_from_regions(axis, region_values, append=append)
        return

    flattened: list[float] = []
    for coords in region_values.values():
        flattened.extend(float(value) for value in coords)
    update_axis_coordinates(guides, axis, flattened, append=append)


def apply_generation_result(
    guides: "Guides",
    result: AxisGenerationResult,
    *,
    append: bool,
) -> None:
    if result.region_coordinates is not None:
        apply_region_axis_values(guides, result.axis, result.region_coordinates, append=append)
        return

    update_axis_coordinates(guides, result.axis, result.coordinates, append=append)
