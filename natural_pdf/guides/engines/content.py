"""Built-in provider engine for content-aligned guides."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

from natural_pdf.analyzers.guides.helpers import (
    Bounds,
    GuidesContext,
    _bounds_from_object,
    _normalize_markers,
    _require_bounds,
)
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.guides.guides_provider import Axis, GuidesDetectionResult, GuidesEngine

logger = logging.getLogger(__name__)


def _coerce_element_bounds(
    element: Any,
    fallback_bounds: Optional[Bounds],
) -> Optional[Bounds]:
    bounds = _bounds_from_object(element)
    if bounds is not None:
        return bounds

    if fallback_bounds is None:
        return None

    default_x0, default_top, default_x1, default_bottom = fallback_bounds

    def _maybe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    x0_val = _maybe_float(getattr(element, "x0", None))
    x1_val = _maybe_float(getattr(element, "x1", None))
    top_val = _maybe_float(getattr(element, "top", None))
    bottom_val = _maybe_float(getattr(element, "bottom", None))

    if x0_val is None and x1_val is None and top_val is None and bottom_val is None:
        return None

    resolved_x0 = x0_val if x0_val is not None else (x1_val if x1_val is not None else default_x0)
    resolved_x1 = x1_val if x1_val is not None else (x0_val if x0_val is not None else default_x1)
    resolved_top = top_val if top_val is not None else default_top
    resolved_bottom = (
        bottom_val
        if bottom_val is not None
        else (top_val if top_val is not None else default_bottom)
    )

    try:
        return (
            float(resolved_x0),
            float(resolved_top),
            float(resolved_x1),
            float(resolved_bottom),
        )
    except (TypeError, ValueError):
        return None


class ContentGuidesEngine(GuidesEngine):
    """Detect guides aligned to textual or element markers."""

    def detect(
        self,
        *,
        axis: Axis,
        method: str,
        context: GuidesContext,
        options: Dict[str, Any],
    ) -> GuidesDetectionResult:
        markers = options.get("markers")
        align = options.get("align", "left")
        outer = options.get("outer", True)
        tolerance = float(options.get("tolerance", 5))
        apply_exclusions = bool(options.get("apply_exclusions", True))

        if axis == "horizontal":
            if align == "top":
                align = "left"
            elif align == "bottom":
                align = "right"

        bounds = _bounds_from_object(context)
        guides_coords: List[float] = []

        elements_to_process: List[Any] = []
        marker_texts: List[str] = []

        if isinstance(markers, ElementCollection):
            elements_to_process = list(markers.elements)
        elif hasattr(markers, "elements"):
            raw = getattr(markers, "elements", [])
            elements_to_process = list(cast(Iterable[Any], raw))
        elif markers is not None and hasattr(markers, "__iter__") and not isinstance(markers, str):
            try:
                markers_list = list(cast(Iterable[Any], markers))
            except TypeError:
                markers_list = []
            if markers_list and hasattr(markers_list[0], "x0"):
                elements_to_process = markers_list

        if not elements_to_process:
            marker_texts = _normalize_markers(
                cast(Optional[Union[str, List[str]]], markers), context
            )

        if elements_to_process:
            for element in elements_to_process:
                elem_bounds = _coerce_element_bounds(element, bounds)
                if not elem_bounds:
                    continue
                x0, top, x1, bottom = elem_bounds
                if axis == "vertical":
                    if align == "left":
                        guides_coords.append(x0)
                    elif align == "right":
                        guides_coords.append(x1)
                    elif align == "center":
                        guides_coords.append((x0 + x1) / 2)
                    elif align == "between":
                        guides_coords.append(x0)
                else:
                    if align == "left":
                        guides_coords.append(top)
                    elif align == "right":
                        guides_coords.append(bottom)
                    elif align == "center":
                        guides_coords.append((top + bottom) / 2)
                    elif align == "between":
                        guides_coords.append(top)
        else:
            for marker in marker_texts:
                finder = getattr(context, "find", None)
                if not finder:
                    break
                element = finder(f'text:contains("{marker}")', apply_exclusions=apply_exclusions)
                if not element:
                    continue
                elem_bounds = _coerce_element_bounds(element, bounds)
                if not elem_bounds:
                    continue
                x0, top, x1, bottom = elem_bounds
                if axis == "vertical":
                    if align == "left":
                        guides_coords.append(x0)
                    elif align == "right":
                        guides_coords.append(x1)
                    elif align == "center":
                        guides_coords.append((x0 + x1) / 2)
                    elif align == "between":
                        guides_coords.append(x0)
                else:
                    if align == "left":
                        guides_coords.append(top)
                    elif align == "right":
                        guides_coords.append(bottom)
                    elif align == "center":
                        guides_coords.append((top + bottom) / 2)
                    elif align == "between":
                        guides_coords.append(top)

        if align == "between" and len(guides_coords) >= 2:
            marker_bounds: List[Tuple[float, float]] = []
            source_elements = elements_to_process
            if not source_elements and marker_texts:
                finder = getattr(context, "find", None)
                if finder:
                    for marker in marker_texts:
                        element = finder(
                            f'text:contains("{marker}")', apply_exclusions=apply_exclusions
                        )
                        if element:
                            source_elements.append(element)

            for element in source_elements:
                elem_bounds = _coerce_element_bounds(element, bounds)
                if not elem_bounds:
                    continue
                if axis == "vertical":
                    marker_bounds.append((elem_bounds[0], elem_bounds[2]))
                else:
                    marker_bounds.append((elem_bounds[1], elem_bounds[3]))

            marker_bounds.sort(key=lambda x: x[0])

            between_coords: List[float] = []
            for idx in range(len(marker_bounds) - 1):
                current_right = marker_bounds[idx][1]
                next_left = marker_bounds[idx + 1][0]
                between_coords.append((current_right + next_left) / 2)
            guides_coords = between_coords

        if outer:
            outer_bounds = bounds or _require_bounds(context, context="content bounds")
            if axis == "vertical":
                if outer is True or outer == "first":
                    guides_coords.insert(0, outer_bounds[0])
                if outer is True or outer == "last":
                    guides_coords.append(outer_bounds[2])
            else:
                if outer is True or outer == "first":
                    guides_coords.insert(0, outer_bounds[1])
                if outer is True or outer == "last":
                    guides_coords.append(outer_bounds[3])

        unique_coords = sorted({float(coord) for coord in guides_coords})
        return GuidesDetectionResult(coordinates=unique_coords)
