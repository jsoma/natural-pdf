"""Built-in provider engine for header-based column detection."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

from natural_pdf.analyzers.guides.helpers import (
    Bounds,
    GuidesContext,
    _bounds_from_object,
    _require_bounds,
)
from natural_pdf.analyzers.guides.separators import (
    find_min_crossing_separator,
    find_seam_carving_separator,
    stabilize_with_rows,
)
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.guides.guides_provider import Axis, GuidesDetectionResult, GuidesEngine


def _normalize_header_input(headers: Union[ElementCollection, Sequence[Any], None]) -> List[Any]:
    if headers is None:
        return []
    if isinstance(headers, ElementCollection):
        return list(headers.elements)
    if isinstance(headers, (str, bytes)):
        return [headers]
    try:
        return list(headers)
    except TypeError:
        return [headers]


def _coerce_bounds(
    element: Any,
    page_bounds: Bounds,
) -> Optional[Bounds]:
    candidate = _bounds_from_object(element)
    if candidate is not None:
        return candidate

    x0 = getattr(element, "x0", None)
    x1 = getattr(element, "x1", None)
    top = getattr(element, "top", None)
    bottom = getattr(element, "bottom", None)

    try:
        resolved_x0 = float(x0 if x0 is not None else page_bounds[0])
        resolved_top = float(top if top is not None else page_bounds[1])
        resolved_x1 = float(x1 if x1 is not None else page_bounds[2])
        resolved_bottom = float(bottom if bottom is not None else page_bounds[3])
    except (TypeError, ValueError):
        return None

    return (resolved_x0, resolved_top, resolved_x1, resolved_bottom)


class HeadersGuidesEngine(GuidesEngine):
    """Detect vertical separators between table headers."""

    def detect(
        self,
        *,
        axis: Axis,
        method: str,
        context: GuidesContext,
        options: Dict[str, Any],
    ) -> GuidesDetectionResult:
        if axis != "vertical":
            raise ValueError("Header detection is only supported for vertical guides.")

        headers = _normalize_header_input(options.get("headers"))
        strategy = options.get("method", "min_crossings")
        min_width = options.get("min_width")
        max_width = options.get("max_width")
        margin = float(options.get("margin", 0.5))
        row_stabilization = bool(options.get("row_stabilization", True))
        num_samples = int(options.get("num_samples", 400))

        bounds = _bounds_from_object(context)
        if bounds is None:
            bounds = _require_bounds(context, context="headers detection")

        page_left, page_top, page_right, page_bottom = bounds

        header_elements: List[Any] = []
        header_strings: List[str] = []

        for item in headers:
            if isinstance(item, str):
                header_strings.append(item)
            elif item is not None:
                header_elements.append(item)

        finder = getattr(context, "find", None)
        if finder and header_strings:
            for text in header_strings:
                if not text:
                    continue
                try:
                    element = finder(f'text:contains("{text}")')
                except Exception:  # pragma: no cover
                    element = None
                if element:
                    header_elements.append(element)

        header_bounds: List[Bounds] = []
        for element in header_elements:
            elem_bounds = _coerce_bounds(element, bounds)
            if elem_bounds is not None:
                header_bounds.append(elem_bounds)

        if len(header_bounds) < 2:
            return GuidesDetectionResult(coordinates=[])

        header_bounds.sort(key=lambda b: (b[0] + b[2]) / 2)
        header_baseline = max(bound[3] for bound in header_bounds)

        text_bboxes: List[Bounds] = []
        finder_all = getattr(context, "find_all", None)
        if finder_all:
            try:
                text_candidates = finder_all("text")
            except Exception:  # pragma: no cover
                text_candidates = None

            iterable: Sequence[Any]
            if isinstance(text_candidates, ElementCollection):
                iterable = text_candidates.elements
            elif isinstance(text_candidates, Sequence):
                iterable = text_candidates
            else:
                iterable = ()

            for elem in iterable:
                elem_bounds = _bounds_from_object(elem)
                if elem_bounds is not None:
                    text_bboxes.append(elem_bounds)

        def _detect_separator(x0: float, x1: float) -> float:
            if x1 <= x0:
                return (x0 + x1) / 2

            intersecting = [
                bbox
                for bbox in text_bboxes
                if bbox[0] < x1 and bbox[2] > x0 and bbox[3] >= header_baseline
            ]

            if strategy == "seam_carving":
                if not intersecting:
                    return (x0 + x1) / 2
                return find_seam_carving_separator(
                    x0,
                    x1,
                    header_baseline,
                    page_bottom,
                    intersecting,
                )

            if not intersecting:
                return (x0 + x1) / 2
            return find_min_crossing_separator(x0, x1, intersecting, num_samples)

        separators: List[float] = [float(page_left)]
        for idx in range(len(header_bounds) - 1):
            left = header_bounds[idx]
            right = header_bounds[idx + 1]
            start = max(page_left, left[2] + margin)
            end = min(page_right, right[0] - margin)
            separators.append(float(_detect_separator(start, end)))
        separators.append(float(page_right))

        separators = sorted(separators)

        if row_stabilization and text_bboxes:
            separators = stabilize_with_rows(separators, text_bboxes, header_baseline)

        separators = sorted({float(value) for value in separators})

        if max_width and max_width > 0 and len(separators) >= 2:
            expanded: List[float] = [separators[0]]
            for idx in range(1, len(separators)):
                segment_start = expanded[-1]
                segment_end = separators[idx]
                width = segment_end - segment_start
                if width <= max_width:
                    expanded.append(segment_end)
                    continue
                partitions = int(math.ceil(width / max_width))
                step = width / partitions
                for seg in range(1, partitions + 1):
                    expanded.append(segment_start + step * seg)
            separators = expanded

        if min_width and min_width > 0 and len(separators) >= 2:
            filtered = [separators[0]]
            last_index = len(separators) - 1
            for idx in range(1, len(separators)):
                candidate = separators[idx]
                width = candidate - filtered[-1]
                if width < min_width and idx != last_index:
                    continue
                filtered.append(candidate)
            if len(filtered) == 1:
                filtered.append(separators[-1])
            if filtered[-1] != separators[-1]:
                filtered[-1] = separators[-1]
            separators = filtered

        separators = sorted({float(value) for value in separators})
        return GuidesDetectionResult(coordinates=separators)
