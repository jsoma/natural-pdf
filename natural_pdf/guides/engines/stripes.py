"""Built-in provider engine for stripe-based guide detection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from natural_pdf.analyzers.guides.helpers import Bounds, GuidesContext, _bounds_from_object
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.guides.guides_provider import Axis, GuidesDetectionResult, GuidesEngine


def _normalize_stripes_input(
    stripes: Optional[Union[ElementCollection, Sequence[Any]]]
) -> List[Any]:
    if stripes is None:
        return []
    if isinstance(stripes, ElementCollection):
        return list(stripes.elements)
    if isinstance(stripes, (str, bytes)):
        return []
    try:
        return list(stripes)
    except TypeError:
        return [stripes]


def _autodetect_stripes(context: GuidesContext, color: Optional[str]) -> List[Any]:
    finder = getattr(context, "find_all", None)
    if not finder:
        return []
    selector = f"rect[fill={color}]" if color else "rect[fill]"
    try:
        raw = finder(selector)
    except Exception:  # pragma: no cover
        raw = []

    if hasattr(raw, "elements"):
        candidates = list(raw.elements)
    else:
        candidates = list(raw or [])

    if color:
        return candidates

    fill_groups: Dict[str, List[Any]] = defaultdict(list)
    for stripe in candidates:
        fill = getattr(stripe, "fill", None)
        if not fill:
            continue
        normalized = str(fill).lower()
        if normalized in {"#ffffff", "white", "none", "transparent"}:
            continue
        fill_groups[str(fill)].append(stripe)

    if not fill_groups:
        return []

    return max(fill_groups.items(), key=lambda item: len(item[1]))[1]


def _coerce_bounds(stripe: Any) -> Optional[Bounds]:
    bounds = _bounds_from_object(stripe)
    if bounds is not None:
        return bounds
    x0 = getattr(stripe, "x0", None)
    x1 = getattr(stripe, "x1", None)
    top = getattr(stripe, "top", None)
    bottom = getattr(stripe, "bottom", None)
    if None in (x0, x1, top, bottom):
        return None
    try:
        return (float(x0), float(top), float(x1), float(bottom))
    except (TypeError, ValueError):
        return None


class StripesGuidesEngine(GuidesEngine):
    """Detect guides from zebra stripes or alternating fills."""

    def detect(
        self,
        *,
        axis: Axis,
        method: str,
        context: GuidesContext,
        options: Dict[str, Any],
    ) -> GuidesDetectionResult:
        stripes = _normalize_stripes_input(options.get("stripes"))
        color = options.get("color")

        if not stripes:
            stripes = _autodetect_stripes(context, color=color)

        if not stripes:
            return GuidesDetectionResult(coordinates=[])

        edges: List[float] = []
        for stripe in stripes:
            stripe_bounds = _coerce_bounds(stripe)
            if stripe_bounds is None:
                continue
            if axis == "horizontal":
                edges.extend([stripe_bounds[1], stripe_bounds[3]])
            else:
                edges.extend([stripe_bounds[0], stripe_bounds[2]])

        unique_edges = sorted({float(edge) for edge in edges})
        return GuidesDetectionResult(coordinates=unique_edges)
