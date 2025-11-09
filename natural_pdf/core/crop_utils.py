from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Tuple, Union

from natural_pdf.core.interfaces import SupportsBBox
from natural_pdf.elements.base import extract_bbox

Bounds = Tuple[float, float, float, float]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _clamp_bbox(
    bbox: Bounds,
    width: float,
    height: float,
) -> Bounds:
    x0, y0, x1, y1 = bbox
    return (
        _clamp(x0, 0.0, width),
        _clamp(y0, 0.0, height),
        _clamp(x1, 0.0, width),
        _clamp(y1, 0.0, height),
    )


def resolve_crop_bbox(
    *,
    width: float,
    height: float,
    crop: Union[
        bool,
        int,
        float,
        SupportsBBox,
        Sequence[float],
        Mapping[str, Any],
        Literal["content", "wide"],
        None,
    ] = False,
    crop_bbox: Optional[Bounds] = None,
    content_bbox_fn: Optional[Callable[[], Optional[Bounds]]] = None,
) -> Optional[Bounds]:
    """Resolve a crop bounding box shared by pages/regions/flow regions."""

    if crop_bbox is not None:
        return _clamp_bbox(crop_bbox, width, height)

    content_bbox = content_bbox_fn() if content_bbox_fn else None

    if isinstance(crop, bool):
        if crop:
            if content_bbox:
                return _clamp_bbox(content_bbox, width, height)
            return (0.0, 0.0, width, height)
        return None

    if isinstance(crop, (int, float)):
        if content_bbox:
            padding = float(crop)
            x0, y0, x1, y1 = content_bbox
            return _clamp_bbox(
                (x0 - padding, y0 - padding, x1 + padding, y1 + padding), width, height
            )
        return None

    if crop == "content" and content_bbox:
        return _clamp_bbox(content_bbox, width, height)

    if crop == "wide" and content_bbox:
        _, y0, _, y1 = content_bbox
        return _clamp_bbox((0.0, y0, width, y1), width, height)

    if crop not in (False, None, "content", "wide"):
        bbox = extract_bbox(crop)
        if bbox:
            return _clamp_bbox(bbox, width, height)

    return None
