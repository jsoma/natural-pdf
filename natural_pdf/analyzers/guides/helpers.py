"""Helper utilities shared across guide modules."""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Protocol, Sequence, Tuple, TypeGuard, Union, cast

import numpy as np
from numpy.typing import NDArray

from natural_pdf.core.interfaces import HasPages, HasSinglePage
from natural_pdf.elements.base import extract_bbox
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.line import LineElement
from natural_pdf.elements.region import Region
from natural_pdf.flows.region import FlowRegion

logger = logging.getLogger(__name__)

Bounds = Tuple[float, float, float, float]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]
GuidesContext = Union["Page", Region, FlowRegion]

try:  # pragma: no cover - typing helper
    from natural_pdf.core.page import Page
except ImportError:  # pragma: no cover
    Page = Any  # type: ignore


class _SupportsSize(Protocol):  # type: ignore[misc]
    width: float
    height: float


def _label_contiguous_regions(mask: np.ndarray) -> Tuple[IntArray, int]:
    """Typed wrapper around scipy.ndimage.label for boolean masks."""
    from scipy.ndimage import label as nd_label

    labeled, count = cast(Tuple[np.ndarray, int], nd_label(mask))
    return cast(IntArray, labeled), int(count)


def _is_flow_region(obj: Any) -> TypeGuard[FlowRegion]:
    return isinstance(obj, FlowRegion)


def _has_size(obj: Any) -> TypeGuard[_SupportsSize]:
    return hasattr(obj, "width") and hasattr(obj, "height")


def _ensure_bounds_tuple(bounds: Tuple[Any, Any, Any, Any]) -> Bounds:
    return (
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3]),
    )


def _union_bounds(bound_list: Iterable[Bounds]) -> Optional[Bounds]:
    coords = list(bound_list)
    if not coords:
        return None
    x0 = min(b[0] for b in coords)
    top = min(b[1] for b in coords)
    x1 = max(b[2] for b in coords)
    bottom = max(b[3] for b in coords)
    return (x0, top, x1, bottom)


def _bounds_from_object(obj: Any) -> Optional[Bounds]:
    if obj is None:
        return None

    if isinstance(obj, tuple) and len(obj) == 4:
        try:
            return _ensure_bounds_tuple(obj)
        except (TypeError, ValueError):
            return None

    bbox = extract_bbox(obj)
    if bbox is not None:
        try:
            return _ensure_bounds_tuple(bbox)
        except (TypeError, ValueError):
            return None

    regions_attr = getattr(obj, "constituent_regions", None)
    if regions_attr is not None:
        try:
            regions = list(regions_attr)
        except TypeError:
            regions = None

        if regions:
            collected: List[Bounds] = []
            for region in regions:
                region_bounds = _bounds_from_object(region)
                if region_bounds is not None:
                    collected.append(region_bounds)
            return _union_bounds(collected)

    if _has_size(obj):
        try:
            return (0.0, 0.0, float(obj.width), float(obj.height))
        except (TypeError, ValueError):
            return None

    return None


def _resolve_single_page(obj: Any) -> "Page":
    if isinstance(obj, HasPages):
        pages = list(obj.pages)
        seen: set[int] = set()
        unique_pages: List["Page"] = []
        for page in pages:
            marker = id(page)
            if marker not in seen:
                seen.add(marker)
                unique_pages.append(page)
        if not unique_pages:
            raise ValueError("Object does not reference any pages")
        if len(unique_pages) > 1:
            raise ValueError("Object spans multiple pages; cannot pick a single page")
        return unique_pages[0]

    if isinstance(obj, HasSinglePage):
        page = obj.page  # type: ignore[attr-defined]
        if page is None:
            raise ValueError("Object reported no page")
        return page

    if hasattr(obj, "_page"):
        return obj._page  # type: ignore[attr-defined]

    raise TypeError(f"Cannot resolve page from {type(obj).__name__}")


def _is_guides_context(value: Any) -> TypeGuard[GuidesContext]:
    if value is None:
        return False
    if isinstance(value, (FlowRegion, Region)):
        return True
    cls = value.__class__
    module_name = getattr(cls, "_module", getattr(cls, "__module__", ""))
    class_name = getattr(cls, "__name__", "")
    if class_name == "Page" and module_name.startswith("natural_pdf.core.page"):
        return True
    return _bounds_from_object(value) is not None


def _require_bounds(obj: Any, *, context: str = "object") -> Bounds:
    bounds = _bounds_from_object(obj)
    if bounds is None:
        raise ValueError(f"Could not determine bounds for {context}")
    return bounds


def _constituent_regions(flow_region: FlowRegion) -> Sequence[Region]:
    return flow_region.constituent_regions


def _collect_line_elements(obj: GuidesContext) -> List[LineElement]:
    if _is_flow_region(obj):
        lines: List[LineElement] = []
        for region in obj.constituent_regions:
            lines.extend(_collect_line_elements(region))
        return lines

    lines_attr = getattr(obj, "lines", None)
    if lines_attr is not None:
        return [cast(LineElement, line) for line in cast(Iterable[Any], lines_attr)]

    finder = getattr(obj, "find_all", None)
    if finder is not None:
        found = finder("line")  # type: ignore[attr-defined]
        if isinstance(found, ElementCollection):
            return [cast(LineElement, line) for line in found.elements]
        if isinstance(found, Iterable):
            return [cast(LineElement, line) for line in found]
    return []


def _normalize_markers(
    markers: Union[str, List[str], ElementCollection, None],
    obj: GuidesContext,
) -> List[str]:
    if markers is None:
        return []

    if _is_flow_region(obj):
        aggregated: List[str] = []
        for region in obj.constituent_regions:
            aggregated.extend(_normalize_markers(markers, region))
        seen: set[str] = set()
        unique: List[str] = []
        for marker in aggregated:
            if marker not in seen:
                seen.add(marker)
                unique.append(marker)
        return unique

    if isinstance(markers, str):
        if markers.startswith(("text", "region", "line", "rect", "blob", "image")):
            finder = getattr(obj, "find_all", None)
            if finder is None:
                raise AttributeError(
                    f"Object {obj} doesn't support find_all for selector '{markers}'"
                )
            elements = finder(markers)
            return [elem.text if hasattr(elem, "text") else str(elem) for elem in elements]
        return [markers]

    if isinstance(markers, ElementCollection):
        return [str(value) for value in markers.extract_each_text()]

    if isinstance(markers, Iterable):
        normalized: List[str] = []
        for marker in markers:
            if isinstance(marker, str):
                if marker.startswith(("text", "region", "line", "rect", "blob", "image")):
                    finder = getattr(obj, "find_all", None)
                    if finder is None:
                        normalized.append(marker)
                        continue
                    elements = finder(marker)
                    normalized.extend(
                        [elem.text if hasattr(elem, "text") else str(elem) for elem in elements]
                    )
                else:
                    normalized.append(marker)
            elif hasattr(marker, "text"):
                normalized.append(marker.text)
            elif hasattr(marker, "extract_text"):
                normalized.append(marker.extract_text())
        return normalized

    raise TypeError("markers must be str, ElementCollection, iterable, or None")


__all__ = [
    "Bounds",
    "GuidesContext",
    "BoolArray",
    "IntArray",
    "_bounds_from_object",
    "_collect_line_elements",
    "_constituent_regions",
    "_ensure_bounds_tuple",
    "_has_size",
    "_is_flow_region",
    "_is_guides_context",
    "_normalize_markers",
    "_require_bounds",
    "_resolve_single_page",
    "_union_bounds",
]
