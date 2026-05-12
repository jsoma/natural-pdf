"""Structure-of-arrays helpers for vectorization experiments.

These helpers are intentionally experiment-only. They provide derived array views
over Natural PDF objects and dictionaries without changing public APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.performance import vector_metrics as metrics


def _get_value(item: Any, name: str, default: Any = 0.0) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _bool_array(items: Sequence[Any], name: str) -> np.ndarray:
    return np.fromiter((bool(_get_value(item, name, False)) for item in items), dtype=bool)


@dataclass
class ElementArray:
    items: Sequence[Any]
    x0: np.ndarray
    top: np.ndarray
    x1: np.ndarray
    bottom: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    width: np.ndarray
    height: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    size: np.ndarray
    strike: np.ndarray
    underline: np.ndarray
    highlight: np.ndarray
    valid: np.ndarray

    @property
    def nbytes(self) -> int:
        return sum(
            int(getattr(array, "nbytes", 0))
            for array in (
                self.x0,
                self.top,
                self.x1,
                self.bottom,
                self.y0,
                self.y1,
                self.width,
                self.height,
                self.cx,
                self.cy,
                self.size,
                self.strike,
                self.underline,
                self.highlight,
                self.valid,
            )
        )

    @property
    def length(self) -> int:
        return len(self.items)

    def select(self, mask: np.ndarray) -> list[Any]:
        return [self.items[int(index)] for index in np.flatnonzero(mask)]


def element_array(items: Iterable[Any], *, label: str) -> ElementArray:
    item_list = list(items)
    with metrics.timed(f"{label}.build_ms"):
        x0 = np.asarray([float(_get_value(item, "x0", np.nan)) for item in item_list], dtype=float)
        top = np.asarray(
            [float(_get_value(item, "top", np.nan)) for item in item_list], dtype=float
        )
        x1 = np.asarray([float(_get_value(item, "x1", np.nan)) for item in item_list], dtype=float)
        bottom = np.asarray(
            [float(_get_value(item, "bottom", np.nan)) for item in item_list], dtype=float
        )
        y0 = np.asarray([float(_get_value(item, "y0", np.nan)) for item in item_list], dtype=float)
        y1 = np.asarray([float(_get_value(item, "y1", np.nan)) for item in item_list], dtype=float)
        size = np.asarray(
            [float(_get_value(item, "size", np.nan) or np.nan) for item in item_list],
            dtype=float,
        )
        width = x1 - x0
        height = bottom - top
        cx = (x0 + x1) / 2.0
        cy = (top + bottom) / 2.0
        valid = np.isfinite(x0) & np.isfinite(top) & np.isfinite(x1) & np.isfinite(bottom)
        valid &= width >= 0
        valid &= height >= 0
        result = ElementArray(
            items=item_list,
            x0=x0,
            top=top,
            x1=x1,
            bottom=bottom,
            y0=y0,
            y1=y1,
            width=width,
            height=height,
            cx=cx,
            cy=cy,
            size=size,
            strike=_bool_array(item_list, "strike"),
            underline=_bool_array(item_list, "underline"),
            highlight=_bool_array(item_list, "highlight"),
            valid=valid,
        )
    metrics.count(f"{label}.build_count")
    metrics.byte_count(f"{label}.array_bytes", result.nbytes)
    return result


class PageArrayView:
    """Lazy derived arrays for page elements."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self._arrays: dict[str, ElementArray] = {}

    @classmethod
    def for_page(cls, page: Any) -> "PageArrayView":
        view = getattr(page, "_npdf_perf_array_view", None)
        if view is None:
            view = cls(page)
            setattr(page, "_npdf_perf_array_view", view)
            metrics.count("page_array_view.create")
        else:
            metrics.count("page_array_view.reuse")
        return view

    def elements(self, element_type: str) -> ElementArray:
        key = element_type.lower()
        if key not in self._arrays:
            source = getattr(self.page, key, [])
            self._arrays[key] = element_array(source, label=f"page_array_view.{key}")
        else:
            metrics.count(f"page_array_view.{key}.reuse")
        return self._arrays[key]


def prepared_char_array(
    page: Any, char_dicts: Sequence[dict[str, Any]], *, label: str
) -> ElementArray:
    cache_name = "_npdf_perf_prepared_char_array"
    cache = getattr(page, cache_name, None)
    cache_key = (id(char_dicts), len(char_dicts))
    if cache is not None and cache.get("key") == cache_key:
        metrics.count(f"{label}.reuse")
        return cache["array"]
    array = element_array(char_dicts, label=label)
    setattr(page, cache_name, {"key": cache_key, "array": array})
    return array


def rectangular_region(region: Any) -> bool:
    return not bool(getattr(region, "has_polygon", False))


def bbox_mask_contains_points(
    x0: float,
    top: float,
    x1: float,
    bottom: float,
    px: np.ndarray,
    py: np.ndarray,
    *,
    right_open: bool = False,
) -> np.ndarray:
    if right_open:
        return (x0 <= px) & (px < x1) & (top <= py) & (py < bottom)
    return (x0 <= px) & (px <= x1) & (top <= py) & (py <= bottom)


def bbox_mask_intersects(
    x0: float,
    top: float,
    x1: float,
    bottom: float,
    arr: ElementArray,
) -> np.ndarray:
    return (x0 < arr.x1) & (x1 > arr.x0) & (top < arr.bottom) & (bottom > arr.top)
