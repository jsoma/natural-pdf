"""Vectorize region/exclusion filtering with memoized derived arrays."""

from contextlib import contextmanager

import numpy as np

from experiments.performance import vector_metrics as metrics
from experiments.performance.vector_helpers import (
    bbox_mask_contains_points,
    bbox_mask_intersects,
    element_array,
    rectangular_region,
)

METADATA = {
    "track": "vector",
    "candidate": "region_overlap_memoized_arrays",
    "cache_only": False,
    "hypothesis": "Memoizing derived arrays for repeated element lists can remove array rebuild overhead without caching query results.",
}

_ARRAY_CACHE = {}


def _array_for(elements, label):
    element_list = list(elements)
    key = tuple(id(element) for element in element_list)
    cache_key = (label, key)
    cached = _ARRAY_CACHE.get(cache_key)
    if cached is not None:
        metrics.count(f"{label}.memo_reuse")
        return cached
    arr = element_array(element_list, label=label)
    _ARRAY_CACHE[cache_key] = arr
    return arr


def _patched_overlap(self, elements, overlap_mode):
    if not elements:
        return []
    if not rectangular_region(self):
        metrics.count("vector.region_overlap_memo.fallback_polygon")
        return _ORIGINAL_OVERLAP(self, elements, overlap_mode)

    arr = _array_for(elements, "vector.region_overlap_memo.elements")
    x0, top, x1, bottom = self.bbox
    same_page = np.asarray(
        [getattr(element, "page", None) is self.page for element in arr.items],
        dtype=bool,
    )
    if overlap_mode == "full":
        mask = (
            same_page
            & arr.valid
            & (x0 <= arr.x0)
            & (arr.x1 <= x1)
            & (top <= arr.top)
            & (arr.bottom <= bottom)
        )
    elif overlap_mode == "partial":
        mask = same_page & arr.valid & bbox_mask_intersects(x0, top, x1, bottom, arr)
    else:
        mask = (
            same_page & arr.valid & bbox_mask_contains_points(x0, top, x1, bottom, arr.cx, arr.cy)
        )

    metrics.count("vector.region_overlap_memo.fast_path")
    metrics.byte_count("vector.region_overlap_memo.mask_bytes", mask.nbytes + same_page.nbytes)
    return arr.select(mask)


def _has_element_exclusions(page):
    exclusions = list(getattr(page, "_exclusions", []) or [])
    parent = getattr(page, "_parent", None)
    exclusions.extend(list(getattr(parent, "_exclusions", []) or []))
    for exclusion in exclusions:
        if len(exclusion) == 2:
            _item, _label = exclusion
            method = "region"
        else:
            _item, _label, method = exclusion
        if method == "element":
            return True
    return False


def _patched_filter_exclusions(self, elements, debug_exclusions=False):
    if getattr(self, "_computing_exclusions", False):
        return elements

    has_page_exclusions = bool(getattr(self, "_exclusions", None))
    parent = getattr(self, "_parent", None)
    has_pdf_exclusions = bool(getattr(parent, "_exclusions", None))
    if not has_page_exclusions and not has_pdf_exclusions:
        return elements
    if _has_element_exclusions(self):
        metrics.count("vector.page_exclusions_memo.fallback_element_exclusion")
        return _ORIGINAL_FILTER_EXCLUSIONS(self, elements, debug_exclusions=debug_exclusions)

    exclusion_regions = self._get_exclusion_regions(
        include_callable=True,
        debug=debug_exclusions,
    )
    if not exclusion_regions:
        return elements
    if any(not rectangular_region(region) for region in exclusion_regions):
        metrics.count("vector.page_exclusions_memo.fallback_polygon")
        return _ORIGINAL_FILTER_EXCLUSIONS(self, elements, debug_exclusions=debug_exclusions)

    arr = _array_for(elements, "vector.page_exclusions_memo.elements")
    excluded = np.zeros(len(arr.items), dtype=bool)
    same_page = np.asarray(
        [getattr(element, "page", None) is self for element in arr.items],
        dtype=bool,
    )
    for region in exclusion_regions:
        x0, top, x1, bottom = region.bbox
        excluded |= (
            same_page & arr.valid & bbox_mask_contains_points(x0, top, x1, bottom, arr.cx, arr.cy)
        )

    keep = ~excluded
    metrics.count("vector.page_exclusions_memo.fast_path")
    metrics.byte_count("vector.page_exclusions_memo.mask_bytes", excluded.nbytes + keep.nbytes)
    return arr.select(keep)


@contextmanager
def install():
    from natural_pdf.core.page import Page
    from natural_pdf.elements.region import Region

    global _ORIGINAL_OVERLAP, _ORIGINAL_FILTER_EXCLUSIONS
    _ARRAY_CACHE.clear()
    _ORIGINAL_OVERLAP = Region._filter_elements_by_overlap_mode
    _ORIGINAL_FILTER_EXCLUSIONS = Page._filter_elements_by_exclusions

    Region._filter_elements_by_overlap_mode = _patched_overlap
    Page._filter_elements_by_exclusions = _patched_filter_exclusions
    try:
        yield
    finally:
        Region._filter_elements_by_overlap_mode = _ORIGINAL_OVERLAP
        Page._filter_elements_by_exclusions = _ORIGINAL_FILTER_EXCLUSIONS
        _ARRAY_CACHE.clear()
