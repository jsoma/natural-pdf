"""Vectorize rectangular char-region filtering."""

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
    "candidate": "spatial_char_filter_ephemeral",
    "cache_only": False,
    "hypothesis": "Rectangular char filtering can use one vector mask instead of per-char bbox loops.",
}


def _vector_filter_chars(char_dicts, exclusion_regions, target_region=None, debug=False):
    if not char_dicts:
        return []
    if target_region is None and not exclusion_regions:
        metrics.count("vector.spatial_char_filter.noop_fast_path")
        return char_dicts
    if target_region is not None and not rectangular_region(target_region):
        metrics.count("vector.spatial_char_filter.fallback_polygon_target")
        return _ORIGINAL(char_dicts, exclusion_regions, target_region=target_region, debug=debug)
    if any(not rectangular_region(region) for region in exclusion_regions):
        metrics.count("vector.spatial_char_filter.fallback_polygon_exclusion")
        return _ORIGINAL(char_dicts, exclusion_regions, target_region=target_region, debug=debug)

    chars = element_array(char_dicts, label="vector.spatial_char_filter.chars")
    keep = chars.valid.copy()
    metrics.byte_count("vector.spatial_char_filter.mask_bytes", keep.nbytes)

    if target_region is not None:
        x0, top, x1, bottom = target_region.bbox
        target_mask = bbox_mask_contains_points(
            x0,
            top,
            x1,
            bottom,
            chars.cx,
            chars.cy,
            right_open=True,
        )
        keep &= target_mask
        metrics.byte_count("vector.spatial_char_filter.mask_bytes", target_mask.nbytes)

    if exclusion_regions and np.any(keep):
        excluded = np.zeros(len(char_dicts), dtype=bool)
        for region in exclusion_regions:
            x0, top, x1, bottom = region.bbox
            overlap = bbox_mask_intersects(x0, top, x1, bottom, chars)
            center_inside = bbox_mask_contains_points(x0, top, x1, bottom, chars.cx, chars.cy)
            excluded |= keep & overlap & center_inside
            metrics.byte_count(
                "vector.spatial_char_filter.mask_bytes",
                overlap.nbytes + center_inside.nbytes,
            )
        keep &= ~excluded
        metrics.byte_count("vector.spatial_char_filter.mask_bytes", excluded.nbytes)

    metrics.count("vector.spatial_char_filter.fast_path")
    metrics.count("vector.spatial_char_filter.input_chars", len(char_dicts))
    metrics.count("vector.spatial_char_filter.output_chars", int(np.count_nonzero(keep)))
    return [char_dicts[int(index)] for index in np.flatnonzero(keep)]


@contextmanager
def install():
    import natural_pdf.core.page as page_module
    import natural_pdf.elements.region as region_module
    import natural_pdf.text.operations as operations

    global _ORIGINAL
    _ORIGINAL = operations.filter_chars_spatially

    operations.filter_chars_spatially = _vector_filter_chars
    page_original = getattr(page_module, "filter_chars_spatially", None)
    region_original = getattr(region_module, "filter_chars_spatially", None)
    page_module.filter_chars_spatially = _vector_filter_chars
    region_module.filter_chars_spatially = _vector_filter_chars
    try:
        yield
    finally:
        operations.filter_chars_spatially = _ORIGINAL
        if page_original is not None:
            page_module.filter_chars_spatially = page_original
        if region_original is not None:
            region_module.filter_chars_spatially = region_original
