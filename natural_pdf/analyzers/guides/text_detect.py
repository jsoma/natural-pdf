"""Text-based detection utilities for guide discovery."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.flows.region import FlowRegion

from .helpers import _label_contiguous_regions

Gap = Tuple[float, float]
Threshold = Union[float, str]

__all__ = [
    "collect_text_elements",
    "find_vertical_whitespace_gaps",
    "find_horizontal_whitespace_gaps",
    "find_vertical_element_gaps",
    "find_horizontal_element_gaps",
]


def collect_text_elements(context: Any) -> List[Any]:
    """Return all text elements from the provided context."""
    if context is None:
        return []

    def _extract(target: Any, *, strict: bool = True) -> List[Any]:
        finder = getattr(target, "find_all", None)
        if finder is None:
            if strict:
                raise AttributeError(f"{target} does not implement find_all")
            return []
        collection = finder("text", apply_exclusions=False)
        elements = collection.elements if isinstance(collection, ElementCollection) else collection
        return list(elements)

    if isinstance(context, FlowRegion):
        elements: List[Any] = []
        for region in context.constituent_regions:
            elements.extend(_extract(region, strict=False))
        return elements

    return _extract(context, strict=True)


def find_vertical_whitespace_gaps(
    bounds: Sequence[float] | None,
    text_elements: Iterable[Any],
    min_gap: float,
    threshold: Threshold = "auto",
    *,
    guide_positions: Sequence[float] | None = None,
) -> List[Gap]:
    """Detect vertical whitespace gaps within the supplied bounds."""
    return _find_whitespace_gaps(
        axis="x",
        bounds=bounds,
        text_elements=text_elements,
        min_gap=min_gap,
        threshold=threshold,
        guide_positions=guide_positions,
    )


def find_horizontal_whitespace_gaps(
    bounds: Sequence[float] | None,
    text_elements: Iterable[Any],
    min_gap: float,
    threshold: Threshold = "auto",
    *,
    guide_positions: Sequence[float] | None = None,
) -> List[Gap]:
    """Detect horizontal whitespace gaps within the supplied bounds."""
    return _find_whitespace_gaps(
        axis="y",
        bounds=bounds,
        text_elements=text_elements,
        min_gap=min_gap,
        threshold=threshold,
        guide_positions=guide_positions,
    )


def _find_whitespace_gaps(
    *,
    axis: str,
    bounds: Sequence[float] | None,
    text_elements: Iterable[Any],
    min_gap: float,
    threshold: Threshold,
    guide_positions: Sequence[float] | None,
) -> List[Gap]:
    if not bounds:
        return []

    if axis == "x":
        low, high = float(bounds[0]), float(bounds[2])
        start_attr, end_attr = "x0", "x1"
    else:
        low, high = float(bounds[1]), float(bounds[3])
        start_attr, end_attr = "top", "bottom"

    span = int(high - low)
    if span <= 0:
        return []

    density = np.zeros(span)
    for element in text_elements:
        start = getattr(element, start_attr, None)
        end = getattr(element, end_attr, None)
        if start is None or end is None:
            continue
        try:
            elem_start = max(low, float(start)) - low
            elem_end = min(high, float(end)) - low
        except (TypeError, ValueError):
            continue
        if elem_end <= elem_start:
            continue
        start_px = int(elem_start)
        end_px = int(elem_end)
        if end_px > start_px:
            density[start_px:end_px] += 1

    if density.max() == 0:
        return []

    guide_positions = list(guide_positions or [])
    guides_needed = max(len(guide_positions) - 2, 0)

    if threshold == "auto":
        if guides_needed == 0:
            threshold_val = 0.5
        else:
            threshold_val = None
            for candidate in np.arange(0.1, 1.0, 0.05):
                candidate_gaps = _find_gaps_with_threshold(density, candidate, min_gap, low)
                if len(candidate_gaps) >= guides_needed:
                    threshold_val = float(candidate)
                    break
            if threshold_val is None:
                threshold_val = 0.8
    else:
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be a number between 0.0 and 1.0, or 'auto'")
        threshold_val = float(threshold)

    return _find_gaps_with_threshold(density, threshold_val, min_gap, low)


def _find_gaps_with_threshold(density, threshold_val, min_gap, origin) -> List[Gap]:
    if density.size == 0:
        return []

    max_density = density.max()
    if max_density <= 0:
        return []

    threshold_density = threshold_val * max_density
    smoothed_density = gaussian_filter1d(density.astype(float), sigma=1.0)
    below_threshold = smoothed_density <= threshold_density

    labeled_regions, num_regions = _label_contiguous_regions(below_threshold)
    gaps: List[Gap] = []
    for region_id in range(1, num_regions + 1):
        region_mask = labeled_regions == region_id
        region_indices = np.where(region_mask)[0]
        if len(region_indices) == 0:
            continue
        start_px = region_indices[0]
        end_px = region_indices[-1] + 1
        start_coord = origin + start_px
        end_coord = origin + end_px
        if end_coord - start_coord >= min_gap:
            gaps.append((start_coord, end_coord))

    return gaps


def find_vertical_element_gaps(
    bounds: Sequence[float] | None,
    text_elements: Iterable[Any],
    min_gap: float,
) -> List[Gap]:
    """Locate vertical whitespace gaps using element edge positions."""
    return _find_element_gaps(
        axis="x",
        bounds=bounds,
        text_elements=text_elements,
        min_gap=min_gap,
    )


def find_horizontal_element_gaps(
    bounds: Sequence[float] | None,
    text_elements: Iterable[Any],
    min_gap: float,
) -> List[Gap]:
    """Locate horizontal whitespace gaps using element edge positions."""
    return _find_element_gaps(
        axis="y",
        bounds=bounds,
        text_elements=text_elements,
        min_gap=min_gap,
    )


def _find_element_gaps(
    *,
    axis: str,
    bounds: Sequence[float] | None,
    text_elements: Iterable[Any],
    min_gap: float,
) -> List[Gap]:
    if not bounds:
        return []

    text_list = list(text_elements)
    if not text_list:
        return []

    if axis == "x":
        low, high = float(bounds[0]), float(bounds[2])
        orth_low, orth_high = float(bounds[1]), float(bounds[3])
        start_attr, end_attr = "x0", "x1"
        orth_start, orth_end = "top", "bottom"
    else:
        low, high = float(bounds[1]), float(bounds[3])
        orth_low, orth_high = float(bounds[0]), float(bounds[2])
        start_attr, end_attr = "top", "bottom"
        orth_start, orth_end = "x0", "x1"

    edges: List[float] = []
    filtered_ranges: List[Tuple[float, float]] = []
    for element in text_list:
        start = getattr(element, start_attr, None)
        end = getattr(element, end_attr, None)
        if start is None or end is None:
            continue
        try:
            start_val = float(start)
            end_val = float(end)
        except (TypeError, ValueError):
            continue
        if end_val <= start_val:
            continue

        orth_start_val = getattr(element, orth_start, None)
        orth_end_val = getattr(element, orth_end, None)
        if orth_start_val is not None and orth_end_val is not None:
            try:
                orth_start_f = float(orth_start_val)
                orth_end_f = float(orth_end_val)
            except (TypeError, ValueError):
                continue
            if orth_end_f < orth_low or orth_start_f > orth_high:
                continue

        edges.extend([start_val, end_val])
        filtered_ranges.append((start_val, end_val))

    if len(edges) < 2:
        return []

    sorted_edges = sorted(set(edges))
    gaps: List[Gap] = []
    for i in range(len(sorted_edges) - 1):
        gap_start = sorted_edges[i]
        gap_end = sorted_edges[i + 1]
        if gap_end - gap_start < min_gap:
            continue

        if not any(start < gap_end and end > gap_start for start, end in filtered_ranges):
            gaps.append((gap_start, gap_end))

    return gaps
