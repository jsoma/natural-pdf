"""Text-based detection utilities for guide discovery."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

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
    """
    Find whitespace gaps using prominence-based valley detection.

    Uses scipy.signal.find_peaks on inverted density to find true valleys
    (local minima) rather than just below-threshold regions. This handles
    cases where text lengths vary across rows (e.g., one long name extending
    into what should be a gap).
    """
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

    # Build density array
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

    return _find_gaps_via_minima(density, min_gap, low)


def _find_gaps_via_minima(density: np.ndarray, min_gap: float, origin: float) -> List[Gap]:
    """
    Find gaps by detecting local minima (valleys) in the density signal.

    Uses scipy.signal.find_peaks on inverted density with prominence filtering
    to find true column separators, even when some text crosses the gap.
    """
    if density.size == 0:
        return []

    max_density = density.max()
    if max_density <= 0:
        return []

    # Smooth the density to merge character-level noise into column-level hills
    smoothed = gaussian_filter1d(density.astype(float), sigma=2.0)

    # Normalize to 0-1 for consistent parameters
    if smoothed.max() > 0:
        normalized = smoothed / smoothed.max()
    else:
        return []

    # Invert: valleys become peaks
    inverted = 1.0 - normalized

    # Find peaks in inverted signal (which are valleys in original)
    # prominence: how much the valley drops relative to surrounding peaks
    # width: minimum gap width
    # distance: minimum separation between valleys (column width)
    peaks, properties = find_peaks(
        inverted,
        prominence=0.1,  # Valley must drop at least 10% relative to surroundings
        width=max(1, int(min_gap * 0.5)),  # At least half the min_gap width
        distance=max(1, int(min_gap)),  # Valleys must be at least min_gap apart
    )

    if len(peaks) == 0:
        return []

    gaps: List[Gap] = []

    for i, peak_idx in enumerate(peaks):
        # Get the width bounds for this valley
        left_ips = properties.get("left_ips", [])
        right_ips = properties.get("right_ips", [])

        if len(left_ips) > i and len(right_ips) > i:
            # Use the interpolated width bounds from find_peaks
            left_bound = int(left_ips[i])
            right_bound = int(right_ips[i]) + 1
        else:
            # Fallback: expand from peak center until we hit rising density
            left_bound = peak_idx
            right_bound = peak_idx + 1

            # Expand left
            while left_bound > 0 and normalized[left_bound - 1] <= normalized[left_bound] + 0.05:
                left_bound -= 1

            # Expand right
            while (
                right_bound < len(normalized) - 1
                and normalized[right_bound] <= normalized[right_bound - 1] + 0.05
            ):
                right_bound += 1

        # Validate: the gap region should have actual zero-density somewhere
        # This ensures we're not just finding a "saddle point" with text throughout
        gap_region = density[left_bound:right_bound]
        if len(gap_region) > 0 and gap_region.min() == 0:
            # Find the actual zero-density portion within this valley
            zero_mask = gap_region == 0
            zero_indices = np.where(zero_mask)[0]

            if len(zero_indices) > 0:
                # Find largest contiguous zero-density run
                splits = np.where(np.diff(zero_indices) != 1)[0] + 1
                groups = np.split(zero_indices, splits)
                largest_group = max(groups, key=len)

                actual_start = left_bound + largest_group[0]
                actual_end = left_bound + largest_group[-1] + 1

                start_coord = origin + actual_start
                end_coord = origin + actual_end

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
