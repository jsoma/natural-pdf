"""Helper functions for computing guide separators."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Bounds = Tuple[float, float, float, float]


def find_min_crossing_separator(
    x0: float,
    x1: float,
    bboxes: Sequence[Bounds],
    num_samples: int,
) -> float:
    """Find an x/y coordinate with minimal crossings through bounding boxes."""
    candidates = np.linspace(x0, x1, max(2, num_samples))

    best_coord = x0
    min_crossings = float("inf")
    best_gap = 0.0

    for coord in candidates:
        crossings = sum(1 for bbox in bboxes if bbox[0] < coord < bbox[2])

        if crossings > 0:
            gaps: List[float] = []
            for bbox in bboxes:
                if bbox[0] < coord < bbox[2]:
                    gaps.extend([abs(coord - bbox[0]), abs(coord - bbox[2])])
            min_gap = min(gaps) if gaps else float("inf")
        else:
            min_gap = float("inf")

        if crossings < min_crossings or (crossings == min_crossings and min_gap > best_gap):
            min_crossings = crossings
            best_coord = coord
            best_gap = min_gap

    return float(best_coord)


def find_seam_carving_separator(
    x0: float,
    x1: float,
    header_y: float,
    page_bottom: float,
    bboxes: Sequence[Bounds],
) -> float:
    """Use seam carving to find a low-cost separator path between text blocks."""
    band_width = max(1, int(x1 - x0))
    band_height = max(1, int(page_bottom - header_y))

    if band_width <= 0 or band_height <= 0:
        return float((x0 + x1) / 2)

    cost_matrix = np.zeros((band_height, band_width))

    for bbox in bboxes:
        if bbox[2] > x0 and bbox[0] < x1 and bbox[3] > header_y:
            left = max(0, int(bbox[0] - x0))
            right = min(band_width, int(bbox[2] - x0))
            top = max(0, int(bbox[1] - header_y))
            bottom = min(band_height, int(bbox[3] - header_y))
            cost_matrix[top:bottom, left:right] = 100

    for i in range(band_width):
        cost_matrix[:, i] += abs(i - band_width // 2) * 0.1

    dp = np.full_like(cost_matrix, np.inf)
    dp[0, :] = cost_matrix[0, :]

    for y in range(1, band_height):
        for x in range(band_width):
            best_prev = dp[y - 1, x]
            if x > 0:
                best_prev = min(best_prev, dp[y - 1, x - 1])
            if x < band_width - 1:
                best_prev = min(best_prev, dp[y - 1, x + 1])
            dp[y, x] = cost_matrix[y, x] + best_prev

    min_x = int(np.argmin(dp[-1, :]))

    path_coords = [min_x]
    for y in range(band_height - 2, -1, -1):
        x = path_coords[-1]
        candidates = [(x, dp[y, x])]
        if x > 0:
            candidates.append((x - 1, dp[y, x - 1]))
        if x < band_width - 1:
            candidates.append((x + 1, dp[y, x + 1]))
        next_x = min(candidates, key=lambda c: c[1])[0]
        path_coords.append(next_x)

    median_x = float(np.median(path_coords))
    return float(x0 + median_x)


def stabilize_with_rows(
    separators: Sequence[float],
    bboxes: Sequence[Bounds],
    header_y: float,
) -> List[float]:
    """Stabilize separator positions using per-row medians."""
    if not bboxes:
        return list(separators)

    y_coords = sorted({bbox[1] for bbox in bboxes} | {bbox[3] for bbox in bboxes})

    gaps: List[Tuple[float, float]] = []
    for idx in range(len(y_coords) - 1):
        gap_size = y_coords[idx + 1] - y_coords[idx]
        if gap_size > 5:
            gaps.append((y_coords[idx], y_coords[idx + 1]))

    if not gaps:
        return list(separators)

    stabilized: List[float] = []
    for sep_index, separator in enumerate(separators):
        row_positions: List[float] = []
        for gap_start, gap_end in gaps:
            row_elements = [bbox for bbox in bboxes if bbox[1] >= gap_start and bbox[3] <= gap_end]
            if not row_elements:
                continue

            if sep_index == 0:
                window_start = 0.0
                window_end = separator + 20
            elif sep_index == len(separators) - 1:
                window_start = separator - 20
                window_end = float("inf")
            else:
                window_start = separator - 20
                window_end = separator + 20

            best_coord = find_min_crossing_separator(
                max(window_start, separator - 20),
                min(window_end, separator + 20),
                row_elements,
                50,
            )
            row_positions.append(best_coord)

        if len(row_positions) >= 3:
            stabilized.append(float(np.median(row_positions)))
        else:
            stabilized.append(float(separator))

    return stabilized
