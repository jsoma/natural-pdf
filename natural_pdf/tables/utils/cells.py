"""Helpers for extracting text from table cells."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

DEFAULT_CELL_OCR_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "min_confidence": 0.1,
    "detection_params": {
        "text_threshold": 0.1,
        "link_threshold": 0.1,
    },
}


def merge_ocr_config(user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge user-provided OCR config with sensible defaults."""

    merged = deepcopy(DEFAULT_CELL_OCR_CONFIG)
    if not user_config:
        return merged

    if not isinstance(user_config, Mapping):
        return merged

    for key, value in user_config.items():
        if isinstance(value, Mapping) and key in merged and isinstance(merged[key], dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def extract_cell_text(
    cell_region,
    *,
    use_ocr: bool = False,
    ocr_config: Optional[Dict[str, Any]] = None,
    content_filter=None,
    apply_exclusions: bool = True,
) -> Optional[str]:
    """Extract text from a single cell region with optional OCR + filtering."""

    if use_ocr:
        resolved_config = merge_ocr_config(ocr_config)
        cell_region.apply_ocr(**resolved_config)
        ocr_text = cell_region.extract_text(apply_exclusions=apply_exclusions).strip()
        if ocr_text:
            if content_filter is not None:
                ocr_text = cell_region._apply_content_filter_to_text(ocr_text, content_filter)
            return ocr_text

    text = cell_region.extract_text(apply_exclusions=apply_exclusions).strip()
    if content_filter is not None:
        text = cell_region._apply_content_filter_to_text(text, content_filter)
    return text or None


def build_table_from_cells(
    cell_regions: Sequence[Any],
    *,
    content_filter=None,
    apply_exclusions: bool = True,
) -> List[List[Optional[str]]]:
    """Construct a table (list-of-lists) from table_cell regions."""

    if not cell_regions:
        return []

    all_row_idxs: List[int] = []
    all_col_idxs: List[int] = []
    for cell in cell_regions:
        try:
            row_idx_value = cell.metadata.get("row_index")
            col_idx_value = cell.metadata.get("col_index")
            if row_idx_value is None or col_idx_value is None:
                raise ValueError("Missing explicit indices")

            r_idx = int(row_idx_value)
            c_idx = int(col_idx_value)
            all_row_idxs.append(r_idx)
            all_col_idxs.append(c_idx)
        except Exception:
            all_row_idxs = []
            all_col_idxs = []
            break

    if all_row_idxs and all_col_idxs:
        num_rows = max(all_row_idxs) + 1
        num_cols = max(all_col_idxs) + 1
        table_grid: List[List[Optional[str]]] = [[None] * num_cols for _ in range(num_rows)]

        for cell in cell_regions:
            row_idx = cell.metadata.get("row_index")
            col_idx = cell.metadata.get("col_index")
            if row_idx is None or col_idx is None:
                raise ValueError("Missing explicit indices")
            cell_text = extract_cell_text(
                cell,
                use_ocr=False,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
            )
            table_grid[int(row_idx)][int(col_idx)] = cell_text

        return table_grid

    centers = np.array([[(c.x0 + c.x1) / 2.0, (c.top + c.bottom) / 2.0] for c in cell_regions])
    xs = centers[:, 0]
    ys = centers[:, 1]

    def _cluster(values: Sequence[float], tol: float = 1.0) -> List[float]:
        sorted_vals = np.sort(values)
        groups = [[sorted_vals[0]]]
        for value in sorted_vals[1:]:
            if abs(value - groups[-1][-1]) <= tol:
                groups[-1].append(value)
            else:
                groups.append([value])
        return [float(np.mean(group)) for group in groups]

    row_centers = _cluster(ys.tolist())
    col_centers = _cluster(xs.tolist())

    num_rows = len(row_centers)
    num_cols = len(col_centers)
    table_grid: List[List[Optional[str]]] = [[None] * num_cols for _ in range(num_rows)]

    for cell, (cx, cy) in zip(cell_regions, centers):
        row_idx = int(np.argmin([abs(cy - rc) for rc in row_centers]))
        col_idx = int(np.argmin([abs(cx - cc) for cc in col_centers]))

        cell_text = extract_cell_text(
            cell,
            use_ocr=False,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
        )
        table_grid[row_idx][col_idx] = cell_text

    return table_grid
