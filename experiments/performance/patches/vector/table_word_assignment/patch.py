"""Vectorize guide table word-to-cell assignment for center/full modes."""

from contextlib import contextmanager

import numpy as np

from experiments.performance import vector_metrics as metrics
from experiments.performance.vector_helpers import element_array

METADATA = {
    "track": "vector",
    "candidate": "table_word_assignment",
    "cache_only": False,
    "hypothesis": "Guide table cells can assign words with searchsorted over row/column bounds.",
}


def _patched_build_table_from_words_fast(
    grid,
    *,
    table_region,
    content_filter,
    apply_exclusions,
    cell_overlap,
    cell_newlines,
):
    if cell_overlap == "partial":
        metrics.count("vector.table_word_assignment.fallback_partial")
        return _ORIGINAL(
            grid,
            table_region=table_region,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
            cell_overlap=cell_overlap,
            cell_newlines=cell_newlines,
        )
    if table_region is None or not hasattr(table_region, "find_all"):
        return None

    bounds = _CELLS._axis_bounds_from_grid(grid)
    if bounds is None:
        return None
    row_tops, row_bottoms, col_lefts, col_rights = bounds
    row_tops_arr = np.asarray(row_tops, dtype=float)
    row_bottoms_arr = np.asarray(row_bottoms, dtype=float)
    col_lefts_arr = np.asarray(col_lefts, dtype=float)
    col_rights_arr = np.asarray(col_rights, dtype=float)

    try:
        words = table_region.find_all(
            "text",
            overlap="partial",
            apply_exclusions=apply_exclusions,
        )
    except Exception:
        return None

    words_list = list(getattr(words, "elements", words))
    word_bins = [[[] for _ in range(len(col_lefts))] for _ in range(len(row_tops))]
    if not words_list:
        return _render_grid(grid, word_bins, content_filter, cell_newlines)

    arr = element_array(words_list, label="vector.table_word_assignment.words")
    row_idx = np.searchsorted(row_tops_arr, arr.cy, side="right") - 1
    col_idx = np.searchsorted(col_lefts_arr, arr.cx, side="right") - 1
    in_bounds = (
        arr.valid
        & (row_idx >= 0)
        & (col_idx >= 0)
        & (row_idx < len(row_tops))
        & (col_idx < len(col_lefts))
    )
    if np.any(in_bounds):
        row_idx_safe = np.clip(row_idx, 0, max(len(row_tops) - 1, 0))
        col_idx_safe = np.clip(col_idx, 0, max(len(col_lefts) - 1, 0))
        center_inside = (
            (col_lefts_arr[col_idx_safe] <= arr.cx)
            & (arr.cx <= col_rights_arr[col_idx_safe])
            & (row_tops_arr[row_idx_safe] <= arr.cy)
            & (arr.cy <= row_bottoms_arr[row_idx_safe])
        )
        in_bounds &= center_inside
        if cell_overlap == "full":
            in_bounds &= (
                (col_lefts_arr[col_idx_safe] <= arr.x0)
                & (arr.x1 <= col_rights_arr[col_idx_safe])
                & (row_tops_arr[row_idx_safe] <= arr.top)
                & (arr.bottom <= row_bottoms_arr[row_idx_safe])
            )
        metrics.byte_count(
            "vector.table_word_assignment.mask_bytes",
            in_bounds.nbytes + center_inside.nbytes,
        )

    for index in np.flatnonzero(in_bounds):
        row = int(row_idx[index])
        col = int(col_idx[index])
        if grid[row][col] is not None:
            word_bins[row][col].append(words_list[int(index)])

    metrics.count("vector.table_word_assignment.fast_path")
    metrics.count("vector.table_word_assignment.words", len(words_list))
    return _render_grid(grid, word_bins, content_filter, cell_newlines)


def _render_grid(grid, word_bins, content_filter, cell_newlines):
    return [
        [
            (
                _CELLS._cell_text_from_words(
                    cell,
                    word_bins[row_idx][col_idx],
                    content_filter=content_filter,
                    cell_newlines=cell_newlines,
                )
                if cell is not None
                else None
            )
            for col_idx, cell in enumerate(row)
        ]
        for row_idx, row in enumerate(grid)
    ]


@contextmanager
def install():
    import natural_pdf.tables.utils.cells as cells

    global _ORIGINAL, _CELLS
    _CELLS = cells
    _ORIGINAL = cells._build_table_from_words_fast

    cells._build_table_from_words_fast = _patched_build_table_from_words_fast
    try:
        yield
    finally:
        cells._build_table_from_words_fast = _ORIGINAL
