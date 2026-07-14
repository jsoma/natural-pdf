"""Helpers for extracting text from table cells."""

from __future__ import annotations

import warnings
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

DEFAULT_CELL_OCR_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "min_confidence": 0.1,
    "detection_params": {
        "text_threshold": 0.1,
        "link_threshold": 0.1,
    },
}

CellExtractMode = Literal["text", "words"]
CellOverlapMode = Literal["center", "full", "partial"]
CellNewlines = Union[bool, str]

SLOW_CALLBACK_CELL_THRESHOLD = 500
SLOW_CALLBACK_WARNING = (
    "cell_extraction_func runs once per table cell and can be slow for large tables. "
    'For text-only extraction, prefer cell_extract="words" with cell_overlap=... '
    "so natural-pdf can batch word assignment."
)


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


def warn_slow_cell_callback(cell_count: int) -> None:
    """Warn when an arbitrary per-cell callback is likely to be expensive."""

    if cell_count > SLOW_CALLBACK_CELL_THRESHOLD:
        warnings.warn(SLOW_CALLBACK_WARNING, UserWarning, stacklevel=3)


def _validate_cell_options(cell_extract: str, cell_overlap: str) -> None:
    if cell_extract not in ("text", "words"):
        raise ValueError("cell_extract must be 'text' or 'words'")
    if cell_overlap not in ("center", "full", "partial"):
        raise ValueError("cell_overlap must be 'center', 'full', or 'partial'")


def _normalize_newlines(text: str, newlines: CellNewlines) -> str:
    if newlines is False:
        return text.replace("\n", " ").replace("\r", " ")
    if isinstance(newlines, str):
        return text.replace("\n", newlines).replace("\r", newlines)
    return text


def _apply_content_filter(cell_region: Any, text: str, content_filter: Any) -> str:
    if content_filter is None:
        return text

    filter_func = getattr(cell_region, "_apply_content_filter_to_text", None)
    if callable(filter_func):
        return filter_func(text, content_filter)
    return text


def extract_cell_value(
    cell_region: Any,
    *,
    cell_extraction_func: Optional[Callable[[Any], Optional[str]]] = None,
    use_ocr: bool = False,
    ocr_config: Optional[Dict[str, Any]] = None,
    content_filter: Any = None,
    apply_exclusions: bool = True,
    cell_extract: CellExtractMode = "text",
    cell_overlap: CellOverlapMode = "center",
    cell_newlines: CellNewlines = True,
    text_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Extract one cell value using callback, OCR, text, or word-level options."""

    _validate_cell_options(cell_extract, cell_overlap)

    if callable(cell_extraction_func):
        # Callback failures must be visible to callers.  Silently converting an
        # exception (or an invalid return value) to ``None`` makes a failed cell
        # indistinguishable from an intentionally blank one.
        value = cell_extraction_func(cell_region)
        if not isinstance(value, (str, type(None))):
            raise TypeError(
                "cell_extraction_func must return str or None, " f"got {type(value).__name__}"
            )
        return value

    if use_ocr:
        resolved_config = merge_ocr_config(ocr_config)
        cell_region.apply_ocr(**resolved_config)
        ocr_text = cell_region.extract_text(apply_exclusions=apply_exclusions).strip()
        if ocr_text:
            ocr_text = _normalize_newlines(ocr_text, cell_newlines)
            ocr_text = _apply_content_filter(cell_region, ocr_text, content_filter)
            return ocr_text or None

    if cell_extract == "words":
        words = cell_region.find_all(
            "text",
            overlap=cell_overlap,
            apply_exclusions=apply_exclusions,
        )
        text = words.extract_text(
            separator=" ",
            newlines=cell_newlines,
        ).strip()
    else:
        text = cell_region.extract_text(
            apply_exclusions=apply_exclusions,
            **(text_kwargs or {}),
        ).strip()
        text = _normalize_newlines(text, cell_newlines)

    text = _apply_content_filter(cell_region, text, content_filter)
    return text or None


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


def _bbox_for(element: Any) -> Optional[Tuple[float, float, float, float]]:
    bbox = getattr(element, "bbox", None)
    if bbox is not None and len(bbox) == 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    attrs = ("x0", "top", "x1", "bottom")
    if all(hasattr(element, attr) for attr in attrs):
        return tuple(float(getattr(element, attr)) for attr in attrs)  # type: ignore[return-value]
    return None


def _bbox_intersects(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _bbox_contains(
    outer: Tuple[float, float, float, float], inner: Tuple[float, float, float, float]
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _indexed_cell_grid(cell_regions: Sequence[Any]) -> Optional[List[List[Optional[Any]]]]:
    row_idxs: List[int] = []
    col_idxs: List[int] = []
    for cell in cell_regions:
        try:
            row_idx_value = cell.metadata.get("row_index")
            col_idx_value = cell.metadata.get("col_index")
            if row_idx_value is None or col_idx_value is None:
                return None
            row_idxs.append(int(row_idx_value))
            col_idxs.append(int(col_idx_value))
        except Exception:
            return None

    if not row_idxs or not col_idxs:
        return None

    grid: List[List[Optional[Any]]] = [
        [None] * (max(col_idxs) + 1) for _ in range(max(row_idxs) + 1)
    ]
    for cell, row_idx, col_idx in zip(cell_regions, row_idxs, col_idxs):
        grid[row_idx][col_idx] = cell
    return grid


def _axis_bounds_from_grid(
    grid: List[List[Optional[Any]]],
) -> Optional[Tuple[List[float], List[float], List[float], List[float]]]:
    row_tops: List[float] = []
    row_bottoms: List[float] = []
    col_lefts: List[float] = []
    col_rights: List[float] = []

    for row in grid:
        row_cells = [cell for cell in row if cell is not None]
        if not row_cells:
            return None
        row_tops.append(float(min(cell.top for cell in row_cells)))
        row_bottoms.append(float(max(cell.bottom for cell in row_cells)))

    num_cols = len(grid[0]) if grid else 0
    for col_idx in range(num_cols):
        col_cells = [row[col_idx] for row in grid if row[col_idx] is not None]
        if not col_cells:
            return None
        col_lefts.append(float(min(cell.x0 for cell in col_cells)))
        col_rights.append(float(max(cell.x1 for cell in col_cells)))

    if row_tops != sorted(row_tops) or col_lefts != sorted(col_lefts):
        return None
    return row_tops, row_bottoms, col_lefts, col_rights


def _cell_text_from_words(
    cell: Any,
    words: Sequence[Any],
    *,
    content_filter: Any,
    cell_newlines: CellNewlines,
) -> Optional[str]:
    text_parts: List[str] = []
    for word in words:
        if hasattr(word, "extract_text"):
            word_text = word.extract_text()
        else:
            word_text = getattr(word, "text", "")
        if word_text:
            text_parts.append(str(word_text))

    text = " ".join(text_parts).strip()
    text = _normalize_newlines(text, cell_newlines)
    text = _apply_content_filter(cell, text, content_filter)
    return text or None


def _build_table_from_words_fast(
    grid: List[List[Optional[Any]]],
    *,
    table_region: Any,
    content_filter: Any,
    apply_exclusions: bool,
    cell_overlap: CellOverlapMode,
    cell_newlines: CellNewlines,
) -> Optional[List[List[Optional[str]]]]:
    if table_region is None or not hasattr(table_region, "find_all"):
        return None

    bounds = _axis_bounds_from_grid(grid)
    if bounds is None:
        return None
    row_tops, row_bottoms, col_lefts, col_rights = bounds

    try:
        words = table_region.find_all(
            "text",
            overlap="partial",
            apply_exclusions=apply_exclusions,
        )
    except Exception:
        return None

    word_bins: List[List[List[Any]]] = [
        [[] for _ in range(len(col_lefts))] for _ in range(len(row_tops))
    ]

    def append_to_cell(row_idx: int, col_idx: int, word: Any) -> None:
        if row_idx < 0 or col_idx < 0:
            return
        if row_idx >= len(row_tops) or col_idx >= len(col_lefts):
            return
        if grid[row_idx][col_idx] is None:
            return
        word_bins[row_idx][col_idx].append(word)

    for word in words:
        word_bbox = _bbox_for(word)
        if word_bbox is None:
            continue

        if cell_overlap in ("center", "full"):
            cx = (word_bbox[0] + word_bbox[2]) / 2.0
            cy = (word_bbox[1] + word_bbox[3]) / 2.0
            col_idx = bisect_right(col_lefts, cx) - 1
            row_idx = bisect_right(row_tops, cy) - 1
            if row_idx < 0 or col_idx < 0:
                continue
            if row_idx >= len(row_tops) or col_idx >= len(col_lefts):
                continue

            cell_bbox = (
                col_lefts[col_idx],
                row_tops[row_idx],
                col_rights[col_idx],
                row_bottoms[row_idx],
            )
            if not _bbox_contains(cell_bbox, (cx, cy, cx, cy)):
                continue
            if cell_overlap == "full" and not _bbox_contains(cell_bbox, word_bbox):
                continue
            append_to_cell(row_idx, col_idx, word)
            continue

        col_start = bisect_right(col_rights, word_bbox[0])
        col_end = bisect_left(col_lefts, word_bbox[2])
        row_start = bisect_right(row_bottoms, word_bbox[1])
        row_end = bisect_left(row_tops, word_bbox[3])

        for row_idx in range(row_start, row_end):
            for col_idx in range(col_start, col_end):
                cell_bbox = (
                    col_lefts[col_idx],
                    row_tops[row_idx],
                    col_rights[col_idx],
                    row_bottoms[row_idx],
                )
                if _bbox_intersects(cell_bbox, word_bbox):
                    append_to_cell(row_idx, col_idx, word)

    return [
        [
            (
                _cell_text_from_words(
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


def build_table_from_cells(
    cell_regions: Sequence[Any],
    *,
    table_region: Any = None,
    cell_extraction_func: Optional[Callable[[Any], Optional[str]]] = None,
    use_ocr: bool = False,
    ocr_config: Optional[Dict[str, Any]] = None,
    content_filter=None,
    apply_exclusions: bool = True,
    cell_extract: CellExtractMode = "text",
    cell_overlap: CellOverlapMode = "center",
    cell_newlines: CellNewlines = True,
) -> List[List[Optional[str]]]:
    """Construct a table (list-of-lists) from table_cell regions."""

    if not cell_regions:
        return []

    _validate_cell_options(cell_extract, cell_overlap)

    if callable(cell_extraction_func):
        warn_slow_cell_callback(len(cell_regions))

    indexed_grid = _indexed_cell_grid(cell_regions)
    if indexed_grid is not None:
        if cell_extract == "words" and not callable(cell_extraction_func) and not use_ocr:
            fast_table = _build_table_from_words_fast(
                indexed_grid,
                table_region=table_region,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
                cell_overlap=cell_overlap,
                cell_newlines=cell_newlines,
            )
            if fast_table is not None:
                return fast_table

        table_grid: List[List[Optional[str]]] = [
            [None] * len(indexed_grid[0]) for _ in range(len(indexed_grid))
        ]
        for cell in cell_regions:
            row_idx = cell.metadata.get("row_index")
            col_idx = cell.metadata.get("col_index")
            if row_idx is None or col_idx is None:
                raise ValueError("Missing explicit indices")
            cell_text = extract_cell_value(
                cell,
                cell_extraction_func=cell_extraction_func,
                use_ocr=use_ocr,
                ocr_config=ocr_config,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
                cell_extract=cell_extract,
                cell_overlap=cell_overlap,
                cell_newlines=cell_newlines,
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

        cell_text = extract_cell_value(
            cell,
            cell_extraction_func=cell_extraction_func,
            use_ocr=use_ocr,
            ocr_config=ocr_config,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
            cell_extract=cell_extract,
            cell_overlap=cell_overlap,
            cell_newlines=cell_newlines,
        )
        table_grid[row_idx][col_idx] = cell_text

    return table_grid
