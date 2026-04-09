"""Single-page grid building and temporary region lifecycle helpers."""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from ._grid_types import GridBuildCounts, GridBuildRegions, GridBuildResult
from ._targets import iter_page_regions, resolve_single_page_grid_target

logger = logging.getLogger(__name__)

TEMP_GRID_REGION_TYPES = frozenset({"table", "table-row", "table-column", "table-cell"})


def _bbox_overlap(
    b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]
) -> bool:
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])


def find_temporary_grid_regions(
    pages: Iterable[Any],
    *,
    source: str,
    overlap_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[Tuple[Any, Any]]:
    matches: List[Tuple[Any, Any]] = []
    seen_regions: set[int] = set()

    for page in pages:
        for region in iter_page_regions(page):
            marker = id(region)
            if marker in seen_regions:
                continue
            seen_regions.add(marker)

            if getattr(region, "source", None) != source:
                continue
            if getattr(region, "region_type", None) not in TEMP_GRID_REGION_TYPES:
                continue
            if overlap_bbox is not None:
                bbox = getattr(region, "bbox", None)
                if bbox is None or not _bbox_overlap(bbox, overlap_bbox):
                    continue

            matches.append((page, region))

    return matches


def remove_temporary_grid_regions(
    pages: Iterable[Any],
    *,
    source: str,
    overlap_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[Any]:
    removed_regions: List[Any] = []
    for page, region in find_temporary_grid_regions(
        pages, source=source, overlap_bbox=overlap_bbox
    ):
        page.remove_element(region, element_type="regions")
        removed_regions.append(region)
    return removed_regions


def build_single_page_grid(
    *,
    target_obj: Any,
    verticals: Sequence[float],
    horizontals: Sequence[float],
    source: str,
    cell_padding: float,
    include_outer_boundaries: bool,
) -> GridBuildResult:
    page, bounds = resolve_single_page_grid_target(target_obj)
    origin_x, origin_y, max_x, max_y = bounds
    context_width = max_x - origin_x
    context_height = max_y - origin_y

    row_boundaries = [float(v) for v in horizontals]
    col_boundaries = [float(v) for v in verticals]

    if include_outer_boundaries:
        if not row_boundaries or row_boundaries[0] > origin_y:
            row_boundaries.insert(0, origin_y)
        if not row_boundaries or row_boundaries[-1] < origin_y + context_height:
            row_boundaries.append(origin_y + context_height)

        if not col_boundaries or col_boundaries[0] > origin_x:
            col_boundaries.insert(0, origin_x)
        if not col_boundaries or col_boundaries[-1] < origin_x + context_width:
            col_boundaries.append(origin_x + context_width)

    row_boundaries = sorted(set(row_boundaries))
    col_boundaries = sorted(set(col_boundaries))

    effective_bbox: Optional[Tuple[float, float, float, float]] = None
    if row_boundaries and col_boundaries:
        effective_bbox = (
            col_boundaries[0],
            row_boundaries[0],
            col_boundaries[-1],
            row_boundaries[-1],
        )
        removed = remove_temporary_grid_regions([page], source=source, overlap_bbox=effective_bbox)
        if removed:
            logger.debug("Removed %d existing grid region(s) prior to rebuild", len(removed))

    logger.debug(
        "Building grid with %d row and %d col boundaries",
        len(row_boundaries),
        len(col_boundaries),
    )

    result = GridBuildResult(effective_bbox=effective_bbox)

    if len(row_boundaries) < 2 or len(col_boundaries) < 2:
        return result

    table_region = page.create_region(
        col_boundaries[0], row_boundaries[0], col_boundaries[-1], row_boundaries[-1]
    )
    table_region.source = source
    table_region.region_type = "table"
    table_region.normalized_type = "table"
    table_region.metadata.update(
        {
            "source_guides": True,
            "num_rows": len(row_boundaries) - 1,
            "num_cols": len(col_boundaries) - 1,
            "boundaries": {"rows": row_boundaries, "cols": col_boundaries},
        }
    )
    page.add_region(table_region, source=source)
    result.counts.table = 1
    result.regions.table = table_region

    for i in range(len(row_boundaries) - 1):
        row_region = page.create_region(
            col_boundaries[0], row_boundaries[i], col_boundaries[-1], row_boundaries[i + 1]
        )
        row_region.source = source
        row_region.region_type = "table-row"
        row_region.normalized_type = "table-row"
        row_region.metadata.update({"row_index": i, "source_guides": True})
        page.add_region(row_region, source=source)
        result.counts.rows += 1
        result.regions.rows.append(row_region)

    for j in range(len(col_boundaries) - 1):
        col_region = page.create_region(
            col_boundaries[j], row_boundaries[0], col_boundaries[j + 1], row_boundaries[-1]
        )
        col_region.source = source
        col_region.region_type = "table-column"
        col_region.normalized_type = "table-column"
        col_region.metadata.update({"col_index": j, "source_guides": True})
        page.add_region(col_region, source=source)
        result.counts.columns += 1
        result.regions.columns.append(col_region)

    for i in range(len(row_boundaries) - 1):
        for j in range(len(col_boundaries) - 1):
            cell_x0 = col_boundaries[j] + cell_padding
            cell_top = row_boundaries[i] + cell_padding
            cell_x1 = col_boundaries[j + 1] - cell_padding
            cell_bottom = row_boundaries[i + 1] - cell_padding

            if cell_x1 <= cell_x0 or cell_bottom <= cell_top:
                continue

            cell_region = page.create_region(cell_x0, cell_top, cell_x1, cell_bottom)
            cell_region.source = source
            cell_region.region_type = "table-cell"
            cell_region.normalized_type = "table-cell"
            cell_region.metadata.update(
                {
                    "row_index": i,
                    "col_index": j,
                    "source_guides": True,
                    "original_boundaries": {
                        "left": col_boundaries[j],
                        "top": row_boundaries[i],
                        "right": col_boundaries[j + 1],
                        "bottom": row_boundaries[i + 1],
                    },
                }
            )
            page.add_region(cell_region, source=source)
            result.counts.cells += 1
            result.regions.cells.append(cell_region)

    logger.info(
        "Created %d table, %d rows, %d columns, and %d cells from guides",
        result.counts.table,
        result.counts.rows,
        result.counts.columns,
        result.counts.cells,
    )

    return result
