"""FlowRegion-specific helper utilities for guide processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from natural_pdf.flows.region import FlowRegion

if True:  # pragma: no cover - circular imports avoided at runtime
    try:
        from natural_pdf.elements.region import Region
    except ImportError:  # pragma: no cover
        Region = Any  # type: ignore

logger = logging.getLogger(__name__)

_BOUNDARY_TOLERANCE = 0.1


@dataclass
class RegionGrid:
    """Container tracking a region and the grid fragments built inside it."""

    region: Region
    table: Region | None
    rows: List[Region]
    columns: List[Region]
    cells: List[Region]


class FlowGuideAdapter:
    """Adapts FlowRegion contexts so grid building logic stays isolated."""

    def __init__(self, guides: Any):
        if not getattr(guides, "is_flow_region", False):
            raise ValueError("FlowGuideAdapter requires guides bound to a FlowRegion context")

        self._guides = guides
        self._flow_region: FlowRegion = guides._flow_context()
        self.flow = self._flow_region.flow
        self._regions = list(guides._flow_constituent_regions())
        self._multi_region = len(self._regions) > 1
        self._guides_cls = guides.__class__
        self.vertical_coords = [float(coord) for coord in guides.vertical]
        self.horizontal_coords = [float(coord) for coord in guides.horizontal]
        self._ensure_region_entries()

    def build_region_grids(
        self,
        source: str,
        cell_padding: float,
    ) -> List[RegionGrid]:
        """Build per-region grids by clipping the unified guide set."""
        region_grids: List[RegionGrid] = []

        for region, verticals, horizontals in self._iter_region_guides():
            region_guides = self._guides_cls(
                verticals=verticals,
                horizontals=horizontals,
                context=region,
            )

            grid_parts = region_guides._build_grid_single_page(
                target=region,
                source=source,
                cell_padding=cell_padding,
                include_outer_boundaries=False,
            )

            if grid_parts["counts"]["table"] <= 0:
                continue

            if self._multi_region:
                self._mark_fragments(grid_parts)

            region_grids.append(
                RegionGrid(
                    region=region,
                    table=grid_parts["regions"]["table"],
                    rows=list(grid_parts["regions"]["rows"]),
                    columns=list(grid_parts["regions"]["columns"]),
                    cells=list(grid_parts["regions"]["cells"]),
                )
            )

        return region_grids

    def stitch_region_results(
        self,
        region_grids: Sequence[RegionGrid],
        orientation: str,
        source: str,
    ) -> Tuple[List[FlowRegion], List[FlowRegion], List[FlowRegion]]:
        """Combine per-region fragments into logical FlowRegion rows/columns/cells."""
        if not region_grids:
            return [], [], []

        if orientation == "vertical":
            return self._stitch_vertical(region_grids, source)
        if orientation == "horizontal":
            return self._stitch_horizontal(region_grids, source)

        rows = [row for grid in region_grids for row in grid.rows]
        columns = [col for grid in region_grids for col in grid.columns]
        cells = [cell for grid in region_grids for cell in grid.cells]
        return rows, columns, cells

    @property
    def regions(self) -> Sequence[Region]:
        return tuple(self._regions)

    def update_axis_from_regions(
        self,
        axis: str,
        region_values: Dict[Any, Sequence[float]],
        *,
        append: bool = False,
    ) -> None:
        """Update a single axis using per-region coordinate lists."""
        axis_key = axis.lower()
        if axis_key not in {"vertical", "horizontal"}:
            raise ValueError("axis must be 'vertical' or 'horizontal'")

        unified_attr, cache_attr, axis_list, start_idx, end_idx = self._axis_metadata(axis_key)
        normalized = {
            region: [float(value) for value in region_values.get(region, [])]
            for region in self._regions
        }

        aggregated: List[float] = []
        if append:
            existing = getattr(self._guides, unified_attr, [])
            aggregated.extend(coord for coord, _ in existing)

        for coords in normalized.values():
            aggregated.extend(coords)

        unique_coords = sorted({float(coord) for coord in aggregated})

        new_unified: List[Tuple[float, Region]] = []
        for coord in unique_coords:
            for region in self._regions:
                bounds = getattr(region, "bbox", None)
                if not bounds:
                    continue
                start = bounds[start_idx]
                end = bounds[end_idx]
                if start <= coord <= end:
                    new_unified.append((float(coord), region))
                    break

        setattr(self._guides, unified_attr, new_unified)
        setattr(self._guides, cache_attr, None)
        axis_list.data = unique_coords

        for region in self._regions:
            verticals, horizontals = self._guides._flow_guides.get(region, ([], []))
            if axis_key == "vertical":
                verticals = [coord for coord, r in new_unified if r == region]
            else:
                horizontals = [coord for coord, r in new_unified if r == region]
            self._guides._flow_guides[region] = (
                sorted(verticals),
                sorted(horizontals),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_region_guides(self) -> Iterable[Tuple[Region, List[float], List[float]]]:
        for region in self._regions:
            bounds = getattr(region, "bbox", None)
            if not bounds:
                continue
            verticals = self._clip_axis(self.vertical_coords, bounds[0], bounds[2])
            horizontals = self._clip_axis(self.horizontal_coords, bounds[1], bounds[3])
            verticals = self._ensure_bounds(verticals, bounds[0], bounds[2])
            horizontals = self._ensure_bounds(horizontals, bounds[1], bounds[3])
            if len(verticals) < 2 or len(horizontals) < 2:
                continue
            yield region, verticals, horizontals

    @staticmethod
    def _clip_axis(values: Sequence[float], start: float, end: float) -> List[float]:
        return [value for value in values if start <= value <= end]

    @staticmethod
    def _ensure_bounds(values: Sequence[float], start: float, end: float) -> List[float]:
        enriched = set(values)
        enriched.add(start)
        enriched.add(end)
        return sorted(enriched)

    def _mark_fragments(self, grid_parts: Dict[str, Any]) -> None:
        fragments = grid_parts["regions"]
        type_map = {
            "table": "table_fragment",
            "rows": "table_row_fragment",
            "columns": "table_column_fragment",
            "cells": "table_cell_fragment",
        }

        for key, region_type in type_map.items():
            value = fragments.get(key)
            if not value:
                continue
            if isinstance(value, list):
                targets = value
            else:
                targets = [value]
            for target in targets:
                target.region_type = region_type
                target.metadata["is_fragment"] = True

    def _stitch_vertical(
        self,
        region_grids: Sequence[RegionGrid],
        source: str,
    ) -> Tuple[List[FlowRegion], List[FlowRegion], List[FlowRegion]]:
        page_rows = [list(grid.rows) for grid in region_grids]
        page_cells = [list(grid.cells) for grid in region_grids]

        for idx in range(len(region_grids) - 1):
            region_a = region_grids[idx].region
            boundary = region_a.bottom
            has_boundary = self._has_boundary(self.horizontal_coords, boundary)

            if has_boundary or not page_rows[idx] or not page_rows[idx + 1]:
                continue

            self._merge_row_boundary(
                page_rows,
                page_cells,
                idx,
                source,
            )

        final_rows = [row for rows in page_rows for row in rows]
        final_cells = [cell for cells in page_cells for cell in cells]
        final_columns = self._stitch_columns_across_pages(region_grids, source)
        return final_rows, final_columns, final_cells

    def _stitch_horizontal(
        self,
        region_grids: Sequence[RegionGrid],
        source: str,
    ) -> Tuple[List[FlowRegion], List[FlowRegion], List[FlowRegion]]:
        page_columns = [list(grid.columns) for grid in region_grids]
        page_cells = [list(grid.cells) for grid in region_grids]

        for idx in range(len(region_grids) - 1):
            region_a = region_grids[idx].region
            boundary = region_a.x1
            has_boundary = self._has_boundary(self.vertical_coords, boundary)
            if has_boundary or not page_columns[idx] or not page_columns[idx + 1]:
                continue
            self._merge_column_boundary(page_columns, page_cells, idx, source)

        final_columns = [col for columns in page_columns for col in columns]
        final_cells = [cell for cells in page_cells for cell in cells]
        final_rows = self._stitch_rows_across_pages(region_grids, source)
        return final_rows, final_columns, final_cells

    def _merge_row_boundary(
        self,
        page_rows: List[List[Region]],
        page_cells: List[List[Region]],
        index: int,
        source: str,
    ) -> None:
        flow = self.flow
        region_rows_a = page_rows[index]
        region_rows_b = page_rows[index + 1]

        last_row = region_rows_a.pop(-1)
        first_row = region_rows_b.pop(0)

        merged_row = FlowRegion(flow, [last_row, first_row], source_flow_element=None)
        merged_row.source = source
        merged_row.region_type = "table_row"
        merged_row.metadata.update(
            {
                "row_index": last_row.metadata.get("row_index"),
                "is_multi_page": True,
            }
        )
        region_rows_a.append(merged_row)

        last_row_idx = last_row.metadata.get("row_index")
        first_row_idx = first_row.metadata.get("row_index")

        last_cells = [
            cell for cell in page_cells[index] if cell.metadata.get("row_index") == last_row_idx
        ]
        next_cells = [
            cell
            for cell in page_cells[index + 1]
            if cell.metadata.get("row_index") == first_row_idx
        ]

        page_cells[index] = [
            cell for cell in page_cells[index] if cell.metadata.get("row_index") != last_row_idx
        ]
        page_cells[index + 1] = [
            cell
            for cell in page_cells[index + 1]
            if cell.metadata.get("row_index") != first_row_idx
        ]

        last_cells.sort(key=lambda cell: cell.metadata.get("col_index", 0))
        next_cells.sort(key=lambda cell: cell.metadata.get("col_index", 0))

        for cell_a, cell_b in zip(last_cells, next_cells):
            merged_cell = FlowRegion(flow, [cell_a, cell_b], source_flow_element=None)
            merged_cell.source = source
            merged_cell.region_type = "table_cell"
            merged_cell.metadata.update(
                {
                    "row_index": cell_a.metadata.get("row_index"),
                    "col_index": cell_a.metadata.get("col_index"),
                    "is_multi_page": True,
                }
            )
            page_cells[index].append(merged_cell)

    def _merge_column_boundary(
        self,
        page_columns: List[List[Region]],
        page_cells: List[List[Region]],
        index: int,
        source: str,
    ) -> None:
        flow = self.flow
        columns_a = page_columns[index]
        columns_b = page_columns[index + 1]

        last_col = columns_a.pop(-1)
        first_col = columns_b.pop(0)

        merged_col = FlowRegion(flow, [last_col, first_col], source_flow_element=None)
        merged_col.source = source
        merged_col.region_type = "table_column"
        merged_col.metadata.update(
            {
                "col_index": last_col.metadata.get("col_index"),
                "is_multi_page": True,
            }
        )
        columns_a.append(merged_col)

        last_col_idx = last_col.metadata.get("col_index")
        first_col_idx = first_col.metadata.get("col_index")

        col_cells_a = [
            cell for cell in page_cells[index] if cell.metadata.get("col_index") == last_col_idx
        ]
        col_cells_b = [
            cell
            for cell in page_cells[index + 1]
            if cell.metadata.get("col_index") == first_col_idx
        ]

        page_cells[index] = [
            cell for cell in page_cells[index] if cell.metadata.get("col_index") != last_col_idx
        ]
        page_cells[index + 1] = [
            cell
            for cell in page_cells[index + 1]
            if cell.metadata.get("col_index") != first_col_idx
        ]

        col_cells_a.sort(key=lambda cell: cell.metadata.get("row_index", 0))
        col_cells_b.sort(key=lambda cell: cell.metadata.get("row_index", 0))

        for cell_a, cell_b in zip(col_cells_a, col_cells_b):
            merged_cell = FlowRegion(flow, [cell_a, cell_b], source_flow_element=None)
            merged_cell.source = source
            merged_cell.region_type = "table_cell"
            merged_cell.metadata.update(
                {
                    "row_index": cell_a.metadata.get("row_index"),
                    "col_index": cell_a.metadata.get("col_index"),
                    "is_multi_page": True,
                }
            )
            page_cells[index].append(merged_cell)

    def _stitch_columns_across_pages(
        self,
        region_grids: Sequence[RegionGrid],
        source: str,
    ) -> List[FlowRegion]:
        stitched_columns: List[FlowRegion] = []
        flow = self.flow
        physical_columns = zip(*(grid.columns for grid in region_grids))

        for col_index, column_group in enumerate(physical_columns):
            column_regions = list(column_group)
            if not column_regions:
                continue
            column = FlowRegion(flow, column_regions, source_flow_element=None)
            column.source = source
            column.region_type = "table_column"
            column.metadata.update({"col_index": col_index, "is_multi_page": True})
            stitched_columns.append(column)

        return stitched_columns

    def _stitch_rows_across_pages(
        self,
        region_grids: Sequence[RegionGrid],
        source: str,
    ) -> List[FlowRegion]:
        stitched_rows: List[FlowRegion] = []
        flow = self.flow
        physical_rows = zip(*(grid.rows for grid in region_grids))

        for row_index, row_group in enumerate(physical_rows):
            row_regions = list(row_group)
            if not row_regions:
                continue
            row = FlowRegion(flow, row_regions, source_flow_element=None)
            row.source = source
            row.region_type = "table_row"
            row.metadata.update({"row_index": row_index, "is_multi_page": True})
            stitched_rows.append(row)

        return stitched_rows

    @staticmethod
    def _has_boundary(coords: Sequence[float], boundary: float) -> bool:
        return any(abs(coord - boundary) < _BOUNDARY_TOLERANCE for coord in coords)

    def _ensure_region_entries(self) -> None:
        flow_guides = getattr(self._guides, "_flow_guides", None)
        if flow_guides is None:
            self._guides._flow_guides = {}
            flow_guides = self._guides._flow_guides
        for region in self._regions:
            flow_guides.setdefault(region, ([], []))

    def _axis_metadata(self, axis: str):
        if axis == "vertical":
            return ("_unified_vertical", "_vertical_cache", self._guides.vertical, 0, 2)
        return ("_unified_horizontal", "_horizontal_cache", self._guides.horizontal, 1, 3)
