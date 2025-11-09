"""Text-alignment table extraction engine."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm


class TextTablesEngine:
    """Runs the text-alignment table extractor via the provider."""

    def extract_tables(
        self,
        *,
        context: Any,
        region: Any,
        table_settings: Optional[Dict[str, Any]] = None,
        text_options: Optional[Dict[str, Any]] = None,
        cell_extraction_func: Optional[Any] = None,
        show_progress: bool = False,
        content_filter: Optional[Any] = None,
        apply_exclusions: bool = True,
        **_: Any,
    ) -> List[List[List[Optional[str]]]]:
        table = self._extract_table(
            region,
            text_options=text_options or {},
            cell_extraction_func=cell_extraction_func,
            show_progress=show_progress,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
        )
        return [table]

    def _extract_table(
        self,
        region,
        *,
        text_options: Dict[str, Any],
        cell_extraction_func: Optional[Any],
        show_progress: bool,
        content_filter,
        apply_exclusions: bool,
    ) -> List[List[Optional[str]]]:
        if "text_table_structure" in region.analyses:
            analysis_results = region.analyses["text_table_structure"]
        else:
            analysis_results = region.analyze_text_table_structure(**text_options)

        if analysis_results is None or not analysis_results.get("cells"):
            return []

        cell_dicts = analysis_results["cells"]
        if not cell_dicts:
            return []

        coord_tolerance = text_options.get("coordinate_grouping_tolerance", 1)
        tops = sorted(
            list(set(round(c["top"] / coord_tolerance) * coord_tolerance for c in cell_dicts))
        )
        lefts = sorted(
            list(set(round(c["left"] / coord_tolerance) * coord_tolerance for c in cell_dicts))
        )

        unique_tops = self._cluster_coords(tops, coord_tolerance)
        unique_lefts = self._cluster_coords(lefts, coord_tolerance)

        cell_iterator: Iterable[Dict[str, float]]
        cell_iterator = (
            tqdm(
                cell_dicts,
                desc=f"Extracting text from {len(cell_dicts)} cells (text method)",
                unit="cell",
                leave=False,
            )
            if show_progress
            else cell_dicts
        )

        cell_text_map: Dict[Tuple[int, int], Optional[str]] = {}
        for cell_data in cell_iterator:
            try:
                cell_region = region.page.region(**cell_data)
                cell_value = None
                if callable(cell_extraction_func):
                    try:
                        cell_value = cell_extraction_func(cell_region)
                        if not isinstance(cell_value, (str, type(None))):
                            cell_value = None
                    except Exception:
                        cell_value = None
                else:
                    cell_value = cell_region.extract_text(
                        layout=False,
                        apply_exclusions=apply_exclusions,
                        content_filter=content_filter,
                    ).strip()

                rounded_top = round(cell_data["top"] / coord_tolerance) * coord_tolerance
                rounded_left = round(cell_data["left"] / coord_tolerance) * coord_tolerance
                cell_text_map[(rounded_top, rounded_left)] = cell_value
            except Exception:
                continue

        final_table: List[List[Optional[str]]] = []
        for row_top in unique_tops:
            row_data = []
            for col_left in unique_lefts:
                best_match_key: Optional[Tuple[int, int]] = None
                min_dist_sq = float("inf")
                for map_top, map_left in cell_text_map.keys():
                    if (
                        abs(map_top - row_top) <= coord_tolerance
                        and abs(map_left - col_left) <= coord_tolerance
                    ):
                        dist_sq = (map_top - row_top) ** 2 + (map_left - col_left) ** 2
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            best_match_key = (map_top, map_left)
                cell_value = cell_text_map.get(best_match_key) if best_match_key else None
                row_data.append(cell_value)
            final_table.append(row_data)

        return final_table

    @staticmethod
    def _cluster_coords(coords: Sequence[float], tolerance: float) -> List[int]:
        if not coords:
            return []
        clustered: List[int] = []
        current_cluster = [coords[0]]
        for coord in coords[1:]:
            if abs(coord - current_cluster[-1]) <= tolerance:
                current_cluster.append(coord)
            else:
                clustered.append(int(min(current_cluster)))
                current_cluster = [coord]
        clustered.append(int(min(current_cluster)))
        return clustered
