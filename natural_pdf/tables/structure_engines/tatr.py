"""Structure detection engine that consumes existing TATR annotations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from natural_pdf.tables.structure_provider import StructureDetectionResult, StructureEngine


class TATRStructureEngine(StructureEngine):
    def detect(
        self,
        *,
        context: Any,
        region: Any,
        options: Optional[Dict[str, Any]] = None,
    ) -> StructureDetectionResult:
        rows = self._filter_elements(region, "region[type=table-row][model=tatr]")
        columns = self._filter_elements(region, "region[type=table-column][model=tatr]")
        headers = self._filter_elements(
            region,
            "region[type=table-column-header][model=tatr]",
        )
        cells = self._filter_elements(region, "region[type=table_cell][model=tatr]")

        capabilities: set[str] = set()
        if rows:
            capabilities.add("rows")
        if columns:
            capabilities.add("columns")
        if headers:
            capabilities.add("headers")
        if cells:
            capabilities.add("cells")

        return StructureDetectionResult(
            capabilities=capabilities,
            rows=rows,
            columns=columns,
            headers=headers,
            cells=cells,
        )

    def _filter_elements(self, region, selector: str) -> List[Any]:
        try:
            candidates = region.page.find_all(selector, apply_exclusions=False)
        except Exception:
            return []
        region_bounds = getattr(region, "bbox", None)
        if not region_bounds:
            return list(candidates)
        x0, top, x1, bottom = region_bounds

        def _is_in_table(element) -> bool:
            elem_center_x = (getattr(element, "x0", 0) + getattr(element, "x1", 0)) / 2
            elem_center_y = (getattr(element, "top", 0) + getattr(element, "bottom", 0)) / 2
            return x0 <= elem_center_x <= x1 and top <= elem_center_y <= bottom

        filtered: List[Any] = []
        for elem in candidates:
            if _is_in_table(elem):
                filtered.append(elem)
        return filtered
