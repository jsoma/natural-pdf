"""TATR-backed table extraction engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from natural_pdf.tables.utils import extract_cell_text, merge_ocr_config


class TATRTableEngine:
    """Wraps TATR-derived structure extraction behind the provider interface."""

    def extract_tables(
        self,
        *,
        context: Any,
        region: Any,
        table_settings: Optional[Dict[str, Any]] = None,
        use_ocr: bool = False,
        ocr_config: Optional[Dict[str, Any]] = None,
        content_filter: Optional[Any] = None,
        apply_exclusions: bool = True,
        **_: Any,
    ) -> List[List[List[str]]]:
        table = self._extract_table(
            region,
            use_ocr=use_ocr,
            ocr_config=ocr_config,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
        )
        return [table]

    def _extract_table(
        self,
        region,
        *,
        use_ocr: bool,
        ocr_config: Optional[Dict[str, Any]],
        content_filter,
        apply_exclusions: bool,
    ) -> List[List[Optional[str]]]:
        rows = region.page.find_all("region[type=table-row][model=tatr]")
        headers = region.page.find_all("region[type=table-column-header][model=tatr]")
        columns = region.page.find_all("region[type=table-column][model=tatr]")

        def is_in_table(target):
            center_x = (target.x0 + target.x1) / 2
            center_y = (target.top + target.bottom) / 2
            return region.x0 <= center_x <= region.x1 and region.top <= center_y <= region.bottom

        rows = [row for row in rows if is_in_table(row)]
        headers = [header for header in headers if is_in_table(header)]
        columns = [column for column in columns if is_in_table(column)]

        rows.sort(key=lambda r: r.top)
        columns.sort(key=lambda c: c.x0)

        table_data: List[List[Optional[str]]] = []
        resolved_ocr_config: Optional[Dict[str, Any]] = (
            merge_ocr_config(ocr_config) if use_ocr else None
        )

        if headers:
            header_texts = []
            for header in headers:
                header_text = extract_cell_text(
                    header,
                    use_ocr=use_ocr,
                    ocr_config=resolved_ocr_config,
                    content_filter=content_filter,
                    apply_exclusions=apply_exclusions,
                )
                header_texts.append(header_text)
            table_data.append(header_texts)

        if columns:
            from natural_pdf.elements.region import Region  # avoid circular import

            for row in rows:
                row_cells: List[Optional[str]] = []
                for column in columns:
                    cell_bbox = (column.x0, row.top, column.x1, row.bottom)
                    cell_region = Region(region.page, cell_bbox)
                    cell_text = extract_cell_text(
                        cell_region,
                        use_ocr=use_ocr,
                        ocr_config=resolved_ocr_config,
                        content_filter=content_filter,
                        apply_exclusions=apply_exclusions,
                    )
                    row_cells.append(cell_text)
                table_data.append(row_cells)
        else:
            for row in rows:
                row_text = extract_cell_text(
                    row,
                    use_ocr=use_ocr,
                    ocr_config=resolved_ocr_config,
                    content_filter=content_filter,
                    apply_exclusions=apply_exclusions,
                )
                table_data.append([row_text])

        return table_data
