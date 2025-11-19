from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

from natural_pdf.services.registry import register_delegate
from natural_pdf.tables import TableResult
from natural_pdf.tables.structure_provider import (
    resolve_structure_engine_name as resolve_table_structure_engine_name,
)
from natural_pdf.tables.structure_provider import run_table_structure_engine
from natural_pdf.tables.table_provider import (
    normalize_table_settings,
    resolve_table_engine_name,
    run_table_engine,
)
from natural_pdf.tables.utils import build_table_from_cells, select_primary_table

# Type aliases for flow table extraction
ContentFilter = Optional[Union[str, Sequence[str], Callable[[str], bool]]]
StitchPredicate = Optional[Callable[[List[Optional[str]], List[Optional[str]], int, Any], bool]]

logger = logging.getLogger(__name__)


class TableService:
    """Service that powers Region.extract_table/extract_tables."""

    def __init__(self, context):
        self._context = context

    @register_delegate("table", "extract_table")
    def extract_table(
        self,
        host,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[[Any], Optional[str]]] = None,
        show_progress: bool = False,
        content_filter=None,
        apply_exclusions: bool = True,
        verticals: Optional[List[float]] = None,
        horizontals: Optional[List[float]] = None,
        structure_engine: Optional[str] = None,
        # Flow-specific arguments
        stitch_rows: StitchPredicate = None,
        merge_headers: Optional[bool] = None,
    ) -> TableResult:
        # Check if host is a FlowRegion (has constituent_regions)
        if hasattr(host, "constituent_regions") and host.constituent_regions:
            return self.extract_flow_table(
                host,
                method=method,
                table_settings=table_settings,
                use_ocr=use_ocr,
                ocr_config=ocr_config,
                text_options=text_options,
                cell_extraction_func=cell_extraction_func,
                show_progress=show_progress,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
                verticals=verticals,
                horizontals=horizontals,
                structure_engine=structure_engine,
                stitch_rows=stitch_rows,
                merge_headers=merge_headers,
            )

        table_settings = table_settings.copy() if table_settings else {}
        text_options = text_options.copy() if text_options else {}

        if verticals is not None:
            table_settings["vertical_strategy"] = "explicit"
            table_settings["explicit_vertical_lines"] = verticals
        if horizontals is not None:
            table_settings["horizontal_strategy"] = "explicit"
            table_settings["explicit_horizontal_lines"] = horizontals

        effective_method = method
        if effective_method is None:
            host_model = getattr(host, "model", None)
            host_region_type = getattr(host, "region_type", None)
            if host_model == "tatr" and host_region_type == "table":
                effective_method = "tatr"
            else:
                logger.debug(
                    "%s: Auto-detecting table extraction method...", getattr(host, "bbox", None)
                )
                try:
                    intersects = cast(
                        Optional[Callable[[Any], bool]], getattr(host, "intersects", None)
                    )
                    cell_regions_in_table = [
                        c
                        for c in host.page.find_all(
                            "region[type=table_cell]", apply_exclusions=False
                        )
                        if intersects and intersects(c)
                    ]
                except Exception:
                    cell_regions_in_table = []

                if cell_regions_in_table:
                    logger.debug(
                        "%s: Found %d table_cell regions – using 'cells' method.",
                        getattr(host, "bbox", None),
                        len(cell_regions_in_table),
                    )
                    return TableResult(
                        build_table_from_cells(
                            cell_regions_in_table,
                            content_filter=content_filter,
                            apply_exclusions=apply_exclusions,
                        )
                    )

                structure_table = self._extract_table_from_structure(
                    host=host,
                    structure_engine=structure_engine,
                    content_filter=content_filter,
                    apply_exclusions=apply_exclusions,
                    strict=structure_engine is not None,
                )
                if structure_table is not None:
                    return structure_table

        effective_method = effective_method or None

        if effective_method == "stream":
            table_settings.setdefault("vertical_strategy", "text")
            table_settings.setdefault("horizontal_strategy", "text")
        elif effective_method == "lattice":
            table_settings.setdefault("vertical_strategy", "lines")
            table_settings.setdefault("horizontal_strategy", "lines")

        logger.debug(
            "%s: Extracting table using method '%s'",
            getattr(host, "bbox", None),
            effective_method or "auto",
        )

        provider_managed_methods = {None, "pdfplumber", "stream", "lattice", "tatr", "text"}
        if effective_method not in provider_managed_methods:
            raise ValueError(
                f"Unknown table extraction method: '{method}'. "
                "Choose from 'tatr', 'pdfplumber', 'text', 'stream', 'lattice'."
            )

        normalized_settings = normalize_table_settings(table_settings)
        engine_name = resolve_table_engine_name(
            context=host,
            requested=effective_method,
            scope="region",
        )
        provider_tables = run_table_engine(
            context=host,
            region=host,
            engine_name=engine_name,
            table_settings=normalized_settings,
            use_ocr=use_ocr,
            ocr_config=ocr_config,
            text_options=text_options,
            cell_extraction_func=cell_extraction_func,
            show_progress=show_progress,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
        )
        table_rows = select_primary_table(provider_tables)
        return TableResult(table_rows)

    @register_delegate("table", "extract_tables")
    def extract_tables(
        self,
        host,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
    ) -> List[List[List[Optional[str]]]]:
        # Check if host is a FlowRegion
        if hasattr(host, "constituent_regions") and host.constituent_regions:
            return self.extract_flow_tables(
                host,
                method=method,
                table_settings=table_settings,
            )

        normalized_settings = normalize_table_settings(table_settings)
        engine_name = resolve_table_engine_name(
            context=host,
            requested=method,
            scope="region",
        )
        return run_table_engine(
            context=host,
            region=host,
            engine_name=engine_name,
            table_settings=normalized_settings,
        )

    def _extract_table_from_structure(
        self,
        host,
        *,
        structure_engine: Optional[str],
        content_filter=None,
        apply_exclusions: bool = True,
        strict: bool = False,
    ) -> Optional[TableResult]:
        engine_name = resolve_table_structure_engine_name(
            host,
            structure_engine,
            scope="region",
        )
        if not engine_name:
            if strict and structure_engine:
                raise ValueError(
                    f"Structure engine '{structure_engine}' could not be resolved for region {getattr(host, 'bbox', None)}"
                )
            return None

        try:
            result = run_table_structure_engine(
                context=host,
                region=host,
                engine_name=engine_name,
                options={"apply_exclusions": apply_exclusions},
            )
        except Exception as exc:
            logger.debug(
                "Region %s: Structure engine '%s' failed",
                getattr(host, "bbox", None),
                engine_name,
            )
            if strict:
                raise RuntimeError(
                    f"Structure engine '{engine_name}' failed for region {getattr(host, 'bbox', None)}"
                ) from exc
            return None

        if not result:
            if strict:
                raise ValueError(
                    f"Structure engine '{engine_name}' returned no structure for region {getattr(host, 'bbox', None)}"
                )
            return None

        if "cells" in result.capabilities and result.cells:
            table_data = build_table_from_cells(
                list(result.cells),
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
            )
            return TableResult(table_data)

        if strict:
            raise ValueError(
                f"Structure engine '{engine_name}' did not provide table cells for region {getattr(host, 'bbox', None)}"
            )
        return None

    def extract_flow_table(
        self,
        host,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[[Any], Optional[str]]] = None,
        show_progress: bool = False,
        content_filter: ContentFilter = None,
        apply_exclusions: bool = True,
        verticals: Optional[Sequence[float]] = None,
        horizontals: Optional[Sequence[float]] = None,
        stitch_rows: StitchPredicate = None,
        merge_headers: Optional[bool] = None,
        structure_engine: Optional[str] = None,
        **kwargs,
    ) -> TableResult:
        """Aggregate table extraction across FlowRegion constituents, preserving semantics."""
        import warnings
        from itertools import zip_longest

        if table_settings is None:
            table_settings = {}
        if text_options is None:
            text_options = {}

        if not host.constituent_regions:
            return TableResult([])

        predicate: StitchPredicate = stitch_rows if callable(stitch_rows) else None

        def _default_merge(
            prev_row: List[Optional[str]], cur_row: List[Optional[str]]
        ) -> List[Optional[str]]:
            merged: List[Optional[str]] = []
            for p, c in zip_longest(prev_row, cur_row, fillvalue=""):
                if (p or "").strip() and (c or "").strip():
                    merged.append(f"{p} {c}".strip())
                else:
                    merged.append((p or "") + (c or ""))
            return merged

        aggregated_rows: List[List[Optional[str]]] = []
        header_row: Optional[List[Optional[str]]] = None
        auto_warning_pending = False
        explicit_warning_pending = False
        auto_repeat_states: List[bool] = []

        def _detect_repeated_header(rows: List[List[Optional[str]]]) -> bool:
            if not rows:
                return False
            first_row = rows[0]
            if header_row is None:
                return False
            if len(first_row) != len(header_row):
                return False
            return all(
                (cell or "").strip() == (header_cell or "").strip()
                for cell, header_cell in zip(first_row, header_row)
            )

        for idx, region in enumerate(host.constituent_regions):
            settings_copy = dict(table_settings)
            text_copy = dict(text_options)
            # Recursive call to extract_table for each constituent region
            # This handles standard regions via the standard path
            table_result = region.extract_table(
                method=method,
                table_settings=settings_copy,
                use_ocr=use_ocr,
                ocr_config=ocr_config,
                text_options=text_copy,
                cell_extraction_func=cell_extraction_func,
                show_progress=show_progress,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
                verticals=verticals,
                horizontals=horizontals,
                structure_engine=structure_engine,
                **kwargs,
            )
            rows = list(table_result)
            if not rows:
                continue

            if merge_headers is None:
                if idx == 0:
                    header_row = list(rows[0])
                elif header_row is not None:
                    repeated = _detect_repeated_header(rows)
                    auto_repeat_states.append(repeated)
                    if repeated:
                        auto_warning_pending = True
                        rows = rows[1:]
                    if True in auto_repeat_states and False in auto_repeat_states:
                        raise ValueError("Inconsistent header pattern detected across segments.")
            elif merge_headers:
                if idx == 0:
                    header_row = list(rows[0])
                else:
                    explicit_warning_pending = True
                    rows = rows[1:]

            if predicate is not None and aggregated_rows:
                prev_row = aggregated_rows[-1]
                merged = predicate(prev_row, rows[0], idx, region)
                if merged:
                    aggregated_rows[-1] = _default_merge(prev_row, rows[0])
                    rows = rows[1:]

            aggregated_rows.extend(rows)

        if auto_warning_pending:
            warnings.warn(
                "Detected repeated headers across FlowRegion segments; removing duplicates.",
                UserWarning,
                stacklevel=2,
            )
        if explicit_warning_pending:
            warnings.warn(
                "Removing repeated headers across FlowRegion segments.",
                UserWarning,
                stacklevel=2,
            )

        return TableResult(aggregated_rows)

    def extract_flow_tables(
        self,
        host,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        **kwargs,
    ) -> List[List[List[Optional[str]]]]:
        if table_settings is None:
            table_settings = {}
        if not host.constituent_regions:
            return []
        result: List[List[List[Optional[str]]]] = []
        for region in host.constituent_regions:
            tables = region.extract_tables(
                method=method,
                table_settings=table_settings.copy(),
                **kwargs,
            )
            if tables:
                result.extend(tables)
        return result
