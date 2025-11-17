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
    ) -> TableResult:
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
