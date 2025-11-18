"""Shared table-extraction wrappers that delegate to the table service."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from natural_pdf.services.base import resolve_service
from natural_pdf.tables import TableResult

ContentFilter = Optional[Union[str, Sequence[str], Callable[[str], bool]]]


def extract_table(
    self,
    method: Optional[str] = None,
    table_settings: Optional[dict] = None,
    use_ocr: bool = False,
    ocr_config: Optional[dict] = None,
    text_options: Optional[Dict[str, Any]] = None,
    cell_extraction_func: Optional[Callable[[Any], Optional[str]]] = None,
    show_progress: bool = False,
    content_filter: ContentFilter = None,
    apply_exclusions: bool = True,
    verticals: Optional[Sequence[float]] = None,
    horizontals: Optional[Sequence[float]] = None,
    structure_engine: Optional[str] = None,
) -> TableResult:
    """Call the table service with the canonical extract_table signature."""

    service = resolve_service(self, "table")
    return service.extract_table(
        self,
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
    )


def extract_tables(
    self,
    method: Optional[str] = None,
    table_settings: Optional[dict] = None,
) -> List[List[List[Optional[str]]]]:
    """Call the table service to extract every table for the host."""

    service = resolve_service(self, "table")
    return service.extract_tables(
        self,
        method=method,
        table_settings=table_settings,
    )


__all__ = ["extract_table", "extract_tables"]
