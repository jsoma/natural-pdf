"""Flow-aware table extraction helpers shared by Flow and FlowRegion variants."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from natural_pdf.tables import TableResult

logger = logging.getLogger(__name__)


ContentFilter = Optional[Union[str, Sequence[str], Callable[[str], bool]]]
StitchPredicate = Optional[Callable[[List[Optional[str]], List[Optional[str]], int, Any], bool]]


def flowregion_extract_table(
    self,
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

    if table_settings is None:
        table_settings = {}
    if text_options is None:
        text_options = {}

    if not self.constituent_regions:
        return TableResult([])

    predicate: StitchPredicate = stitch_rows if callable(stitch_rows) else None

    def _default_merge(
        prev_row: List[Optional[str]], cur_row: List[Optional[str]]
    ) -> List[Optional[str]]:
        from itertools import zip_longest

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

    for idx, region in enumerate(self.constituent_regions):
        settings_copy = dict(table_settings)
        text_copy = dict(text_options)
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


def flowregion_extract_tables(
    self,
    method: Optional[str] = None,
    table_settings: Optional[dict] = None,
    **kwargs,
) -> List[List[List[Optional[str]]]]:
    if table_settings is None:
        table_settings = {}
    if not self.constituent_regions:
        return []
    result: List[List[List[Optional[str]]]] = []
    for region in self.constituent_regions:
        tables = region.extract_tables(
            method=method,
            table_settings=table_settings.copy(),
            **kwargs,
        )
        if tables:
            result.extend(tables)
    return result


def flow_extract_table(self, **kwargs) -> TableResult:
    """Flow.extract_table simply routes through the FlowRegion helper."""

    if not self.segments:
        return TableResult([])
    combined_region = self._analysis_region()
    return flowregion_extract_table(combined_region, **kwargs)


def flow_extract_tables(self, **kwargs) -> List[List[List[Optional[str]]]]:
    if not self.segments:
        return []
    tables: List[List[List[Optional[str]]]] = []
    for segment in self.segments:
        segment_kwargs = dict(kwargs)
        settings = segment_kwargs.get("table_settings")
        if settings is not None:
            segment_kwargs["table_settings"] = dict(settings)
        tables.extend(segment.extract_tables(**segment_kwargs) or [])
    return tables


def flow_collection_extract_table(self, **kwargs) -> List[TableResult]:
    results: List[TableResult] = []
    for fr in self._flow_regions:
        fr_kwargs = dict(kwargs)
        settings = fr_kwargs.get("table_settings")
        if settings is not None:
            fr_kwargs["table_settings"] = dict(settings)
        results.append(fr.extract_table(**fr_kwargs))
    return results


def flow_collection_extract_tables(self, **kwargs) -> List[List[List[Optional[str]]]]:
    tables: List[List[List[Optional[str]]]] = []
    for fr in self._flow_regions:
        fr_kwargs = dict(kwargs)
        settings = fr_kwargs.get("table_settings")
        if settings is not None:
            fr_kwargs["table_settings"] = dict(settings)
        tables.extend(fr.extract_tables(**fr_kwargs) or [])
    return tables


__all__ = [
    "flowregion_extract_table",
    "flowregion_extract_tables",
    "flow_extract_table",
    "flow_extract_tables",
    "flow_collection_extract_table",
    "flow_collection_extract_tables",
]
