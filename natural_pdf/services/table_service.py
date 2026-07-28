from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

from natural_pdf.tables import TableResult
from natural_pdf.tables.structure_provider import (
    resolve_structure_engine_name as resolve_table_structure_engine_name,
)
from natural_pdf.tables.structure_provider import (
    run_table_structure_engine,
)
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

    # Methods whose region-level extraction may resolve to ruling-based
    # (pdfplumber lattice) detection, which can silently drop the first row of
    # a flow segment: at a seam the ruling above that row was drawn in the
    # *previous* segment (renderers do not redraw table borders after a
    # column/page break), so lattice starts at the next ruling instead.
    _SEAM_RECOVERY_METHODS = frozenset(
        {None, "auto", "default", "pdfplumber", "pdfplumber_auto", "lattice"}
    )

    def __init__(self, context):
        self._context = context

    @staticmethod
    def _is_attributable_cell(host: Any, cell: Any) -> bool:
        host_source = getattr(host, "source", None)
        cell_source = getattr(cell, "source", None)
        if host_source and cell_source:
            return host_source == cell_source

        host_model = getattr(host, "model", None)
        cell_model = getattr(cell, "model", None)
        if host_model and cell_model:
            return host_model == cell_model

        return False

    def extract_table(
        self,
        host,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[[Any], Optional[str]]] = None,
        cell_extract: Literal["text", "words"] = "text",
        cell_overlap: Literal["center", "full", "partial"] = "center",
        cell_newlines: Union[bool, str] = True,
        show_progress: bool = False,
        content_filter=None,
        apply_exclusions: bool = True,
        verticals: Optional[Union[List[float], Sequence[Any]]] = None,
        horizontals: Optional[Union[List[float], Sequence[Any]]] = None,
        outer: bool = False,
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
                cell_extract=cell_extract,
                cell_overlap=cell_overlap,
                cell_newlines=cell_newlines,
                show_progress=show_progress,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
                verticals=verticals,
                horizontals=horizontals,
                outer=outer,
                structure_engine=structure_engine,
                stitch_rows=stitch_rows,
                merge_headers=merge_headers,
            )

        table_settings = table_settings.copy() if table_settings else {}
        text_options = text_options.copy() if text_options else {}

        # Convert verticals/horizontals from elements to float positions if needed
        verticals_floats = (
            self._resolve_guide_positions(verticals, "x0") if verticals is not None else None
        )
        horizontals_floats = (
            self._resolve_guide_positions(horizontals, "top") if horizontals is not None else None
        )

        # Handle outer=True - add region boundaries
        if outer:
            if verticals_floats is not None:
                verticals_floats = self._add_outer_boundaries(
                    host, verticals_floats, axis="vertical", apply_exclusions=apply_exclusions
                )
            if horizontals_floats is not None:
                horizontals_floats = self._add_outer_boundaries(
                    host, horizontals_floats, axis="horizontal", apply_exclusions=apply_exclusions
                )

        if verticals_floats is not None:
            table_settings["vertical_strategy"] = "explicit"
            table_settings["explicit_vertical_lines"] = verticals_floats
        if horizontals_floats is not None:
            table_settings["horizontal_strategy"] = "explicit"
            table_settings["explicit_horizontal_lines"] = horizontals_floats
        has_partial_explicit_guides = (verticals_floats is not None) != (
            horizontals_floats is not None
        )

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
                    candidate_cells = [
                        c
                        for c in host.page.find_all(
                            "region[type=table_cell]", apply_exclusions=False
                        )
                        if intersects and intersects(c)
                    ]
                except Exception:
                    candidate_cells = []

                cell_regions_in_table = [
                    cell for cell in candidate_cells if self._is_attributable_cell(host, cell)
                ]

                if cell_regions_in_table:
                    logger.debug(
                        "%s: Found %d table_cell regions – using 'cells' method.",
                        getattr(host, "bbox", None),
                        len(cell_regions_in_table),
                    )
                    return TableResult(
                        build_table_from_cells(
                            cell_regions_in_table,
                            table_region=host,
                            cell_extraction_func=cell_extraction_func,
                            use_ocr=use_ocr,
                            ocr_config=ocr_config,
                            content_filter=content_filter,
                            apply_exclusions=apply_exclusions,
                            cell_extract=cell_extract,
                            cell_overlap=cell_overlap,
                            cell_newlines=cell_newlines,
                        )
                    )

                structure_table = self._extract_table_from_structure(
                    host=host,
                    structure_engine=structure_engine,
                    cell_extraction_func=cell_extraction_func,
                    use_ocr=use_ocr,
                    ocr_config=ocr_config,
                    cell_extract=cell_extract,
                    cell_overlap=cell_overlap,
                    cell_newlines=cell_newlines,
                    content_filter=content_filter,
                    apply_exclusions=apply_exclusions,
                    strict=structure_engine is not None,
                )
                if structure_table is not None:
                    return structure_table

                if has_partial_explicit_guides:
                    effective_method = "auto"

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

        provider_managed_methods = {
            None,
            "auto",
            "default",
            "pdfplumber_auto",
            "pdfplumber",
            "stream",
            "lattice",
            "tatr",
            "text",
        }
        if effective_method not in provider_managed_methods:
            raise ValueError(
                f"Unknown table extraction method: '{method}'. "
                "Choose from 'auto', 'tatr', 'pdfplumber', 'text', 'stream', 'lattice'."
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
            cell_extract=cell_extract,
            cell_overlap=cell_overlap,
            cell_newlines=cell_newlines,
            show_progress=show_progress,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
        )
        table_rows = select_primary_table(provider_tables)
        return TableResult(table_rows)

    def extract_tables(
        self,
        host,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
    ) -> List[TableResult]:
        logger.warning(
            "extract_tables() extracts all tables at once without targeting. "
            "Consider using page.find('region[type=table]').extract_table() "
            "to isolate and extract a specific table."
        )

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
        raw_tables = run_table_engine(
            context=host,
            region=host,
            engine_name=engine_name,
            table_settings=normalized_settings,
        )
        return [TableResult(table) for table in raw_tables]

    def _extract_table_from_structure(
        self,
        host,
        *,
        structure_engine: Optional[str],
        cell_extraction_func: Optional[Callable[[Any], Optional[str]]] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        cell_extract: Literal["text", "words"] = "text",
        cell_overlap: Literal["center", "full", "partial"] = "center",
        cell_newlines: Union[bool, str] = True,
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
                table_region=host,
                cell_extraction_func=cell_extraction_func,
                use_ocr=use_ocr,
                ocr_config=ocr_config,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
                cell_extract=cell_extract,
                cell_overlap=cell_overlap,
                cell_newlines=cell_newlines,
            )
            return TableResult(table_data)

        if strict:
            raise ValueError(
                f"Structure engine '{engine_name}' did not provide table cells for region {getattr(host, 'bbox', None)}"
            )
        return None

    def _resolve_guide_positions(
        self,
        guides: Union[List[float], Sequence[Any]],
        attr: str,
    ) -> List[float]:
        """Convert guides to float positions.

        Args:
            guides: List of floats, or sequence of elements with positional attributes
            attr: Attribute to extract ('x0' for verticals, 'top' for horizontals)

        Returns:
            Sorted list of float positions
        """
        positions: List[float] = []
        for item in guides:
            if isinstance(item, (int, float)):
                positions.append(float(item))
            elif hasattr(item, attr):
                positions.append(float(getattr(item, attr)))
            else:
                raise TypeError(
                    f"Cannot convert {type(item).__name__} to guide position. "
                    f"Expected float or element with '{attr}' attribute."
                )
        return sorted(positions)

    def _add_outer_boundaries(
        self,
        host,
        positions: List[float],
        axis: str,
        apply_exclusions: bool = True,
    ) -> List[float]:
        """Add outer boundaries to guide positions based on content extent.

        Args:
            host: The region being processed
            positions: Sorted list of guide positions
            axis: 'vertical' or 'horizontal'
            apply_exclusions: Whether to respect exclusions when finding content bounds

        Returns:
            Positions with outer boundaries added
        """
        if not positions:
            return positions

        # Find content bounds
        text_elements = host.find_all("text", apply_exclusions=apply_exclusions)
        if not text_elements:
            # Fall back to region bounds
            if axis == "vertical":
                return [host.x0] + positions + [host.x1]
            else:
                return [host.top] + positions + [host.bottom]

        if axis == "vertical":
            # For vertical guides, we need x-axis bounds
            content_left = min(t.x0 for t in text_elements)
            content_right = max(t.x1 for t in text_elements)

            # Add left boundary if there's content before first guide
            left_boundary = min(content_left, positions[0])
            # Add right boundary after last guide to capture all content
            right_boundary = content_right + 1  # +1 for padding

            result = (
                [left_boundary] + [p for p in positions if p > left_boundary] + [right_boundary]
            )
        else:
            # For horizontal guides, we need y-axis bounds
            content_top = min(t.top for t in text_elements)
            content_bottom = max(t.bottom for t in text_elements)

            top_boundary = min(content_top, positions[0])
            bottom_boundary = content_bottom + 1

            result = [top_boundary] + [p for p in positions if p > top_boundary] + [bottom_boundary]

        return sorted(set(result))  # Remove duplicates and sort

    def extract_flow_table(
        self,
        host,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[[Any], Optional[str]]] = None,
        cell_extract: Literal["text", "words"] = "text",
        cell_overlap: Literal["center", "full", "partial"] = "center",
        cell_newlines: Union[bool, str] = True,
        show_progress: bool = False,
        content_filter: ContentFilter = None,
        apply_exclusions: bool = True,
        verticals: Optional[Sequence[float]] = None,
        horizontals: Optional[Sequence[float]] = None,
        outer: bool = False,
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
                cell_extract=cell_extract,
                cell_overlap=cell_overlap,
                cell_newlines=cell_newlines,
                show_progress=show_progress,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
                verticals=verticals,
                horizontals=horizontals,
                outer=outer,
                structure_engine=structure_engine,
                **kwargs,
            )
            rows = list(table_result)
            if not rows:
                continue

            if (
                idx > 0
                and method in self._SEAM_RECOVERY_METHODS
                and horizontals is None
                and table_settings.get("horizontal_strategy") != "text"
            ):
                seam_evidence = self._find_dropped_seam_top(
                    region, rows, apply_exclusions=apply_exclusions
                )
                if seam_evidence is not None:
                    recovered = self._retry_segment_with_seam_line(
                        region,
                        seam_evidence,
                        rows,
                        method=method,
                        table_settings=table_settings,
                        use_ocr=use_ocr,
                        ocr_config=ocr_config,
                        text_options=text_options,
                        cell_extraction_func=cell_extraction_func,
                        cell_extract=cell_extract,
                        cell_overlap=cell_overlap,
                        cell_newlines=cell_newlines,
                        show_progress=show_progress,
                        content_filter=content_filter,
                        apply_exclusions=apply_exclusions,
                        verticals=verticals,
                        outer=outer,
                        structure_engine=structure_engine,
                        **kwargs,
                    )
                    if recovered is not None:
                        rows = recovered

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

    @staticmethod
    def _cell_tokens(cell: Optional[str]) -> Tuple[str, ...]:
        """Normalize a cell to exact whitespace-delimited text tokens."""

        if cell is None:
            return ()
        return tuple(str(cell).split())

    @classmethod
    def _row_tokens(cls, row: Sequence[Optional[str]]) -> Tuple[str, ...]:
        return tuple(token for cell in row for token in cls._cell_tokens(cell))

    @staticmethod
    def _cell_shape(cell: Optional[str]) -> Tuple[bool, bool]:
        """Return ``(is_blank, is_numeric_or_date_like)`` for one cell."""

        if cell is None or not str(cell).strip():
            return True, False
        compact = "".join(str(cell).split())
        numeric_punctuation = set("0123456789+-.,$%()/:\u2212")
        numeric_or_date_like = any(character.isdigit() for character in compact) and all(
            character in numeric_punctuation for character in compact
        )
        return False, numeric_or_date_like

    @classmethod
    def _find_dropped_seam_top(
        cls, region, rows, *, apply_exclusions: bool
    ) -> Optional[Tuple[float, Tuple[str, ...]]]:
        """Return evidence for a first text line omitted at a flow seam.

        This is the signature of a seam drop: the segment's text layer starts
        with a row whose cells never made it into the extracted table because
        the ruling line above it lives in the previous flow segment.

        Detection is deliberately conservative.  The next text line must
        exactly match the first extracted row (token boundaries included), and
        the candidate-to-row spacing must match subsequent table row spacing.
        This prevents a title or page header somewhere above a table from
        becoming a synthetic row merely because its text is absent from the
        extraction.
        """
        try:
            words = [
                w
                for w in region.find_all("text", apply_exclusions=apply_exclusions)
                if (getattr(w, "text", "") or "").strip()
            ]
            if not words:
                return None
            words.sort(key=lambda word: (float(word.top), float(getattr(word, "x0", 0.0))))

            lines: List[Tuple[float, List[Any]]] = []
            for word in words:
                word_top = float(word.top)
                if lines and abs(word_top - lines[-1][0]) <= 2.0:
                    lines[-1][1].append(word)
                else:
                    lines.append((word_top, [word]))
        except Exception:
            # Regions without text elements/geometry (mocks, exotic hosts):
            # never attempt recovery.
            return None

        if len(lines) < 3 or len(rows) < 2:
            return None

        def line_tokens(line_words: Sequence[Any]) -> Tuple[str, ...]:
            return tuple(
                token for word in line_words for token in str(getattr(word, "text", "")).split()
            )

        candidate_top, candidate_words = lines[0]
        candidate_tokens = line_tokens(candidate_words)
        first_row_tokens = cls._row_tokens(rows[0])
        if not candidate_tokens or candidate_tokens == first_row_tokens:
            return None

        # The first extracted row must be the *immediately following* text
        # line.  Exact tuples avoid the old substring bug ("1" in "10").
        first_row_top, first_row_words = lines[1]
        if line_tokens(first_row_words) != first_row_tokens:
            return None

        # Establish the normal row pitch using consecutive extracted rows.
        # If the text layout cannot prove that cadence, leave extraction alone.
        matched_tops = [first_row_top]
        line_index = 2
        for row in rows[1:6]:
            if line_index >= len(lines):
                break
            line_top, line_words = lines[line_index]
            if line_tokens(line_words) != cls._row_tokens(row):
                break
            matched_tops.append(line_top)
            line_index += 1
        if len(matched_tops) < 2:
            return None

        normal_gaps = [
            following - preceding
            for preceding, following in zip(matched_tops, matched_tops[1:])
            if following > preceding
        ]
        if not normal_gaps:
            return None
        ordered_gaps = sorted(normal_gaps)
        normal_gap = ordered_gaps[len(ordered_gaps) // 2]
        candidate_gap = first_row_top - candidate_top
        if candidate_gap < normal_gap * 0.5 or candidate_gap > normal_gap * 1.5:
            return None

        return candidate_top, candidate_tokens

    def _retry_segment_with_seam_line(
        self,
        region,
        seam_evidence: Tuple[float, Tuple[str, ...]],
        rows: List[List[Optional[str]]],
        *,
        method: Optional[str],
        table_settings: dict,
        **extract_kwargs,
    ) -> Optional[List[List[Optional[str]]]]:
        """Re-extract a flow segment with an explicit horizontal line injected
        just above its first text line, closing the top of the seam row so
        ruling-based detection can recover it.

        Returns the recovered rows, or None when the retry did not improve on
        the original extraction (in which case the caller keeps ``rows``).
        """
        seam_top, candidate_tokens = seam_evidence
        retry_settings = dict(table_settings)
        explicit = list(retry_settings.get("explicit_horizontal_lines") or [])
        seam_line = seam_top - 0.5
        region_top = getattr(region, "top", None)
        if region_top is not None and seam_line < region_top:
            seam_line = region_top
        explicit.append(seam_line)
        retry_settings["explicit_horizontal_lines"] = explicit
        try:
            retry_rows = list(
                region.extract_table(
                    method=method,
                    table_settings=retry_settings,
                    **extract_kwargs,
                )
            )
        except Exception:
            logger.debug(
                "Seam-row recovery retry failed for region %s",
                getattr(region, "bbox", None),
                exc_info=True,
            )
            return None
        if not retry_rows or len(retry_rows) != len(rows) + 1:
            return None

        expected_width = len(rows[0])
        if expected_width == 0 or any(len(row) != expected_width for row in rows + retry_rows):
            return None

        # The retry may add exactly the candidate row, but it must not alter,
        # merge, split, or reorder anything the original extraction returned.
        if retry_rows[1:] != rows:
            return None

        candidate_row = retry_rows[0]
        if self._row_tokens(candidate_row) != candidate_tokens:
            return None

        # A seam row must have the same immediate per-column shape as the
        # first surviving row.  At least one numeric/date-like cell is required:
        # without it, an all-text heading is indistinguishable from all-text
        # table data and recovery cannot be fail-closed.
        candidate_shape = tuple(self._cell_shape(cell) for cell in candidate_row)
        immediate_row_shape = tuple(self._cell_shape(cell) for cell in rows[0])
        if candidate_shape != immediate_row_shape:
            return None
        if not any(is_numeric_or_date for _is_blank, is_numeric_or_date in candidate_shape):
            return None

        return retry_rows

    def extract_flow_tables(
        self,
        host,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        **kwargs,
    ) -> List[TableResult]:
        if table_settings is None:
            table_settings = {}
        if not host.constituent_regions:
            return []
        result: List[TableResult] = []
        for region in host.constituent_regions:
            tables = region.extract_tables(
                method=method,
                table_settings=table_settings.copy(),
                **kwargs,
            )
            if tables:
                result.extend(tables)
        return result
