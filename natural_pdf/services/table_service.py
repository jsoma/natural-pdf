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
            # When outer=True and verticals provided but no horizontals,
            # add content-based horizontal boundaries to capture edge rows
            elif verticals_floats is not None and horizontals_floats is None:
                # Try to detect stripe boundaries (alternating row shading)
                stripe_boundaries = self._detect_stripe_boundaries(host)
                if stripe_boundaries:
                    # Use stripes with outer boundaries added
                    horizontals_floats = self._add_outer_boundaries(
                        host,
                        stripe_boundaries,
                        axis="horizontal",
                        apply_exclusions=apply_exclusions,
                    )
                else:
                    # No stripes - just add top/bottom content boundaries
                    horizontals_floats = self._get_content_boundaries(
                        host, axis="horizontal", apply_exclusions=apply_exclusions
                    )

        if verticals_floats is not None:
            table_settings["vertical_strategy"] = "explicit"
            table_settings["explicit_vertical_lines"] = verticals_floats
        if horizontals_floats is not None:
            table_settings["horizontal_strategy"] = "explicit"
            table_settings["explicit_horizontal_lines"] = horizontals_floats

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

    def _get_content_boundaries(
        self,
        host,
        axis: str,
        apply_exclusions: bool = True,
    ) -> List[float]:
        """Get just the outer content boundaries (no intermediate positions).

        This is used when outer=True but no explicit guides are provided
        and no stripes are detected. Returns just the min/max content bounds.

        Args:
            host: The region being processed
            axis: 'vertical' or 'horizontal'
            apply_exclusions: Whether to respect exclusions when finding content bounds

        Returns:
            List with just [min, max] content boundaries
        """
        text_elements = host.find_all("text", apply_exclusions=apply_exclusions)
        if not text_elements:
            # Fall back to region bounds
            if axis == "vertical":
                return [host.x0, host.x1]
            else:
                return [host.top, host.bottom]

        if axis == "vertical":
            content_left = min(t.x0 for t in text_elements)
            content_right = max(t.x1 for t in text_elements) + 1
            return [content_left, content_right]
        else:
            content_top = min(t.top for t in text_elements)
            content_bottom = max(t.bottom for t in text_elements) + 1
            return [content_top, content_bottom]

    def _detect_stripe_boundaries(self, host) -> List[float]:
        """Detect horizontal stripe boundaries from pdfplumber rectangles.

        Many PDF tables use alternating row shading (zebra stripes). This method
        extracts the top/bottom boundaries of those rectangles to use as
        horizontal guides for table extraction.

        Args:
            host: The region being processed

        Returns:
            Sorted list of y-positions from stripe rectangle boundaries,
            or empty list if no stripes detected
        """
        try:
            # Get the pdfplumber page object
            page = getattr(host, "page", host)
            plumber_page = getattr(page, "_page", None)
            if plumber_page is None:
                return []

            rects = plumber_page.rects
            if not rects:
                return []

            # Filter rectangles to those within the host region
            region_top = getattr(host, "top", 0)
            region_bottom = getattr(host, "bottom", float("inf"))

            # Collect boundaries from rectangles that look like row stripes
            # (rectangles that span horizontally and are within region bounds)
            boundaries = set()
            for r in rects:
                rect_top = r.get("top", 0)
                rect_bottom = r.get("bottom", 0)

                # Check if rectangle is within region bounds (with some tolerance)
                if rect_top >= region_top - 5 and rect_bottom <= region_bottom + 5:
                    # Check if it's a horizontal stripe (wider than tall)
                    rect_width = r.get("x1", 0) - r.get("x0", 0)
                    rect_height = rect_bottom - rect_top
                    if rect_width > rect_height * 2:  # At least 2x wider than tall
                        boundaries.add(rect_top)
                        boundaries.add(rect_bottom)

            return sorted(boundaries) if boundaries else []

        except Exception:
            # If anything goes wrong, just return empty list
            return []

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
