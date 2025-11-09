from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union, cast

from natural_pdf.core.interfaces import SupportsGeometry
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

if TYPE_CHECKING:
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region

logger = logging.getLogger(__name__)


class TableExtractionMixin(SupportsGeometry):
    """Shared table extraction logic for Region-like classes."""

    def extract_table(
        self,
        method: Optional[str] = None,  # Make method optional
        table_settings: Optional[dict] = None,  # Use Optional
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,  # Use Optional
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[["Region"], Optional[str]]] = None,
        show_progress: bool = False,  # Controls progress bar for text method
        content_filter: Optional[
            Union[str, Callable[[str], bool], List[str]]
        ] = None,  # NEW: Content filtering
        apply_exclusions: bool = True,  # Whether to apply exclusion regions during extraction
        verticals: Optional[List] = None,  # Explicit vertical lines
        horizontals: Optional[List] = None,  # Explicit horizontal lines
        structure_engine: Optional[str] = None,
    ) -> TableResult:  # Return type allows Optional[str] for cells
        """
        Extract a table from this region.

        Args:
            method: Method to use: 'tatr', 'pdfplumber', 'text', 'stream', 'lattice', or None (auto-detect).
                    'stream' is an alias for 'pdfplumber' with text-based strategies (equivalent to
                    setting `vertical_strategy` and `horizontal_strategy` to 'text').
                    'lattice' is an alias for 'pdfplumber' with line-based strategies (equivalent to
                    setting `vertical_strategy` and `horizontal_strategy` to 'lines').
            table_settings: Settings for pdfplumber table extraction (used with 'pdfplumber', 'stream', or 'lattice' methods).
            use_ocr: Whether to use OCR for text extraction (currently only applicable with 'tatr' method).
            ocr_config: OCR configuration parameters.
            text_options: Dictionary of options for the 'text' method, corresponding to arguments
                          of analyze_text_table_structure (e.g., snap_tolerance, expand_bbox).
            cell_extraction_func: Optional callable function that takes a cell Region object
                                  and returns its string content. Overrides default text extraction
                                  for the 'text' method.
            show_progress: If True, display a progress bar during cell text extraction for the 'text' method.
            content_filter: Optional content filter to apply during cell text extraction. Can be:
                - A regex pattern string (characters matching the pattern are EXCLUDED)
                - A callable that takes text and returns True to KEEP the character
                - A list of regex patterns (characters matching ANY pattern are EXCLUDED)
                Works with all extraction methods by filtering cell content.
            apply_exclusions: Whether to apply exclusion regions during text extraction (default: True).
                When True, text within excluded regions (e.g., headers/footers) will not be extracted.
            verticals: Optional list of explicit vertical lines for table extraction. When provided,
                       automatically sets vertical_strategy='explicit' and explicit_vertical_lines.
            horizontals: Optional list of explicit horizontal lines for table extraction. When provided,
                         automatically sets horizontal_strategy='explicit' and explicit_horizontal_lines.
            structure_engine: Optional structure detection engine to use when the region already
                contains detected table cells/rows/columns (e.g., "tatr"). When explicitly
                provided, the engine must succeed; otherwise a ValueError is raised. When left
                as None, the mixin will attempt any auto-configured structure engine before
                falling back to the standard table engines.

        Returns:
            Table data as a list of rows, where each row is a list of cell values (str or None).
        """
        # Default settings if none provided (copy to avoid caller mutation)
        table_settings = table_settings.copy() if table_settings else {}
        text_options = text_options.copy() if text_options else {}

        # Handle explicit vertical and horizontal lines
        if verticals is not None:
            table_settings["vertical_strategy"] = "explicit"
            table_settings["explicit_vertical_lines"] = verticals
        if horizontals is not None:
            table_settings["horizontal_strategy"] = "explicit"
            table_settings["explicit_horizontal_lines"] = horizontals

        # Auto-detect method if not specified
        if method is None:
            # If this is a TATR-detected region, use TATR method
            host_model = getattr(self, "model", None)
            host_region_type = getattr(self, "region_type", None)
            if host_model == "tatr" and host_region_type == "table":
                effective_method = "tatr"
            else:
                logger.debug(f"Region {self.bbox}: Auto-detecting table extraction method...")

                try:
                    intersects = cast(
                        Optional[Callable[[Any], bool]], getattr(self, "intersects", None)
                    )
                    cell_regions_in_table = [
                        c
                        for c in self.page.find_all(
                            "region[type=table_cell]", apply_exclusions=False
                        )
                        if intersects and intersects(c)
                    ]
                except Exception:
                    cell_regions_in_table = []  # Fallback silently

                if cell_regions_in_table:
                    logger.debug(
                        f"Region {self.bbox}: Found {len(cell_regions_in_table)} pre-computed table_cell regions – using 'cells' method."
                    )
                    return TableResult(
                        build_table_from_cells(
                            cell_regions_in_table,
                            content_filter=content_filter,
                            apply_exclusions=apply_exclusions,
                        )
                    )

                structure_table = self._extract_table_from_structure(
                    structure_engine=structure_engine,
                    content_filter=content_filter,
                    apply_exclusions=apply_exclusions,
                    strict=structure_engine is not None,
                )
                if structure_table is not None:
                    return structure_table

                effective_method = None  # Let provider auto-engine decide
        else:
            effective_method = method

        # Handle method aliases for pdfplumber-style engines
        if effective_method == "stream":
            logger.debug("Using 'stream' method alias for 'pdfplumber' with text-based strategies.")
            # Set default text strategies if not already provided by the user
            table_settings.setdefault("vertical_strategy", "text")
            table_settings.setdefault("horizontal_strategy", "text")
        elif effective_method == "lattice":
            logger.debug(
                "Using 'lattice' method alias for 'pdfplumber' with line-based strategies."
            )
            # Set default line strategies if not already provided by the user
            table_settings.setdefault("vertical_strategy", "lines")
            table_settings.setdefault("horizontal_strategy", "lines")

        logger.debug(
            "Region %s: Extracting table using method '%s'",
            getattr(self, "bbox", None),
            effective_method or "auto",
        )

        provider_managed_methods = {None, "pdfplumber", "stream", "lattice", "tatr", "text"}

        if effective_method not in provider_managed_methods:
            raise ValueError(
                f"Unknown table extraction method: '{method}'. Choose from 'tatr', 'pdfplumber', 'text', 'stream', 'lattice'."
            )

        normalized_settings = normalize_table_settings(table_settings)
        engine_name = resolve_table_engine_name(
            context=self,
            requested=effective_method,
            scope="region",
        )
        provider_tables = run_table_engine(
            context=self,
            region=self,
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

    def extract_tables(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
    ) -> List[List[List[Optional[str]]]]:
        """
        Extract all tables from this region using pdfplumber-based methods.

        Note: Only 'pdfplumber', 'stream', and 'lattice' methods are supported for extract_tables.
        'tatr' and 'text' methods are designed for single table extraction only.

        Args:
            method: Method to use: 'pdfplumber', 'stream', 'lattice', or None (auto-detect).
                    'stream' uses text-based strategies, 'lattice' uses line-based strategies.
            table_settings: Settings for pdfplumber table extraction.

        Returns:
            List of tables, where each table is a list of rows, and each row is a list of cell values.
        """
        normalized_settings = normalize_table_settings(table_settings)
        engine_name = resolve_table_engine_name(
            context=self,
            requested=method,
            scope="region",
        )
        return run_table_engine(
            context=self,
            region=self,
            engine_name=engine_name,
            table_settings=normalized_settings,
        )

    def _extract_table_tatr(
        self, use_ocr=False, ocr_config=None, content_filter=None, apply_exclusions=True
    ) -> List[List[Optional[str]]]:
        from natural_pdf.tables.engines.tatr import TATRTableEngine

        engine = TATRTableEngine()
        tables = engine.extract_tables(
            context=self,
            region=self,
            use_ocr=use_ocr,
            ocr_config=ocr_config,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
        )
        return tables[0] if tables else []

    def _extract_table_from_structure(
        self,
        *,
        structure_engine: Optional[str],
        content_filter=None,
        apply_exclusions: bool = True,
        strict: bool = False,
    ) -> Optional[TableResult]:
        engine_name = resolve_table_structure_engine_name(
            self,
            structure_engine,
            scope="region",
        )
        if not engine_name:
            if strict and structure_engine:
                raise ValueError(
                    f"Structure engine '{structure_engine}' could not be resolved for region {getattr(self, 'bbox', None)}"
                )
            return None

        try:
            result = run_table_structure_engine(
                context=self,
                region=self,
                engine_name=engine_name,
                options={"apply_exclusions": apply_exclusions},
            )
        except Exception as exc:
            logger.debug(
                "Region %s: Structure engine '%s' failed",
                getattr(self, "bbox", None),
                engine_name,
            )
            if strict:
                raise RuntimeError(
                    f"Structure engine '{engine_name}' failed for region {getattr(self, 'bbox', None)}"
                ) from exc
            return None

        if not result:
            if strict:
                raise ValueError(
                    f"Structure engine '{engine_name}' returned no structure for region {getattr(self, 'bbox', None)}"
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
                f"Structure engine '{engine_name}' did not provide table cells for region {getattr(self, 'bbox', None)}"
            )
        return None

    def _extract_table_text(self, **text_options) -> List[List[Optional[str]]]:
        raise NotImplementedError("Text table extraction is now handled via provider engines")
