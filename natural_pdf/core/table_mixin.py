from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

from natural_pdf.core.interfaces import SupportsGeometry
from natural_pdf.services.base import resolve_service
from natural_pdf.tables import TableResult

if TYPE_CHECKING:
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region


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
        return resolve_service(self, "table").extract_table(
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
        return resolve_service(self, "table").extract_tables(
            self,
            method=method,
            table_settings=table_settings,
        )
