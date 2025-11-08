from __future__ import annotations

import logging
from collections.abc import Mapping as MappingABC
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from tqdm.auto import tqdm

from natural_pdf.tables import TableResult
from natural_pdf.tables.table_provider import (
    normalize_table_settings,
    resolve_table_engine_name,
    run_table_engine,
)

logger = logging.getLogger(__name__)


class TableExtractionMixin:
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

        Returns:
            Table data as a list of rows, where each row is a list of cell values (str or None).
        """
        # Default settings if none provided (copy to avoid caller mutation)
        table_settings = table_settings.copy() if table_settings else {}
        if text_options is None:
            text_options = {}  # Initialize empty dict

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
            if hasattr(self, "model") and self.model == "tatr" and self.region_type == "table":
                effective_method = "tatr"
            else:
                logger.debug(f"Region {self.bbox}: Auto-detecting table extraction method...")

                try:
                    cell_regions_in_table = [
                        c
                        for c in self.page.find_all(
                            "region[type=table_cell]", apply_exclusions=False
                        )
                        if self.intersects(c)
                    ]
                except Exception:
                    cell_regions_in_table = []  # Fallback silently

                if cell_regions_in_table:
                    logger.debug(
                        f"Region {self.bbox}: Found {len(cell_regions_in_table)} pre-computed table_cell regions – using 'cells' method."
                    )
                    return TableResult(
                        self._extract_table_from_cells(
                            cell_regions_in_table,
                            content_filter=content_filter,
                            apply_exclusions=apply_exclusions,
                        )
                    )

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

        provider_managed_methods = {None, "pdfplumber", "stream", "lattice"}

        # Use the selected method
        if effective_method == "tatr":
            table_rows = self._extract_table_tatr(
                use_ocr=use_ocr,
                ocr_config=ocr_config,
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
            )
        elif effective_method == "text":
            current_text_options = text_options.copy()
            current_text_options["cell_extraction_func"] = cell_extraction_func
            current_text_options["show_progress"] = show_progress
            current_text_options["content_filter"] = content_filter
            current_text_options["apply_exclusions"] = apply_exclusions
            table_rows = self._extract_table_text(**current_text_options)
        elif effective_method in provider_managed_methods:
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
            )
            table_rows = self._select_primary_table(provider_tables)
        else:
            raise ValueError(
                f"Unknown table extraction method: '{method}'. Choose from 'tatr', 'pdfplumber', 'text', 'stream', 'lattice'."
            )

        return TableResult(table_rows)

    def extract_tables(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
    ) -> List[List[List[str]]]:
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

    def _extract_tables_plumber(self, table_settings: dict) -> List[List[List[str]]]:
        """
        Extract all tables using pdfplumber's table extraction.

        Args:
            table_settings: Settings for pdfplumber table extraction

        Returns:
            List of tables, where each table is a list of rows, and each row is a list of cell values
        """
        # Inject global PDF-level text tolerances if not explicitly present
        page_cfg = self.page._config
        pdf_cfg = page_cfg if page_cfg else self.page._parent._config
        _uses_text = "text" in (
            table_settings.get("vertical_strategy"),
            table_settings.get("horizontal_strategy"),
        )
        if (
            _uses_text
            and "text_x_tolerance" not in table_settings
            and "x_tolerance" not in table_settings
        ):
            x_tol = pdf_cfg.get("x_tolerance")
            if x_tol is not None:
                table_settings.setdefault("text_x_tolerance", x_tol)
        if (
            _uses_text
            and "text_y_tolerance" not in table_settings
            and "y_tolerance" not in table_settings
        ):
            y_tol = pdf_cfg.get("y_tolerance")
            if y_tol is not None:
                table_settings.setdefault("text_y_tolerance", y_tol)

        if (
            _uses_text
            and "snap_tolerance" not in table_settings
            and "snap_x_tolerance" not in table_settings
        ):
            snap = max(1, round((pdf_cfg.get("y_tolerance", 1)) * 0.9))
            table_settings.setdefault("snap_tolerance", snap)
        if (
            _uses_text
            and "join_tolerance" not in table_settings
            and "join_x_tolerance" not in table_settings
        ):
            join = table_settings.get("snap_tolerance", 1)
            table_settings.setdefault("join_tolerance", join)
            table_settings.setdefault("join_x_tolerance", join)
            table_settings.setdefault("join_y_tolerance", join)

        self._adjust_explicit_vertical_guides(table_settings, apply_exclusions=True)

        # Apply char-level exclusion filtering, if any exclusions are
        # defined on the parent Page.  We create a lightweight
        # pdfplumber.Page copy whose .chars list omits characters that
        # fall inside any exclusion Region.  Other object types are
        # left untouched for now ("chars-only" strategy).
        base_plumber_page = self.page._page

        if getattr(self.page, "_exclusions", None):
            # Resolve exclusion Regions (callables already evaluated)
            exclusion_regions = self._get_exclusion_regions(include_callable=True)

            def _keep_char(obj):
                """Return True if pdfplumber obj should be kept."""
                if obj.get("object_type") != "char":
                    # Keep non-char objects unchanged – lattice grids etc.
                    return True

                # Compute character centre point
                cx = (obj["x0"] + obj["x1"]) / 2.0
                cy = (obj["top"] + obj["bottom"]) / 2.0

                # Reject if the centre lies inside ANY exclusion Region
                for reg in exclusion_regions:
                    if reg.x0 <= cx <= reg.x1 and reg.top <= cy <= reg.bottom:
                        return False
                return True

            try:
                filtered_page = base_plumber_page.filter(_keep_char)
            except Exception as _filter_err:
                # Fallback – if filtering fails, log and proceed unfiltered
                logger.warning(
                    f"Region {self.bbox}: Failed to filter pdfplumber chars for exclusions: {_filter_err}"
                )
                filtered_page = base_plumber_page
        else:
            filtered_page = base_plumber_page

        # Ensure bbox is within pdfplumber page bounds
        page_bbox = filtered_page.bbox
        clipped_bbox = (
            max(self.bbox[0], page_bbox[0]),  # x0
            max(self.bbox[1], page_bbox[1]),  # y0
            min(self.bbox[2], page_bbox[2]),  # x1
            min(self.bbox[3], page_bbox[3]),  # y1
        )

        # Only crop if the clipped bbox is valid (has positive width and height)
        if clipped_bbox[2] > clipped_bbox[0] and clipped_bbox[3] > clipped_bbox[1]:
            cropped = filtered_page.crop(clipped_bbox)
        else:
            # If the region is completely outside the page bounds, return empty list
            return []

        # Extract all tables from the cropped area
        tables = cropped.extract_tables(table_settings)

        # Apply RTL text processing to all tables
        if tables:
            processed_tables = []
            for table in tables:
                processed_table = []
                for row in table:
                    processed_row = []
                    for cell in row:
                        if cell is not None:
                            # Apply RTL text processing to each cell
                            rtl_processed_cell = self._apply_rtl_processing_to_text(cell)
                            processed_row.append(rtl_processed_cell)
                        else:
                            processed_row.append(cell)
                    processed_table.append(processed_row)
                processed_tables.append(processed_table)
            return processed_tables

        # Return empty list if no tables found
        return []

    def _extract_table_plumber(
        self, table_settings: dict, content_filter=None, apply_exclusions=True
    ) -> List[List[str]]:
        """
        Extract table using pdfplumber's table extraction.
        This method extracts the largest table within the region.

        Args:
            table_settings: Settings for pdfplumber table extraction
            content_filter: Optional content filter to apply to cell values

        Returns:
            Table data as a list of rows, where each row is a list of cell values
        """
        # Inject global PDF-level text tolerances if not explicitly present
        page_cfg = self.page._config
        pdf_cfg = page_cfg if page_cfg else self.page._parent._config
        _uses_text = "text" in (
            table_settings.get("vertical_strategy"),
            table_settings.get("horizontal_strategy"),
        )
        if (
            _uses_text
            and "text_x_tolerance" not in table_settings
            and "x_tolerance" not in table_settings
        ):
            x_tol = pdf_cfg.get("x_tolerance")
            if x_tol is not None:
                table_settings.setdefault("text_x_tolerance", x_tol)
        if (
            _uses_text
            and "text_y_tolerance" not in table_settings
            and "y_tolerance" not in table_settings
        ):
            y_tol = pdf_cfg.get("y_tolerance")
            if y_tol is not None:
                table_settings.setdefault("text_y_tolerance", y_tol)

        if (
            _uses_text
            and "snap_tolerance" not in table_settings
            and "snap_x_tolerance" not in table_settings
        ):
            snap = max(1, round((pdf_cfg.get("y_tolerance", 1)) * 0.9))
            table_settings.setdefault("snap_tolerance", snap)
        if (
            _uses_text
            and "join_tolerance" not in table_settings
            and "join_x_tolerance" not in table_settings
        ):
            join = table_settings.get("snap_tolerance", 1)
            table_settings.setdefault("join_tolerance", join)
            table_settings.setdefault("join_x_tolerance", join)
            table_settings.setdefault("join_y_tolerance", join)

        self._adjust_explicit_vertical_guides(table_settings, apply_exclusions=apply_exclusions)

        # Apply char-level exclusion filtering (chars only) just like in
        # _extract_tables_plumber so header/footer text does not appear
        # in extracted tables.
        base_plumber_page = self.page._page

        if apply_exclusions and getattr(self.page, "_exclusions", None):
            exclusion_regions = self._get_exclusion_regions(include_callable=True)

            def _keep_char(obj):
                if obj.get("object_type") != "char":
                    return True
                cx = (obj["x0"] + obj["x1"]) / 2.0
                cy = (obj["top"] + obj["bottom"]) / 2.0
                for reg in exclusion_regions:
                    if reg.x0 <= cx <= reg.x1 and reg.top <= cy <= reg.bottom:
                        return False
                return True

            try:
                filtered_page = base_plumber_page.filter(_keep_char)
            except Exception as _filter_err:
                logger.warning(
                    f"Region {self.bbox}: Failed to filter pdfplumber chars for exclusions (single table): {_filter_err}"
                )
                filtered_page = base_plumber_page
        else:
            filtered_page = base_plumber_page

        # Now crop the (possibly filtered) page to the region bbox
        # Ensure bbox is within pdfplumber page bounds
        page_bbox = filtered_page.bbox
        clipped_bbox = (
            max(self.bbox[0], page_bbox[0]),  # x0
            max(self.bbox[1], page_bbox[1]),  # y0
            min(self.bbox[2], page_bbox[2]),  # x1
            min(self.bbox[3], page_bbox[3]),  # y1
        )

        # Only crop if the clipped bbox is valid (has positive width and height)
        if clipped_bbox[2] > clipped_bbox[0] and clipped_bbox[3] > clipped_bbox[1]:
            cropped = filtered_page.crop(clipped_bbox)
        else:
            # If the region is completely outside the page bounds, return empty table
            return []

        # Extract the single largest table from the cropped area
        table = cropped.extract_table(table_settings)

        # Return the table or an empty list if none found
        if table:
            # Apply RTL text processing and content filtering if provided
            processed_table = []
            for row in table:
                processed_row = []
                for cell in row:
                    if cell is not None:
                        # Apply RTL text processing first
                        rtl_processed_cell = self._apply_rtl_processing_to_text(cell)

                        # Then apply content filter if provided
                        if content_filter is not None:
                            filtered_cell = self._apply_content_filter_to_text(
                                rtl_processed_cell, content_filter
                            )
                            processed_row.append(filtered_cell)
                        else:
                            processed_row.append(rtl_processed_cell)
                    else:
                        processed_row.append(cell)
                processed_table.append(processed_row)
            return processed_table
        return []

    def _extract_table_tatr(
        self, use_ocr=False, ocr_config=None, content_filter=None, apply_exclusions=True
    ) -> List[List[str]]:
        """
        Extract table using TATR structure detection.

        Args:
            use_ocr: Whether to apply OCR to each cell for better text extraction
            ocr_config: Optional OCR configuration parameters
            content_filter: Optional content filter to apply to cell values

        Returns:
            Table data as a list of rows, where each row is a list of cell values
        """
        # Find all rows and headers in this table
        rows = self.page.find_all("region[type=table-row][model=tatr]")
        headers = self.page.find_all("region[type=table-column-header][model=tatr]")
        columns = self.page.find_all("region[type=table-column][model=tatr]")

        # Filter to only include rows/headers/columns that overlap with this table region
        def is_in_table(region):
            # Check for overlap - simplifying to center point for now
            region_center_x = (region.x0 + region.x1) / 2
            region_center_y = (region.top + region.bottom) / 2
            return (
                self.x0 <= region_center_x <= self.x1 and self.top <= region_center_y <= self.bottom
            )

        rows = [row for row in rows if is_in_table(row)]
        headers = [header for header in headers if is_in_table(header)]
        columns = [column for column in columns if is_in_table(column)]

        # Sort rows by vertical position (top to bottom)
        rows.sort(key=lambda r: r.top)

        # Sort columns by horizontal position (left to right)
        columns.sort(key=lambda c: c.x0)

        # Create table data structure
        table_data = []

        # Prepare OCR config if needed
        ocr_kwargs: Dict[str, Any] = {}
        if use_ocr:
            # Default OCR config focuses on small text with low confidence
            default_ocr_config = {
                "enabled": True,
                "min_confidence": 0.1,  # Lower than default to catch more text
                "detection_params": {
                    "text_threshold": 0.1,  # Lower threshold for low-contrast text
                    "link_threshold": 0.1,  # Lower threshold for connecting text components
                },
            }

            merged_config: Dict[str, Any] = deepcopy(default_ocr_config)

            # Merge with provided config if any
            if ocr_config:
                if isinstance(ocr_config, MappingABC):
                    for key, value in ocr_config.items():
                        if (
                            isinstance(value, MappingABC)
                            and key in merged_config
                            and isinstance(merged_config[key], dict)
                        ):
                            nested = dict(merged_config[key])
                            nested.update(value)
                            merged_config[key] = nested
                        else:
                            merged_config[key] = value
                else:
                    logger.warning(
                        "Ignoring ocr_config of unsupported type %s; expected mapping.",
                        type(ocr_config),
                    )

            # Use the merged config
            ocr_kwargs = merged_config

        # Add header row if headers were detected
        if headers:
            header_texts = []
            for header in headers:
                if use_ocr:
                    # Try OCR for better text extraction
                    header.apply_ocr(**ocr_kwargs)
                    ocr_text = header.extract_text(apply_exclusions=apply_exclusions).strip()
                    if ocr_text:
                        if content_filter is not None:
                            ocr_text = self._apply_content_filter_to_text(ocr_text, content_filter)
                        header_texts.append(ocr_text)
                        continue

                # Fallback to normal extraction
                header_text = header.extract_text(apply_exclusions=apply_exclusions).strip()
                if content_filter is not None:
                    header_text = self._apply_content_filter_to_text(header_text, content_filter)
                header_texts.append(header_text)
            table_data.append(header_texts)

        # Process rows
        for row in rows:
            row_cells = []

            # If we have columns, use them to extract cells
            if columns:
                for column in columns:
                    # Create a cell region at the intersection of row and column
                    cell_bbox = (column.x0, row.top, column.x1, row.bottom)

                    # Create a region for this cell
                    from natural_pdf.elements.region import (  # Import here to avoid circular imports
                        Region,
                    )

                    cell_region = Region(self.page, cell_bbox)

                    # Extract text from the cell
                    if use_ocr:
                        # Apply OCR to the cell
                        cell_region.apply_ocr(**ocr_kwargs)
                        # Get text from newly created OCR elements
                        ocr_text = cell_region.extract_text(
                            apply_exclusions=apply_exclusions
                        ).strip()
                        if ocr_text:
                            if content_filter is not None:
                                ocr_text = self._apply_content_filter_to_text(
                                    ocr_text, content_filter
                                )
                            row_cells.append(ocr_text)
                            continue

                    # Fallback to normal extraction
                    cell_text = cell_region.extract_text(apply_exclusions=apply_exclusions).strip()
                    if content_filter is not None:
                        cell_text = self._apply_content_filter_to_text(cell_text, content_filter)
                    row_cells.append(cell_text)
            else:
                # No column information, just extract the whole row text
                if use_ocr:
                    # Try OCR on the whole row
                    row.apply_ocr(**ocr_kwargs)
                    ocr_text = row.extract_text(apply_exclusions=apply_exclusions).strip()
                    if ocr_text:
                        if content_filter is not None:
                            ocr_text = self._apply_content_filter_to_text(ocr_text, content_filter)
                        row_cells.append(ocr_text)
                        continue

                # Fallback to normal extraction
                row_text = row.extract_text(apply_exclusions=apply_exclusions).strip()
                if content_filter is not None:
                    row_text = self._apply_content_filter_to_text(row_text, content_filter)
                row_cells.append(row_text)

            table_data.append(row_cells)

        return table_data

    def _extract_table_text(self, **text_options) -> List[List[Optional[str]]]:
        """
        Extracts table content based on text alignment analysis.

        Args:
            **text_options: Options passed to analyze_text_table_structure,
                          plus optional 'cell_extraction_func', 'coordinate_grouping_tolerance',
                          'show_progress', and 'content_filter'.

        Returns:
            Table data as list of lists of strings (or None for empty cells).
        """
        cell_extraction_func = text_options.pop("cell_extraction_func", None)
        show_progress = text_options.pop("show_progress", False)
        content_filter = text_options.pop("content_filter", None)
        apply_exclusions = text_options.pop("apply_exclusions", True)

        # Analyze structure first (or use cached results)
        if "text_table_structure" in self.analyses:
            analysis_results = self.analyses["text_table_structure"]
            logger.debug("Using cached text table structure analysis results.")
        else:
            analysis_results = self.analyze_text_table_structure(**text_options)

        if analysis_results is None or not analysis_results.get("cells"):
            logger.warning(f"Region {self.bbox}: No cells found using 'text' method.")
            return []

        cell_dicts = analysis_results["cells"]

        if not cell_dicts:
            return []

        # 1. Get unique sorted top and left coordinates (cell boundaries)
        coord_tolerance = text_options.get("coordinate_grouping_tolerance", 1)
        tops = sorted(
            list(set(round(c["top"] / coord_tolerance) * coord_tolerance for c in cell_dicts))
        )
        lefts = sorted(
            list(set(round(c["left"] / coord_tolerance) * coord_tolerance for c in cell_dicts))
        )

        # Refine boundaries (cluster_coords helper remains the same)
        def cluster_coords(coords):
            if not coords:
                return []
            clustered = []
            current_cluster = [coords[0]]
            for c in coords[1:]:
                if abs(c - current_cluster[-1]) <= coord_tolerance:
                    current_cluster.append(c)
                else:
                    clustered.append(min(current_cluster))
                    current_cluster = [c]
            clustered.append(min(current_cluster))
            return clustered

        unique_tops = cluster_coords(tops)
        unique_lefts = cluster_coords(lefts)

        # Determine iterable for tqdm
        cell_iterator = cell_dicts
        if show_progress:
            # Only wrap if progress should be shown
            cell_iterator = tqdm(
                cell_dicts,
                desc=f"Extracting text from {len(cell_dicts)} cells (text method)",
                unit="cell",
                leave=False,  # Optional: Keep bar after completion
            )

        # 2. Create a lookup map for cell text: {(rounded_top, rounded_left): cell_text}
        cell_text_map = {}
        for cell_data in cell_iterator:
            try:
                cell_region = self.page.region(**cell_data)
                cell_value = None  # Initialize
                if callable(cell_extraction_func):
                    try:
                        cell_value = cell_extraction_func(cell_region)
                        if not isinstance(cell_value, (str, type(None))):
                            logger.warning(
                                f"Custom cell_extraction_func returned non-string/None type ({type(cell_value)}) for cell {cell_data}. Treating as None."
                            )
                            cell_value = None
                    except Exception as func_err:
                        logger.error(
                            f"Error executing custom cell_extraction_func for cell {cell_data}: {func_err}",
                            exc_info=True,
                        )
                        cell_value = None
                else:
                    cell_value = cell_region.extract_text(
                        layout=False,
                        apply_exclusions=apply_exclusions,
                        content_filter=content_filter,
                    ).strip()

                rounded_top = round(cell_data["top"] / coord_tolerance) * coord_tolerance
                rounded_left = round(cell_data["left"] / coord_tolerance) * coord_tolerance
                cell_text_map[(rounded_top, rounded_left)] = cell_value
            except Exception as e:
                logger.warning(f"Could not process cell {cell_data} for text extraction: {e}")

        # 3. Build the final list-of-lists table (loop remains the same)
        final_table = []
        for row_top in unique_tops:
            row_data = []
            for col_left in unique_lefts:
                best_match_key = None
                min_dist_sq = float("inf")
                for map_top, map_left in cell_text_map.keys():
                    if (
                        abs(map_top - row_top) <= coord_tolerance
                        and abs(map_left - col_left) <= coord_tolerance
                    ):
                        dist_sq = (map_top - row_top) ** 2 + (map_left - col_left) ** 2
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            best_match_key = (map_top, map_left)
                cell_value = cell_text_map.get(best_match_key)
                row_data.append(cell_value)
            final_table.append(row_data)

        return final_table

    def _extract_table_from_cells(
        self, cell_regions: List["Region"], content_filter=None, apply_exclusions=True
    ) -> List[List[Optional[str]]]:
        """Construct a table (list-of-lists) from table_cell regions.

        This assumes each cell Region has metadata.row_index / col_index as written by
        detect_table_structure_from_lines().  If these keys are missing we will
        fall back to sorting by geometry.

        Args:
            cell_regions: List of table cell Region objects to extract text from
            content_filter: Optional content filter to apply to cell text extraction
        """
        if not cell_regions:
            return []

        # Attempt to use explicit indices first
        all_row_idxs = []
        all_col_idxs = []
        for cell in cell_regions:
            try:
                row_idx_value = cell.metadata.get("row_index")
                col_idx_value = cell.metadata.get("col_index")
                if row_idx_value is None or col_idx_value is None:
                    raise ValueError("Missing explicit indices")

                r_idx = int(row_idx_value)
                c_idx = int(col_idx_value)
                all_row_idxs.append(r_idx)
                all_col_idxs.append(c_idx)
            except Exception:
                # Not all cells have indices – clear the lists so we switch to geometric sorting
                all_row_idxs = []
                all_col_idxs = []
                break

        if all_row_idxs and all_col_idxs:
            num_rows = max(all_row_idxs) + 1
            num_cols = max(all_col_idxs) + 1

            # Initialise blank grid
            table_grid: List[List[Optional[str]]] = [[None] * num_cols for _ in range(num_rows)]

            for cell in cell_regions:
                row_idx_value = cell.metadata.get("row_index")
                col_idx_value = cell.metadata.get("col_index")
                if row_idx_value is None or col_idx_value is None:
                    raise ValueError("Missing explicit indices")

                r_idx = int(row_idx_value)
                c_idx = int(col_idx_value)
                text_val = cell.extract_text(
                    layout=False,
                    apply_exclusions=apply_exclusions,
                    content_filter=content_filter,
                ).strip()
                table_grid[r_idx][c_idx] = text_val if text_val else None

            return table_grid

        import numpy as np

        # Build arrays of centers
        centers = np.array([[(c.x0 + c.x1) / 2.0, (c.top + c.bottom) / 2.0] for c in cell_regions])
        xs = centers[:, 0]
        ys = centers[:, 1]

        # Cluster unique row Y positions and column X positions with a tolerance
        def _cluster(vals, tol=1.0):
            sorted_vals = np.sort(vals)
            groups = [[sorted_vals[0]]]
            for v in sorted_vals[1:]:
                if abs(v - groups[-1][-1]) <= tol:
                    groups[-1].append(v)
                else:
                    groups.append([v])
            return [np.mean(g) for g in groups]

        row_centers = _cluster(ys)
        col_centers = _cluster(xs)

        num_rows = len(row_centers)
        num_cols = len(col_centers)

        table_grid: List[List[Optional[str]]] = [[None] * num_cols for _ in range(num_rows)]

        # Assign each cell to nearest row & col center
        for cell, (cx, cy) in zip(cell_regions, centers):
            row_idx = int(np.argmin([abs(cy - rc) for rc in row_centers]))
            col_idx = int(np.argmin([abs(cx - cc) for cc in col_centers]))

            text_val = cell.extract_text(
                layout=False, apply_exclusions=apply_exclusions, content_filter=content_filter
            ).strip()
            table_grid[row_idx][col_idx] = text_val if text_val else None

        return table_grid

    def _adjust_explicit_vertical_guides(
        self, table_settings: dict, *, apply_exclusions: bool = True
    ) -> None:
        """Clamp explicit vertical guides to detected text bounds for text strategies."""

        if (
            table_settings.get("horizontal_strategy") != "text"
            or table_settings.get("vertical_strategy") != "explicit"
            or "explicit_vertical_lines" not in table_settings
        ):
            return

        text_elements = self.find_all("text", apply_exclusions=apply_exclusions)
        if not text_elements:
            return

        text_bounds = text_elements.merge().bbox
        text_left = text_bounds[0]
        text_right = text_bounds[2]

        adjusted_verticals: List[float] = []
        for guide in table_settings["explicit_vertical_lines"]:
            if guide < text_left:
                adjusted_verticals.append(text_left)
                logger.debug(
                    "Region %s: Adjusted left guide from %.1f to %.1f",
                    getattr(self, "bbox", None),
                    guide,
                    text_left,
                )
            elif guide > text_right:
                adjusted_verticals.append(text_right)
                logger.debug(
                    "Region %s: Adjusted right guide from %.1f to %.1f",
                    getattr(self, "bbox", None),
                    guide,
                    text_right,
                )
            else:
                adjusted_verticals.append(guide)

        table_settings["explicit_vertical_lines"] = adjusted_verticals

    def _select_primary_table(
        self, tables: Sequence[Sequence[Sequence[Optional[str]]]]
    ) -> List[List[Optional[str]]]:
        """Pick the largest table (by rows*cols) from a provider response."""

        best_table: List[List[Optional[str]]] = []
        best_score = -1
        for table in tables or []:
            if not table:
                continue
            row_count = len(table)
            col_count = max((len(row) for row in table), default=0)
            score = row_count * col_count
            if score > best_score:
                best_table = [list(row) for row in table]
                best_score = score
        return best_table

    def _tables_have_content(self, tables: Sequence[Sequence[Sequence[Optional[str]]]]) -> bool:
        return any(
            any(any((cell or "").strip() for cell in row) for row in table if table)
            for table in tables
        )
