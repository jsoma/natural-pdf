import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PIL_Image

    from natural_pdf.core.page import Page
    from natural_pdf.core.page_collection import PageCollection
    from natural_pdf.elements.base import Element as PhysicalElement
    from natural_pdf.elements.element_collection import (
        ElementCollection as PhysicalElementCollection,
    )
    from natural_pdf.elements.region import Region as PhysicalRegion

    from .collections import FlowElementCollection
    from .element import FlowElement

# Import required classes for the new methods
# For runtime image manipulation
from PIL import Image as PIL_Image_Runtime

from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.tables import TableResult

logger = logging.getLogger(__name__)


class Flow(Visualizable):
    """Defines a logical flow or sequence of physical Page or Region objects.

    A Flow represents a continuous logical document structure that spans across
    multiple pages or regions, enabling operations on content that flows across
    boundaries. This is essential for handling multi-page tables, articles that
    span columns, or any content that requires reading order across segments.

    Flows specify arrangement (vertical/horizontal) and alignment rules to create
    a unified coordinate system for element extraction and text processing. They
    enable natural-pdf to treat fragmented content as a single continuous area
    for analysis and extraction operations.

    The Flow system is particularly useful for:
    - Multi-page tables that break across page boundaries
    - Multi-column articles with complex reading order
    - Forms that span multiple pages
    - Any content requiring logical continuation across segments

    Attributes:
        segments: List of Page or Region objects in flow order.
        arrangement: Primary flow direction ('vertical' or 'horizontal').
        alignment: Cross-axis alignment for segments of different sizes.
        segment_gap: Virtual gap between segments in PDF points.

    Example:
        Multi-page table flow:
        ```python
        pdf = npdf.PDF("multi_page_table.pdf")

        # Create flow for table spanning pages 2-4
        table_flow = Flow(
            segments=[pdf.pages[1], pdf.pages[2], pdf.pages[3]],
            arrangement='vertical',
            alignment='left',
            segment_gap=10.0
        )

        # Extract table as if it were continuous
        table_data = table_flow.extract_table()
        text_content = table_flow.get_text()
        ```

        Multi-column article flow:
        ```python
        page = pdf.pages[0]
        left_column = page.region(0, 0, 300, page.height)
        right_column = page.region(320, 0, page.width, page.height)

        # Create horizontal flow for columns
        article_flow = Flow(
            segments=[left_column, right_column],
            arrangement='horizontal',
            alignment='top'
        )

        # Read in proper order
        article_text = article_flow.get_text()
        ```

    Note:
        Flows create virtual coordinate systems that map element positions across
        segments, enabling spatial navigation and element selection to work
        seamlessly across boundaries.
    """

    def __init__(
        self,
        segments: Union[List[Union["Page", "PhysicalRegion"]], "PageCollection"],
        arrangement: Literal["vertical", "horizontal"],
        alignment: Literal["start", "center", "end", "top", "left", "bottom", "right"] = "start",
        segment_gap: float = 0.0,
    ):
        """
        Initializes a Flow object.

        Args:
            segments: An ordered list of natural_pdf.core.page.Page or
                      natural_pdf.elements.region.Region objects that constitute the flow,
                      or a PageCollection containing pages.
            arrangement: The primary direction of the flow.
                         - "vertical": Segments are stacked top-to-bottom.
                         - "horizontal": Segments are arranged left-to-right.
            alignment: How segments are aligned on their cross-axis if they have
                       differing dimensions. For a "vertical" arrangement:
                       - "left" (or "start"): Align left edges.
                       - "center": Align centers.
                       - "right" (or "end"): Align right edges.
                       For a "horizontal" arrangement:
                       - "top" (or "start"): Align top edges.
                       - "center": Align centers.
                       - "bottom" (or "end"): Align bottom edges.
            segment_gap: The virtual gap (in PDF points) between segments.
        """
        # Handle PageCollection input
        if hasattr(segments, "pages"):  # It's a PageCollection
            segments = list(segments.pages)

        if not segments:
            raise ValueError("Flow segments cannot be empty.")
        if arrangement not in ["vertical", "horizontal"]:
            raise ValueError("Arrangement must be 'vertical' or 'horizontal'.")

        self.segments: List["PhysicalRegion"] = self._normalize_segments(segments)
        self.arrangement: Literal["vertical", "horizontal"] = arrangement
        self.alignment: Literal["start", "center", "end", "top", "left", "bottom", "right"] = (
            alignment
        )
        self.segment_gap: float = segment_gap

        self._validate_alignment()

        # TODO: Pre-calculate segment offsets for faster lookups if needed

    def _normalize_segments(
        self, segments: List[Union["Page", "PhysicalRegion"]]
    ) -> List["PhysicalRegion"]:
        """Converts all Page segments to full-page Region objects for uniform processing."""
        normalized = []
        from natural_pdf.core.page import Page as CorePage
        from natural_pdf.elements.region import Region as ElementsRegion

        for i, segment in enumerate(segments):
            if isinstance(segment, CorePage):
                normalized.append(segment.region(0, 0, segment.width, segment.height))
            elif isinstance(segment, ElementsRegion):
                normalized.append(segment)
            elif hasattr(segment, "object_type") and segment.object_type == "page":
                if not isinstance(segment, CorePage):
                    raise TypeError(
                        f"Segment {i} has object_type 'page' but is not an instance of natural_pdf.core.page.Page. Got {type(segment)}"
                    )
                normalized.append(segment.region(0, 0, segment.width, segment.height))
            elif hasattr(segment, "object_type") and segment.object_type == "region":
                if not isinstance(segment, ElementsRegion):
                    raise TypeError(
                        f"Segment {i} has object_type 'region' but is not an instance of natural_pdf.elements.region.Region. Got {type(segment)}"
                    )
                normalized.append(segment)
            else:
                raise TypeError(
                    f"Segment {i} is not a valid Page or Region object. Got {type(segment)}."
                )
        return normalized

    def _validate_alignment(self) -> None:
        """Validates the alignment based on the arrangement."""
        valid_alignments = {
            "vertical": ["start", "center", "end", "left", "right"],
            "horizontal": ["start", "center", "end", "top", "bottom"],
        }
        if self.alignment not in valid_alignments[self.arrangement]:
            raise ValueError(
                f"Invalid alignment '{self.alignment}' for '{self.arrangement}' arrangement. "
                f"Valid options are: {valid_alignments[self.arrangement]}"
            )

    def _get_highlighter(self):
        """Get the highlighting service from the first segment."""
        if not self.segments:
            raise RuntimeError("Flow has no segments to get highlighter from")

        # Get highlighter from first segment
        first_segment = self.segments[0]
        if hasattr(first_segment, "_highlighter"):
            return first_segment._highlighter
        elif hasattr(first_segment, "page") and hasattr(first_segment.page, "_highlighter"):
            return first_segment.page._highlighter
        else:
            raise RuntimeError(
                f"Cannot find HighlightingService from Flow segments. "
                f"First segment type: {type(first_segment).__name__}"
            )

    def show(
        self,
        *,
        # Basic rendering options
        resolution: Optional[float] = None,
        width: Optional[int] = None,
        # Highlight options
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        labels: bool = True,
        label_format: Optional[str] = None,
        highlights: Optional[List[Dict[str, Any]]] = None,
        # Layout options for multi-page/region
        layout: Literal["stack", "grid", "single"] = "stack",
        stack_direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 5,
        columns: Optional[int] = None,  # For grid layout
        # Cropping options
        crop: Union[bool, Literal["content"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        # Flow-specific options
        in_context: bool = False,
        separator_color: Optional[Tuple[int, int, int]] = None,
        separator_thickness: int = 2,
        **kwargs,
    ) -> Optional["PIL_Image"]:
        """Generate a preview image with highlights.

        If in_context=True, shows segments as cropped images stacked together
        with separators between segments.

        Args:
            resolution: DPI for rendering (default from global settings)
            width: Target width in pixels (overrides resolution)
            color: Default highlight color
            labels: Whether to show labels for highlights
            label_format: Format string for labels
            highlights: Additional highlight groups to show
            layout: How to arrange multiple pages/regions
            stack_direction: Direction for stack layout
            gap: Pixels between stacked images
            columns: Number of columns for grid layout
            crop: Whether to crop
            crop_bbox: Explicit crop bounds
            in_context: If True, use special Flow visualization with separators
            separator_color: RGB color for separator lines (default: red)
            separator_thickness: Thickness of separator lines
            **kwargs: Additional parameters passed to rendering

        Returns:
            PIL Image object or None if nothing to render
        """
        if in_context:
            # Use the special in_context visualization
            return self._show_in_context(
                resolution=resolution or 150,
                width=width,
                stack_direction=stack_direction,
                stack_gap=gap,
                separator_color=separator_color or (255, 0, 0),
                separator_thickness=separator_thickness,
                **kwargs,
            )

        # Otherwise use the standard show method
        return super().show(
            resolution=resolution,
            width=width,
            color=color,
            labels=labels,
            label_format=label_format,
            highlights=highlights,
            layout=layout,
            stack_direction=stack_direction,
            gap=gap,
            columns=columns,
            crop=crop,
            crop_bbox=crop_bbox,
            **kwargs,
        )

    def find(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> Optional["FlowElement"]:
        """
        Finds the first element within the flow that matches the given selector or text criteria.

        Elements found are wrapped as FlowElement objects, anchored to this Flow.

        Args:
            selector: CSS-like selector string.
            text: Text content to search for.
            apply_exclusions: Whether to respect exclusion zones on the original pages/regions.
            regex: Whether the text search uses regex.
            case: Whether the text search is case-sensitive.
            **kwargs: Additional filter parameters for the underlying find operation.

        Returns:
            A FlowElement if a match is found, otherwise None.
        """
        results = self.find_all(
            selector=selector,
            text=text,
            apply_exclusions=apply_exclusions,
            regex=regex,
            case=case,
            **kwargs,
        )
        return results.first if results else None

    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        **kwargs,
    ) -> "FlowElementCollection":
        """
        Finds all elements within the flow that match the given selector or text criteria.

        This method efficiently groups segments by their parent pages, searches at the page level,
        then filters results appropriately for each segment. This ensures elements that intersect
        with flow segments (but aren't fully contained) are still found.

        Elements found are wrapped as FlowElement objects, anchored to this Flow,
        and returned in a FlowElementCollection.
        """
        from .collections import FlowElementCollection
        from .element import FlowElement

        # Step 1: Group segments by their parent pages (like in analyze_layout)
        segments_by_page = {}  # Dict[Page, List[Segment]]

        for i, segment in enumerate(self.segments):
            # Determine the page for this segment - fix type detection
            if hasattr(segment, "page") and hasattr(segment.page, "find_all"):
                # It's a Region object (has a parent page)
                page_obj = segment.page
                segment_type = "region"
            elif (
                hasattr(segment, "find_all")
                and hasattr(segment, "width")
                and hasattr(segment, "height")
                and not hasattr(segment, "page")
            ):
                # It's a Page object (has find_all but no parent page)
                page_obj = segment
                segment_type = "page"
            else:
                logger.warning(f"Segment {i+1} does not support find_all, skipping")
                continue

            if page_obj not in segments_by_page:
                segments_by_page[page_obj] = []
            segments_by_page[page_obj].append((segment, segment_type))

        if not segments_by_page:
            logger.warning("No segments with searchable pages found")
            return FlowElementCollection([])

        # Step 2: Search each unique page only once
        all_flow_elements: List["FlowElement"] = []

        for page_obj, page_segments in segments_by_page.items():
            # Find all matching elements on this page
            page_matches = page_obj.find_all(
                selector=selector,
                text=text,
                apply_exclusions=apply_exclusions,
                regex=regex,
                case=case,
                **kwargs,
            )

            if not page_matches:
                continue

            # Step 3: For each segment on this page, collect relevant elements
            for segment, segment_type in page_segments:
                if segment_type == "page":
                    # Full page segment: include all elements
                    for phys_elem in page_matches.elements:
                        all_flow_elements.append(FlowElement(physical_object=phys_elem, flow=self))

                elif segment_type == "region":
                    # Region segment: filter to only intersecting elements
                    for phys_elem in page_matches.elements:
                        try:
                            # Check if element intersects with this flow segment
                            if segment.intersects(phys_elem):
                                all_flow_elements.append(
                                    FlowElement(physical_object=phys_elem, flow=self)
                                )
                        except Exception as intersect_error:
                            logger.debug(
                                f"Error checking intersection for element: {intersect_error}"
                            )
                            # Include the element anyway if intersection check fails
                            all_flow_elements.append(
                                FlowElement(physical_object=phys_elem, flow=self)
                            )

        # Step 4: Remove duplicates (can happen if multiple segments intersect the same element)
        unique_flow_elements = []
        seen_element_ids = set()

        for flow_elem in all_flow_elements:
            # Create a unique identifier for the underlying physical element
            phys_elem = flow_elem.physical_object
            elem_id = (
                (
                    getattr(phys_elem.page, "index", id(phys_elem.page))
                    if hasattr(phys_elem, "page")
                    else id(phys_elem)
                ),
                phys_elem.bbox if hasattr(phys_elem, "bbox") else id(phys_elem),
            )

            if elem_id not in seen_element_ids:
                unique_flow_elements.append(flow_elem)
                seen_element_ids.add(elem_id)

        return FlowElementCollection(unique_flow_elements)

    def __repr__(self) -> str:
        return (
            f"<Flow segments={len(self.segments)}, "
            f"arrangement='{self.arrangement}', alignment='{self.alignment}', gap={self.segment_gap}>"
        )

    @overload
    def extract_table(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[dict] = None,
        cell_extraction_func: Optional[Any] = None,
        show_progress: bool = False,
        content_filter: Optional[Any] = None,
        stitch_rows: Callable[[List[Optional[str]]], bool] = None,
    ) -> TableResult: ...

    @overload
    def extract_table(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[dict] = None,
        cell_extraction_func: Optional[Any] = None,
        show_progress: bool = False,
        content_filter: Optional[Any] = None,
        stitch_rows: Callable[
            [List[Optional[str]], List[Optional[str]], int, Union["Page", "PhysicalRegion"]],
            bool,
        ] = None,
    ) -> TableResult: ...

    def extract_table(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[dict] = None,
        cell_extraction_func: Optional[Any] = None,
        show_progress: bool = False,
        content_filter: Optional[Any] = None,
        stitch_rows: Optional[Callable] = None,
        merge_headers: Optional[bool] = None,
    ) -> TableResult:
        """
        Extract table data from all segments in the flow, combining results sequentially.

        This method extracts table data from each segment in flow order and combines
        the results into a single logical table. This is particularly useful for
        multi-page tables or tables that span across columns.

        Args:
            method: Method to use: 'tatr', 'pdfplumber', 'text', 'stream', 'lattice', or None (auto-detect).
            table_settings: Settings for pdfplumber table extraction.
            use_ocr: Whether to use OCR for text extraction (currently only applicable with 'tatr' method).
            ocr_config: OCR configuration parameters.
            text_options: Dictionary of options for the 'text' method.
            cell_extraction_func: Optional callable function that takes a cell Region object
                                  and returns its string content. For 'text' method only.
            show_progress: If True, display a progress bar during cell text extraction for the 'text' method.
            content_filter: Optional content filter to apply during cell text extraction.
            merge_headers: Whether to merge tables by removing repeated headers from subsequent
                segments. If None (default), auto-detects by checking if the first row
                of each segment matches the first row of the first segment. If segments have
                inconsistent header patterns (some repeat, others don't), raises ValueError.
                Useful for multi-page tables where headers repeat on each page.
            stitch_rows: Optional callable to determine when rows should be merged across
                         segment boundaries. Applied AFTER header removal if merge_headers
                         is enabled. Two overloaded signatures are supported:

                         • func(current_row) -> bool
                           Called only on the first row of each segment (after the first).
                           Return True to merge this first row with the last row from
                           the previous segment.

                         • func(prev_row, current_row, row_index, segment) -> bool
                           Called for every row. Return True to merge current_row with
                           the previous row in the aggregated results.

                         When True is returned, rows are concatenated cell-by-cell.
                         This is useful for handling table rows split across page
                         boundaries or segments. If None, rows are never merged.

        Returns:
            TableResult object containing the aggregated table data from all segments.

        Example:
            Multi-page table extraction:
            ```python
            pdf = npdf.PDF("multi_page_table.pdf")

            # Create flow for table spanning pages 2-4
            table_flow = Flow(
                segments=[pdf.pages[1], pdf.pages[2], pdf.pages[3]],
                arrangement='vertical'
            )

            # Extract table as if it were continuous
            table_data = table_flow.extract_table()
            df = table_data.df  # Convert to pandas DataFrame

            # Custom row stitching - single parameter (simple case)
            table_data = table_flow.extract_table(
                stitch_rows=lambda row: row and not (row[0] or "").strip()
            )

            # Custom row stitching - full parameters (advanced case)
            table_data = table_flow.extract_table(
                stitch_rows=lambda prev, curr, idx, seg: idx == 0 and curr and not (curr[0] or "").strip()
            )
            ```
        """
        logger.info(
            f"Extracting table from Flow with {len(self.segments)} segments (method: {method or 'auto'})"
        )

        if not self.segments:
            logger.warning("Flow has no segments, returning empty table")
            return TableResult([])

        # Resolve predicate and determine its signature
        predicate: Optional[Callable] = None
        predicate_type: str = "none"

        if callable(stitch_rows):
            import inspect

            sig = inspect.signature(stitch_rows)
            param_count = len(sig.parameters)

            if param_count == 1:
                predicate = stitch_rows
                predicate_type = "single_param"
            elif param_count == 4:
                predicate = stitch_rows
                predicate_type = "full_params"
            else:
                logger.warning(
                    f"stitch_rows function has {param_count} parameters, expected 1 or 4. Ignoring."
                )
                predicate = None
                predicate_type = "none"

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
        processed_segments = 0
        header_row: Optional[List[Optional[str]]] = None
        merge_headers_enabled = False
        headers_warned = False  # Track if we've already warned about dropping headers
        segment_has_repeated_header = []  # Track which segments have repeated headers

        for seg_idx, segment in enumerate(self.segments):
            try:
                logger.debug(f"  Extracting table from segment {seg_idx+1}/{len(self.segments)}")

                segment_result = segment.extract_table(
                    method=method,
                    table_settings=table_settings.copy() if table_settings else None,
                    use_ocr=use_ocr,
                    ocr_config=ocr_config,
                    text_options=text_options.copy() if text_options else None,
                    cell_extraction_func=cell_extraction_func,
                    show_progress=show_progress,
                    content_filter=content_filter,
                )

                if not segment_result:
                    continue

                if hasattr(segment_result, "_rows"):
                    segment_rows = list(segment_result._rows)
                else:
                    segment_rows = list(segment_result)

                if not segment_rows:
                    logger.debug(f"    No table data found in segment {seg_idx+1}")
                    continue

                # Handle header detection and merging for multi-page tables
                if seg_idx == 0:
                    # First segment: capture potential header row
                    if segment_rows:
                        header_row = segment_rows[0]
                        # Determine if we should merge headers
                        if merge_headers is None:
                            # Auto-detect: we'll check all subsequent segments
                            merge_headers_enabled = False  # Will be determined later
                        else:
                            merge_headers_enabled = merge_headers
                        # Track that first segment exists (for consistency checking)
                        segment_has_repeated_header.append(False)  # First segment doesn't "repeat"
                elif seg_idx == 1 and merge_headers is None:
                    # Auto-detection: check if first row of second segment matches header
                    has_header = segment_rows and header_row and segment_rows[0] == header_row
                    segment_has_repeated_header.append(has_header)

                    if has_header:
                        merge_headers_enabled = True
                        # Remove the detected repeated header from this segment
                        segment_rows = segment_rows[1:]
                        logger.debug(
                            f"    Auto-detected repeated header in segment {seg_idx+1}, removed"
                        )
                        if not headers_warned:
                            warnings.warn(
                                "Detected repeated headers in multi-page table. Merging by removing "
                                "repeated headers from subsequent pages.",
                                UserWarning,
                                stacklevel=2,
                            )
                            headers_warned = True
                    else:
                        merge_headers_enabled = False
                        logger.debug(f"    No repeated header detected in segment {seg_idx+1}")
                elif seg_idx > 1:
                    # Check consistency: all segments should have same pattern
                    has_header = segment_rows and header_row and segment_rows[0] == header_row
                    segment_has_repeated_header.append(has_header)

                    # Remove header if merging is enabled and header is present
                    if merge_headers_enabled and has_header:
                        segment_rows = segment_rows[1:]
                        logger.debug(f"    Removed repeated header from segment {seg_idx+1}")
                elif seg_idx > 0 and merge_headers_enabled:
                    # Explicit merge_headers=True: remove headers from subsequent segments
                    if segment_rows and header_row and segment_rows[0] == header_row:
                        segment_rows = segment_rows[1:]
                        logger.debug(f"    Removed repeated header from segment {seg_idx+1}")
                        if not headers_warned:
                            warnings.warn(
                                "Removing repeated headers from multi-page table during merge.",
                                UserWarning,
                                stacklevel=2,
                            )
                            headers_warned = True

                for row_idx, row in enumerate(segment_rows):
                    should_merge = False

                    if predicate is not None and aggregated_rows:
                        if predicate_type == "single_param":
                            # For single param: only call on first row of segment (row_idx == 0)
                            # and pass the current row
                            if row_idx == 0:
                                should_merge = predicate(row)
                        elif predicate_type == "full_params":
                            # For full params: call with all arguments
                            should_merge = predicate(aggregated_rows[-1], row, row_idx, segment)

                    if should_merge:
                        aggregated_rows[-1] = _default_merge(aggregated_rows[-1], row)
                    else:
                        aggregated_rows.append(row)

                processed_segments += 1
                logger.debug(
                    f"    Added {len(segment_rows)} rows (post-merge) from segment {seg_idx+1}"
                )

            except Exception as e:
                logger.error(f"Error extracting table from segment {seg_idx+1}: {e}", exc_info=True)
                continue

        # Check for inconsistent header patterns after processing all segments
        if merge_headers is None and len(segment_has_repeated_header) > 2:
            # During auto-detection, check for consistency across all segments
            expected_pattern = segment_has_repeated_header[1]  # Pattern from second segment
            for seg_idx, has_header in enumerate(segment_has_repeated_header[2:], 2):
                if has_header != expected_pattern:
                    # Inconsistent pattern detected
                    segments_with_headers = [
                        i for i, has_h in enumerate(segment_has_repeated_header[1:], 1) if has_h
                    ]
                    segments_without_headers = [
                        i for i, has_h in enumerate(segment_has_repeated_header[1:], 1) if not has_h
                    ]
                    raise ValueError(
                        f"Inconsistent header pattern in multi-page table: "
                        f"segments {segments_with_headers} have repeated headers, "
                        f"but segments {segments_without_headers} do not. "
                        f"All segments must have the same header pattern for reliable merging."
                    )

        logger.info(
            f"Flow table extraction complete: {len(aggregated_rows)} total rows from {processed_segments}/{len(self.segments)} segments"
        )
        return TableResult(aggregated_rows)

    def analyze_layout(
        self,
        engine: Optional[str] = None,
        options: Optional[Any] = None,
        confidence: Optional[float] = None,
        classes: Optional[List[str]] = None,
        exclude_classes: Optional[List[str]] = None,
        device: Optional[str] = None,
        existing: str = "replace",
        model_name: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> "PhysicalElementCollection":
        """
        Analyze layout across all segments in the flow.

        This method efficiently groups segments by their parent pages, runs layout analysis
        only once per unique page, then filters results appropriately for each segment.
        This avoids redundant analysis when multiple flow segments come from the same page.

        Args:
            engine: Name of the layout engine (e.g., 'yolo', 'tatr'). Uses manager's default if None.
            options: Specific LayoutOptions object for advanced configuration.
            confidence: Minimum confidence threshold.
            classes: Specific classes to detect.
            exclude_classes: Classes to exclude.
            device: Device for inference.
            existing: How to handle existing detected regions: 'replace' (default) or 'append'.
            model_name: Optional model name for the engine.
            client: Optional client for API-based engines.

        Returns:
            ElementCollection containing all detected Region objects from all segments.

        Example:
            Multi-page layout analysis:
            ```python
            pdf = npdf.PDF("document.pdf")

            # Create flow for first 3 pages
            page_flow = Flow(
                segments=pdf.pages[:3],
                arrangement='vertical'
            )

            # Analyze layout across all pages (efficiently)
            all_regions = page_flow.analyze_layout(engine='yolo')

            # Find all tables across the flow
            tables = all_regions.filter('region[type=table]')
            ```
        """
        from natural_pdf.elements.element_collection import ElementCollection

        logger.info(
            f"Analyzing layout across Flow with {len(self.segments)} segments (engine: {engine or 'default'})"
        )

        if not self.segments:
            logger.warning("Flow has no segments, returning empty collection")
            return ElementCollection([])

        # Step 1: Group segments by their parent pages to avoid redundant analysis
        segments_by_page = {}  # Dict[Page, List[Segment]]

        for i, segment in enumerate(self.segments):
            # Determine the page for this segment
            if hasattr(segment, "analyze_layout"):
                # It's a Page object
                page_obj = segment
                segment_type = "page"
            elif hasattr(segment, "page") and hasattr(segment.page, "analyze_layout"):
                # It's a Region object
                page_obj = segment.page
                segment_type = "region"
            else:
                logger.warning(f"Segment {i+1} does not support layout analysis, skipping")
                continue

            if page_obj not in segments_by_page:
                segments_by_page[page_obj] = []
            segments_by_page[page_obj].append((segment, segment_type))

        if not segments_by_page:
            logger.warning("No segments with analyzable pages found")
            return ElementCollection([])

        logger.debug(
            f"  Grouped {len(self.segments)} segments into {len(segments_by_page)} unique pages"
        )

        # Step 2: Analyze each unique page only once
        all_detected_regions: List["PhysicalRegion"] = []
        processed_pages = 0

        for page_obj, page_segments in segments_by_page.items():
            try:
                logger.debug(
                    f"  Analyzing layout for page {getattr(page_obj, 'number', '?')} with {len(page_segments)} segments"
                )

                # Run layout analysis once for this page
                page_results = page_obj.analyze_layout(
                    engine=engine,
                    options=options,
                    confidence=confidence,
                    classes=classes,
                    exclude_classes=exclude_classes,
                    device=device,
                    existing=existing,
                    model_name=model_name,
                    client=client,
                )

                # Extract regions from results
                if hasattr(page_results, "elements"):
                    # It's an ElementCollection
                    page_regions = page_results.elements
                elif isinstance(page_results, list):
                    # It's a list of regions
                    page_regions = page_results
                else:
                    logger.warning(
                        f"Page {getattr(page_obj, 'number', '?')} returned unexpected layout analysis result type: {type(page_results)}"
                    )
                    continue

                if not page_regions:
                    logger.debug(
                        f"    No layout regions found on page {getattr(page_obj, 'number', '?')}"
                    )
                    continue

                # Step 3: For each segment on this page, collect relevant regions
                segments_processed_on_page = 0
                for segment, segment_type in page_segments:
                    if segment_type == "page":
                        # Full page segment: include all detected regions
                        all_detected_regions.extend(page_regions)
                        segments_processed_on_page += 1
                        logger.debug(f"    Added {len(page_regions)} regions for full-page segment")

                    elif segment_type == "region":
                        # Region segment: filter to only intersecting regions
                        intersecting_regions = []
                        for region in page_regions:
                            try:
                                if segment.intersects(region):
                                    intersecting_regions.append(region)
                            except Exception as intersect_error:
                                logger.debug(
                                    f"Error checking intersection for region: {intersect_error}"
                                )
                                # Include the region anyway if intersection check fails
                                intersecting_regions.append(region)

                        all_detected_regions.extend(intersecting_regions)
                        segments_processed_on_page += 1
                        logger.debug(
                            f"    Added {len(intersecting_regions)} intersecting regions for region segment {segment.bbox}"
                        )

                processed_pages += 1
                logger.debug(
                    f"    Processed {segments_processed_on_page} segments on page {getattr(page_obj, 'number', '?')}"
                )

            except Exception as e:
                logger.error(
                    f"Error analyzing layout for page {getattr(page_obj, 'number', '?')}: {e}",
                    exc_info=True,
                )
                continue

        # Step 4: Remove duplicates (can happen if multiple segments intersect the same region)
        unique_regions = []
        seen_region_ids = set()

        for region in all_detected_regions:
            # Create a unique identifier for this region (page + bbox)
            region_id = (
                getattr(region.page, "index", id(region.page)),
                region.bbox if hasattr(region, "bbox") else id(region),
            )

            if region_id not in seen_region_ids:
                unique_regions.append(region)
                seen_region_ids.add(region_id)

        dedupe_removed = len(all_detected_regions) - len(unique_regions)
        if dedupe_removed > 0:
            logger.debug(f"  Removed {dedupe_removed} duplicate regions")

        logger.info(
            f"Flow layout analysis complete: {len(unique_regions)} unique regions from {processed_pages} pages"
        )
        return ElementCollection(unique_regions)

    def _get_render_specs(
        self,
        mode: Literal["show", "render"] = "show",
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        highlights: Optional[List[Dict[str, Any]]] = None,
        crop: Union[bool, Literal["content"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        label_prefix: Optional[str] = "FlowSegment",
        **kwargs,
    ) -> List[RenderSpec]:
        """Get render specifications for this flow.

        Args:
            mode: Rendering mode - 'show' includes highlights, 'render' is clean
            color: Color for highlighting segments in show mode
            highlights: Additional highlight groups to show
            crop: Whether to crop to segments
            crop_bbox: Explicit crop bounds
            label_prefix: Prefix for segment labels
            **kwargs: Additional parameters

        Returns:
            List of RenderSpec objects, one per page with segments
        """
        if not self.segments:
            return []

        # Group segments by their physical pages
        segments_by_page = {}  # Dict[Page, List[PhysicalRegion]]

        for i, segment in enumerate(self.segments):
            # Get the page for this segment
            if hasattr(segment, "page") and segment.page is not None:
                # It's a Region, use its page
                page_obj = segment.page
                if page_obj not in segments_by_page:
                    segments_by_page[page_obj] = []
                segments_by_page[page_obj].append(segment)
            elif (
                hasattr(segment, "index")
                and hasattr(segment, "width")
                and hasattr(segment, "height")
            ):
                # It's a full Page object, create a full-page region for it
                page_obj = segment
                full_page_region = segment.region(0, 0, segment.width, segment.height)
                if page_obj not in segments_by_page:
                    segments_by_page[page_obj] = []
                segments_by_page[page_obj].append(full_page_region)
            else:
                logger.warning(f"Segment {i+1} has no identifiable page, skipping")
                continue

        if not segments_by_page:
            return []

        # Create RenderSpec for each page
        specs = []

        # Sort pages by index for consistent output order
        sorted_pages = sorted(
            segments_by_page.keys(),
            key=lambda p: p.index if hasattr(p, "index") else getattr(p, "page_number", 0),
        )

        for page_idx, page_obj in enumerate(sorted_pages):
            segments_on_this_page = segments_by_page[page_obj]
            if not segments_on_this_page:
                continue

            spec = RenderSpec(page=page_obj)

            # Handle cropping
            if crop_bbox:
                spec.crop_bbox = crop_bbox
            elif crop == "content" or crop is True:
                # Calculate bounds of segments on this page
                x_coords = []
                y_coords = []
                for segment in segments_on_this_page:
                    if hasattr(segment, "bbox") and segment.bbox:
                        x0, y0, x1, y1 = segment.bbox
                        x_coords.extend([x0, x1])
                        y_coords.extend([y0, y1])

                if x_coords and y_coords:
                    spec.crop_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

            # Add highlights in show mode
            if mode == "show":
                # Highlight segments
                for i, segment in enumerate(segments_on_this_page):
                    segment_label = None
                    if label_prefix:
                        # Create label for this segment
                        global_segment_idx = None
                        try:
                            # Find the global index of this segment in the original flow
                            global_segment_idx = self.segments.index(segment)
                        except ValueError:
                            # If it's a generated full-page region, find its source page
                            for idx, orig_segment in enumerate(self.segments):
                                if (
                                    hasattr(orig_segment, "index")
                                    and hasattr(segment, "page")
                                    and orig_segment.index == segment.page.index
                                ):
                                    global_segment_idx = idx
                                    break

                        if global_segment_idx is not None:
                            segment_label = f"{label_prefix}_{global_segment_idx + 1}"
                        else:
                            segment_label = f"{label_prefix}_p{page_idx + 1}s{i + 1}"

                    spec.add_highlight(
                        bbox=segment.bbox,
                        polygon=segment.polygon if segment.has_polygon else None,
                        color=color or "blue",
                        label=segment_label,
                    )

                # Add additional highlight groups if provided
                if highlights:
                    for group in highlights:
                        group_elements = group.get("elements", [])
                        group_color = group.get("color", color)
                        group_label = group.get("label")

                        for elem in group_elements:
                            # Only add if element is on this page
                            if hasattr(elem, "page") and elem.page == page_obj:
                                spec.add_highlight(
                                    element=elem, color=group_color, label=group_label
                                )

            specs.append(spec)

        return specs

    def _show_in_context(
        self,
        resolution: float,
        width: Optional[int] = None,
        stack_direction: str = "vertical",
        stack_gap: int = 5,
        stack_background_color: Tuple[int, int, int] = (255, 255, 255),
        separator_color: Tuple[int, int, int] = (255, 0, 0),
        separator_thickness: int = 2,
        **kwargs,
    ) -> Optional["PIL_Image"]:
        """
        Show segments as cropped images stacked together with separators between segments.

        Args:
            resolution: Resolution in DPI for rendering segment images
            width: Optional width for segment images
            stack_direction: Direction to stack segments ('vertical' or 'horizontal')
            stack_gap: Gap in pixels between segments
            stack_background_color: RGB background color for the final image
            separator_color: RGB color for separator lines between segments
            separator_thickness: Thickness in pixels of separator lines
            **kwargs: Additional arguments passed to segment rendering

        Returns:
            PIL Image with all segments stacked together
        """
        from PIL import Image, ImageDraw

        segment_images = []
        segment_pages = []

        # Determine stacking direction
        final_stack_direction = stack_direction
        if stack_direction == "auto":
            final_stack_direction = self.arrangement

        # Get cropped images for each segment
        for i, segment in enumerate(self.segments):
            # Get the page reference for this segment
            if hasattr(segment, "page") and segment.page is not None:
                segment_page = segment.page
                # Get cropped image of the segment
                # Use render() for clean image without highlights
                segment_image = segment.render(
                    resolution=resolution,
                    crop=True,
                    width=width,
                    **kwargs,
                )

            elif (
                hasattr(segment, "index")
                and hasattr(segment, "width")
                and hasattr(segment, "height")
            ):
                # It's a full Page object
                segment_page = segment
                # Use render() for clean image without highlights
                segment_image = segment.render(resolution=resolution, width=width, **kwargs)
            else:
                raise ValueError(
                    f"Segment {i+1} has no identifiable page. Segment type: {type(segment)}, attributes: {dir(segment)}"
                )

            if segment_image is not None:
                segment_images.append(segment_image)
                segment_pages.append(segment_page)
            else:
                logger.warning(f"Segment {i+1} render() returned None, skipping")

        # Check if we have any valid images
        if not segment_images:
            logger.error("No valid segment images could be rendered")
            return None

        # We should have at least one segment image by now (or an exception would have been raised)
        if len(segment_images) == 1:
            return segment_images[0]

        # Calculate dimensions for the final stacked image
        if final_stack_direction == "vertical":
            # Stack vertically
            final_width = max(img.width for img in segment_images)

            # Calculate total height including gaps and separators
            total_height = sum(img.height for img in segment_images)
            total_height += (len(segment_images) - 1) * stack_gap

            # Add separator thickness between all segments
            num_separators = len(segment_images) - 1 if len(segment_images) > 1 else 0
            total_height += num_separators * separator_thickness

            # Create the final image
            final_image = Image.new("RGB", (final_width, total_height), stack_background_color)
            draw = ImageDraw.Draw(final_image)

            current_y = 0

            for i, img in enumerate(segment_images):
                # Add separator line before each segment (except the first one)
                if i > 0:
                    # Draw separator line
                    draw.rectangle(
                        [(0, current_y), (final_width, current_y + separator_thickness)],
                        fill=separator_color,
                    )
                    current_y += separator_thickness

                # Paste the segment image
                paste_x = (final_width - img.width) // 2  # Center horizontally
                final_image.paste(img, (paste_x, current_y))
                current_y += img.height

                # Add gap after segment (except for the last one)
                if i < len(segment_images) - 1:
                    current_y += stack_gap

            return final_image

        elif final_stack_direction == "horizontal":
            # Stack horizontally
            final_height = max(img.height for img in segment_images)

            # Calculate total width including gaps and separators
            total_width = sum(img.width for img in segment_images)
            total_width += (len(segment_images) - 1) * stack_gap

            # Add separator thickness between all segments
            num_separators = len(segment_images) - 1 if len(segment_images) > 1 else 0
            total_width += num_separators * separator_thickness

            # Create the final image
            final_image = Image.new("RGB", (total_width, final_height), stack_background_color)
            draw = ImageDraw.Draw(final_image)

            current_x = 0

            for i, img in enumerate(segment_images):
                # Add separator line before each segment (except the first one)
                if i > 0:
                    # Draw separator line
                    draw.rectangle(
                        [(current_x, 0), (current_x + separator_thickness, final_height)],
                        fill=separator_color,
                    )
                    current_x += separator_thickness

                # Paste the segment image
                paste_y = (final_height - img.height) // 2  # Center vertically
                final_image.paste(img, (current_x, paste_y))
                current_x += img.width

                # Add gap after segment (except for the last one)
                if i < len(segment_images) - 1:
                    current_x += stack_gap

            return final_image

        else:
            raise ValueError(
                f"Invalid stack_direction '{final_stack_direction}' for in_context. Must be 'vertical' or 'horizontal'."
            )

    # --- Helper methods for coordinate transformations and segment iteration ---
    # These will be crucial for FlowElement's directional methods.

    def get_segment_bounding_box_in_flow(
        self, segment_index: int
    ) -> Optional[tuple[float, float, float, float]]:
        """
        Calculates the conceptual bounding box of a segment within the flow's coordinate system.
        This considers arrangement, alignment, and segment gaps.
        (This is a placeholder for more complex logic if a true virtual coordinate system is needed)
        For now, it might just return the physical segment's bbox if gaps are 0 and alignment is simple.
        """
        if segment_index < 0 or segment_index >= len(self.segments):
            return None

        # This is a simplified version. A full implementation would calculate offsets.
        # For now, we assume FlowElement directional logic handles segment traversal and uses physical coords.
        # If we were to *draw* the flow or get a FlowRegion bbox that spans gaps, this would be critical.
        # physical_segment = self.segments[segment_index]
        # return physical_segment.bbox
        raise NotImplementedError(
            "Calculating a segment's bbox *within the flow's virtual coordinate system* is not yet fully implemented."
        )

    def get_element_flow_coordinates(
        self, physical_element: "PhysicalElement"
    ) -> Optional[tuple[float, float, float, float]]:
        """
        Translates a physical element's coordinates into the flow's virtual coordinate system.
        (Placeholder - very complex if segment_gap > 0 or complex alignments)
        """
        # For now, elements operate in their own physical coordinates. This method would be needed
        # if FlowRegion.bbox or other operations needed to present a unified coordinate space.
        # As per our discussion, elements *within* a FlowRegion retain original physical coordinates.
        # So, this might not be strictly necessary for the current design's core functionality.
        raise NotImplementedError(
            "Translating element coordinates to a unified flow coordinate system is not yet implemented."
        )

    def get_sections(
        self,
        start_elements=None,
        end_elements=None,
        new_section_on_page_break: bool = False,
        include_boundaries: str = "both",
        orientation: str = "vertical",
    ) -> "ElementCollection":
        """
        Extract logical sections from the Flow based on *start* and *end* boundary
        elements, mirroring the behaviour of PDF/PageCollection.get_sections().

        This implementation is a thin wrapper that converts the Flow into a
        temporary PageCollection (constructed from the unique pages that the
        Flow spans) and then delegates the heavy‐lifting to that existing
        implementation.  Any FlowElement / FlowElementCollection inputs are
        automatically unwrapped to their underlying physical elements so that
        PageCollection can work with them directly.

        Args:
            start_elements: Elements or selector string that mark the start of
                sections (optional).
            end_elements: Elements or selector string that mark the end of
                sections (optional).
            new_section_on_page_break: Whether to start a new section at page
                boundaries (default: False).
            include_boundaries: How to include boundary elements: 'start',
                'end', 'both', or 'none' (default: 'both').
            orientation: 'vertical' (default) or 'horizontal' - determines section direction.

        Returns:
            ElementCollection of Region/FlowRegion objects representing the
            extracted sections.
        """
        # ------------------------------------------------------------------
        # Unwrap FlowElement(-Collection) inputs and selector strings so we
        # can reason about them generically.
        # ------------------------------------------------------------------
        from natural_pdf.flows.collections import FlowElementCollection
        from natural_pdf.flows.element import FlowElement

        def _unwrap(obj):
            """Convert Flow-specific wrappers to their underlying physical objects.

            Keeps selector strings as-is; converts FlowElement to its physical
            element; converts FlowElementCollection to list of physical
            elements; passes through ElementCollection by taking .elements.
            """

            if obj is None or isinstance(obj, str):
                return obj

            if isinstance(obj, FlowElement):
                return obj.physical_object

            if isinstance(obj, FlowElementCollection):
                return [fe.physical_object for fe in obj.flow_elements]

            if hasattr(obj, "elements"):
                return obj.elements

            if isinstance(obj, (list, tuple, set)):
                out = []
                for item in obj:
                    if isinstance(item, FlowElement):
                        out.append(item.physical_object)
                    else:
                        out.append(item)
                return out

            return obj  # Fallback – unknown type

        start_elements_unwrapped = _unwrap(start_elements)
        end_elements_unwrapped = _unwrap(end_elements)

        # ------------------------------------------------------------------
        # For Flow, we need to handle sections that may span segments
        # We'll process all segments together, not independently
        # ------------------------------------------------------------------
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.elements.region import Region
        from natural_pdf.flows.element import FlowElement
        from natural_pdf.flows.region import FlowRegion

        # Helper to check if element is in segment
        def _element_in_segment(elem, segment):
            # Simple bbox check
            return (
                elem.page == segment.page
                and elem.top >= segment.top
                and elem.bottom <= segment.bottom
                and elem.x0 >= segment.x0
                and elem.x1 <= segment.x1
            )

        # Collect all boundary elements with their segment info
        all_starts = []
        all_ends = []

        for seg_idx, segment in enumerate(self.segments):
            # Find starts in this segment
            if isinstance(start_elements_unwrapped, str):
                seg_starts = segment.find_all(start_elements_unwrapped).elements
            elif start_elements_unwrapped:
                seg_starts = [
                    e for e in start_elements_unwrapped if _element_in_segment(e, segment)
                ]
            else:
                seg_starts = []

            for elem in seg_starts:
                all_starts.append((elem, seg_idx, segment))

            # Find ends in this segment
            if isinstance(end_elements_unwrapped, str):
                seg_ends = segment.find_all(end_elements_unwrapped).elements
            elif end_elements_unwrapped:
                seg_ends = [e for e in end_elements_unwrapped if _element_in_segment(e, segment)]
            else:
                seg_ends = []

            for elem in seg_ends:
                all_ends.append((elem, seg_idx, segment))

        # Sort by segment index, then position
        all_starts.sort(key=lambda x: (x[1], x[0].top, x[0].x0))
        all_ends.sort(key=lambda x: (x[1], x[0].top, x[0].x0))

        # If no boundary elements found, return empty collection
        if not all_starts and not all_ends:
            return ElementCollection([])

        sections = []

        # Case 1: Only start elements provided
        if all_starts and not all_ends:
            for i in range(len(all_starts)):
                start_elem, start_seg_idx, start_seg = all_starts[i]

                # Find end (next start or end of flow)
                if i + 1 < len(all_starts):
                    # Section ends at next start
                    end_elem, end_seg_idx, end_seg = all_starts[i + 1]

                    if start_seg_idx == end_seg_idx:
                        # Same segment - create regular Region
                        section = start_seg.get_section_between(
                            start_elem, end_elem, include_boundaries, orientation
                        )
                        if section:
                            sections.append(section)
                    else:
                        # Cross-segment - create FlowRegion
                        regions = []

                        # First segment: from start to bottom
                        if include_boundaries in ["both", "start"]:
                            top = start_elem.top
                        else:
                            top = start_elem.bottom
                        regions.append(
                            Region(
                                start_seg.page, (start_seg.x0, top, start_seg.x1, start_seg.bottom)
                            )
                        )

                        # Middle segments (full)
                        for idx in range(start_seg_idx + 1, end_seg_idx):
                            regions.append(self.segments[idx])

                        # Last segment: from top to end element
                        if include_boundaries in ["both", "end"]:
                            bottom = end_elem.bottom
                        else:
                            bottom = end_elem.top
                        regions.append(
                            Region(end_seg.page, (end_seg.x0, end_seg.top, end_seg.x1, bottom))
                        )

                        # Create FlowRegion
                        flow_element = FlowElement(physical_object=start_elem, flow=self)
                        flow_region = FlowRegion(
                            flow=self,
                            constituent_regions=regions,
                            source_flow_element=flow_element,
                            boundary_element_found=end_elem,
                        )
                        flow_region.start_element = start_elem
                        flow_region.end_element = end_elem
                        flow_region._boundary_exclusions = include_boundaries
                        sections.append(flow_region)
                else:
                    # Last section - goes to end of flow
                    if start_seg_idx == len(self.segments) - 1:
                        # Within last segment
                        section = start_seg.get_section_between(
                            start_elem, None, include_boundaries, orientation
                        )
                        if section:
                            sections.append(section)
                    else:
                        # Spans to end
                        regions = []

                        # First segment: from start to bottom
                        if include_boundaries in ["both", "start"]:
                            top = start_elem.top
                        else:
                            top = start_elem.bottom
                        regions.append(
                            Region(
                                start_seg.page, (start_seg.x0, top, start_seg.x1, start_seg.bottom)
                            )
                        )

                        # Remaining segments (full)
                        for idx in range(start_seg_idx + 1, len(self.segments)):
                            regions.append(self.segments[idx])

                        # Create FlowRegion
                        flow_element = FlowElement(physical_object=start_elem, flow=self)
                        flow_region = FlowRegion(
                            flow=self,
                            constituent_regions=regions,
                            source_flow_element=flow_element,
                            boundary_element_found=None,
                        )
                        flow_region.start_element = start_elem
                        flow_region._boundary_exclusions = include_boundaries
                        sections.append(flow_region)

        # Case 2: Both start and end elements
        elif all_starts and all_ends:
            # Match starts with ends
            used_ends = set()

            for start_elem, start_seg_idx, start_seg in all_starts:
                # Find matching end
                best_end = None

                for end_elem, end_seg_idx, end_seg in all_ends:
                    if id(end_elem) in used_ends:
                        continue

                    # End must come after start
                    if end_seg_idx > start_seg_idx or (
                        end_seg_idx == start_seg_idx and end_elem.top >= start_elem.bottom
                    ):
                        best_end = (end_elem, end_seg_idx, end_seg)
                        break

                if best_end:
                    end_elem, end_seg_idx, end_seg = best_end
                    used_ends.add(id(end_elem))

                    if start_seg_idx == end_seg_idx:
                        # Same segment
                        section = start_seg.get_section_between(
                            start_elem, end_elem, include_boundaries, orientation
                        )
                        if section:
                            sections.append(section)
                    else:
                        # Cross-segment FlowRegion
                        regions = []

                        # First segment
                        if include_boundaries in ["both", "start"]:
                            top = start_elem.top
                        else:
                            top = start_elem.bottom
                        regions.append(
                            Region(
                                start_seg.page, (start_seg.x0, top, start_seg.x1, start_seg.bottom)
                            )
                        )

                        # Middle segments
                        for idx in range(start_seg_idx + 1, end_seg_idx):
                            regions.append(self.segments[idx])

                        # Last segment
                        if include_boundaries in ["both", "end"]:
                            bottom = end_elem.bottom
                        else:
                            bottom = end_elem.top
                        regions.append(
                            Region(end_seg.page, (end_seg.x0, end_seg.top, end_seg.x1, bottom))
                        )

                        # Create FlowRegion
                        flow_element = FlowElement(physical_object=start_elem, flow=self)
                        flow_region = FlowRegion(
                            flow=self,
                            constituent_regions=regions,
                            source_flow_element=flow_element,
                            boundary_element_found=end_elem,
                        )
                        flow_region.start_element = start_elem
                        flow_region.end_element = end_elem
                        flow_region._boundary_exclusions = include_boundaries
                        sections.append(flow_region)

        # Case 3: Only end elements (sections from beginning to each end)
        elif not all_starts and all_ends:
            # TODO: Handle this case if needed
            pass

        return ElementCollection(sections)

    def highlights(self, show: bool = False):
        """
        Create a highlight context for accumulating highlights.

        This allows for clean syntax to show multiple highlight groups:

        Example:
            with flow.highlights() as h:
                h.add(flow.find_all('table'), label='tables', color='blue')
                h.add(flow.find_all('text:bold'), label='bold text', color='red')
                h.show()

        Or with automatic display:
            with flow.highlights(show=True) as h:
                h.add(flow.find_all('table'), label='tables')
                h.add(flow.find_all('text:bold'), label='bold')
                # Automatically shows when exiting the context

        Args:
            show: If True, automatically show highlights when exiting context

        Returns:
            HighlightContext for accumulating highlights
        """
        from natural_pdf.core.highlighting_service import HighlightContext

        return HighlightContext(self, show_on_exit=show)
