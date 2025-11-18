import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
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

# Import required classes for the new methods
# For runtime image manipulation

from natural_pdf.core.context import PDFContext
from natural_pdf.core.highlighter_utils import resolve_highlighter
from natural_pdf.core.interfaces import SupportsSections
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.flows.collections import FlowElementCollection
from natural_pdf.flows.element import FlowElement
from natural_pdf.flows.region import FlowRegion
from natural_pdf.selectors.host_mixin import SelectorHostMixin, delegate_signature
from natural_pdf.services.base import ServiceHostMixin, resolve_service
from natural_pdf.services.methods import flow_selector_methods as _flow_selector_methods
from natural_pdf.services.methods import flow_table_methods as _flow_table_methods
from natural_pdf.services.methods import layout_methods as _layout_methods
from natural_pdf.services.methods import qa_methods as _qa_methods
from natural_pdf.tables import TableResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from natural_pdf.elements.base import Element as SelectorElement
    from natural_pdf.elements.element_collection import ElementCollection as SelectorCollection
else:  # pragma: no cover - runtime aliases for flow-specific selectors
    SelectorElement = FlowElement  # type: ignore[assignment]
    SelectorCollection = FlowElementCollection  # type: ignore[assignment]


class Flow(ServiceHostMixin, Visualizable, SelectorHostMixin):
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
        segments: Union[Sequence[SupportsSections], "PageCollection"],
        arrangement: Literal["vertical", "horizontal"],
        alignment: Literal["start", "center", "end", "top", "left", "bottom", "right"] = "start",
        segment_gap: float = 0.0,
    ):
        """
        Initializes a Flow object.

        Args:
            segments: An ordered sequence of objects implementing SupportsSections (e.g., Page,
                      Region) that constitute the flow, or a PageCollection containing pages.
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
        from natural_pdf.core.page_collection import PageCollection as _PageCollection

        if isinstance(segments, _PageCollection):
            segment_list: List[SupportsSections] = list(segments.pages)
        else:
            segment_list = list(segments)

        if not segment_list:
            raise ValueError("Flow segments cannot be empty.")
        if arrangement not in ["vertical", "horizontal"]:
            raise ValueError("Arrangement must be 'vertical' or 'horizontal'.")

        self.segments: List["PhysicalRegion"] = self._normalize_segments(segment_list)
        self.arrangement: Literal["vertical", "horizontal"] = arrangement
        self.alignment: Literal["start", "center", "end", "top", "left", "bottom", "right"] = (
            alignment
        )
        self.segment_gap: float = segment_gap
        self._analysis_region_cache: Optional["FlowRegion"] = None

        self._bind_service_context(self.segments)

        self._validate_alignment()

        # TODO: Pre-calculate segment offsets for faster lookups if needed

    def _normalize_segments(self, segments: Sequence[SupportsSections]) -> List["PhysicalRegion"]:
        """Materialize all segments as Region objects for uniform processing."""
        normalized: List["PhysicalRegion"] = []
        from natural_pdf.elements.region import Region as ElementsRegion

        for index, segment in enumerate(segments):
            if not isinstance(segment, SupportsSections):
                raise TypeError(
                    f"Segment {index} must implement SupportsSections; found {type(segment)}."
                )

            region_candidate = segment.to_region()
            if not isinstance(region_candidate, ElementsRegion):
                raise TypeError(
                    f"Segment {index} returned unsupported region type {type(region_candidate)}."
                )
            normalized.append(region_candidate)
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

    def _analysis_region(self) -> "FlowRegion":
        """
        Create (and cache) a FlowRegion representing the entire flow for analysis tasks.
        """

        if self._analysis_region_cache is None:
            from natural_pdf.flows.region import FlowRegion

            self._analysis_region_cache = FlowRegion(
                flow=self,
                constituent_regions=list(self.segments),
            )
        return self._analysis_region_cache

    # ------------------------------------------------------------------
    # Context + capability helpers
    # ------------------------------------------------------------------
    def _bind_service_context(self, segments: Sequence["PhysicalRegion"]) -> None:
        context = self._resolve_service_context(segments)
        self._init_service_host(context)

    def _resolve_service_context(self, segments: Sequence["PhysicalRegion"]) -> PDFContext:
        for segment in segments:
            context = getattr(segment, "_context", None)
            if context is not None:
                return context
            page = getattr(segment, "page", None)
            if page is not None:
                page_context = getattr(page, "_context", None)
                if page_context is not None:
                    return page_context
                pdf_obj = getattr(page, "pdf", getattr(page, "_parent", None))
                if pdf_obj is not None:
                    pdf_context = getattr(pdf_obj, "_context", None)
                    if pdf_context is not None:
                        return pdf_context
        return PDFContext.with_defaults()

    def _analysis_region_host(self):
        return self._analysis_region()

    def _ocr_element_manager(self):
        return self._analysis_region_host()._ocr_element_manager()

    def _qa_segments(self):
        return self._analysis_region_host()._qa_segments()

    def _qa_target_region(self):
        return self._analysis_region_host()._qa_target_region()

    def _qa_context_page_number(self) -> int:
        return self._analysis_region_host()._qa_context_page_number()

    def _qa_source_elements(self):
        return self._analysis_region_host()._qa_source_elements()

    def _qa_normalize_result(self, result: Any):
        return self._analysis_region_host()._qa_normalize_result(result)

    def _qa_confidence(self, candidate: Any) -> float:
        return self._analysis_region_host()._qa_confidence(candidate)

    def _get_highlighter(self):
        """Get the highlighting service from the first segment."""
        if not self.segments:
            raise RuntimeError("Flow has no segments to get highlighter from")

        return resolve_highlighter(self.segments[0])

    def show(
        self,
        *,
        resolution: Optional[float] = None,
        width: Optional[int] = None,
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        labels: bool = True,
        label_format: Optional[str] = None,
        highlights: Optional[Union[List[Dict[str, Any]], bool]] = None,
        legend_position: str = "right",
        annotate: Optional[Union[str, List[str]]] = None,
        layout: Optional[Literal["stack", "grid", "single"]] = None,
        stack_direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 5,
        columns: Optional[int] = 6,
        crop: Union[bool, int, str, "PhysicalRegion", Literal["wide"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
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
        resolved_resolution = self._resolve_image_resolution(resolution)

        if in_context:
            # Use the special in_context visualization
            return self._show_in_context(
                resolution=resolved_resolution,
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
            legend_position=legend_position,
            annotate=annotate,
            layout=layout,
            stack_direction=stack_direction,
            gap=gap,
            columns=columns,
            crop=crop,
            crop_bbox=crop_bbox,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Analysis helpers (delegated to a FlowRegion spanning all segments)
    # ------------------------------------------------------------------

    def apply_ocr(self, *args: Any, **kwargs: Any) -> "Flow":
        """
        Apply OCR across every segment in the flow.
        """

        self._analysis_region().apply_ocr(*args, **kwargs)
        return self

    def extract_ocr_elements(self, *args: Any, **kwargs: Any) -> List[Any]:
        """
        Extract OCR-derived text elements from all segments.
        """

        return self._analysis_region().extract_ocr_elements(*args, **kwargs)

    def remove_ocr_elements(self) -> int:
        """
        Remove OCR elements that were previously added to constituent pages.
        """

        return self._analysis_region().remove_ocr_elements()

    def clear_text_layer(self) -> Tuple[int, int]:
        """
        Clear the underlying text layers (words/chars) for every segment page.
        """

        return self._analysis_region().clear_text_layer()

    def create_text_elements_from_ocr(
        self,
        ocr_results: Any,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
    ) -> List[Any]:
        """
        Utility for constructing text elements from OCR output.
        """

        return self._analysis_region().create_text_elements_from_ocr(
            ocr_results,
            scale_x=scale_x,
            scale_y=scale_y,
        )

    @delegate_signature(_qa_methods.ask)
    def ask(self, *args, **kwargs):
        return _qa_methods.ask(self, *args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<Flow segments={len(self.segments)}, "
            f"arrangement='{self.arrangement}', alignment='{self.alignment}', gap={self.segment_gap}>"
        )

    @delegate_signature(_flow_table_methods.flow_extract_table)
    def extract_table(self, *args, **kwargs) -> TableResult:
        return _flow_table_methods.flow_extract_table(self, *args, **kwargs)

    @delegate_signature(_flow_table_methods.flow_extract_tables)
    def extract_tables(self, *args, **kwargs) -> List[List[List[Optional[str]]]]:
        return _flow_table_methods.flow_extract_tables(self, *args, **kwargs)

    @delegate_signature(_layout_methods.analyze_layout)
    def analyze_layout(self, *args, **kwargs):
        return _layout_methods.analyze_layout(self, *args, **kwargs)

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
                raise RuntimeError(
                    f"No segments recorded for page {getattr(page_obj, 'number', '?')}"
                )

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
                            seg_page = getattr(segment, "page", None)
                            if seg_page is not None:
                                for idx, orig_segment in enumerate(self.segments):
                                    if getattr(orig_segment, "page", None) is seg_page:
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

            if segment_image is None:
                raise RuntimeError(f"Segment {i+1} render() returned None")

            segment_images.append(segment_image)
            segment_pages.append(segment_page)

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
    ) -> "PhysicalElementCollection":
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
                if isinstance(start_elements_unwrapped, Iterable):
                    candidates = list(start_elements_unwrapped)
                else:
                    candidates = [start_elements_unwrapped]
                seg_starts = [e for e in candidates if _element_in_segment(e, segment)]
            else:
                seg_starts = []

            for elem in seg_starts:
                all_starts.append((elem, seg_idx, segment))

            # Find ends in this segment
            if isinstance(end_elements_unwrapped, str):
                seg_ends = segment.find_all(end_elements_unwrapped).elements
            elif end_elements_unwrapped:
                if isinstance(end_elements_unwrapped, Iterable):
                    candidates_end = list(end_elements_unwrapped)
                else:
                    candidates_end = [end_elements_unwrapped]
                seg_ends = [e for e in candidates_end if _element_in_segment(e, segment)]
            else:
                seg_ends = []

            for elem in seg_ends:
                all_ends.append((elem, seg_idx, segment))

        # Sort by segment index, then position
        all_starts.sort(key=lambda x: (x[1], x[0].top, x[0].x0))
        all_ends.sort(key=lambda x: (x[1], x[0].top, x[0].x0))

        # When only end elements are supplied we synthesise implicit start
        # markers so we can reuse the "start-only" logic below.  This mirrors
        # the historical behaviour of PageCollection.get_sections which
        # created implicit starts at the top of the document and immediately
        # after each end boundary.
        if not all_starts and all_ends:

            def _create_implicit(segment, position):
                thickness = 0.1
                if orientation == "vertical":
                    top = max(segment.top, min(position, segment.bottom))
                    bottom = min(segment.bottom, top + thickness)
                    if bottom <= top:
                        bottom = min(segment.bottom, top + thickness)
                        if bottom <= top:
                            bottom = top
                    implicit_region = Region(segment.page, (segment.x0, top, segment.x1, bottom))
                else:
                    left = max(segment.x0, min(position, segment.x1))
                    right = min(segment.x1, left + thickness)
                    if right <= left:
                        right = min(segment.x1, left + thickness)
                        if right <= left:
                            right = left
                    implicit_region = Region(
                        segment.page, (left, segment.top, right, segment.bottom)
                    )

                implicit_region.metadata["is_implicit_start"] = True
                return implicit_region

            synthetic_starts: List[Tuple[Region, int, Region]] = []

            first_segment = self.segments[0]
            initial_position = first_segment.top if orientation == "vertical" else first_segment.x0
            synthetic_starts.append(
                (
                    _create_implicit(first_segment, initial_position),
                    0,
                    first_segment,
                )
            )

            seen_end_ids = set()
            for end_elem, end_seg_idx, _ in all_ends:
                if id(end_elem) in seen_end_ids:
                    continue
                seen_end_ids.add(id(end_elem))

                position = end_elem.bottom if orientation == "vertical" else end_elem.x1
                if include_boundaries in ["start", "none"]:
                    position = end_elem.top if orientation == "vertical" else end_elem.x0

                segment = self.segments[end_seg_idx]
                synthetic_starts.append(
                    (
                        _create_implicit(segment, position),
                        end_seg_idx,
                        segment,
                    )
                )

            all_starts = synthetic_starts
            all_ends = []

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

        # Case 3 is handled by synthesising implicit start elements above.
        elif not all_starts and all_ends:
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
