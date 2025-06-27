import logging
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element as PhysicalElement
    from natural_pdf.elements.collections import ElementCollection as PhysicalElementCollection
    from natural_pdf.elements.region import Region as PhysicalRegion

    from .collections import FlowElementCollection
    from .element import FlowElement

logger = logging.getLogger(__name__)


class Flow:
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
        segments: List[Union["Page", "PhysicalRegion"]],
        arrangement: Literal["vertical", "horizontal"],
        alignment: Literal["start", "center", "end", "top", "left", "bottom", "right"] = "start",
        segment_gap: float = 0.0,
    ):
        """
        Initializes a Flow object.

        Args:
            segments: An ordered list of natural_pdf.core.page.Page or
                      natural_pdf.elements.region.Region objects that constitute the flow.
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
        Elements are collected segment by segment, preserving the flow order.

        Elements found are wrapped as FlowElement objects, anchored to this Flow,
        and returned in a FlowElementCollection.
        """
        from .collections import FlowElementCollection
        from .element import FlowElement

        all_flow_elements: List["FlowElement"] = []

        # Iterate through segments in their defined flow order
        for physical_segment in self.segments:
            # Find all matching physical elements within the current segment
            # Region.find_all() should return elements in local reading order.
            matches_in_segment: "PhysicalElementCollection" = physical_segment.find_all(
                selector=selector,
                text=text,
                apply_exclusions=apply_exclusions,
                regex=regex,
                case=case,
                **kwargs,
            )
            if matches_in_segment:
                # Wrap each found physical element as a FlowElement and add to the list
                # This preserves the order from matches_in_segment.elements
                for phys_elem in matches_in_segment.elements:
                    all_flow_elements.append(FlowElement(physical_object=phys_elem, flow=self))

        # The global sort that was here previously has been removed.
        # The order is now determined by segment sequence, then by local order within each segment.

        return FlowElementCollection(all_flow_elements)

    def __repr__(self) -> str:
        return (
            f"<Flow segments={len(self.segments)}, "
            f"arrangement='{self.arrangement}', alignment='{self.alignment}', gap={self.segment_gap}>"
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
