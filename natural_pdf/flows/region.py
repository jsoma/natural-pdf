import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable

from pdfplumber.utils.geometry import objects_to_bbox  # For calculating combined bbox

# For runtime image manipulation
from PIL import Image as PIL_Image_Runtime

if TYPE_CHECKING:
    from PIL.Image import Image as PIL_Image  # For type hints

    from natural_pdf.core.page import Page as PhysicalPage
    from natural_pdf.elements.base import Element as PhysicalElement
    from natural_pdf.elements.collections import ElementCollection
    from natural_pdf.elements.region import Region as PhysicalRegion

    from .element import FlowElement
    from .flow import Flow

logger = logging.getLogger(__name__)


class FlowRegion:
    """
    Represents a selected area within a Flow, potentially composed of multiple
    physical Region objects (constituent_regions) that might span across
    different original pages or disjoint physical regions defined in the Flow.

    A FlowRegion is the result of a directional operation (e.g., .below(), .above())
    on a FlowElement.
    """

    def __init__(
        self,
        flow: "Flow",
        constituent_regions: List["PhysicalRegion"],
        source_flow_element: "FlowElement",
        boundary_element_found: Optional["PhysicalElement"] = None,
    ):
        """
        Initializes a FlowRegion.

        Args:
            flow: The Flow instance this region belongs to.
            constituent_regions: A list of physical natural_pdf.elements.region.Region
                                 objects that make up this FlowRegion.
            source_flow_element: The FlowElement that created this FlowRegion.
            boundary_element_found: The physical element that stopped an 'until' search,
                                    if applicable.
        """
        self.flow: "Flow" = flow
        self.constituent_regions: List["PhysicalRegion"] = constituent_regions
        self.source_flow_element: "FlowElement" = source_flow_element
        self.boundary_element_found: Optional["PhysicalElement"] = boundary_element_found

        # Cache for expensive operations
        self._cached_text: Optional[str] = None
        self._cached_elements: Optional["ElementCollection"] = None  # Stringized
        self._cached_bbox: Optional[Tuple[float, float, float, float]] = None

    @property
    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates a conceptual bounding box that encompasses all constituent physical regions.
        This is the union of the bounding boxes of the constituent regions in their
        original physical coordinates.
        Returns None if there are no constituent regions.
        """
        if self._cached_bbox is not None:
            return self._cached_bbox
        if not self.constituent_regions:
            return None

        # Use objects_to_bbox from pdfplumber.utils.geometry to merge bboxes
        # This helper expects a list of objects that have .x0, .top, .x1, .bottom attributes.
        # Our PhysicalRegion objects satisfy this.
        self._cached_bbox = objects_to_bbox(self.constituent_regions)
        return self._cached_bbox

    @property
    def x0(self) -> Optional[float]:
        return self.bbox[0] if self.bbox else None

    @property
    def top(self) -> Optional[float]:
        return self.bbox[1] if self.bbox else None

    @property
    def x1(self) -> Optional[float]:
        return self.bbox[2] if self.bbox else None

    @property
    def bottom(self) -> Optional[float]:
        return self.bbox[3] if self.bbox else None

    @property
    def width(self) -> Optional[float]:
        return self.x1 - self.x0 if self.bbox else None

    @property
    def height(self) -> Optional[float]:
        return self.bottom - self.top if self.bbox else None

    def extract_text(self, apply_exclusions: bool = True, **kwargs) -> str:
        """
        Extracts and concatenates text from all constituent physical regions.
        The order of concatenation respects the flow's arrangement.

        Args:
            apply_exclusions: Whether to respect PDF exclusion zones within each
                              constituent physical region during text extraction.
            **kwargs: Additional arguments passed to the underlying extract_text method
                      of each constituent region.

        Returns:
            The combined text content as a string.
        """
        if (
            self._cached_text is not None and apply_exclusions
        ):  # Simple cache check, might need refinement if kwargs change behavior
            return self._cached_text

        if not self.constituent_regions:
            return ""

        texts: List[str] = []
        # For now, simple concatenation. Order depends on how constituent_regions were added.
        # The FlowElement._flow_direction method is responsible for ordering constituent_regions correctly.
        for region in self.constituent_regions:
            texts.append(region.extract_text(apply_exclusions=apply_exclusions, **kwargs))

        # Join based on flow arrangement (e.g., newline for vertical, space for horizontal)
        # This is a simplification; true layout-aware joining would be more complex.
        joiner = (
            "\n" if self.flow.arrangement == "vertical" else " "
        )  # TODO: Make this smarter, consider segment_gap
        extracted = joiner.join(t for t in texts if t)

        if apply_exclusions:  # Only cache if standard exclusion behavior
            self._cached_text = extracted
        return extracted

    def elements(self, apply_exclusions: bool = True) -> "ElementCollection":  # Stringized return
        """
        Collects all unique physical elements from all constituent physical regions.

        Args:
            apply_exclusions: Whether to respect PDF exclusion zones within each
                              constituent physical region when gathering elements.

        Returns:
            An ElementCollection containing all unique elements.
        """
        from natural_pdf.elements.collections import (
            ElementCollection as RuntimeElementCollection,  # Local import
        )

        if self._cached_elements is not None and apply_exclusions:  # Simple cache check
            return self._cached_elements

        if not self.constituent_regions:
            return RuntimeElementCollection([])

        all_physical_elements: List["PhysicalElement"] = []  # Stringized item type
        seen_elements = (
            set()
        )  # To ensure uniqueness if elements are shared or duplicated by region definitions

        for region in self.constituent_regions:
            # Region.get_elements() returns a list, not ElementCollection
            elements_in_region: List["PhysicalElement"] = region.get_elements(
                apply_exclusions=apply_exclusions
            )
            for elem in elements_in_region:
                if elem not in seen_elements:  # Check for uniqueness based on object identity
                    all_physical_elements.append(elem)
                    seen_elements.add(elem)

        # Basic reading order sort based on original page and coordinates.
        def get_sort_key(phys_elem: "PhysicalElement"):  # Stringized param type
            page_idx = -1
            if hasattr(phys_elem, "page") and hasattr(phys_elem.page, "index"):
                page_idx = phys_elem.page.index
            return (page_idx, phys_elem.top, phys_elem.x0)

        try:
            sorted_physical_elements = sorted(all_physical_elements, key=get_sort_key)
        except AttributeError:
            logger.warning(
                "Could not sort elements in FlowRegion by reading order; some elements might be missing page, top or x0 attributes."
            )
            sorted_physical_elements = all_physical_elements

        result_collection = RuntimeElementCollection(sorted_physical_elements)
        if apply_exclusions:
            self._cached_elements = result_collection
        return result_collection

    def find(
        self, selector: Optional[str] = None, *, text: Optional[str] = None, **kwargs
    ) -> Optional["PhysicalElement"]:  # Stringized
        """
        Finds the first physical element within this FlowRegion that matches the selector or text.
        """
        # Uses self.elements() which respects exclusions if apply_exclusions=True by default
        all_elems = self.elements(apply_exclusions=kwargs.get("apply_exclusions", True))
        return all_elems.find(selector=selector, text=text, **kwargs)  # ElementCollection.find

    def find_all(
        self, selector: Optional[str] = None, *, text: Optional[str] = None, **kwargs
    ) -> "ElementCollection":  # Stringized
        """
        Finds all physical elements within this FlowRegion that match the selector or text.
        """
        all_elems = self.elements(apply_exclusions=kwargs.get("apply_exclusions", True))
        return all_elems.find_all(
            selector=selector, text=text, **kwargs
        )  # ElementCollection.find_all

    def highlight(
        self, label: Optional[str] = None, color: Optional[Union[Tuple, str]] = None, **kwargs
    ) -> "FlowRegion":  # Stringized
        """
        Highlights all constituent physical regions on their respective pages.

        Args:
            label: A base label for the highlights. Each constituent region might get an indexed label.
            color: Color for the highlight.
            **kwargs: Additional arguments for the underlying highlight method.

        Returns:
            Self for method chaining.
        """
        if not self.constituent_regions:
            return self

        base_label = label if label else "FlowRegionPart"
        for i, region in enumerate(self.constituent_regions):
            current_label = (
                f"{base_label}_{i+1}" if len(self.constituent_regions) > 1 else base_label
            )
            region.highlight(label=current_label, color=color, **kwargs)
        return self

    def show(
        self,
        scale: float = 2.0,
        labels: bool = True,
        legend_position: str = "right",
        color: Optional[Union[Tuple, str]] = "fuchsia",
        label_prefix: Optional[str] = "FlowPart",
        width: Optional[int] = None,
        stack_direction: str = "vertical",
        stack_gap: int = 5,
        stack_background_color: Tuple[int, int, int] = (255, 255, 255),
        **kwargs,
    ) -> Optional["PIL_Image"]:
        """
        Generates and returns a PIL Image of relevant pages with constituent regions highlighted.
        If multiple pages are involved, they are stacked into a single image.
        """
        if not self.constituent_regions:
            logger.info("FlowRegion.show() called with no constituent regions.")
            return None

        # 1. Group constituent regions by their physical page
        regions_by_page: Dict["PhysicalPage", List["PhysicalRegion"]] = {}
        for region in self.constituent_regions:
            if region.page:
                if region.page not in regions_by_page:
                    regions_by_page[region.page] = []
                regions_by_page[region.page].append(region)
            else:
                raise ValueError(f"Constituent region {region.bbox} has no page.")

        if not regions_by_page:
            logger.info("FlowRegion.show() found no constituent regions with associated pages.")
            return None

        # 2. Get a highlighter service (e.g., from the first page involved)
        first_page_with_regions = next(iter(regions_by_page.keys()), None)
        highlighter_service = None
        if first_page_with_regions and hasattr(first_page_with_regions, "_highlighter"):
            highlighter_service = first_page_with_regions._highlighter

        if not highlighter_service:
            raise ValueError(
                "Cannot get highlighter service for FlowRegion.show(). "
                "Ensure constituent regions' pages are initialized with a highlighter."
            )

        output_page_images: List["PIL_Image_Runtime"] = []

        # Sort pages by index for consistent output order
        sorted_pages = sorted(
            regions_by_page.keys(),
            key=lambda p: p.index if hasattr(p, "index") else getattr(p, "page_number", 0),
        )

        # 3. Render each page with its relevant constituent regions highlighted
        for page_idx, page_obj in enumerate(sorted_pages):
            constituent_regions_on_this_page = regions_by_page[page_obj]
            if not constituent_regions_on_this_page:
                continue

            temp_highlights_for_page = []
            for i, region_part in enumerate(constituent_regions_on_this_page):
                part_label = None
                if labels and label_prefix:  # Ensure labels is True for label_prefix to apply
                    # If FlowRegion consists of multiple parts on this page, or overall
                    count_indicator = ""
                    if (
                        len(self.constituent_regions) > 1
                    ):  # If flow region has multiple parts overall
                        # Find global index of this region_part in self.constituent_regions
                        try:
                            global_idx = self.constituent_regions.index(region_part)
                            count_indicator = f"_{global_idx + 1}"
                        except ValueError:  # Should not happen if region_part is from the list
                            count_indicator = f"_p{page_idx}i{i+1}"  # fallback local index
                    elif (
                        len(constituent_regions_on_this_page) > 1
                    ):  # If multiple parts on *this* page, but FR is single part overall
                        count_indicator = f"_{i+1}"

                    part_label = f"{label_prefix}{count_indicator}" if label_prefix else None

                temp_highlights_for_page.append(
                    {
                        "page_index": (
                            page_obj.index
                            if hasattr(page_obj, "index")
                            else getattr(page_obj, "page_number", 1) - 1
                        ),
                        "bbox": region_part.bbox,
                        "polygon": region_part.polygon if region_part.has_polygon else None,
                        "color": color,  # Use the passed color
                        "label": part_label,
                        "use_color_cycling": False,  # Keep specific color
                    }
                )

            if not temp_highlights_for_page:
                continue

            page_image = highlighter_service.render_preview(
                page_index=(
                    page_obj.index
                    if hasattr(page_obj, "index")
                    else getattr(page_obj, "page_number", 1) - 1
                ),
                temporary_highlights=temp_highlights_for_page,
                scale=scale,
                width=width,
                labels=labels,  # Pass through labels
                legend_position=legend_position,
                **kwargs,
            )
            if page_image:
                output_page_images.append(page_image)

        # 4. Stack the generated page images if multiple
        if not output_page_images:
            logger.info("FlowRegion.show() produced no page images to concatenate.")
            return None

        if len(output_page_images) == 1:
            return output_page_images[0]

        # Stacking logic (same as in FlowRegionCollection.show)
        if stack_direction == "vertical":
            final_width = max(img.width for img in output_page_images)
            final_height = (
                sum(img.height for img in output_page_images)
                + (len(output_page_images) - 1) * stack_gap
            )
            if final_width == 0 or final_height == 0:
                raise ValueError("Cannot create concatenated image with zero width or height.")

            concatenated_image = PIL_Image_Runtime.new(
                "RGB", (final_width, final_height), stack_background_color
            )
            current_y = 0
            for img in output_page_images:
                paste_x = (final_width - img.width) // 2
                concatenated_image.paste(img, (paste_x, current_y))
                current_y += img.height + stack_gap
            return concatenated_image
        elif stack_direction == "horizontal":
            final_width = (
                sum(img.width for img in output_page_images)
                + (len(output_page_images) - 1) * stack_gap
            )
            final_height = max(img.height for img in output_page_images)
            if final_width == 0 or final_height == 0:
                raise ValueError("Cannot create concatenated image with zero width or height.")

            concatenated_image = PIL_Image_Runtime.new(
                "RGB", (final_width, final_height), stack_background_color
            )
            current_x = 0
            for img in output_page_images:
                paste_y = (final_height - img.height) // 2
                concatenated_image.paste(img, (current_x, paste_y))
                current_x += img.width + stack_gap
            return concatenated_image
        else:
            raise ValueError(
                f"Invalid stack_direction '{stack_direction}' for FlowRegion.show(). Must be 'vertical' or 'horizontal'."
            )

    def to_images(
        self,
        resolution: float = 150,
        **kwargs,
    ) -> List["PIL_Image"]:
        """
        Generates and returns a list of cropped PIL Images,
        one for each constituent physical region of this FlowRegion.
        """
        if not self.constituent_regions:
            logger.info("FlowRegion.to_images() called on an empty FlowRegion.")
            return []

        cropped_images: List["PIL_Image"] = []
        for region_part in self.constituent_regions:
            try:
                img = region_part.to_image(
                    resolution=resolution, crop=True, include_highlights=False, **kwargs
                )
                if img:
                    cropped_images.append(img)
            except Exception as e:
                logger.error(
                    f"Error generating image for constituent region {region_part.bbox}: {e}",
                    exc_info=True,
                )

        return cropped_images

    def to_image(self, background_color=(255, 255, 255), **kwargs) -> Optional["PIL_Image"]:
        """
        Creates a single composite image by stacking the images of its constituent regions.
        Stacking direction is based on the Flow's arrangement.
        Individual region images are obtained by calling to_images(**kwargs).

        Args:
            background_color: Tuple for RGB background color of the composite image.
            **kwargs: Additional arguments passed to to_images() for rendering individual parts
                      (e.g., resolution).

        Returns:
            A single PIL.Image.Image object, or None if no constituent images.
        """
        # Use PIL_Image_Runtime for creating new images at runtime
        images = self.to_images(**kwargs)
        if not images:
            return None
        if len(images) == 1:
            return images[0]

        if self.flow.arrangement == "vertical":
            # Stack vertically
            composite_width = max(img.width for img in images)
            composite_height = sum(img.height for img in images)
            if composite_width == 0 or composite_height == 0:
                return None  # Avoid zero-size image

            new_image = PIL_Image_Runtime.new(
                "RGB", (composite_width, composite_height), background_color
            )
            current_y = 0
            for img in images:
                # Default to left alignment for vertical stacking
                new_image.paste(img, (0, current_y))
                current_y += img.height
            return new_image

        elif self.flow.arrangement == "horizontal":
            # Stack horizontally
            composite_width = sum(img.width for img in images)
            composite_height = max(img.height for img in images)
            if composite_width == 0 or composite_height == 0:
                return None

            new_image = PIL_Image_Runtime.new(
                "RGB", (composite_width, composite_height), background_color
            )
            current_x = 0
            for img in images:
                # Default to top alignment for horizontal stacking
                new_image.paste(img, (current_x, 0))
                current_x += img.width
            return new_image
        else:
            # Should not happen if flow.arrangement is validated
            logger.warning(
                f"Unknown flow arrangement: {self.flow.arrangement}. Cannot stack images."
            )
            return None

    def __repr__(self) -> str:
        return (
            f"<FlowRegion constituents={len(self.constituent_regions)}, flow={self.flow}, "
            f"source_bbox={self.source_flow_element.bbox if self.source_flow_element else 'N/A'}>"
        )

    @property
    def is_empty(self) -> bool:
        """Checks if the FlowRegion contains no constituent regions or if all are empty."""
        if not self.constituent_regions:
            return True
        # A more robust check might see if extract_text() is empty and elements() is empty.
        # For now, if it has regions, it's not considered empty by this simple check.
        # User Point 4: FlowRegion can be empty (no text, no elements). This implies checking content.
        try:
            return not bool(self.extract_text(apply_exclusions=False).strip()) and not bool(
                self.elements(apply_exclusions=False)
            )
        except Exception:
            return True  # If error during check, assume empty to be safe

    # ------------------------------------------------------------------
    # Table extraction helpers (delegates to underlying physical regions)
    # ------------------------------------------------------------------

    def extract_table(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[["PhysicalRegion"], Optional[str]]] = None,
        show_progress: bool = False,
        **kwargs,
    ) -> List[List[Optional[str]]]:
        """Extracts a single logical table from the FlowRegion.

        This is a convenience wrapper that iterates through the constituent
        physical regions **in flow order**, calls their ``extract_table``
        method, and concatenates the resulting rows.  It mirrors the public
        interface of :pymeth:`natural_pdf.elements.region.Region.extract_table`.

        Args:
            method, table_settings, use_ocr, ocr_config, text_options, cell_extraction_func, show_progress:
                Same as in :pymeth:`Region.extract_table` and are forwarded as-is
                to each physical region.
            **kwargs: Additional keyword arguments forwarded to the underlying
                ``Region.extract_table`` implementation.

        Returns:
            A list of rows (``List[List[Optional[str]]]``).  Rows returned from
            consecutive constituent regions are appended in document order.  If
            no tables are detected in any region, an empty list is returned.
        """

        if table_settings is None:
            table_settings = {}
        if text_options is None:
            text_options = {}

        if not self.constituent_regions:
            return []

        aggregated_rows: List[List[Optional[str]]] = []

        for region in self.constituent_regions:
            try:
                region_rows = region.extract_table(
                    method=method,
                    table_settings=table_settings.copy(),  # Avoid side-effects
                    use_ocr=use_ocr,
                    ocr_config=ocr_config,
                    text_options=text_options.copy(),
                    cell_extraction_func=cell_extraction_func,
                    show_progress=show_progress,
                    **kwargs,
                )

                # ``region_rows`` can legitimately be [] if no table found.
                if region_rows:
                    aggregated_rows.extend(region_rows)
            except Exception as e:
                logger.error(
                    f"FlowRegion.extract_table: Error extracting table from constituent region {region}: {e}",
                    exc_info=True,
                )

        return aggregated_rows

    def extract_tables(
        self,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        **kwargs,
    ) -> List[List[List[Optional[str]]]]:
        """Extract **all** tables from the FlowRegion.

        This simply chains :pymeth:`Region.extract_tables` over each physical
        region and concatenates their results, preserving flow order.

        Args:
            method, table_settings: Forwarded to underlying ``Region.extract_tables``.
            **kwargs: Additional keyword arguments forwarded.

        Returns:
            A list where each item is a full table (list of rows).  The order of
            tables follows the order of the constituent regions in the flow.
        """

        if table_settings is None:
            table_settings = {}

        if not self.constituent_regions:
            return []

        all_tables: List[List[List[Optional[str]]]] = []

        for region in self.constituent_regions:
            try:
                region_tables = region.extract_tables(
                    method=method,
                    table_settings=table_settings.copy(),
                    **kwargs,
                )
                # ``region_tables`` is a list (possibly empty).
                if region_tables:
                    all_tables.extend(region_tables)
            except Exception as e:
                logger.error(
                    f"FlowRegion.extract_tables: Error extracting tables from constituent region {region}: {e}",
                    exc_info=True,
                )

        return all_tables
