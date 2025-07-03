import logging
from collections.abc import MutableSequence
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, TypeVar, Union

from PIL import Image  # Single import for PIL.Image module

if TYPE_CHECKING:
    # from PIL.Image import Image as PIL_Image # No longer needed with Image.Image type hint
    from natural_pdf.core.page import Page as PhysicalPage
    from natural_pdf.elements.base import Element as PhysicalElement
    from natural_pdf.elements.element_collection import ElementCollection

    from .element import FlowElement
    from .flow import Flow  # Though not directly used in __init__, FlowRegion needs it.
    from .region import FlowRegion


logger = logging.getLogger(__name__)

T_FEC = TypeVar("T_FEC", bound="FlowElement")
T_FRC = TypeVar("T_FRC", bound="FlowRegion")


class FlowElementCollection(MutableSequence[T_FEC]):
    """
    A collection of FlowElement objects, typically the result of Flow.find_all().
    Provides directional methods that operate on its contained FlowElements and
    return FlowRegionCollection objects.
    """

    def __init__(self, flow_elements: List["FlowElement"]):
        self._flow_elements: List["FlowElement"] = (
            flow_elements if flow_elements is not None else []
        )

    def __getitem__(self, index: int) -> "FlowElement":
        return self._flow_elements[index]

    def __setitem__(self, index: int, value: "FlowElement") -> None:
        self._flow_elements[index] = value

    def __delitem__(self, index: int) -> None:
        del self._flow_elements[index]

    def __len__(self) -> int:
        return len(self._flow_elements)

    def insert(self, index: int, value: "FlowElement") -> None:
        self._flow_elements.insert(index, value)

    @property
    def flow_elements(self) -> List["FlowElement"]:
        return self._flow_elements

    @property
    def first(self) -> Optional["FlowElement"]:
        return self._flow_elements[0] if self._flow_elements else None

    @property
    def last(self) -> Optional["FlowElement"]:
        return self._flow_elements[-1] if self._flow_elements else None

    def __repr__(self) -> str:
        return f"<FlowElementCollection(count={len(self)})>"

    def _execute_directional_on_all(self, method_name: str, **kwargs) -> "FlowRegionCollection":
        results: List["FlowRegion"] = []
        if not self._flow_elements:
            return FlowRegionCollection([])  # Return empty FlowRegionCollection

        # Assuming all flow_elements share the same flow context
        # (which should be true if they came from the same Flow.find_all())

        for fe in self._flow_elements:
            method_to_call = getattr(fe, method_name)
            flow_region_result: "FlowRegion" = method_to_call(**kwargs)
            # FlowElement directional methods always return a FlowRegion (even if empty)
            results.append(flow_region_result)
        return FlowRegionCollection(results)

    def above(
        self,
        height: Optional[float] = None,
        width_ratio: Optional[float] = None,
        width_absolute: Optional[float] = None,
        width_alignment: str = "center",
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegionCollection":
        return self._execute_directional_on_all(
            "above",
            height=height,
            width_ratio=width_ratio,
            width_absolute=width_absolute,
            width_alignment=width_alignment,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def below(
        self,
        height: Optional[float] = None,
        width_ratio: Optional[float] = None,
        width_absolute: Optional[float] = None,
        width_alignment: str = "center",
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegionCollection":
        return self._execute_directional_on_all(
            "below",
            height=height,
            width_ratio=width_ratio,
            width_absolute=width_absolute,
            width_alignment=width_alignment,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def left(
        self,
        width: Optional[float] = None,
        height_ratio: Optional[float] = None,
        height_absolute: Optional[float] = None,
        height_alignment: str = "center",
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegionCollection":
        return self._execute_directional_on_all(
            "left",
            width=width,
            height_ratio=height_ratio,
            height_absolute=height_absolute,
            height_alignment=height_alignment,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def right(
        self,
        width: Optional[float] = None,
        height_ratio: Optional[float] = None,
        height_absolute: Optional[float] = None,
        height_alignment: str = "center",
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "FlowRegionCollection":
        return self._execute_directional_on_all(
            "right",
            width=width,
            height_ratio=height_ratio,
            height_absolute=height_absolute,
            height_alignment=height_alignment,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def show(
        self,
        resolution: Optional[float] = None,
        labels: bool = True,
        legend_position: str = "right",
        default_color: Optional[Union[Tuple, str]] = "orange",  # A distinct color for FEC show
        label_prefix: Optional[str] = "FEC_Element",
        width: Optional[int] = None,
        stack_direction: str = "vertical",  # "vertical" or "horizontal"
        stack_gap: int = 5,  # Gap between stacked page images
        stack_background_color: Tuple[int, int, int] = (255, 255, 255),  # Background for stacking
        **kwargs,
    ) -> Optional[Image.Image]:
        """
        Shows all FlowElements in this collection by highlighting them on their respective pages.
        If multiple pages are involved, they are stacked into a single image.
        """
        if not self._flow_elements:
            logger.info("FlowElementCollection.show() called on an empty collection.")
            return None

        # Group flow elements by their physical page
        elements_by_page: dict["PhysicalPage", List["FlowElement"]] = {}
        for flow_element in self._flow_elements:
            page_obj = flow_element.page
            if page_obj:
                if page_obj not in elements_by_page:
                    elements_by_page[page_obj] = []
                elements_by_page[page_obj].append(flow_element)
            else:
                raise ValueError(f"FlowElement {flow_element} has no page.")

        if not elements_by_page:
            logger.info(
                "FlowElementCollection.show() found no flow elements with associated pages."
            )
            return None

        # Get a highlighter service from the first page
        first_page_with_elements = next(iter(elements_by_page.keys()), None)
        highlighter_service = None
        if first_page_with_elements and hasattr(first_page_with_elements, "_highlighter"):
            highlighter_service = first_page_with_elements._highlighter

        if not highlighter_service:
            raise ValueError(
                "Cannot get highlighter service for FlowElementCollection.show(). "
                "Ensure flow elements' pages are initialized with a highlighter."
            )

        output_page_images: List[Image.Image] = []

        # Sort pages by index for consistent output order
        sorted_pages = sorted(
            elements_by_page.keys(),
            key=lambda p: p.index if hasattr(p, "index") else getattr(p, "page_number", 0),
        )

        # Render each page with its relevant flow elements highlighted
        for page_idx, page_obj in enumerate(sorted_pages):
            flow_elements_on_this_page = elements_by_page[page_obj]
            if not flow_elements_on_this_page:
                continue

            temp_highlights_for_page = []
            for i, flow_element in enumerate(flow_elements_on_this_page):
                element_label = None
                if labels and label_prefix:
                    count_indicator = ""
                    if len(self._flow_elements) > 1:
                        # Find global index of this flow_element in self._flow_elements
                        try:
                            global_idx = self._flow_elements.index(flow_element)
                            count_indicator = f"_{global_idx + 1}"
                        except ValueError:
                            count_indicator = f"_p{page_idx}i{i+1}"  # fallback local index
                    elif len(flow_elements_on_this_page) > 1:
                        count_indicator = f"_{i+1}"

                    element_label = f"{label_prefix}{count_indicator}" if label_prefix else None

                temp_highlights_for_page.append(
                    {
                        "page_index": (
                            page_obj.index
                            if hasattr(page_obj, "index")
                            else getattr(page_obj, "page_number", 1) - 1
                        ),
                        "bbox": flow_element.bbox,
                        "polygon": (
                            getattr(flow_element.physical_object, "polygon", None)
                            if hasattr(flow_element.physical_object, "has_polygon")
                            and flow_element.physical_object.has_polygon
                            else None
                        ),
                        "color": default_color,
                        "label": element_label,
                        "use_color_cycling": False,
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
                resolution=resolution,
                width=width,
                labels=labels,
                legend_position=legend_position,
                **kwargs,
            )
            if page_image:
                output_page_images.append(page_image)

        # Stack the generated page images if multiple
        if not output_page_images:
            logger.info("FlowElementCollection.show() produced no page images to concatenate.")
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

            concatenated_image = Image.new(
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

            concatenated_image = Image.new(
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
                f"Invalid stack_direction '{stack_direction}' for FlowElementCollection.show(). Must be 'vertical' or 'horizontal'."
            )


class FlowRegionCollection(MutableSequence[T_FRC]):
    """
    A collection of FlowRegion objects, typically the result of directional
    operations on a FlowElementCollection.
    Provides methods for querying and visualizing the aggregated content.
    """

    def __init__(self, flow_regions: List["FlowRegion"]):
        self._flow_regions: List["FlowRegion"] = flow_regions if flow_regions is not None else []

    def __getitem__(self, index: int) -> "FlowRegion":
        return self._flow_regions[index]

    def __setitem__(self, index: int, value: "FlowRegion") -> None:
        self._flow_regions[index] = value

    def __delitem__(self, index: int) -> None:
        del self._flow_regions[index]

    def __len__(self) -> int:
        return len(self._flow_regions)

    def insert(self, index: int, value: "FlowRegion") -> None:
        self._flow_regions.insert(index, value)

    def __repr__(self) -> str:
        return f"<FlowRegionCollection(count={len(self)})>"

    def __add__(self, other: "FlowRegionCollection") -> "FlowRegionCollection":
        if not isinstance(other, FlowRegionCollection):
            return NotImplemented
        return FlowRegionCollection(self._flow_regions + other._flow_regions)

    @property
    def flow_regions(self) -> List["FlowRegion"]:
        return self._flow_regions

    @property
    def first(self) -> Optional["FlowRegion"]:
        return self._flow_regions[0] if self._flow_regions else None

    @property
    def last(self) -> Optional["FlowRegion"]:
        return self._flow_regions[-1] if self._flow_regions else None

    @property
    def is_empty(self) -> bool:
        if not self._flow_regions:
            return True
        return all(fr.is_empty for fr in self._flow_regions)

    def filter(self, func: Callable[["FlowRegion"], bool]) -> "FlowRegionCollection":
        return FlowRegionCollection([fr for fr in self._flow_regions if func(fr)])

    def sort(
        self, key: Optional[Callable[["FlowRegion"], Any]] = None, reverse: bool = False
    ) -> "FlowRegionCollection":
        """Sorts the collection in-place. Default sort is by flow order if possible."""
        # A default key could try to sort by first constituent region's page then top/left,
        # but FlowRegions can be complex. For now, require explicit key or rely on list.sort default.
        if key is None:
            # Attempt a sensible default sort: by page of first constituent, then its top, then its x0
            def default_sort_key(fr: "FlowRegion"):
                if fr.constituent_regions:
                    first_constituent = fr.constituent_regions[0]
                    page_idx = first_constituent.page.index if first_constituent.page else -1
                    return (page_idx, first_constituent.top, first_constituent.x0)
                return (float("inf"), float("inf"), float("inf"))  # Push empty ones to the end

            self._flow_regions.sort(key=default_sort_key, reverse=reverse)
        else:
            self._flow_regions.sort(key=key, reverse=reverse)
        return self

    def extract_text(self, separator: str = "\n", apply_exclusions: bool = True, **kwargs) -> str:
        texts = [
            fr.extract_text(apply_exclusions=apply_exclusions, **kwargs)
            for fr in self._flow_regions
        ]
        return separator.join(t for t in texts if t)  # Filter out empty strings from concatenation

    def extract_each_text(self, apply_exclusions: bool = True, **kwargs) -> List[str]:
        return [
            fr.extract_text(apply_exclusions=apply_exclusions, **kwargs)
            for fr in self._flow_regions
        ]

    def find(
        self, selector: Optional[str] = None, *, text: Optional[str] = None, **kwargs
    ) -> Optional["PhysicalElement"]:
        from natural_pdf.elements.base import Element as PhysicalElement  # Runtime import

        for fr in self._flow_regions:
            found = fr.find(selector=selector, text=text, **kwargs)
            if found:
                return found
        return None

    def find_all(
        self, selector: Optional[str] = None, *, text: Optional[str] = None, **kwargs
    ) -> "ElementCollection":
        from natural_pdf.elements.collections import (
            ElementCollection as RuntimeElementCollection,  # Runtime import
        )

        all_physical_elements: List["PhysicalElement"] = []
        for fr in self._flow_regions:
            # FlowRegion.find_all returns an ElementCollection
            elements_in_fr: "RuntimeElementCollection" = fr.find_all(
                selector=selector, text=text, **kwargs
            )
            if elements_in_fr:  # ElementCollection has boolean True if not empty
                all_physical_elements.extend(
                    elements_in_fr.elements
                )  # Access .elements to get list

        # Deduplicate while preserving order as much as possible (simple set doesn't preserve order)
        seen = set()
        unique_elements = []
        for el in all_physical_elements:
            if el not in seen:
                unique_elements.append(el)
                seen.add(el)
        return RuntimeElementCollection(unique_elements)

    def highlight(
        self,
        label_prefix: Optional[str] = "FRC",
        color: Optional[Union[Tuple, str]] = None,
        **kwargs,
    ) -> "FlowRegionCollection":
        if not self._flow_regions:
            return self

        num_flow_regions = len(self._flow_regions)
        for i, fr in enumerate(self._flow_regions):
            current_label = None
            if label_prefix:
                current_label = f"{label_prefix}_{i+1}" if num_flow_regions > 1 else label_prefix

            # Pass the specific color to each FlowRegion's highlight method.
            # FlowRegion.highlight will then pass it to its constituent regions.
            fr.highlight(label=current_label, color=color, **kwargs)
        return self

    def show(
        self,
        resolution: Optional[float] = None,
        labels: bool = True,
        legend_position: str = "right",
        default_color: Optional[Union[Tuple, str]] = "darkviolet",  # A distinct color for FRC show
        label_prefix: Optional[str] = "FRC_Part",
        width: Optional[int] = None,
        stack_direction: str = "vertical",  # New: "vertical" or "horizontal"
        stack_gap: int = 5,  # New: Gap between stacked page images
        stack_background_color: Tuple[int, int, int] = (
            255,
            255,
            255,
        ),  # New: Background for stacking
        **kwargs,
    ) -> Optional[Image.Image]:  # Return type changed
        if not self._flow_regions:
            logger.info("FlowRegionCollection.show() called on an empty collection.")
            return None  # Changed from []

        regions_by_page: dict["PhysicalPage", List[dict[str, Any]]] = {}

        first_flow_region = self._flow_regions[0]
        highlighter_service = None
        if first_flow_region and first_flow_region.flow and first_flow_region.flow.segments:
            first_segment_page = first_flow_region.flow.segments[0].page
            if first_segment_page and hasattr(first_segment_page, "_highlighter"):
                highlighter_service = first_segment_page._highlighter

        if not highlighter_service:
            logger.error("Cannot get highlighter service for FlowRegionCollection.show().")
            return None  # Changed from []

        constituent_idx = 0
        for fr_idx, fr in enumerate(self._flow_regions):
            for constituent_region in fr.constituent_regions:
                page_obj = constituent_region.page
                if not page_obj:
                    logger.warning(
                        f"Constituent region {constituent_region.bbox} has no page. Skipping in show()."
                    )
                    continue

                if page_obj not in regions_by_page:
                    regions_by_page[page_obj] = []

                part_label = None
                if label_prefix:
                    part_label = f"{label_prefix}_{constituent_idx}"

                regions_by_page[page_obj].append(
                    {
                        "page_index": (
                            page_obj.index
                            if hasattr(page_obj, "index")
                            else getattr(page_obj, "page_number", 1) - 1
                        ),
                        "bbox": constituent_region.bbox,
                        "polygon": (
                            constituent_region.polygon if constituent_region.has_polygon else None
                        ),
                        "color": default_color,
                        "label": part_label,
                        "use_color_cycling": False,
                    }
                )
                constituent_idx += 1

        output_page_images: List[Image.Image] = []
        sorted_pages = sorted(
            regions_by_page.keys(),
            key=lambda p: p.index if hasattr(p, "index") else getattr(p, "page_number", 0),
        )

        for page_obj in sorted_pages:
            temp_highlights_for_page = regions_by_page[page_obj]
            if not temp_highlights_for_page:
                continue

            page_image = highlighter_service.render_preview(
                page_index=(
                    page_obj.index
                    if hasattr(page_obj, "index")
                    else getattr(page_obj, "page_number", 1) - 1
                ),
                temporary_highlights=temp_highlights_for_page,
                resolution=resolution,
                width=width,
                labels=labels,
                legend_position=legend_position,
                **kwargs,
            )
            if page_image:
                output_page_images.append(page_image)

        if not output_page_images:
            logger.info("FlowRegionCollection.show() produced no page images to concatenate.")
            return None

        if len(output_page_images) == 1:
            return output_page_images[0]

        if stack_direction == "vertical":
            final_width = max(img.width for img in output_page_images)
            final_height = (
                sum(img.height for img in output_page_images)
                + (len(output_page_images) - 1) * stack_gap
            )
            if final_width == 0 or final_height == 0:
                logger.warning("Cannot create concatenated image with zero width or height.")
                return None

            concatenated_image = Image.new(
                "RGB", (final_width, final_height), stack_background_color
            )
            current_y = 0
            for img in output_page_images:
                paste_x = (final_width - img.width) // 2  # Center horizontally
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
                logger.warning("Cannot create concatenated image with zero width or height.")
                return None

            concatenated_image = Image.new(
                "RGB", (final_width, final_height), stack_background_color
            )
            current_x = 0
            for img in output_page_images:
                paste_y = (final_height - img.height) // 2  # Center vertically
                concatenated_image.paste(img, (current_x, paste_y))
                current_x += img.width + stack_gap
            return concatenated_image
        else:
            logger.error(
                f"Invalid stack_direction '{stack_direction}' for FlowRegionCollection.show(). Must be 'vertical' or 'horizontal'."
            )
            return None

    def to_images(self, resolution: float = 150, **kwargs) -> List[Image.Image]:
        """Returns a flat list of cropped images of all constituent physical regions."""
        all_cropped_images: List[Image.Image] = []
        for fr in self._flow_regions:
            all_cropped_images.extend(fr.to_images(resolution=resolution, **kwargs))
        return all_cropped_images

    def apply(self, func: Callable[["FlowRegion"], Any]) -> List[Any]:
        return [func(fr) for fr in self._flow_regions]
