import logging
from collections.abc import MutableSequence
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
    Set,
    SupportsIndex,
    Tuple,
    Union,
    cast,
    overload,
)

from PIL import Image  # Single import for PIL.Image module

from natural_pdf.collections.mixins import QACollectionMixin, SectionsCollectionMixin, _SectionHost
from natural_pdf.core.context import PDFContext
from natural_pdf.core.highlighter_utils import resolve_highlighter
from natural_pdf.core.render_spec import RenderSpec, Visualizable
from natural_pdf.selectors.host_mixin import delegate_signature
from natural_pdf.services.base import ServiceHostMixin
from natural_pdf.services.delegates import attach_capability
from natural_pdf.services.methods import flow_table_methods as _flow_table_methods
from natural_pdf.services.methods import navigation_methods as _navigation_methods
from natural_pdf.tables import TableResult

if TYPE_CHECKING:
    # from PIL.Image import Image as PIL_Image # No longer needed with Image.Image type hint
    from natural_pdf.core.page import Page as PhysicalPage
    from natural_pdf.elements.base import Element as PhysicalElement
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region

    from .element import FlowElement
    from .region import FlowRegion


logger = logging.getLogger(__name__)


class FlowElementCollection(MutableSequence["FlowElement"]):
    """
    A collection of FlowElement objects, typically the result of Flow.find_all().
    Provides directional methods that operate on its contained FlowElements and
    return FlowRegionCollection objects.
    """

    def __init__(self, flow_elements: Optional[Sequence["FlowElement"]] = None):
        self._flow_elements: List["FlowElement"] = (
            list(flow_elements) if flow_elements is not None else []
        )

    def __getitem__(self, index: Union[int, slice, SupportsIndex]) -> Any:
        if isinstance(index, slice):
            return FlowElementCollection(self._flow_elements[index])
        numeric_index = int(index)
        return self._flow_elements[numeric_index]

    def __setitem__(self, index: Union[int, slice, SupportsIndex], value: Any) -> None:
        if isinstance(index, slice):
            if isinstance(value, FlowElementCollection):
                replacement = list(value._flow_elements)
            elif isinstance(value, Iterable):
                replacement = list(value)
            else:
                raise TypeError(
                    "Slice assignment requires a FlowElementCollection or sequence of FlowElement instances."
                )
            self._flow_elements[index] = replacement
        else:
            numeric_index = int(index)
            self._flow_elements[numeric_index] = cast("FlowElement", value)

    def __delitem__(self, index: Union[int, slice, SupportsIndex]) -> None:
        if isinstance(index, slice):
            del self._flow_elements[index]
        else:
            del self._flow_elements[int(index)]

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
        return _navigation_methods.flow_element_collection_above(
            self,
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
        return _navigation_methods.flow_element_collection_below(
            self,
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
        return _navigation_methods.flow_element_collection_left(
            self,
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
        return _navigation_methods.flow_element_collection_right(
            self,
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

            effective_resolution = float(resolution) if resolution is not None else 150.0

            page_image = highlighter_service.render_preview(
                page_index=(
                    page_obj.index
                    if hasattr(page_obj, "index")
                    else getattr(page_obj, "page_number", 1) - 1
                ),
                temporary_highlights=temp_highlights_for_page,
                resolution=effective_resolution,
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


class FlowRegionCollection(
    ServiceHostMixin,
    Visualizable,
    SectionsCollectionMixin,
    QACollectionMixin,
    MutableSequence["FlowRegion"],
):
    """
    A collection of FlowRegion objects, typically the result of directional
    operations on a FlowElementCollection.
    Provides methods for querying and visualizing the aggregated content.
    """

    def __init__(self, flow_regions: Optional[Sequence["FlowRegion"]] = None):
        self._flow_regions: List["FlowRegion"] = (
            list(flow_regions) if flow_regions is not None else []
        )
        self._context_placeholder: bool = False
        self._bind_initial_context(self._flow_regions)

    def __getitem__(self, index: Union[int, slice, SupportsIndex]) -> Any:
        if isinstance(index, slice):
            return FlowRegionCollection(self._flow_regions[index])
        return self._flow_regions[int(index)]

    def __setitem__(self, index: Union[int, slice, SupportsIndex], value: Any) -> None:
        if isinstance(index, slice):
            if isinstance(value, FlowRegionCollection):
                replacement = list(value._flow_regions)
            elif isinstance(value, Iterable):
                replacement = list(value)
            else:
                raise TypeError(
                    "Slice assignment requires a FlowRegionCollection or sequence of FlowRegion instances."
                )
            self._flow_regions[index] = replacement
            for region in replacement:
                self._maybe_update_context_from(region)
        else:
            region_value = cast("FlowRegion", value)
            self._flow_regions[int(index)] = region_value
            self._maybe_update_context_from(region_value)

    def __delitem__(self, index: Union[int, slice, SupportsIndex]) -> None:
        if isinstance(index, slice):
            del self._flow_regions[index]
        else:
            del self._flow_regions[int(index)]

    def __len__(self) -> int:
        return len(self._flow_regions)

    def insert(self, index: int, value: "FlowRegion") -> None:
        self._flow_regions.insert(index, value)
        self._maybe_update_context_from(value)

    def __repr__(self) -> str:
        return f"<FlowRegionCollection(count={len(self)})>"

    # ------------------------------------------------------------------
    # Service context helpers
    # ------------------------------------------------------------------
    def _bind_initial_context(self, regions: Sequence["FlowRegion"]) -> None:
        context = self._find_context(regions)
        if context is None:
            self._context_placeholder = True
            context = PDFContext.with_defaults()
        else:
            self._context_placeholder = False
        self._init_service_host(context)

    def _maybe_update_context_from(self, region: "FlowRegion") -> None:
        if not self._context_placeholder:
            return
        context = getattr(region, "_context", None)
        if context is None:
            return
        self._init_service_host(context)
        self._context_placeholder = False

    @staticmethod
    def _find_context(regions: Sequence["FlowRegion"]):
        for region in regions:
            context = getattr(region, "_context", None)
            if context is not None:
                return context
        return None

    def _normalize_within(self, within: Optional[Any]) -> Optional["Region"]:
        if within is None:
            return None
        from natural_pdf.elements.region import Region
        from natural_pdf.flows.region import FlowRegion

        if isinstance(within, Region):
            return within
        if isinstance(within, FlowRegion):
            raise TypeError(
                "FlowRegionCollection directional 'within' expects a Region; FlowRegion is not supported. "
                "TODO: support FlowRegion-to-FlowRegion constraints by intersecting constituent pages."
            )
        raise TypeError(
            f"Unsupported within argument type '{type(within).__name__}' for FlowRegionCollection."
        )

    def above(
        self,
        height: Optional[float] = None,
        width: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional[Any] = None,
        anchor: str = "start",
        **kwargs,
    ) -> "FlowRegionCollection":
        normalized_within = self._normalize_within(within)
        return _navigation_methods.flow_region_collection_above(
            self,
            within=normalized_within,
            height=height,
            width=width,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=offset,
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            anchor=anchor,
            **kwargs,
        )

    def below(
        self,
        height: Optional[float] = None,
        width: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional[Any] = None,
        anchor: str = "start",
        **kwargs,
    ) -> "FlowRegionCollection":
        normalized_within = self._normalize_within(within)
        return _navigation_methods.flow_region_collection_below(
            self,
            within=normalized_within,
            height=height,
            width=width,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=offset,
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            anchor=anchor,
            **kwargs,
        )

    def left(
        self,
        width: Optional[float] = None,
        height: str = "element",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional[Any] = None,
        anchor: str = "start",
        **kwargs,
    ) -> "FlowRegionCollection":
        normalized_within = self._normalize_within(within)
        return _navigation_methods.flow_region_collection_left(
            self,
            within=normalized_within,
            width=width,
            height=height,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=offset,
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            anchor=anchor,
            **kwargs,
        )

    def right(
        self,
        width: Optional[float] = None,
        height: str = "element",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional[Any] = None,
        anchor: str = "start",
        **kwargs,
    ) -> "FlowRegionCollection":
        normalized_within = self._normalize_within(within)
        return _navigation_methods.flow_region_collection_right(
            self,
            within=normalized_within,
            width=width,
            height=height,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=offset,
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            anchor=anchor,
            **kwargs,
        )

    def _get_highlighter(self):
        if not self._flow_regions:
            raise RuntimeError("Cannot locate highlighting service for empty FlowRegionCollection.")
        return resolve_highlighter(self._flow_regions)

    @staticmethod
    def _merge_crop_bbox(
        existing: Optional[Tuple[float, float, float, float]],
        incoming: Optional[Tuple[float, float, float, float]],
    ) -> Optional[Tuple[float, float, float, float]]:
        if existing is None:
            return incoming
        if incoming is None:
            return existing
        return (
            min(existing[0], incoming[0]),
            min(existing[1], incoming[1]),
            max(existing[2], incoming[2]),
            max(existing[3], incoming[3]),
        )

    def _get_render_specs(
        self,
        mode: Literal["show", "render"] = "show",
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        highlights: Optional[List[Dict[str, Any]]] = None,
        crop: Union[bool, int, Literal["content", "wide"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> List[RenderSpec]:
        if not self._flow_regions:
            return []

        label_prefix = kwargs.pop("label_prefix", None)
        fr_kwargs = dict(kwargs)
        if label_prefix is not None:
            fr_kwargs["label_prefix"] = label_prefix

        specs_by_page: Dict[Any, RenderSpec] = {}
        ordered_pages: List[Any] = []
        normalized_crop = self._normalize_crop_mode(crop)

        for fr in self._flow_regions:
            fr_specs = fr._get_render_specs(
                mode=mode,
                color=color,
                highlights=highlights,
                crop=normalized_crop,
                crop_bbox=crop_bbox,
                **fr_kwargs,
            )
            for spec in fr_specs:
                page = spec.page
                if page in specs_by_page:
                    existing = specs_by_page[page]
                    existing.highlights.extend(spec.highlights)
                    existing.crop_bbox = self._merge_crop_bbox(existing.crop_bbox, spec.crop_bbox)
                else:
                    specs_by_page[page] = spec
                    ordered_pages.append(page)

        return [specs_by_page[page] for page in ordered_pages]

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

    # ------------------------------------------------------------------
    # Table extraction helpers
    # ------------------------------------------------------------------

    @delegate_signature(_flow_table_methods.flow_collection_extract_table)
    def extract_table(self, *args, **kwargs) -> List[TableResult]:
        return _flow_table_methods.flow_collection_extract_table(self, *args, **kwargs)

    @delegate_signature(_flow_table_methods.flow_collection_extract_tables)
    def extract_tables(self, *args, **kwargs) -> List[List[List[Optional[str]]]]:
        return _flow_table_methods.flow_collection_extract_tables(self, *args, **kwargs)

    def _iter_sections(self) -> Iterable["_SectionHost"]:
        return cast(Iterable["_SectionHost"], iter(self._flow_regions))

    def highlight(
        self,
        label_prefix: Optional[str] = "FRC",
        color: Optional[Union[Tuple, str]] = None,
        **kwargs,
    ) -> Optional[Image.Image]:
        if not self._flow_regions:
            return None

        num_flow_regions = len(self._flow_regions)
        for i, fr in enumerate(self._flow_regions):
            current_label = None
            if label_prefix:
                current_label = f"{label_prefix}_{i+1}" if num_flow_regions > 1 else label_prefix

            # Pass the specific color to each FlowRegion's highlight method.
            # FlowRegion.highlight will then pass it to its constituent regions.
            fr.highlight(label=current_label, color=color, **kwargs)
        return None

    def show(
        self,
        resolution: Optional[float] = None,
        labels: bool = True,
        legend_position: str = "right",
        default_color: Optional[Union[Tuple, str]] = "darkviolet",  # A distinct color for FRC show
        label_prefix: Optional[str] = "FRC_Part",
        width: Optional[int] = None,
        stack_direction: Literal["vertical", "horizontal"] = "vertical",  # New
        stack_gap: int = 5,  # New: Gap between stacked page images
        stack_background_color: Tuple[int, int, int] = (
            255,
            255,
            255,
        ),  # New: Background for stacking
        **kwargs,
    ) -> Optional[Image.Image]:
        if not self._flow_regions:
            logger.info("FlowRegionCollection.show() called on an empty collection.")
            return None

        if label_prefix is not None and "label_prefix" not in kwargs:
            kwargs["label_prefix"] = label_prefix
        kwargs.setdefault("stack_background_color", stack_background_color)
        return super().show(
            resolution=resolution,
            width=width,
            labels=labels,
            legend_position=legend_position,
            color=default_color,
            stack_direction=stack_direction,
            gap=stack_gap,
            **kwargs,
        )

    def to_images(self, resolution: float = 150, **kwargs) -> List[Image.Image]:
        """Returns a flat list of cropped images of all constituent physical regions."""
        all_cropped_images: List[Image.Image] = []
        for fr in self._flow_regions:
            all_cropped_images.extend(fr.to_images(resolution=resolution, **kwargs))
        return all_cropped_images

    def apply(self, func: Callable[["FlowRegion"], Any]) -> List[Any]:
        return [func(fr) for fr in self._flow_regions]

    # ------------------------------------------------------------------
    # QA service hooks
    # ------------------------------------------------------------------
    def _qa_segment_iterable(self) -> Sequence["FlowRegion"]:
        return self._flow_regions

    def _qa_target_region(self):
        if not self._flow_regions:
            raise RuntimeError("FlowRegionCollection has no FlowRegion data for QA.")
        first_region = self._flow_regions[0]
        return first_region._qa_target_region()

    def _qa_context_page_number(self) -> int:
        first_region = self.first
        if first_region is None:
            return -1
        getter = getattr(first_region, "_qa_context_page_number", None)
        if callable(getter):
            try:
                value = getter()
                if isinstance(value, (int, float, str)):
                    return int(value)
                return -1
            except Exception:
                return -1
        return -1

    def _qa_source_elements(self):
        from natural_pdf.elements.element_collection import ElementCollection

        first_region = self.first
        if first_region is None:
            return ElementCollection([])
        getter = getattr(first_region, "_qa_source_elements", None)
        if callable(getter):
            try:
                result = getter()
                if isinstance(result, ElementCollection):
                    return result
            except Exception:
                pass
        return ElementCollection([])

    @staticmethod
    def _normalize_crop_mode(
        value: Union[bool, int, Literal["content", "wide"]]
    ) -> Union[bool, Literal["content"]]:
        if value == "content":
            return "content"
        return bool(value)


attach_capability(FlowRegionCollection, "qa")
