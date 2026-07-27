from __future__ import annotations

"""Unified rendering infrastructure for natural-pdf.

This module provides the core components for the unified image generation system:
- RenderSpec: Data structure describing what to render
- Visualizable: Mixin providing show/render/export methods
"""

import logging
from collections.abc import Iterable as IterableABC
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from natural_pdf.core.page import Page
    from natural_pdf.elements.region import Region

from natural_pdf.core.highlighter_utils import resolve_highlighter

ColorInput = Union[
    str,
    Tuple[int, int, int],
    Tuple[int, int, int, int],
    Tuple[float, float, float],
    Tuple[float, float, float, float],
]

logger = logging.getLogger(__name__)


class ImageOptionsProtocol(Protocol):
    resolution: Optional[float]


def _image_options() -> Optional[ImageOptionsProtocol]:
    try:
        import natural_pdf
    except Exception:  # pragma: no cover - defensive guard
        return None

    return cast(ImageOptionsProtocol, getattr(natural_pdf.options, "image", None))


@dataclass
class RenderSpec:
    """Specification for rendering a single page or region.

    This is the core data structure that unifies all rendering operations.
    Every visual object in natural-pdf converts its display requirements
    into one or more RenderSpecs, which are then processed by the
    unified rendering pipeline.

    Attributes:
        page: The page to render
        crop_bbox: Optional bounding box (x0, y0, x1, y1) to crop to
        highlights: List of highlight specifications, each containing:
            - bbox or polygon: The geometry to highlight
            - color: Optional color for the highlight
            - label: Optional label text
            - element: Optional reference to the source element
    """

    page: "Page"
    crop_bbox: Optional[Tuple[float, float, float, float]] = None
    highlights: List[Dict[str, Any]] = field(default_factory=list)

    def add_highlight(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        polygon: Optional[List[Tuple[float, float]]] = None,
        color: Optional[ColorInput] = None,
        label: Optional[str] = None,
        element: Optional[Any] = None,
        attributes_to_draw: Optional[Dict[str, Any]] = None,
        quantitative_metadata: Optional[Dict[str, Any]] = None,
        fill: Optional[bool] = None,
        outline: Optional[bool] = None,
        line_width: Optional[float] = None,
        vertices: Optional[bool] = None,
    ) -> None:
        """Add a highlight to this render spec.

        Args:
            bbox: Bounding box to highlight
            polygon: Polygon points to highlight (alternative to bbox)
            color: Color for the highlight
            label: Label text for the highlight
            element: Source element reference
            attributes_to_draw: Key-value pairs to draw on the highlight
            quantitative_metadata: Colorbar metadata for quantitative attributes
        """
        if bbox is None and polygon is None and element is not None:
            # Extract geometry from element
            if (
                hasattr(element, "polygon")
                and hasattr(element, "has_polygon")
                and element.has_polygon
            ):
                polygon = element.polygon
            elif hasattr(element, "bbox"):
                bbox = element.bbox

        if bbox is None and polygon is None:
            raise ValueError("Must provide bbox, polygon, or element with geometry")

        highlight: Dict[str, Any] = {
            "bbox": bbox,
            "polygon": polygon,
            "color": color,
            "label": label,
            "element": element,
            "attributes_to_draw": attributes_to_draw,
            "quantitative_metadata": quantitative_metadata,
            "fill": fill,
            "outline": outline,
            "line_width": line_width,
            "vertices": vertices,
        }
        # Remove None values
        highlight = {k: v for k, v in highlight.items() if v is not None}
        self.highlights.append(highlight)


def add_explicit_highlights_to_spec(
    spec: RenderSpec,
    highlights: Optional[Union[List[Dict[str, Any]], bool]],
    *,
    default_color: Optional[ColorInput] = None,
    page: Optional[Any] = None,
) -> None:
    """Add caller-provided highlight specs/groups to a render spec.

    Supports both grouped highlights of the form ``{"elements": [...]}`` and
    direct highlight entries such as ``{"bbox": (...), "color": "red"}``.
    """
    if not highlights or highlights is False:
        return

    for entry in highlights:
        if not isinstance(entry, MappingABC):
            spec.add_highlight(element=entry, color=default_color)
            continue

        entry_dict = dict(entry)
        entry_color = entry_dict.get("color", default_color)
        entry_label = entry_dict.get("label")

        if "elements" in entry_dict:
            raw_elements = entry_dict.get("elements")
            if raw_elements is None:
                continue
            if hasattr(raw_elements, "elements"):
                elements_iter = raw_elements.elements
            elif isinstance(raw_elements, IterableABC) and not isinstance(
                raw_elements, (str, bytes, MappingABC)
            ):
                elements_iter = raw_elements
            else:
                elements_iter = (raw_elements,)

            for elem in elements_iter:
                elem_page = getattr(elem, "page", None)
                if page is not None and elem_page is not None and elem_page != page:
                    continue
                spec.add_highlight(element=elem, color=entry_color, label=entry_label)
            continue

        target_page = entry_dict.get("page")
        target_page_index = entry_dict.get("page_index")
        if page is not None:
            if target_page is not None and target_page != page:
                continue
            if target_page_index is not None and getattr(page, "index", None) != target_page_index:
                continue

        elem = entry_dict.get("element")
        elem_page = getattr(elem, "page", None)
        if page is not None and elem_page is not None and elem_page != page:
            continue

        spec.add_highlight(
            bbox=entry_dict.get("bbox"),
            polygon=entry_dict.get("polygon"),
            element=elem,
            color=entry_color,
            label=entry_label,
            attributes_to_draw=entry_dict.get("attributes_to_draw"),
            quantitative_metadata=entry_dict.get("quantitative_metadata"),
            fill=entry_dict.get("fill"),
            outline=entry_dict.get("outline"),
            line_width=entry_dict.get("line_width", entry_dict.get("linewidth")),
            vertices=entry_dict.get("vertices"),
        )


class Visualizable:
    """Mixin class providing unified show/render/export methods.

    Classes that inherit from Visualizable need only implement
    _get_render_specs() to gain full image generation capabilities.
    """

    def highlight(self, *elements, **kwargs):
        """
        Convenience method for highlighting elements in Jupyter/Colab.

        This method creates a highlight context, adds the elements, and returns
        the resulting image. It's designed for simple one-liner usage in notebooks.

        Args:
            *elements: Elements or element collections to highlight
            **kwargs: Additional parameters passed to show()

        Returns:
            PIL Image with highlights

        Example:
            # Simple one-liner highlighting
            page.highlight(left, mid, right)

            # With custom colors
            page.highlight(
                (tables, 'blue'),
                (headers, 'red'),
                (footers, 'green')
            )
        """
        from natural_pdf.core.highlighting_service import HighlightContext

        # Create context and add elements
        ctx = HighlightContext(self, show_on_exit=False)

        for element in elements:
            if isinstance(element, tuple) and len(element) == 2:
                # Element with color: (element, color)
                ctx.add(element[0], color=element[1])
            elif isinstance(element, tuple) and len(element) == 3:
                # Element with color and label: (element, color, label)
                ctx.add(element[0], color=element[1], label=element[2])
            else:
                # Just element
                ctx.add(element)

        # Return the image directly
        return ctx.show(**kwargs)

    def _get_render_specs(
        self, mode: Literal["show", "render"] = "show", **kwargs
    ) -> List[RenderSpec]:
        """Get render specifications for this object.

        This is the only method subclasses need to implement.
        It should return a list of RenderSpec objects describing
        what needs to be rendered.

        Args:
            mode: Rendering mode - 'show' includes highlights, 'render' is clean
            **kwargs: Additional parameters from show/render methods

        Returns:
            List of RenderSpec objects
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _get_render_specs()")

    def _get_highlighter(self):
        """Get the highlighting service for rendering.

        This method should be overridden by classes that have
        a different way of accessing the highlighter.
        """
        try:
            return resolve_highlighter(self)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Cannot find HighlightingService for {self.__class__.__name__}. "
                "Override _get_highlighter() to provide access."
            ) from exc

    def _get_rendering_service(self):
        """Resolve the RenderingService for hosts that mix in Visualizable."""
        services = getattr(self, "services", None)
        rendering = getattr(services, "rendering", None) if services else None
        if rendering is not None:
            return rendering
        from natural_pdf.services.base import resolve_service  # Local import to avoid cycles

        return resolve_service(self, "rendering")

    def get_rendering_service(self):
        """Public accessor for the rendering service (primarily for tests)."""
        cls = getattr(self, "__class__", type(self))
        method = getattr(cls, "_get_rendering_service", None)
        if method is None:
            method = Visualizable._get_rendering_service
        return method(self)

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
        render_ocr: bool = False,
        layout: Optional[Literal["stack", "grid", "single"]] = None,
        stack_direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 5,
        columns: Optional[int] = 6,
        limit: Optional[int] = 30,
        crop: Union[bool, int, str, "Region", Literal["wide"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> Optional[PILImage]:
        """Generate a preview image with highlights.

        This method is for interactive debugging and visualization.
        Elements are highlighted to show what's selected or being worked with.

        Args:
            resolution: DPI for rendering (default from global settings)
            width: Target width in pixels (overrides resolution)
            color: Default highlight color
            labels: Whether to show labels for highlights
            label_format: Format string for labels (e.g., "Element {index}")
            highlights: Additional highlight groups to show, or False to disable all highlights
            legend_position: Position of legend/colorbar ('right', 'left', 'top', 'bottom')
            annotate: Attribute name(s) to display on highlights (string or list)
            render_ocr: Whether to render OCR text overlay on the image
            layout: How to arrange multiple pages/regions (defaults to 'grid' for multi-page, 'single' for single page)
            stack_direction: Direction for stack layout
            gap: Pixels between stacked images
            columns: Number of columns for grid layout (defaults to 6)
            limit: Maximum number of pages to display (default 30, None for all)
            crop: Cropping mode:
                - False: No cropping (default)
                - True: Tight crop to element bounds
                - int: Padding in pixels around element
                - 'wide': Full page width, cropped vertically to element
                - Region: Crop to the bounds of another region
            crop_bbox: Explicit crop bounds
            **kwargs: Additional parameters passed to rendering

        Returns:
            PIL Image object or None if nothing to render
        """
        return self.get_rendering_service().show(
            self,
            resolution=resolution,
            width=width,
            color=color,
            labels=labels,
            label_format=label_format,
            highlights=highlights,
            legend_position=legend_position,
            annotate=annotate,
            render_ocr=render_ocr,
            layout=layout,
            stack_direction=stack_direction,
            gap=gap,
            columns=columns,
            limit=limit,
            crop=crop,
            crop_bbox=crop_bbox,
            **kwargs,
        )

    def render(
        self,
        *,
        resolution: Optional[float] = None,
        width: Optional[int] = None,
        highlights: Optional[Union[List[Dict[str, Any]], bool]] = None,
        labels: bool = False,
        label_format: Optional[str] = None,
        render_ocr: bool = False,
        layout: Literal["stack", "grid", "single"] = "stack",
        stack_direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 5,
        columns: Optional[int] = None,
        crop: Union[bool, int, str, "Region", Literal["wide"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> Optional[PILImage]:
        """Generate a clean image, with optional explicit highlights.

        This method produces publication-ready images without
        any debugging annotations or persistent highlights.

        Args:
            resolution: DPI for rendering (default from global settings)
            width: Target width in pixels (overrides resolution)
            highlights: Optional explicit highlight groups/specs to render
            labels: Whether to render a legend for explicit highlights
            label_format: Format string for generated highlight labels
            render_ocr: Whether to render OCR text overlay on the image
            layout: How to arrange multiple pages/regions
            stack_direction: Direction for stack layout
            gap: Pixels between stacked images
            columns: Number of columns for grid layout
            crop: Cropping mode (False, True, int for padding, 'wide', or Region)
            crop_bbox: Explicit crop bounds
            **kwargs: Additional parameters passed to rendering

        Returns:
            PIL Image object or None if nothing to render
        """
        return self.get_rendering_service().render(
            self,
            resolution=resolution,
            width=width,
            highlights=highlights,
            labels=labels,
            label_format=label_format,
            render_ocr=render_ocr,
            layout=layout,
            stack_direction=stack_direction,
            gap=gap,
            columns=columns,
            crop=crop,
            crop_bbox=crop_bbox,
            **kwargs,
        )

    def export(
        self,
        path: Union[str, Path],
        *,
        resolution: Optional[float] = None,
        width: Optional[int] = None,
        layout: Literal["stack", "grid", "single"] = "stack",
        stack_direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 5,
        columns: Optional[int] = None,
        crop: Union[bool, Literal["content"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        format: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Export a clean image to file.

        This is a convenience method that renders and saves in one step.

        Args:
            path: Output file path
            resolution: DPI for rendering
            width: Target width in pixels
            layout: How to arrange multiple pages/regions
            stack_direction: Direction for stack layout
            gap: Pixels between stacked images
            columns: Number of columns for grid layout
            crop: Cropping mode (False, True, int for padding, 'wide', or Region)
            crop_bbox: Explicit crop bounds
            format: Image format (inferred from path if not specified)
            **kwargs: Additional parameters passed to rendering
        """
        self.get_rendering_service().export(
            self,
            path=path,
            resolution=resolution,
            width=width,
            layout=layout,
            stack_direction=stack_direction,
            gap=gap,
            columns=columns,
            crop=crop,
            crop_bbox=crop_bbox,
            format=format,
            **kwargs,
        )

    def _resolve_image_resolution(self, requested: Optional[float]) -> float:
        """Resolve an explicit resolution or fall back to configured defaults."""

        if requested is not None:
            return float(requested)

        options = _image_options()
        if options is not None and options.resolution is not None:
            return float(options.resolution)

        return 150.0
