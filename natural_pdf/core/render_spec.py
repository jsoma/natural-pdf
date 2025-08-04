"""Unified rendering infrastructure for natural-pdf.

This module provides the core components for the unified image generation system:
- RenderSpec: Data structure describing what to render
- Visualizable: Mixin providing show/render/export methods
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    from PIL import Image as PIL_Image

    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element

logger = logging.getLogger(__name__)


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
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        label: Optional[str] = None,
        element: Optional["Element"] = None,
    ) -> None:
        """Add a highlight to this render spec.

        Args:
            bbox: Bounding box to highlight
            polygon: Polygon points to highlight (alternative to bbox)
            color: Color for the highlight
            label: Label text for the highlight
            element: Source element reference
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

        highlight = {
            "bbox": bbox,
            "polygon": polygon,
            "color": color,
            "label": label,
            "element": element,
        }
        # Remove None values
        highlight = {k: v for k, v in highlight.items() if v is not None}
        self.highlights.append(highlight)


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
        # Try common patterns
        if hasattr(self, "_highlighter"):
            return self._highlighter
        elif hasattr(self, "page") and hasattr(self.page, "_highlighter"):
            return self.page._highlighter
        elif hasattr(self, "pages") and self.pages:
            # For collections, use first page's highlighter
            first_page = next(iter(self.pages))
            if hasattr(first_page, "_highlighter"):
                return first_page._highlighter

        raise RuntimeError(
            f"Cannot find HighlightingService for {self.__class__.__name__}. "
            "Override _get_highlighter() to provide access."
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
        highlights: Optional[Union[List[Dict[str, Any]], bool]] = None,
        legend_position: str = "right",
        annotate: Optional[Union[str, List[str]]] = None,
        # Layout options for multi-page/region
        layout: Optional[Literal["stack", "grid", "single"]] = None,
        stack_direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 5,
        columns: Optional[int] = 6,  # For grid layout, defaults to 6 columns
        limit: Optional[int] = 30,  # Max pages to show (default 30)
        # Cropping options
        crop: Union[bool, int, str, "Region", Literal["wide"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> Optional["PIL_Image"]:
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
        # Convert string to list if needed
        if isinstance(annotate, str):
            annotate = [annotate]

        # Handle 'cols' as an alias for 'columns' for backward compatibility
        if "cols" in kwargs and columns == 6:  # Only use cols if columns wasn't explicitly set
            columns = kwargs.pop("cols")
            logger.info(f"Using 'cols' parameter as alias for 'columns': {columns}")

        # Pass limit as max_pages to _get_render_specs
        if limit is not None:
            kwargs["max_pages"] = limit

        specs = self._get_render_specs(
            mode="show",
            color=color,
            highlights=highlights,
            crop=crop,
            crop_bbox=crop_bbox,
            annotate=annotate,
            **kwargs,
        )

        if not specs:
            logger.warning(f"{self.__class__.__name__}.show() generated no render specs")
            return None

        # Determine default layout based on content and parameters
        if layout is None:
            # For PDFs and multi-page collections, default to grid with 6 columns
            if len(specs) > 1:
                layout = "grid"
            else:
                layout = "single"

        highlighter = self._get_highlighter()
        return highlighter.unified_render(
            specs=specs,
            resolution=resolution,
            width=width,
            labels=labels,
            label_format=label_format,
            legend_position=legend_position,
            layout=layout,
            stack_direction=stack_direction,
            gap=gap,
            columns=columns,
            **kwargs,
        )

    def render(
        self,
        *,
        # Basic rendering options
        resolution: Optional[float] = None,
        width: Optional[int] = None,
        # Layout options for multi-page/region
        layout: Literal["stack", "grid", "single"] = "stack",
        stack_direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 5,
        columns: Optional[int] = None,
        # Cropping options
        crop: Union[bool, int, str, "Region", Literal["wide"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> Optional["PIL_Image"]:
        """Generate a clean image without highlights.

        This method produces publication-ready images without
        any debugging annotations or highlights.

        Args:
            resolution: DPI for rendering (default from global settings)
            width: Target width in pixels (overrides resolution)
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
        # Handle 'cols' as an alias for 'columns' for backward compatibility
        if "cols" in kwargs and columns is None:  # Only use cols if columns wasn't explicitly set
            columns = kwargs.pop("cols")
            logger.info(f"Using 'cols' parameter as alias for 'columns': {columns}")

        specs = self._get_render_specs(mode="render", crop=crop, crop_bbox=crop_bbox, **kwargs)

        if not specs:
            logger.warning(f"{self.__class__.__name__}.render() generated no render specs")
            return None

        highlighter = self._get_highlighter()
        return highlighter.unified_render(
            specs=specs,
            resolution=resolution,
            width=width,
            labels=False,  # Never show labels in render mode
            layout=layout,
            stack_direction=stack_direction,
            gap=gap,
            columns=columns,
            **kwargs,
        )

    def export(
        self,
        path: Union[str, Path],
        *,
        # All the same options as render()
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
        image = self.render(
            resolution=resolution,
            width=width,
            layout=layout,
            stack_direction=stack_direction,
            gap=gap,
            columns=columns,
            crop=crop,
            crop_bbox=crop_bbox,
            **kwargs,
        )

        if image is None:
            raise ValueError(f"No image generated by {self.__class__.__name__}.render()")

        # Ensure path is a Path object
        path = Path(path)

        # Determine format
        if format is None:
            format = path.suffix.lstrip(".").upper()
            if format == "JPG":
                format = "JPEG"

        # Save image
        save_kwargs = {}
        if format == "JPEG":
            save_kwargs["quality"] = kwargs.get("quality", 95)
        elif format == "PNG":
            save_kwargs["compress_level"] = kwargs.get("compress_level", 6)

        image.save(path, format=format, **save_kwargs)
        logger.info(f"Exported {self.__class__.__name__} to {path}")
