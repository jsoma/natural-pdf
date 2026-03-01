"""
Centralized service for managing and rendering highlights in a PDF document.
"""

import logging  # Added
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, cast

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf.core.page import Page
else:  # pragma: no cover - runtime fallback
    Page = Any  # type: ignore[assignment]

from natural_pdf.core.render_spec import ColorInput, RenderSpec
from natural_pdf.elements.base import extract_bbox

# Import ColorManager and related utils
from natural_pdf.utils.visualization import (
    ColorManager,
    create_legend,
    merge_images_with_legend,
    render_plain_page,
)

# Constants for drawing (Can be potentially moved to ColorManager/Renderer if desired)
BORDER_ALPHA = 180  # Default alpha for highlight border
DEFAULT_FALLBACK_COLOR = (255, 255, 0)  # Yellow fallback (RGB only, alpha added by ColorManager)

# Setup logger
logger = logging.getLogger(__name__)

RGBAColor = Tuple[int, int, int, int]

_FONT_FALLBACK = ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf", "FreeSans.ttf"]

# Cached font path: None = not yet probed, False = all failed, str = valid path
_cached_font_path: Union[str, bool, None] = None


def _get_font_path() -> Optional[str]:
    """Find and cache the first available truetype font from the fallback list."""
    global _cached_font_path
    if _cached_font_path is not None:
        return _cached_font_path if _cached_font_path else None
    for fname in _FONT_FALLBACK:
        try:
            ImageFont.truetype(fname, 10)
            _cached_font_path = fname
            return fname
        except (IOError, OSError):
            continue
    _cached_font_path = False
    return None


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try fonts in fallback order, return default if all fail."""
    for name in _FONT_FALLBACK:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def draw_vertices(
    draw: ImageDraw.ImageDraw,
    vertices: List[Tuple[float, float]],
    color: RGBAColor,
    vertex_size: int,
) -> None:
    """Draw small markers at each vertex of a highlight shape.

    Args:
        draw: PIL ImageDraw object to draw on.
        vertices: List of (x, y) pixel coordinates.
        color: RGBA color for the vertex markers.
        vertex_size: Radius in pixels for the ellipse markers.
    """
    for x, y in vertices:
        draw.ellipse(
            [x - vertex_size, y - vertex_size, x + vertex_size, y + vertex_size],
            fill=color,
        )


def render_ocr_overlay(
    page: "Page",
    base_image: Image.Image,
    scale_factor: float,
) -> Image.Image:
    """Render OCR text elements onto an image.

    Pure function that composites OCR text boxes onto the provided image.

    Args:
        page: The Page object to read OCR elements from.
        base_image: The RGBA image to composite onto.
        scale_factor: Scale factor from PDF coordinates to pixel coordinates.

    Returns:
        A new RGBA image with OCR text composited on top.
    """
    result_image = base_image if base_image.mode == "RGBA" else base_image.convert("RGBA")

    # Find OCR elements on the page
    ocr_elements = page.find_all("text[source=ocr]")
    if not ocr_elements:
        ocr_elements = [el for el in page.words if getattr(el, "source", None) == "ocr"]

    if not ocr_elements:
        logger.debug(f"No OCR elements found for page {page.number} to render.")
        return result_image

    overlay = Image.new("RGBA", result_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Find a suitable font path (cached at module level)
    font_path = _get_font_path()

    for element in ocr_elements:
        x0, top, x1, bottom = element.bbox
        x0_s = x0 * scale_factor
        top_s = top * scale_factor
        x1_s = x1 * scale_factor
        bottom_s = bottom * scale_factor
        box_w, box_h = x1_s - x0_s, bottom_s - top_s

        if box_h <= 0:
            continue

        font_size = max(9, int(box_h * 0.85))

        sized_font = _load_font(font_size) if not font_path else None
        if font_path:
            try:
                sized_font = ImageFont.truetype(font_path, font_size)
            except (IOError, OSError):
                sized_font = _load_font(font_size)

        # Adjust font size if text overflows
        try:
            text_w = draw.textlength(element.text, font=sized_font)
            if text_w > box_w * 1.1:
                ratio = max(0.5, (box_w * 1.0) / text_w)
                font_size = max(9, int(font_size * ratio))
                if font_path:
                    try:
                        sized_font = ImageFont.truetype(font_path, font_size)
                    except IOError:
                        pass
        except AttributeError:
            pass

        # Draw background and text
        padding = max(1, int(font_size * 0.05))
        draw.rectangle(
            [x0_s - padding, top_s - padding, x1_s + padding, bottom_s + padding],
            fill=(255, 255, 255, 230),
        )

        text_top_offset = 0
        if hasattr(sized_font, "getbbox"):
            _, text_top_offset, _, text_bottom_offset = sized_font.getbbox(element.text)
            text_h = text_bottom_offset - text_top_offset
        else:
            text_h = font_size
        text_y = top_s + (box_h - text_h) / 2
        text_y -= text_top_offset
        text_x = x0_s + padding

        draw.text((text_x, text_y), element.text, fill=(0, 0, 0, 255), font=sized_font)

    return Image.alpha_composite(result_image, overlay)


@dataclass
class Highlight:
    """
    Represents a single highlight to be drawn.
    Stores geometric data, color, label, and extracted attributes.
    """

    page_index: int
    bbox: Tuple[float, float, float, float]
    color: RGBAColor  # Final RGBA color determined by service
    label: Optional[str] = None
    polygon: Optional[List[Tuple[float, float]]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)  # Store extracted attribute values
    quantitative_metadata: Optional[Dict[str, Any]] = None

    @property
    def is_polygon(self) -> bool:
        """Check if this highlight uses polygon coordinates."""
        return self.polygon is not None and len(self.polygon) >= 3

    @property
    def border_color(self) -> Tuple[int, int, int, int]:
        """Calculate a slightly darker/more opaque border color."""
        # Use base color but increase alpha for border
        return (self.color[0], self.color[1], self.color[2], BORDER_ALPHA)


class HighlightContext:
    """
    Context manager for accumulating highlights before displaying them together.

    This allows for a clean syntax to show multiple highlight groups:

    Example:
        with pdf.highlights() as h:
            h.add(page.find_all('table'), label='tables', color='blue')
            h.add(page.find_all('text:bold'), label='bold text', color='red')
            h.show()  # Display all highlights together

    Or for automatic display on exit:
        with pdf.highlights(show=True) as h:
            h.add(page.find_all('table'), label='tables')
            h.add(page.find_all('text:bold'), label='bold')
            # Automatically shows when exiting the context
    """

    def __init__(self, source, show_on_exit: bool = False):
        """
        Initialize the highlight context.

        Args:
            source: The source object (PDF, Page, PageCollection, etc.)
            show_on_exit: If True, automatically show highlights when exiting context
        """
        self.source = source
        self.show_on_exit = show_on_exit
        self.highlight_groups = []
        self._color_manager = ColorManager()
        self._exit_image = None  # Store image for Jupyter display

    def add(
        self,
        elements,
        label: Optional[str] = None,
        color: Optional[ColorInput] = None,
        **kwargs,
    ) -> "HighlightContext":
        """
        Add a group of elements to highlight.

        Args:
            elements: Elements to highlight (can be ElementCollection, list, or single element)
            label: Label for this highlight group
            color: Color for this group (if None, uses color cycling)
            **kwargs: Additional highlight parameters

        Returns:
            Self for method chaining
        """
        # Convert single element to list
        if hasattr(elements, "elements"):
            # It's an ElementCollection
            element_list = elements.elements
        elif isinstance(elements, list):
            element_list = elements
        else:
            # Single element
            element_list = [elements]

        # Determine color if not specified
        if color is None:
            color = self._color_manager.get_color(label=label, force_cycle=True)

        self.highlight_groups.append(
            {"elements": element_list, "label": label, "color": color, **kwargs}
        )

        return self

    def show(self, **kwargs) -> Optional[Image.Image]:
        """
        Display all accumulated highlights.

        Args:
            **kwargs: Additional parameters passed to the show method

        Returns:
            PIL Image with all highlights, or None if no source
        """
        if not self.source:
            return None

        # If source has the new unified show method, use it with highlights parameter
        if hasattr(self.source, "show"):
            return self.source.show(highlights=self.highlight_groups, **kwargs)
        else:
            # Fallback for objects without the new show method
            logger.warning(
                f"Object {type(self.source)} does not support unified show() with highlights"
            )
            return None

    def render(self, **kwargs) -> Optional[Image.Image]:
        """
        Render all accumulated highlights (clean image without debug elements).

        Args:
            **kwargs: Additional parameters passed to the render method

        Returns:
            PIL Image with all highlights, or None if no source
        """
        if not self.source:
            return None

        # If source has the new unified render method, use it with highlights parameter
        if hasattr(self.source, "render"):
            return self.source.render(highlights=self.highlight_groups, **kwargs)
        else:
            # Fallback for objects without the new render method
            logger.warning(
                f"Object {type(self.source)} does not support unified render() with highlights"
            )
            return None

    @property
    def image(self) -> Optional[Image.Image]:
        """Get the last generated image (useful after context exit)."""
        return self._exit_image

    def __enter__(self) -> "HighlightContext":
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, optionally showing highlights."""
        if self.show_on_exit and not exc_type:
            self._exit_image = self.show()

            # Check if we're in a Jupyter/IPython environment
            try:
                # Try to get IPython instance
                from IPython.core.getipython import get_ipython

                ipython = get_ipython()
                if ipython is not None:
                    # We're in IPython/Jupyter
                    from IPython.display import display

                    if self._exit_image is not None:
                        display(self._exit_image)
            except (ImportError, NameError):
                # Not in Jupyter or IPython not available - that's OK
                pass

        # __exit__ must return False to not suppress exceptions
        return False


class HighlightingService:
    """
    Central service to manage highlight data and orchestrate rendering.
    Holds the state of all highlights across the document.
    """

    def __init__(self, pdf_object):
        self._pdf = pdf_object  # Reference to the parent PDF object
        self._highlights_by_page: Dict[int, List[Highlight]] = {}
        self._color_manager = ColorManager()  # Instantiate the color manager
        logger.info("HighlightingService initialized with ColorManager.")

    # Removed _get_next_color - logic moved to ColorManager
    # Removed _color_cycle, _labels_colors - managed by ColorManager

    def _process_color_input(self, color_input: Optional[ColorInput]) -> Optional[RGBAColor]:
        """
        Parses various color input formats into a standard RGBA tuple (0-255).
        Returns None if input is invalid.
        """
        if color_input is None:
            return None

        if isinstance(color_input, tuple):
            # Convert float values (0.0-1.0) to int (0-255)
            processed = []
            all_float = all(isinstance(c, float) and 0.0 <= c <= 1.0 for c in color_input[:3])

            for i, c in enumerate(color_input):
                if isinstance(c, float):
                    val = (
                        int(c * 255)
                        if (i < 3 and all_float) or (i == 3 and 0.0 <= c <= 1.0)
                        else int(c)
                    )
                elif isinstance(c, int):
                    val = c
                else:
                    logger.warning(f"Invalid color component type: {c} in {color_input}")
                    return None  # Invalid type
                processed.append(max(0, min(255, val)))  # Clamp to 0-255

            # Check length and add default alpha if needed
            if len(processed) == 3:
                # Use alpha from ColorManager instance
                processed.append(self._color_manager._alpha)
                return cast(RGBAColor, tuple(processed))
            elif len(processed) == 4:
                return cast(RGBAColor, tuple(processed))
            else:
                logger.warning(f"Invalid color tuple length: {color_input}")
                return None  # Invalid length

        elif isinstance(color_input, str):
            try:
                # Convert color name/hex string to RGB tuple (0.0-1.0 floats)
                from colour import Color  # type: ignore[import-untyped]

                color_obj = Color(color_input)
                # Convert floats (0.0-1.0) to integers (0-255)
                r = int(color_obj.red * 255)
                g = int(color_obj.green * 255)
                b = int(color_obj.blue * 255)
                # Clamp values just in case
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                # Add alpha
                rgba: RGBAColor = (r, g, b, self._color_manager._alpha)
                return rgba
            except ImportError:
                logger.error("Color utility class not found. Cannot process string colors.")
                return None
            except ValueError:
                logger.warning(f"Invalid color string: '{color_input}'")
                return None
        else:
            logger.warning(f"Invalid color input type: {type(color_input)}")
            return None

    def _determine_highlight_color(
        self,
        color_input: Optional[ColorInput] = None,
        label: Optional[str] = None,
        use_color_cycling: bool = False,
    ) -> RGBAColor:
        """
        Determines the final RGBA color for a highlight using the ColorManager.

        Args:
            color_input: User-provided color (tuple or string).
            label: Label associated with the highlight.
            use_color_cycling: Whether to force cycling (ignores label).

        Returns:
            RGBA color tuple (0-255).
        """
        explicit_color = self._process_color_input(color_input)

        if explicit_color:
            # If a valid color was explicitly provided, use it
            return explicit_color
        else:
            # Otherwise, use the color manager to get a color based on label/cycling
            return cast(
                RGBAColor, self._color_manager.get_color(label=label, force_cycle=use_color_cycling)
            )

    def add(
        self,
        page_index: int,
        bbox: Any,
        color: Optional[ColorInput] = None,
        label: Optional[str] = None,
        use_color_cycling: bool = False,
        element: Optional[Any] = None,
        annotate: Optional[List[str]] = None,
        existing: str = "append",
    ):
        """Adds a rectangular highlight."""

        bbox_tuple = extract_bbox(bbox)
        if bbox_tuple is None:
            logger.error(
                f"Invalid bbox type or structure provided for page {page_index}: {type(bbox)} - {bbox}."
            )
            return

        processed_bbox = cast(Tuple[float, float, float, float], bbox_tuple)

        self._add_internal(
            page_index=page_index,
            bbox=processed_bbox,  # Use the processed tuple
            polygon=None,
            color_input=color,
            label=label,
            use_color_cycling=use_color_cycling,
            element=element,
            annotate=annotate,
            existing=existing,
        )

    def add_polygon(
        self,
        page_index: int,
        polygon: List[Tuple[float, float]],
        color: Optional[Union[Tuple, str]] = None,
        label: Optional[str] = None,
        use_color_cycling: bool = False,
        element: Optional[Any] = None,
        annotate: Optional[List[str]] = None,
        existing: str = "append",
    ):
        """Adds a polygonal highlight."""
        # Calculate bounding box from polygon for internal storage
        if polygon and len(polygon) >= 3:
            x_coords = [p[0] for p in polygon]
            y_coords = [p[1] for p in polygon]
            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        else:
            logger.warning(f"Invalid polygon provided for page {page_index}. Cannot add highlight.")
            return

        self._add_internal(
            page_index=page_index,
            bbox=bbox,
            polygon=polygon,
            color_input=color,
            label=label,
            use_color_cycling=use_color_cycling,
            element=element,
            annotate=annotate,
            existing=existing,
        )

    def _add_internal(
        self,
        page_index: int,
        bbox: Tuple[float, float, float, float],
        polygon: Optional[List[Tuple[float, float]]],
        color_input: Optional[ColorInput],
        label: Optional[str],
        use_color_cycling: bool,
        element: Optional[Any],
        annotate: Optional[List[str]],
        existing: str,
    ):
        """Internal method to create and store a Highlight object."""
        if page_index < 0 or page_index >= len(self._pdf.pages):
            logger.error(f"Invalid page index {page_index}. Cannot add highlight.")
            return

        # Handle 'replace' logic - clear highlights for this page *before* adding new one
        if existing == "replace":
            self.clear_page(page_index)

        # Determine the final color using the ColorManager
        final_color = self._determine_highlight_color(
            color_input=color_input, label=label, use_color_cycling=use_color_cycling
        )

        # Extract attributes from the element if requested
        attributes_to_draw = {}
        if element and annotate:
            for attr_name in annotate:
                try:
                    attr_value = getattr(element, attr_name, None)
                    if attr_value is not None:
                        attributes_to_draw[attr_name] = attr_value
                except AttributeError:
                    logger.warning(f"Attribute '{attr_name}' not found on element {element}")

        # Create the highlight data object
        highlight = Highlight(
            page_index=page_index,
            bbox=bbox,
            color=final_color,
            label=label,
            polygon=polygon,
            attributes=attributes_to_draw,
        )

        # Add to the list for the specific page
        if page_index not in self._highlights_by_page:
            self._highlights_by_page[page_index] = []
        self._highlights_by_page[page_index].append(highlight)
        logger.debug(f"Added highlight to page {page_index}: {highlight}")

        # --- Invalidate page-level image cache --------------------------------
        # The Page.render method maintains an internal cache keyed by rendering
        # parameters.  Because the cache key currently does **not** incorporate
        # any information about the highlights themselves, it can return stale
        # images after highlights are added or removed.  To ensure the next
        # render reflects the new highlights, we clear the cache for the
        # affected page here.
        try:
            page_obj = self._pdf[page_index]
            if hasattr(page_obj, "_to_image_cache"):
                page_obj._to_image_cache.clear()
                logger.debug(
                    f"Cleared cached render images for page {page_index} after adding a highlight."
                )
        except Exception as cache_err:  # pragma: no cover – never fail highlight creation
            logger.warning(
                f"Failed to invalidate render cache for page {page_index}: {cache_err}",
                exc_info=True,
            )

    def clear_all(self):
        """Clears all highlights from all pages and resets the color manager."""
        self._highlights_by_page = {}
        self._color_manager.reset()
        logger.info("Cleared all highlights and reset ColorManager.")

        # Clear cached images for *all* pages because their visual state may
        # depend on highlight visibility.
        for idx, page in enumerate(self._pdf.pages):
            try:
                if hasattr(page, "_to_image_cache"):
                    page._to_image_cache.clear()
            except Exception:
                # Non-critical – keep going for remaining pages
                continue

    def clear_page(self, page_index: int):
        """Clears all highlights from a specific page."""
        if page_index in self._highlights_by_page:
            del self._highlights_by_page[page_index]
            logger.debug(f"Cleared highlights for page {page_index}.")

        # Also clear any cached rendered images for this page so the next render
        # reflects the removal of highlights.
        try:
            page_obj = self._pdf[page_index]
            if hasattr(page_obj, "_to_image_cache"):
                page_obj._to_image_cache.clear()
                logger.debug(
                    f"Cleared cached render images for page {page_index} after removing highlights."
                )
        except Exception as cache_err:  # pragma: no cover
            logger.warning(
                f"Failed to invalidate render cache for page {page_index}: {cache_err}",
                exc_info=True,
            )

    def get_highlights_for_page(self, page_index: int) -> List[Highlight]:
        """Returns a list of Highlight objects for a specific page."""
        return self._highlights_by_page.get(page_index, [])

    def get_labels_and_colors(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Returns a mapping of labels used to their assigned colors (for persistent highlights)."""
        return self._color_manager.get_label_colors()

    def render_preview(
        self,
        page_index: int,
        temporary_highlights: List[Dict],
        resolution: float = 144,
        labels: bool = True,
        legend_position: str = "right",
        render_ocr: bool = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> Optional[Image.Image]:
        """
        Renders a preview image for a specific page containing only the
        provided temporary highlights. Does not affect persistent state.

        Delegates to ``_render_spec`` via a temporary ``RenderSpec``.

        Args:
            page_index: Index of the page to render.
            temporary_highlights: List of highlight data dicts.
            resolution: Resolution (DPI) for base page image rendering.
            labels: Whether to include a legend.
            legend_position: Position of the legend.
            render_ocr: Whether to render OCR text.
            crop_bbox: Optional bounding box in PDF coordinates to crop to.
            **kwargs: Additional args (e.g., width, height).

        Returns:
            PIL Image of the preview, or None if rendering fails.
        """
        if page_index < 0 or page_index >= len(self._pdf.pages):
            logger.error(f"Invalid page index {page_index} for render_preview.")
            return None

        page_obj = self._pdf.pages[page_index]

        # Build a RenderSpec from the temporary highlights
        from natural_pdf.core.render_spec import RenderSpec

        spec = RenderSpec(page=page_obj, crop_bbox=crop_bbox)

        for hl_data in temporary_highlights:
            # Resolve color using the same logic as before
            final_color = self._determine_highlight_color(
                color_input=hl_data.get("color"),
                label=hl_data.get("label"),
                use_color_cycling=hl_data.get("use_color_cycling", False),
            )

            # Extract attributes from element if requested
            attrs_to_draw = {}
            element = hl_data.get("element")
            annotate = hl_data.get("annotate")
            if element and annotate:
                for attr_name in annotate:
                    try:
                        attr_value = getattr(element, attr_name, None)
                        if attr_value is not None:
                            attrs_to_draw[attr_name] = attr_value
                    except AttributeError:
                        logger.warning(f"Attribute '{attr_name}' not found on element {element}")

            # Resolve bbox
            bbox_value = extract_bbox(cast(Any, hl_data.get("bbox")))
            polygon_value = hl_data.get("polygon")
            if bbox_value is None and not polygon_value:
                continue

            highlight_entry: Dict[str, Any] = {
                "color": final_color,
                "label": hl_data.get("label"),
            }
            if bbox_value is not None:
                highlight_entry["bbox"] = cast(Tuple[float, float, float, float], bbox_value)
            if polygon_value:
                highlight_entry["polygon"] = polygon_value
                if bbox_value is None:
                    xs = [pt[0] for pt in polygon_value]
                    ys = [pt[1] for pt in polygon_value]
                    highlight_entry["bbox"] = (min(xs), min(ys), max(xs), max(ys))
            if attrs_to_draw:
                highlight_entry["attributes_to_draw"] = attrs_to_draw

            quantitative_metadata = hl_data.get("quantitative_metadata")
            if quantitative_metadata:
                highlight_entry["quantitative_metadata"] = quantitative_metadata

            spec.highlights.append(highlight_entry)

        # Extract width from kwargs for _render_spec
        render_width = kwargs.pop("width", None)

        return self._render_spec(
            spec=spec,
            resolution=resolution if resolution is not None else 144,
            width=render_width,
            labels=labels,
            label_format=None,
            render_ocr=render_ocr,
            legend_position=legend_position,
            spec_index=0,
            **kwargs,
        )

    def unified_render(
        self,
        specs: List["RenderSpec"],
        resolution: float = 150,
        width: Optional[int] = None,
        labels: bool = True,
        label_format: Optional[str] = None,
        render_ocr: bool = False,
        layout: Literal["stack", "grid", "single"] = "stack",
        stack_direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 5,
        columns: Optional[int] = None,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        legend_position: str = "right",
        **kwargs,
    ) -> Optional[Image.Image]:
        """
        Unified rendering method that processes RenderSpec objects.

        This is the single entry point for all image generation in natural-pdf.
        It handles page rendering, cropping, highlighting, and layout of multiple images.

        Args:
            specs: List of RenderSpec objects describing what to render
            resolution: DPI for rendering (default 150)
            width: Target width in pixels (overrides resolution)
            labels: Whether to show labels for highlights
            label_format: Format string for labels
            render_ocr: Whether to render OCR text overlay
            layout: How to arrange multiple images
            stack_direction: Direction for stack layout
            gap: Pixels between images
            columns: Number of columns for grid layout
            background_color: RGB color for background
            **kwargs: Additional parameters

        Returns:
            PIL Image or None if nothing to render
        """
        from natural_pdf.core.render_spec import RenderSpec

        if not specs:
            raise ValueError("unified_render requires at least one RenderSpec")

        # Process each spec into an image
        images = []

        for spec_idx, spec in enumerate(specs):
            if not isinstance(spec, RenderSpec):
                raise TypeError(f"Spec index {spec_idx} expected RenderSpec, got {type(spec)}")

            page_image = self._render_spec(
                spec=spec,
                resolution=resolution,
                width=width,
                labels=labels,
                label_format=label_format,
                render_ocr=render_ocr,
                legend_position=legend_position,
                spec_index=spec_idx,
                **kwargs,
            )

            if page_image is None:
                raise RuntimeError(f"Render spec {spec_idx} produced no image")

            images.append(page_image)

        if not images:
            raise RuntimeError("unified_render produced no images from specs")

        # Single image - return directly
        if len(images) == 1:
            return images[0]

        # Multiple images - apply layout
        if layout == "stack":
            return self._stack_images(
                images, direction=stack_direction, gap=gap, background_color=background_color
            )
        elif layout == "grid":
            return self._grid_images(
                images, columns=columns, gap=gap, background_color=background_color
            )
        else:  # "single"
            if len(images) > 1:
                raise ValueError("layout='single' cannot be used with multiple specs")
            return images[0]

    def _render_spec(
        self,
        spec: "RenderSpec",
        resolution: float,
        width: Optional[int],
        labels: bool,
        label_format: Optional[str],
        render_ocr: bool = False,
        legend_position: str = "right",
        spec_index: int = 0,
        **kwargs,
    ) -> Optional[Image.Image]:
        """Render a single RenderSpec to an image."""
        # Get the page
        page = spec.page
        if not hasattr(page, "width") or not hasattr(page, "height"):
            logger.error(f"Spec {spec_index} page does not have width/height attributes")
            return None

        # Calculate actual resolution/width
        target_width = width  # may be None
        base_resolution = resolution if resolution is not None else 150

        if target_width is not None and page.width > 0:
            # Resolution needed for exact pixel width
            width_resolution = (target_width / page.width) * 72
            # Never render below the base resolution — render high, resize after
            actual_resolution = max(width_resolution, base_resolution)
        else:
            actual_resolution = base_resolution

        scale_factor = actual_resolution / 72

        # Get base page image
        logger.debug(f"Calling render_plain_page with page={page}, resolution={actual_resolution}")
        page_image = render_plain_page(page, resolution=actual_resolution)
        if page_image is None:
            raise RuntimeError(f"render_plain_page returned None for page {page}")

        # Apply crop if specified
        if spec.crop_bbox:
            page_image = self._crop_image(page_image, spec.crop_bbox, page, scale_factor)

        # Apply highlights if any
        if spec.highlights:
            page_image = self._apply_spec_highlights(
                page_image,
                spec.highlights,
                page,
                scale_factor,
                labels=labels,
                label_format=label_format,
                spec_index=spec_index,
                crop_offset=spec.crop_bbox[:2] if spec.crop_bbox else None,
            )

        # Apply OCR overlay (after highlights, before legend)
        if render_ocr:
            page_image = render_ocr_overlay(page, page_image, scale_factor)

        # Add legend or colorbar if labels are enabled
        if spec.highlights and labels:
            from natural_pdf.utils.visualization import (
                create_colorbar,
                create_legend,
                merge_images_with_legend,
            )

            # Check if we have quantitative metadata (for colorbar)
            quantitative_metadata = None
            for highlight_data in spec.highlights:
                if (
                    "quantitative_metadata" in highlight_data
                    and highlight_data["quantitative_metadata"]
                ):
                    quantitative_metadata = highlight_data["quantitative_metadata"]
                    break

            if quantitative_metadata:
                colorbar = create_colorbar(
                    values=quantitative_metadata["values"],
                    colormap=quantitative_metadata["colormap"],
                    bins=quantitative_metadata["bins"],
                    orientation=(
                        "horizontal" if legend_position in ["top", "bottom"] else "vertical"
                    ),
                )
                page_image = merge_images_with_legend(
                    page_image, colorbar, position=legend_position
                )
                logger.debug(
                    f"Added colorbar for quantitative attribute '{quantitative_metadata['attribute']}' in spec {spec_index}."
                )

            if not quantitative_metadata:
                # Create regular categorical legend
                spec_labels = {}
                for hl in spec.highlights:
                    label = hl.get("label")
                    color = hl.get("color")
                    if label and color:
                        processed_color = self._process_color_input(color)
                        if processed_color:
                            spec_labels[label] = processed_color
                        else:
                            spec_labels[label] = self._color_manager.get_color(label=label)

                if spec_labels:
                    legend = create_legend(spec_labels)
                    if legend:
                        page_image = merge_images_with_legend(
                            page_image, legend, position=legend_position
                        )
                        logger.debug(
                            f"Added legend with {len(spec_labels)} labels for spec {spec_index}."
                        )

        # If we rendered at higher resolution than needed for the target width,
        # resize down with LANCZOS for sharp output.
        if target_width is not None and page_image.width != target_width:
            aspect = page_image.height / page_image.width
            target_height = int(target_width * aspect)
            page_image = page_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        return page_image

    def _crop_image(
        self,
        image: Image.Image,
        crop_bbox: Tuple[float, float, float, float],
        page: "Page",
        scale_factor: float,
    ) -> Image.Image:
        """Crop an image to the specified bbox."""
        # Convert PDF coordinates to pixel coordinates
        x0, y0, x1, y1 = crop_bbox
        pixel_bbox = (
            int(x0 * scale_factor),
            int(y0 * scale_factor),
            int(x1 * scale_factor),
            int(y1 * scale_factor),
        )

        # Ensure valid crop bounds
        pixel_bbox = (
            max(0, pixel_bbox[0]),
            max(0, pixel_bbox[1]),
            min(image.width, pixel_bbox[2]),
            min(image.height, pixel_bbox[3]),
        )

        if pixel_bbox[2] <= pixel_bbox[0] or pixel_bbox[3] <= pixel_bbox[1]:
            raise ValueError(f"Invalid crop bounds: {crop_bbox}")

        return image.crop(pixel_bbox)

    def _apply_spec_highlights(
        self,
        image: Image.Image,
        highlights: List[Dict[str, Any]],
        page: "Page",
        scale_factor: float,
        labels: bool,
        label_format: Optional[str],
        spec_index: int,
        crop_offset: Optional[Tuple[float, float]] = None,
    ) -> Image.Image:
        """Apply highlights from a RenderSpec to an image."""
        # Convert to RGBA for transparency
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Create overlay for highlights
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Process each highlight
        for idx, highlight_dict in enumerate(highlights):
            # Get geometry
            bbox = highlight_dict.get("bbox")
            polygon = highlight_dict.get("polygon")

            if bbox is None and polygon is None:
                raise ValueError(f"Highlight {idx} lacks geometry (bbox or polygon required)")

            # Get color
            color = highlight_dict.get("color")
            label = highlight_dict.get("label")

            if color is None:
                # Use label-based color assignment for consistency
                color = self._color_manager.get_color(label=label, force_cycle=False)
            else:
                # Process color input
                color = self._process_color_input(color)
                if color is None:
                    color = self._color_manager.get_color(label=label, force_cycle=False)

            # Generate label if needed
            if label is None and labels and label_format:
                # Generate label from format
                label = label_format.format(index=idx, spec_index=spec_index, total=len(highlights))

            # Calculate offset for cropped images
            offset_x = 0
            offset_y = 0
            if crop_offset:
                offset_x = crop_offset[0] * scale_factor
                offset_y = crop_offset[1] * scale_factor

            # Add pdfplumber page offset for coordinate translation
            page_offset_x = 0
            page_offset_y = 0
            if hasattr(page, "_page") and hasattr(page._page, "bbox"):
                # PDFPlumber page bbox might have negative offsets
                page_offset_x = -page._page.bbox[0]
                page_offset_y = -page._page.bbox[1]

            # Draw the highlight
            border_color = (color[0], color[1], color[2], BORDER_ALPHA)
            vertex_size = max(3, int(2 * scale_factor))
            scaled_bbox = None

            if polygon is not None:
                # Scale polygon points and apply offset
                scaled_polygon = [
                    (
                        (p[0] + page_offset_x) * scale_factor - offset_x,
                        (p[1] + page_offset_y) * scale_factor - offset_y,
                    )
                    for p in polygon
                ]
                draw.polygon(scaled_polygon, fill=color, outline=border_color)
                draw_vertices(draw, scaled_polygon, border_color, vertex_size)
                # Compute bounding box from polygon for attribute drawing
                xs = [p[0] for p in scaled_polygon]
                ys = [p[1] for p in scaled_polygon]
                scaled_bbox = [min(xs), min(ys), max(xs), max(ys)]
            elif bbox is not None:
                # Scale bbox and apply offset
                x0, y0, x1, y1 = bbox
                scaled_bbox = [
                    (x0 + page_offset_x) * scale_factor - offset_x,
                    (y0 + page_offset_y) * scale_factor - offset_y,
                    (x1 + page_offset_x) * scale_factor - offset_x,
                    (y1 + page_offset_y) * scale_factor - offset_y,
                ]
                draw.rectangle(scaled_bbox, fill=color, outline=border_color)
                vertices = [
                    (scaled_bbox[0], scaled_bbox[1]),
                    (scaled_bbox[2], scaled_bbox[1]),
                    (scaled_bbox[2], scaled_bbox[3]),
                    (scaled_bbox[0], scaled_bbox[3]),
                ]
                draw_vertices(draw, vertices, border_color, vertex_size)

            # Draw attributes if present
            if scaled_bbox is not None:
                attributes_to_draw = highlight_dict.get("attributes_to_draw")
                if attributes_to_draw:
                    self._draw_spec_attributes(draw, attributes_to_draw, scaled_bbox, scale_factor)

        # Composite overlay onto image
        return Image.alpha_composite(image, overlay)

    def _draw_spec_attributes(
        self,
        draw: ImageDraw.ImageDraw,
        attributes: Dict[str, Any],
        bbox_scaled: List[float],
        scale_factor: float,
    ) -> None:
        """Draw attribute key-value pairs on the highlight."""
        font_size = max(10, int(8 * scale_factor))
        font = _load_font(font_size)

        line_height = font_size + int(4 * scale_factor)  # Scaled line spacing
        bg_padding = int(3 * scale_factor)
        max_width = 0
        text_lines = []

        # Format attribute lines
        for name, value in attributes.items():
            if isinstance(value, float):
                value_str = f"{value:.2f}"  # Format floats
            else:
                value_str = str(value)
            line = f"{name}: {value_str}"
            text_lines.append(line)
            try:
                # Calculate max width for background box
                max_width = max(max_width, draw.textlength(line, font=font))
            except AttributeError:
                # Fallback for older PIL versions
                bbox = draw.textbbox((0, 0), line, font=font)
                max_width = max(max_width, bbox[2] - bbox[0])

        if not text_lines:
            return  # Nothing to draw

        total_height = line_height * len(text_lines)

        # Position near top-right corner with padding
        x = bbox_scaled[2] - int(2 * scale_factor) - max_width
        y = bbox_scaled[1] + int(2 * scale_factor)

        # Draw background rectangle (semi-transparent white)
        bg_x0 = x - bg_padding
        bg_y0 = y - bg_padding
        bg_x1 = x + max_width + bg_padding
        bg_y1 = y + total_height + bg_padding
        draw.rectangle(
            [bg_x0, bg_y0, bg_x1, bg_y1],
            fill=(255, 255, 255, 240),
            outline=(0, 0, 0, 180),  # Light black outline
            width=1,
        )

        # Draw text lines (black)
        current_y = y
        for line in text_lines:
            draw.text((x, current_y), line, fill=(0, 0, 0, 255), font=font)
            current_y += line_height

    def _stack_images(
        self,
        images: List[Image.Image],
        direction: str,
        gap: int,
        background_color: Tuple[int, int, int],
    ) -> Image.Image:
        """Stack images vertically or horizontally."""
        if direction == "vertical":
            # Calculate dimensions
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images) + gap * (len(images) - 1)

            # Create canvas
            canvas = Image.new("RGB", (max_width, total_height), background_color)

            # Paste images
            y_offset = 0
            for img in images:
                # Center horizontally
                x_offset = (max_width - img.width) // 2
                # Convert RGBA to RGB if needed
                if img.mode == "RGBA":
                    # Create white background
                    bg = Image.new("RGB", img.size, background_color)
                    bg.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                    img = bg
                canvas.paste(img, (x_offset, y_offset))
                y_offset += img.height + gap

        else:  # horizontal
            # Calculate dimensions
            total_width = sum(img.width for img in images) + gap * (len(images) - 1)
            max_height = max(img.height for img in images)

            # Create canvas
            canvas = Image.new("RGB", (total_width, max_height), background_color)

            # Paste images
            x_offset = 0
            for img in images:
                # Center vertically
                y_offset = (max_height - img.height) // 2
                # Convert RGBA to RGB if needed
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, background_color)
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                canvas.paste(img, (x_offset, y_offset))
                x_offset += img.width + gap

        return canvas

    def _grid_images(
        self,
        images: List[Image.Image],
        columns: Optional[int],
        gap: int,
        background_color: Tuple[int, int, int],
    ) -> Image.Image:
        """Arrange images in a grid."""
        n_images = len(images)

        # Determine grid dimensions
        if columns is None:
            # Auto-calculate columns for roughly square grid
            columns = int(n_images**0.5)
            if columns * columns < n_images:
                columns += 1

        rows = (n_images + columns - 1) // columns  # Ceiling division

        # Get max dimensions for cells
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)

        # Calculate canvas size
        canvas_width = columns * max_width + (columns - 1) * gap
        canvas_height = rows * max_height + (rows - 1) * gap

        # Create canvas
        canvas = Image.new("RGB", (canvas_width, canvas_height), background_color)

        # Place images
        for idx, img in enumerate(images):
            row = idx // columns
            col = idx % columns

            # Calculate position (centered in cell)
            cell_x = col * (max_width + gap)
            cell_y = row * (max_height + gap)
            x_offset = cell_x + (max_width - img.width) // 2
            y_offset = cell_y + (max_height - img.height) // 2

            # Convert RGBA to RGB if needed
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, background_color)
                bg.paste(img, mask=img.split()[3])
                img = bg

            canvas.paste(img, (x_offset, y_offset))

        return canvas
