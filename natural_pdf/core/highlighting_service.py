"""
Centralized service for managing and rendering highlights in a PDF document.
"""

import io
import logging  # Added
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from colour import Color
from PIL import Image, ImageDraw, ImageFont

# Attempt to import Page for type hinting safely
try:
    from .page import Page
except ImportError:
    Page = Any  # Fallback if circular import issue arises during type checking

from natural_pdf.core.render_spec import RenderSpec

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


@dataclass
class Highlight:
    """
    Represents a single highlight to be drawn.
    Stores geometric data, color, label, and extracted attributes.
    """

    page_index: int
    bbox: Tuple[float, float, float, float]
    color: Tuple[int, int, int, int]  # Final RGBA color determined by service
    label: Optional[str] = None
    polygon: Optional[List[Tuple[float, float]]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)  # Store extracted attribute values

    @property
    def is_polygon(self) -> bool:
        """Check if this highlight uses polygon coordinates."""
        return self.polygon is not None and len(self.polygon) >= 3

    @property
    def border_color(self) -> Tuple[int, int, int, int]:
        """Calculate a slightly darker/more opaque border color."""
        # Use base color but increase alpha for border
        return (self.color[0], self.color[1], self.color[2], BORDER_ALPHA)


class HighlightRenderer:
    """
    Handles the drawing logic for highlights on a single page image.
    Instantiated by HighlightingService for each render request.
    """

    def __init__(
        self,
        page: Page,
        base_image: Image.Image,
        highlights: List[Highlight],
        scale_factor: float,
        render_ocr: bool,
    ):
        self.page = page  # Keep page reference for OCR rendering
        self.base_image = base_image.convert("RGBA")  # Ensure RGBA
        self.highlights = highlights
        self.scale_factor = scale_factor  # Renamed from scale to scale_factor for clarity
        self.render_ocr = render_ocr
        self.result_image = self.base_image.copy()
        self.vertex_size = max(3, int(2 * self.scale_factor))  # Size of corner markers

    def render(self) -> Image.Image:
        """Executes the rendering process."""
        self._draw_highlights()
        if self.render_ocr:
            self._render_ocr_text()
        return self.result_image

    def _draw_highlights(self):
        """Draws all highlight shapes, borders, vertices, and attributes."""
        for highlight in self.highlights:
            # Create a transparent overlay for this single highlight
            overlay = Image.new("RGBA", self.base_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            scaled_bbox = None

            if highlight.is_polygon:
                scaled_polygon = [
                    (p[0] * self.scale_factor, p[1] * self.scale_factor) for p in highlight.polygon
                ]
                # Draw polygon fill and border
                draw.polygon(
                    scaled_polygon, fill=highlight.color, outline=highlight.border_color, width=2
                )
                self._draw_vertices(draw, scaled_polygon, highlight.border_color)

                # Calculate scaled bbox for attribute drawing
                x_coords = [p[0] for p in scaled_polygon]
                y_coords = [p[1] for p in scaled_polygon]
                scaled_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

            else:  # Rectangle
                x0, top, x1, bottom = highlight.bbox
                x0_s, top_s, x1_s, bottom_s = (
                    x0 * self.scale_factor,
                    top * self.scale_factor,
                    x1 * self.scale_factor,
                    bottom * self.scale_factor,
                )
                scaled_bbox = [x0_s, top_s, x1_s, bottom_s]
                # Draw rectangle fill and border
                draw.rectangle(
                    scaled_bbox, fill=highlight.color, outline=highlight.border_color, width=2
                )

                vertices = [(x0_s, top_s), (x1_s, top_s), (x1_s, bottom_s), (x0_s, bottom_s)]
                self._draw_vertices(draw, vertices, highlight.border_color)

            # Draw attributes if present on the highlight object
            if highlight.attributes and scaled_bbox:  # Ensure bbox is calculated
                self._draw_attributes(draw, highlight.attributes, scaled_bbox)

            # Composite this highlight's overlay onto the result using alpha blending
            self.result_image = Image.alpha_composite(self.result_image, overlay)

    def _draw_vertices(
        self,
        draw: ImageDraw.Draw,
        vertices: List[Tuple[float, float]],
        color: Tuple[int, int, int, int],
    ):
        """Draw small markers at each vertex."""
        for x, y in vertices:
            # Draw ellipse centered at vertex
            draw.ellipse(
                [
                    x - self.vertex_size,
                    y - self.vertex_size,
                    x + self.vertex_size,
                    y + self.vertex_size,
                ],
                fill=color,  # Use border color for vertices
            )

    def _draw_attributes(
        self, draw: ImageDraw.Draw, attributes: Dict[str, Any], bbox_scaled: List[float]
    ):
        """Draws attribute key-value pairs on the highlight."""
        try:
            # Slightly larger font, scaled
            font_size = max(10, int(8 * self.scale_factor))
            # Prioritize monospace fonts for better alignment
            font = ImageFont.truetype("Arial.ttf", font_size)  # Fallback sans-serif
        except IOError:
            font = ImageFont.load_default()
            font_size = 10  # Reset size for default font

        line_height = font_size + int(4 * self.scale_factor)  # Scaled line spacing
        bg_padding = int(3 * self.scale_factor)
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
                pass  # Ignore if textlength not available

        if not text_lines:
            return  # Nothing to draw

        total_height = line_height * len(text_lines)

        # Position near top-right corner with padding
        x = bbox_scaled[2] - int(2 * self.scale_factor) - max_width
        y = bbox_scaled[1] + int(2 * self.scale_factor)

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

    def _render_ocr_text(self):
        """Renders OCR text onto the image. (Adapted from old HighlightManager)"""
        # Use the page reference to get OCR elements
        # Try finding first, then extracting if necessary
        ocr_elements = self.page.find_all("text[source=ocr]")
        if not ocr_elements:
            # Don't run full OCR here, just extract if already run
            ocr_elements = [el for el in self.page.words if getattr(el, "source", None) == "ocr"]
            # Alternative: self.page.extract_ocr_elements() - but might be slow

        if not ocr_elements:
            logger.debug(f"No OCR elements found for page {self.page.number} to render.")
            return

        overlay = Image.new("RGBA", self.base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Find a suitable font
        font_path = None
        default_font = ImageFont.load_default()
        common_fonts = ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf", "FreeSans.ttf"]
        for fname in common_fonts:
            try:
                ImageFont.truetype(fname, 10)  # Test load
                font_path = fname
                break
            except IOError:
                continue

        for element in ocr_elements:
            x0, top, x1, bottom = element.bbox
            x0_s, top_s, x1_s, bottom_s = (
                x0 * self.scale_factor,
                top * self.scale_factor,
                x1 * self.scale_factor,
                bottom * self.scale_factor,
            )
            box_w, box_h = x1_s - x0_s, bottom_s - top_s

            if box_h <= 0:
                continue  # Skip zero-height boxes

            # --- Font Size Calculation ---
            font_size = max(9, int(box_h * 0.85))  # Min size 9, 85% of box height

            try:
                sized_font = ImageFont.truetype(font_path, font_size) if font_path else default_font
            except IOError:
                sized_font = default_font

            # --- Adjust Font Size if Text Overflows ---
            try:
                text_w = draw.textlength(element.text, font=sized_font)
                if text_w > box_w * 1.1:  # Allow 10% overflow
                    ratio = max(0.5, (box_w * 1.0) / text_w)  # Don't shrink below 50%
                    font_size = max(9, int(font_size * ratio))
                    if font_path:
                        try:
                            sized_font = ImageFont.truetype(font_path, font_size)
                        except IOError:
                            pass  # Keep previous if error
            except AttributeError:
                pass  # Skip adjustment if textlength fails

            # --- Draw Background and Text ---
            padding = max(1, int(font_size * 0.05))  # Minimal padding
            draw.rectangle(
                [x0_s - padding, top_s - padding, x1_s + padding, bottom_s + padding],
                fill=(255, 255, 255, 230),  # Highly transparent white background
            )

            # Calculate text position (centered vertically, slightly offset from left)
            if hasattr(sized_font, "getbbox"):  # Modern PIL
                _, text_top_offset, _, text_bottom_offset = sized_font.getbbox(element.text)
                text_h = text_bottom_offset - text_top_offset
            else:  # Older PIL approximation
                text_h = font_size
            text_y = top_s + (box_h - text_h) / 2
            # Adjust for vertical offset in some fonts
            text_y -= text_top_offset if hasattr(sized_font, "getbbox") else 0
            text_x = x0_s + padding  # Start near left edge with padding

            draw.text((text_x, text_y), element.text, fill=(0, 0, 0, 255), font=sized_font)

        # Composite the OCR text overlay onto the result image
        self.result_image = Image.alpha_composite(self.result_image, overlay)


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
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
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
                from IPython import get_ipython

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

    def _process_color_input(
        self, color_input: Optional[Union[Tuple, str]]
    ) -> Optional[Tuple[int, int, int, int]]:
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
                return tuple(processed)
            elif len(processed) == 4:
                return tuple(processed)
            else:
                logger.warning(f"Invalid color tuple length: {color_input}")
                return None  # Invalid length

        elif isinstance(color_input, str):
            try:
                # Convert color name/hex string to RGB tuple (0.0-1.0 floats)
                from colour import Color  # Import here if not at top

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
                rgba = (r, g, b, self._color_manager._alpha)
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
        color_input: Optional[Union[Tuple, str]] = None,
        label: Optional[str] = None,
        use_color_cycling: bool = False,
    ) -> Tuple[int, int, int, int]:
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
            return self._color_manager.get_color(label=label, force_cycle=use_color_cycling)

    def add(
        self,
        page_index: int,
        bbox: Union[Tuple[float, float, float, float], Any],  # Relax input type hint
        color: Optional[Union[Tuple, str]] = None,
        label: Optional[str] = None,
        use_color_cycling: bool = False,
        element: Optional[Any] = None,
        annotate: Optional[List[str]] = None,
        existing: str = "append",
    ):
        """Adds a rectangular highlight."""

        processed_bbox: Tuple[float, float, float, float]
        # Check if bbox is an object with expected attributes (likely a Region)
        # Assuming Region object has x0, top, x1, bottom attributes based on error context
        if (
            hasattr(bbox, "x0")
            and hasattr(bbox, "top")
            and hasattr(bbox, "x1")
            and hasattr(bbox, "bottom")
        ):
            try:
                # Ensure attributes are numeric before creating tuple
                processed_bbox = (
                    float(bbox.x0),
                    float(bbox.top),
                    float(bbox.x1),
                    float(bbox.bottom),
                )
            except (ValueError, TypeError):
                logger.error(
                    f"Invalid attribute types in bbox object for page {page_index}: {bbox}. Expected numeric values."
                )
                return
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                # Ensure elements are numeric and convert to tuple
                processed_bbox = tuple(float(v) for v in bbox)
            except (ValueError, TypeError):
                logger.error(
                    f"Invalid values in bbox sequence for page {page_index}: {bbox}. Expected numeric values."
                )
                return
        else:
            logger.error(
                f"Invalid bbox type or structure provided for page {page_index}: {type(bbox)} - {bbox}. Expected tuple/list of 4 numbers or Region-like object."
            )
            return  # Don't proceed if bbox is invalid

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
        color_input: Optional[Union[Tuple, str]],
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

    def render_page(
        self,
        page_index: int,
        resolution: float = 144,
        labels: bool = True,
        legend_position: str = "right",
        render_ocr: bool = False,
        **kwargs,  # Pass other args to pdfplumber.page.to_image if needed (internal API)
    ) -> Optional[Image.Image]:
        """
        Renders a specific page with its highlights.
        Legend is now generated based only on highlights present on this page.

        Args:
            page_index: The 0-based index of the page to render.
            resolution: Resolution (DPI) for the base page image if width/height not in kwargs.
                       Defaults to 144 DPI (equivalent to previous scale=2.0).
            labels: Whether to include a legend for highlights.
            legend_position: Position of the legend.
            render_ocr: Whether to render OCR text on the image.
            kwargs: Additional keyword arguments for pdfplumber's internal page.to_image (e.g., width, height).

        Returns:
            A PIL Image object of the rendered page, or None if rendering fails.
        """
        if page_index < 0 or page_index >= len(self._pdf.pages):
            logger.error(f"Invalid page index {page_index} for rendering.")
            return None

        page_obj = self._pdf[page_index]  # Renamed to avoid conflict
        highlights_on_page = self.get_highlights_for_page(page_index)

        to_image_args = kwargs.copy()
        actual_scale_x = None
        actual_scale_y = None

        if "width" in to_image_args and to_image_args["width"] is not None:
            logger.debug(f"Rendering page {page_index} with width={to_image_args['width']}.")
            if "height" in to_image_args:
                to_image_args.pop("height", None)
            # Actual scale will be calculated after image creation
        elif "height" in to_image_args and to_image_args["height"] is not None:
            logger.debug(f"Rendering page {page_index} with height={to_image_args['height']}.")
            # Actual scale will be calculated after image creation
        else:
            # Use explicit resolution if provided via kwargs, otherwise fallback to the
            # `resolution` parameter (which might be None).  If we still end up with
            # `None`, default to 144 DPI to avoid downstream errors.
            render_resolution = to_image_args.pop("resolution", resolution)
            if render_resolution is None:
                render_resolution = 144

            # Reinstate into kwargs for pdfplumber
            to_image_args["resolution"] = render_resolution

            actual_scale_x = render_resolution / 72.0
            actual_scale_y = render_resolution / 72.0
            logger.debug(
                f"Rendering page {page_index} with resolution {render_resolution} (scale: {actual_scale_x:.2f})."
            )

        try:
            img_object = page_obj._page.to_image(**to_image_args)
            base_image_pil = (
                img_object.annotated
                if hasattr(img_object, "annotated")
                else img_object._repr_png_()
            )
            if isinstance(base_image_pil, bytes):
                from io import BytesIO

                base_image_pil = Image.open(BytesIO(base_image_pil))
            base_image_pil = base_image_pil.convert("RGBA")  # Ensure RGBA for renderer
            logger.debug(f"Base image for page {page_index} rendered. Size: {base_image_pil.size}.")

            if actual_scale_x is None or actual_scale_y is None:  # If not set by resolution path
                if page_obj.width > 0:
                    actual_scale_x = base_image_pil.width / page_obj.width
                else:
                    actual_scale_x = resolution / 72.0  # Fallback to resolution-based scale
                if page_obj.height > 0:
                    actual_scale_y = base_image_pil.height / page_obj.height
                else:
                    actual_scale_y = resolution / 72.0  # Fallback to resolution-based scale
                logger.debug(
                    f"Calculated actual scales for page {page_index}: x={actual_scale_x:.2f}, y={actual_scale_y:.2f}"
                )

        except IOError as e:
            logger.error(f"IOError creating base image for page {page_index}: {e}")
            raise
        except AttributeError as e:
            logger.error(f"AttributeError creating base image for page {page_index}: {e}")
            raise

        renderer_scale = actual_scale_x  # Assuming aspect ratio maintained, use x_scale

        # --- Render Highlights ---
        rendered_image: Image.Image
        if highlights_on_page:
            renderer = HighlightRenderer(
                page=page_obj,
                base_image=base_image_pil,
                highlights=highlights_on_page,
                scale_factor=renderer_scale,  # Use the determined actual scale
                render_ocr=render_ocr,
            )
            rendered_image = renderer.render()
        else:
            if render_ocr:
                # Still render OCR even if no highlights, using the determined actual scale
                renderer = HighlightRenderer(
                    page=page_obj,
                    base_image=base_image_pil,
                    highlights=[],
                    scale_factor=renderer_scale,
                    render_ocr=True,
                )
                rendered_image = renderer.render()
            else:
                rendered_image = base_image_pil  # No highlights, no OCR requested

        # --- Add Legend or Colorbar (Based ONLY on this page's highlights) ---
        if labels:
            # Check if we have quantitative metadata (for colorbar)
            quantitative_metadata = None
            for hl in highlights_on_page:
                if hasattr(hl, "quantitative_metadata") and hl.quantitative_metadata:
                    quantitative_metadata = hl.quantitative_metadata
                    break

            if quantitative_metadata:
                # Create colorbar for quantitative data
                from natural_pdf.utils.visualization import create_colorbar

                try:
                    colorbar = create_colorbar(
                        values=quantitative_metadata["values"],
                        colormap=quantitative_metadata["colormap"],
                        bins=quantitative_metadata["bins"],
                        orientation=(
                            "horizontal" if legend_position in ["top", "bottom"] else "vertical"
                        ),
                    )
                    rendered_image = merge_images_with_legend(
                        rendered_image, colorbar, legend_position
                    )
                    logger.debug(
                        f"Added colorbar for quantitative attribute '{quantitative_metadata['attribute']}' on page {page_index}."
                    )
                except Exception as e:
                    logger.warning(f"Failed to create colorbar for page {page_index}: {e}")
                    # Fall back to regular legend
                    quantitative_metadata = None

            if not quantitative_metadata:
                # Create regular categorical legend
                labels_colors_on_page: Dict[str, Tuple[int, int, int, int]] = {}
                for hl in highlights_on_page:
                    if hl.label and hl.label not in labels_colors_on_page:
                        labels_colors_on_page[hl.label] = hl.color

                if labels_colors_on_page:  # Only add legend if there are labels on this page
                    legend = create_legend(labels_colors_on_page)
                    if legend:  # Ensure create_legend didn't return None
                        rendered_image = merge_images_with_legend(
                            rendered_image, legend, legend_position
                        )
                        logger.debug(
                            f"Added legend with {len(labels_colors_on_page)} labels for page {page_index}."
                        )
                    else:
                        logger.debug(f"Legend creation returned None for page {page_index}.")
                else:
                    logger.debug(f"No labels found on page {page_index}, skipping legend.")

        return rendered_image

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

        Args:
            page_index: Index of the page to render.
            temporary_highlights: List of highlight data dicts (from ElementCollection._prepare).
            resolution: Resolution (DPI) for base page image rendering if width/height not used.
                       Defaults to 144 DPI (equivalent to previous scale=2.0).
            labels: Whether to include a legend.
            legend_position: Position of the legend.
            render_ocr: Whether to render OCR text.
            crop_bbox: Optional bounding box (x0, top, x1, bottom) in PDF coordinate
                space to crop the output image to, before legends or other overlays are
                applied. If None, no cropping is performed.
            **kwargs: Additional args for pdfplumber's internal to_image (e.g., width, height).

        Returns:
            PIL Image of the preview, or None if rendering fails.
        """
        if page_index < 0 or page_index >= len(self._pdf.pages):
            logger.error(f"Invalid page index {page_index} for render_preview.")
            return None

        page_obj = self._pdf.pages[page_index]

        to_image_args = kwargs.copy()
        actual_scale_x = None
        actual_scale_y = None

        # Determine arguments for page._page.to_image()
        if "width" in to_image_args and to_image_args["width"] is not None:
            logger.debug(
                f"Rendering preview for page {page_index} with width={to_image_args['width']}."
            )
            # Resolution is implicitly handled by pdfplumber when width is set
            if "height" in to_image_args:
                to_image_args.pop("height", None)
            # after image is created, we will calculate actual_scale_x and actual_scale_y

        elif "height" in to_image_args and to_image_args["height"] is not None:
            logger.debug(
                f"Rendering preview for page {page_index} with height={to_image_args['height']}."
            )
            # Resolution is implicitly handled by pdfplumber when height is set
            # after image is created, we will calculate actual_scale_x and actual_scale_y
        else:
            # Neither width nor height is provided, rely on `resolution`.
            # If `resolution` was explicitly passed as `None`, fall back to 144 DPI.
            render_resolution = 144 if resolution is None else resolution
            to_image_args["resolution"] = render_resolution

            actual_scale_x = render_resolution / 72.0
            actual_scale_y = render_resolution / 72.0
            logger.debug(
                f"Rendering preview for page {page_index} with resolution={render_resolution} (scale: {actual_scale_x:.2f})."
            )

        try:
            img_object = page_obj._page.to_image(**to_image_args)
            base_image_pil = (
                img_object.annotated
                if hasattr(img_object, "annotated")
                else img_object._repr_png_()
            )
            if isinstance(base_image_pil, bytes):
                from io import BytesIO

                base_image_pil = Image.open(BytesIO(base_image_pil))
            base_image_pil = base_image_pil.convert("RGB")

            # If scale was not determined by resolution, calculate it now from base_image_pil dimensions
            if actual_scale_x is None or actual_scale_y is None:
                if page_obj.width > 0:
                    actual_scale_x = base_image_pil.width / page_obj.width
                else:
                    actual_scale_x = resolution / 72.0  # Fallback to resolution-based scale
                if page_obj.height > 0:
                    actual_scale_y = base_image_pil.height / page_obj.height
                else:
                    actual_scale_y = resolution / 72.0  # Fallback to resolution-based scale
                logger.debug(
                    f"Calculated actual scales for page {page_index}: x={actual_scale_x:.2f}, y={actual_scale_y:.2f} from image size {base_image_pil.size} and page size ({page_obj.width}, {page_obj.height})"
                )

            # Convert temporary highlight dicts to Highlight objects
            preview_highlights = []
            for hl_data in temporary_highlights:
                final_color = self._determine_highlight_color(
                    color_input=hl_data.get("color"),
                    label=hl_data.get("label"),
                    use_color_cycling=hl_data.get("use_color_cycling", False),
                )
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
                            logger.warning(
                                f"Attribute '{attr_name}' not found on element {element}"
                            )
                if hl_data.get("bbox") or hl_data.get("polygon"):
                    preview_highlights.append(
                        Highlight(
                            page_index=hl_data["page_index"],
                            bbox=hl_data.get("bbox"),
                            polygon=hl_data.get("polygon"),
                            color=final_color,
                            label=hl_data.get("label"),
                            attributes=attrs_to_draw,
                        )
                    )

            # Use the calculated actual_scale_x for the HighlightRenderer
            # Assuming HighlightRenderer can handle a single scale or we adapt it.
            # For now, pdfplumber usually maintains aspect ratio, so one scale should be okay.
            # If not, HighlightRenderer needs to accept scale_x and scale_y.
            # We will use actual_scale_x assuming aspect ratio is maintained by pdfplumber,
            # or if not, it's a reasonable approximation for highlight scaling.
            renderer_scale = actual_scale_x

            renderer = HighlightRenderer(
                page=page_obj,
                base_image=base_image_pil,
                highlights=preview_highlights,
                scale_factor=renderer_scale,
                render_ocr=render_ocr,
            )
            rendered_image = renderer.render()

            # --- Optional Cropping BEFORE legend addition ---
            if crop_bbox is not None:
                cb_x0, cb_top, cb_x1, cb_bottom = crop_bbox
                # Convert to pixel coordinates using actual scales
                left_px = int(cb_x0 * actual_scale_x) - 1
                top_px = int(cb_top * actual_scale_y) - 1
                right_px = int(cb_x1 * actual_scale_x) + 1
                bottom_px = int(cb_bottom * actual_scale_y) + 1

                # Safeguard coordinates within bounds
                left_px = max(0, min(left_px, rendered_image.width - 1))
                top_px = max(0, min(top_px, rendered_image.height - 1))
                right_px = max(left_px + 1, min(right_px, rendered_image.width))
                bottom_px = max(top_px + 1, min(bottom_px, rendered_image.height))

                rendered_image = rendered_image.crop((left_px, top_px, right_px, bottom_px))

            legend = None
            if labels:
                # Check if we have quantitative metadata (for colorbar)
                quantitative_metadata = None
                for hl_data in temporary_highlights:
                    if "quantitative_metadata" in hl_data and hl_data["quantitative_metadata"]:
                        quantitative_metadata = hl_data["quantitative_metadata"]
                        break

                if quantitative_metadata:
                    # Create colorbar for quantitative data
                    from natural_pdf.utils.visualization import create_colorbar

                    try:
                        colorbar = create_colorbar(
                            values=quantitative_metadata["values"],
                            colormap=quantitative_metadata["colormap"],
                            bins=quantitative_metadata["bins"],
                            orientation=(
                                "horizontal" if legend_position in ["top", "bottom"] else "vertical"
                            ),
                        )
                        final_image = merge_images_with_legend(
                            rendered_image, colorbar, position=legend_position
                        )
                        logger.debug(
                            f"Added colorbar for quantitative attribute '{quantitative_metadata['attribute']}' on page {page_index}."
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create colorbar for page {page_index}: {e}")
                        # Fall back to regular legend
                        quantitative_metadata = None

                if not quantitative_metadata:
                    # Create regular categorical legend
                    preview_labels = {h.label: h.color for h in preview_highlights if h.label}
                    if preview_labels:
                        legend = create_legend(preview_labels)
                        final_image = merge_images_with_legend(
                            rendered_image, legend, position=legend_position
                        )
                    else:
                        final_image = rendered_image
            else:
                final_image = rendered_image

        except IOError as e:
            logger.error(f"IOError rendering preview for page {page_index}: {e}")
            raise
        except AttributeError as e:
            logger.error(f"AttributeError rendering preview for page {page_index}: {e}")
            raise

        return final_image

    def unified_render(
        self,
        specs: List["RenderSpec"],
        resolution: float = 150,
        width: Optional[int] = None,
        labels: bool = True,
        label_format: Optional[str] = None,
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
            logger.warning("unified_render called with empty specs list")
            return None

        # Process each spec into an image
        images = []

        for spec_idx, spec in enumerate(specs):
            if not isinstance(spec, RenderSpec):
                logger.error(f"Invalid spec type at index {spec_idx}: {type(spec)}")
                continue

            try:
                # Render the page
                page_image = self._render_spec(
                    spec=spec,
                    resolution=resolution,
                    width=width,
                    labels=labels,
                    label_format=label_format,
                    legend_position=legend_position,
                    spec_index=spec_idx,
                    **kwargs,
                )

                if page_image:
                    images.append(page_image)

            except Exception as e:
                logger.error(f"Error rendering spec {spec_idx}: {e}", exc_info=True)
                continue

        if not images:
            logger.warning("No images generated from specs")
            return None

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
        else:  # "single" - just return first image
            logger.warning(f"Multiple specs with layout='single', returning first image only")
            return images[0]

    def _render_spec(
        self,
        spec: "RenderSpec",
        resolution: float,
        width: Optional[int],
        labels: bool,
        label_format: Optional[str],
        legend_position: str,
        spec_index: int,
        **kwargs,
    ) -> Optional[Image.Image]:
        """Render a single RenderSpec to an image."""
        # Get the page
        page = spec.page
        if not hasattr(page, "width") or not hasattr(page, "height"):
            logger.error(f"Spec {spec_index} page does not have width/height attributes")
            return None

        # Calculate actual resolution/width
        if width is not None and page.width > 0:
            # Calculate resolution from width
            actual_resolution = (width / page.width) * 72
        else:
            # Use provided resolution or default
            actual_resolution = resolution if resolution is not None else 150

        # Get base page image
        try:
            # Use render_plain_page for clean rendering
            logger.debug(
                f"Calling render_plain_page with page={page}, resolution={actual_resolution}"
            )
            page_image = render_plain_page(page, resolution=actual_resolution)
        except Exception as e:
            logger.error(f"Failed to render page: {e}")
            logger.error(f"Page: {page}, Resolution: {actual_resolution}, Width: {width}")
            import traceback

            traceback.print_exc()
            return None

        if page_image is None:
            return None

        # Apply crop if specified
        if spec.crop_bbox:
            page_image = self._crop_image(
                page_image, spec.crop_bbox, page, actual_resolution / 72  # scale factor
            )

        # Apply highlights if any
        if spec.highlights:
            page_image = self._apply_spec_highlights(
                page_image,
                spec.highlights,
                page,
                actual_resolution / 72,  # scale factor
                labels=labels,
                label_format=label_format,
                spec_index=spec_index,
                crop_offset=spec.crop_bbox[:2] if spec.crop_bbox else None,  # Pass crop offset
            )

            # Add legend or colorbar if labels are enabled
            if labels:
                # Import visualization functions
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
                    # Create colorbar for quantitative data
                    try:
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
                    except Exception as e:
                        logger.warning(f"Failed to create colorbar for spec {spec_index}: {e}")
                        # Fall back to regular legend
                        quantitative_metadata = None

                if not quantitative_metadata:
                    # Create regular categorical legend
                    spec_labels = {}
                    for hl in spec.highlights:
                        label = hl.get("label")
                        color = hl.get("color")
                        if label and color:
                            # Process color to ensure it's an RGBA tuple
                            processed_color = self._process_color_input(color)
                            if processed_color:
                                spec_labels[label] = processed_color
                            else:
                                # Fallback to color manager if processing fails
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
            logger.warning(f"Invalid crop bounds: {crop_bbox}")
            return image

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
                logger.warning(f"Highlight {idx} has no geometry")
                continue

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

            # Draw the highlight
            if polygon:
                # Scale polygon points and apply offset
                scaled_polygon = [
                    (p[0] * scale_factor - offset_x, p[1] * scale_factor - offset_y)
                    for p in polygon
                ]
                draw.polygon(
                    scaled_polygon, fill=color, outline=(color[0], color[1], color[2], BORDER_ALPHA)
                )
            else:
                # Scale bbox and apply offset
                x0, y0, x1, y1 = bbox
                scaled_bbox = [
                    x0 * scale_factor - offset_x,
                    y0 * scale_factor - offset_y,
                    x1 * scale_factor - offset_x,
                    y1 * scale_factor - offset_y,
                ]
                draw.rectangle(
                    scaled_bbox, fill=color, outline=(color[0], color[1], color[2], BORDER_ALPHA)
                )

                # Draw attributes if present
                attributes_to_draw = highlight_dict.get("attributes_to_draw")
                if attributes_to_draw and scaled_bbox:
                    self._draw_spec_attributes(draw, attributes_to_draw, scaled_bbox, scale_factor)

        # Composite overlay onto image
        return Image.alpha_composite(image, overlay)

    def _draw_spec_attributes(
        self,
        draw: ImageDraw.Draw,
        attributes: Dict[str, Any],
        bbox_scaled: List[float],
        scale_factor: float,
    ) -> None:
        """Draw attribute key-value pairs on the highlight."""
        try:
            # Slightly larger font, scaled
            font_size = max(10, int(8 * scale_factor))
            # Try to load a font
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default()
                    font_size = 10  # Reset size for default font
        except Exception:
            font = ImageFont.load_default()
            font_size = 10

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
