"""
Centralized service for managing and rendering highlights in a PDF document.
"""
import io
import os
import logging # Added
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union

from PIL import Image, ImageDraw, ImageFont
from colour import Color

# Attempt to import Page for type hinting safely
try:
    from .page import Page
except ImportError:
    Page = Any # Fallback if circular import issue arises during type checking

# Import ColorManager and related utils
from natural_pdf.utils.visualization import ColorManager, create_legend, merge_images_with_legend

# Constants for drawing (Can be potentially moved to ColorManager/Renderer if desired)
BORDER_ALPHA = 180 # Default alpha for highlight border
DEFAULT_FALLBACK_COLOR = (255, 255, 0) # Yellow fallback (RGB only, alpha added by ColorManager)

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
    color: Tuple[int, int, int, int] # Final RGBA color determined by service
    label: Optional[str] = None
    polygon: Optional[List[Tuple[float, float]]] = None
    attributes: Dict[str, Any] = field(default_factory=dict) # Store extracted attribute values

    @property
    def is_polygon(self) -> bool:
        """Check if this highlight uses polygon coordinates."""
        return self.polygon is not None and len(self.polygon) >= 3

    @property
    def border_color(self) -> Tuple[int, int, int, int]:
         """Calculate a slightly darker/more opaque border color."""
         # Use base color but increase alpha for border
         return (
            self.color[0],
            self.color[1],
            self.color[2],
            BORDER_ALPHA
        )


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
        scale: float,
        render_ocr: bool,
    ):
        self.page = page # Keep page reference for OCR rendering
        self.base_image = base_image.convert('RGBA') # Ensure RGBA
        self.highlights = highlights
        self.scale = scale
        self.render_ocr = render_ocr
        self.result_image = self.base_image.copy()
        self.vertex_size = max(3, int(2 * self.scale)) # Size of corner markers

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
            overlay = Image.new('RGBA', self.base_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            scaled_bbox = None

            if highlight.is_polygon:
                scaled_polygon = [(p[0] * self.scale, p[1] * self.scale) for p in highlight.polygon]
                # Draw polygon fill and border
                draw.polygon(scaled_polygon, fill=highlight.color, outline=highlight.border_color, width=2)
                self._draw_vertices(draw, scaled_polygon, highlight.border_color)

                # Calculate scaled bbox for attribute drawing
                x_coords = [p[0] for p in scaled_polygon]
                y_coords = [p[1] for p in scaled_polygon]
                scaled_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

            else: # Rectangle
                x0, top, x1, bottom = highlight.bbox
                x0_s, top_s, x1_s, bottom_s = x0 * self.scale, top * self.scale, x1 * self.scale, bottom * self.scale
                scaled_bbox = [x0_s, top_s, x1_s, bottom_s]
                # Draw rectangle fill and border
                draw.rectangle(scaled_bbox, fill=highlight.color, outline=highlight.border_color, width=2)

                vertices = [(x0_s, top_s), (x1_s, top_s), (x1_s, bottom_s), (x0_s, bottom_s)]
                self._draw_vertices(draw, vertices, highlight.border_color)

            # Draw attributes if present on the highlight object
            if highlight.attributes and scaled_bbox: # Ensure bbox is calculated
                self._draw_attributes(draw, highlight.attributes, scaled_bbox)

            # Composite this highlight's overlay onto the result using alpha blending
            self.result_image = Image.alpha_composite(self.result_image, overlay)

    def _draw_vertices(self, draw: ImageDraw.Draw, vertices: List[Tuple[float, float]], color: Tuple[int, int, int, int]):
        """Draw small markers at each vertex."""
        for x, y in vertices:
            # Draw ellipse centered at vertex
            draw.ellipse(
                [x - self.vertex_size, y - self.vertex_size, x + self.vertex_size, y + self.vertex_size],
                fill=color # Use border color for vertices
            )

    def _draw_attributes(self, draw: ImageDraw.Draw, attributes: Dict[str, Any], bbox_scaled: List[float]):
        """Draws attribute key-value pairs on the highlight."""
        try:
            # Slightly larger font, scaled
            font_size = max(10, int(8 * self.scale))
            # Prioritize monospace fonts for better alignment
            font = ImageFont.truetype("Arial.ttf", font_size) # Fallback sans-serif
        except IOError:
            font = ImageFont.load_default()
            font_size = 10 # Reset size for default font

        line_height = font_size + int(4 * self.scale) # Scaled line spacing
        bg_padding = int(3 * self.scale)
        max_width = 0
        text_lines = []

        # Format attribute lines
        for name, value in attributes.items():
            if isinstance(value, float):
                value_str = f"{value:.2f}" # Format floats
            else:
                value_str = str(value)
            line = f"{name}: {value_str}"
            text_lines.append(line)
            try:
                 # Calculate max width for background box
                 max_width = max(max_width, draw.textlength(line, font=font))
            except AttributeError: pass # Ignore if textlength not available

        if not text_lines: return # Nothing to draw

        total_height = line_height * len(text_lines)

        # Position near top-right corner with padding
        x = bbox_scaled[2] - int(2 * self.scale) - max_width
        y = bbox_scaled[1] + int(2 * self.scale)

        # Draw background rectangle (semi-transparent white)
        bg_x0 = x - bg_padding
        bg_y0 = y - bg_padding
        bg_x1 = x + max_width + bg_padding
        bg_y1 = y + total_height + bg_padding
        draw.rectangle(
            [bg_x0, bg_y0, bg_x1, bg_y1],
            fill=(255, 255, 255, 240),
            outline=(0, 0, 0, 180), # Light black outline
            width=1
        )

        # Draw text lines (black)
        current_y = y
        for line in text_lines:
            draw.text((x, current_y), line, fill=(0, 0, 0, 255), font=font)
            current_y += line_height

    def _render_ocr_text(self):
        """Renders OCR text onto the image. (Adapted from old HighlightManager)"""
        # Use the page reference to get OCR elements
        try:
            # Try finding first, then extracting if necessary
            ocr_elements = self.page.find_all('text[source=ocr]')
            if not ocr_elements:
                 # Don't run full OCR here, just extract if already run
                 ocr_elements = [el for el in self.page.words if getattr(el, 'source', None) == 'ocr']
                 # Alternative: self.page.extract_ocr_elements() - but might be slow

        except Exception as e:
            logger.warning(f"Could not get OCR elements for page {self.page.number}: {e}", exc_info=True)
            return # Don't modify image if OCR elements aren't available

        if not ocr_elements:
            logger.debug(f"No OCR elements found for page {self.page.number} to render.")
            return

        overlay = Image.new('RGBA', self.base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Find a suitable font
        font_path = None
        default_font = ImageFont.load_default()
        common_fonts = ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf", "FreeSans.ttf"]
        for fname in common_fonts:
             try:
                 ImageFont.truetype(fname, 10) # Test load
                 font_path = fname
                 break
             except IOError:
                 continue

        for element in ocr_elements:
            x0, top, x1, bottom = element.bbox
            x0_s, top_s, x1_s, bottom_s = x0 * self.scale, top * self.scale, x1 * self.scale, bottom * self.scale
            box_w, box_h = x1_s - x0_s, bottom_s - top_s

            if box_h <= 0: continue # Skip zero-height boxes

            # --- Font Size Calculation ---
            font_size = max(9, int(box_h * 0.85)) # Min size 9, 85% of box height

            try:
                sized_font = ImageFont.truetype(font_path, font_size) if font_path else default_font
            except IOError:
                sized_font = default_font

            # --- Adjust Font Size if Text Overflows ---
            try:
                 text_w = draw.textlength(element.text, font=sized_font)
                 if text_w > box_w * 1.1: # Allow 10% overflow
                     ratio = max(0.5, (box_w * 1.0) / text_w) # Don't shrink below 50%
                     font_size = max(9, int(font_size * ratio))
                     if font_path:
                         try: sized_font = ImageFont.truetype(font_path, font_size)
                         except IOError: pass # Keep previous if error
            except AttributeError: pass # Skip adjustment if textlength fails

            # --- Draw Background and Text ---
            padding = max(1, int(font_size * 0.05)) # Minimal padding
            draw.rectangle(
                [x0_s - padding, top_s - padding, x1_s + padding, bottom_s + padding],
                fill=(255, 255, 255, 230) # Highly transparent white background
            )

            # Calculate text position (centered vertically, slightly offset from left)
            try:
                 if hasattr(sized_font, "getbbox"): # Modern PIL
                     _, text_top_offset, _, text_bottom_offset = sized_font.getbbox(element.text)
                     text_h = text_bottom_offset - text_top_offset
                 else: # Older PIL approximation
                     text_h = font_size
                 text_y = top_s + (box_h - text_h) / 2
                 # Adjust for vertical offset in some fonts
                 text_y -= text_top_offset if hasattr(sized_font, "getbbox") else 0
                 text_x = x0_s + padding # Start near left edge with padding

            except Exception:
                # Fallback positioning
                text_x, text_y = x0_s + padding, top_s + padding

            draw.text((text_x, text_y), element.text, fill=(0, 0, 0, 255), font=sized_font)

        # Composite the OCR text overlay onto the result image
        self.result_image = Image.alpha_composite(self.result_image, overlay)


class HighlightingService:
    """
    Central service to manage highlight data and orchestrate rendering.
    Holds the state of all highlights across the document.
    """
    def __init__(self, pdf_object):
        self._pdf = pdf_object # Reference to the parent PDF object
        self._highlights_by_page: Dict[int, List[Highlight]] = {}
        self._color_manager = ColorManager() # Instantiate the color manager
        logger.info("HighlightingService initialized with ColorManager.")

    # Removed _get_next_color - logic moved to ColorManager
    # Removed _color_cycle, _labels_colors - managed by ColorManager

    def _process_color_input(
        self, 
        color_input: Optional[Union[Tuple, str]]
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
                      val = int(c * 255) if (i < 3 and all_float) or (i==3 and 0.0 <= c <= 1.0) else int(c)
                 elif isinstance(c, int):
                      val = c
                 else:
                      logger.warning(f"Invalid color component type: {c} in {color_input}")
                      return None # Invalid type
                 processed.append(max(0, min(255, val))) # Clamp to 0-255

            # Check length and add default alpha if needed
            if len(processed) == 3:
                 # Use alpha from ColorManager instance
                 processed.append(self._color_manager._alpha)
                 return tuple(processed)
            elif len(processed) == 4:
                 return tuple(processed)
            else:
                 logger.warning(f"Invalid color tuple length: {color_input}")
                 return None # Invalid length

        elif isinstance(color_input, str):
            try:
                # Convert color name/hex string to RGB tuple (0.0-1.0 floats)
                from colour import Color # Import here if not at top
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
            except Exception as e:
                 logger.error(f"Error processing color string '{color_input}': {e}")
                 return None
        else:
             logger.warning(f"Invalid color input type: {type(color_input)}")
             return None

    def _determine_highlight_color(
        self,
        color_input: Optional[Union[Tuple, str]] = None,
        label: Optional[str] = None,
        use_color_cycling: bool = False
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
        bbox: Union[Tuple[float, float, float, float], Any], # Relax input type hint
        color: Optional[Union[Tuple, str]] = None,
        label: Optional[str] = None,
        use_color_cycling: bool = False,
        element: Optional[Any] = None,
        include_attrs: Optional[List[str]] = None,
        existing: str = 'append'
    ):
        """Adds a rectangular highlight."""
        
        processed_bbox: Tuple[float, float, float, float]
        # Check if bbox is an object with expected attributes (likely a Region)
        # Assuming Region object has x0, top, x1, bottom attributes based on error context
        if (hasattr(bbox, 'x0') and hasattr(bbox, 'top') and
            hasattr(bbox, 'x1') and hasattr(bbox, 'bottom')):
             try:
                # Ensure attributes are numeric before creating tuple
                processed_bbox = (float(bbox.x0), float(bbox.top), float(bbox.x1), float(bbox.bottom))
             except (ValueError, TypeError):
                 logger.error(f"Invalid attribute types in bbox object for page {page_index}: {bbox}. Expected numeric values.")
                 return
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
             try:
                 # Ensure elements are numeric and convert to tuple
                 processed_bbox = tuple(float(v) for v in bbox)
             except (ValueError, TypeError):
                 logger.error(f"Invalid values in bbox sequence for page {page_index}: {bbox}. Expected numeric values.")
                 return
        else:
            logger.error(f"Invalid bbox type or structure provided for page {page_index}: {type(bbox)} - {bbox}. Expected tuple/list of 4 numbers or Region-like object.")
            return # Don't proceed if bbox is invalid
            
        self._add_internal(
            page_index=page_index,
            bbox=processed_bbox, # Use the processed tuple
            polygon=None,
            color_input=color,
            label=label,
            use_color_cycling=use_color_cycling,
            element=element,
            include_attrs=include_attrs,
            existing=existing
        )

    def add_polygon(
        self,
        page_index: int,
        polygon: List[Tuple[float, float]],
        color: Optional[Union[Tuple, str]] = None,
        label: Optional[str] = None,
        use_color_cycling: bool = False,
        element: Optional[Any] = None,
        include_attrs: Optional[List[str]] = None,
        existing: str = 'append'
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
            include_attrs=include_attrs,
            existing=existing
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
        include_attrs: Optional[List[str]],
        existing: str
    ):
        """Internal method to create and store a Highlight object."""
        if page_index < 0 or page_index >= len(self._pdf.pages):
             logger.error(f"Invalid page index {page_index}. Cannot add highlight.")
             return

        # Handle 'replace' logic - clear highlights for this page *before* adding new one
        if existing == 'replace':
            self.clear_page(page_index)

        # Determine the final color using the ColorManager
        final_color = self._determine_highlight_color(
            color_input=color_input,
            label=label,
            use_color_cycling=use_color_cycling
        )

        # Extract attributes from the element if requested
        attributes_to_draw = {}
        if element and include_attrs:
             for attr_name in include_attrs:
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
            attributes=attributes_to_draw
        )

        # Add to the list for the specific page
        if page_index not in self._highlights_by_page:
            self._highlights_by_page[page_index] = []
        self._highlights_by_page[page_index].append(highlight)
        logger.debug(f"Added highlight to page {page_index}: {highlight}")

    def clear_all(self):
        """Clears all highlights from all pages and resets the color manager."""
        self._highlights_by_page = {}
        self._color_manager.reset()
        logger.info("Cleared all highlights and reset ColorManager.")

    def clear_page(self, page_index: int):
        """Clears all highlights from a specific page."""
        if page_index in self._highlights_by_page:
            del self._highlights_by_page[page_index]
            logger.debug(f"Cleared highlights for page {page_index}.")
            # Note: We typically don't reset the color manager when clearing a single page
            # to maintain color consistency if highlights are added back.

    def get_highlights_for_page(self, page_index: int) -> List[Highlight]:
        """Returns a list of Highlight objects for a specific page."""
        return self._highlights_by_page.get(page_index, [])

    def get_labels_and_colors(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Returns a mapping of labels used to their assigned colors (for persistent highlights)."""
        return self._color_manager.get_label_colors()

    def render_page(
        self,
        page_index: int,
        scale: float = 2.0,
        labels: bool = True,
        legend_position: str = 'right',
        render_ocr: bool = False,
        resolution: Optional[float] = None,
        **kwargs # Pass other args to pdfplumber.page.to_image if needed
    ) -> Optional[Image.Image]:
        """
        Renders a specific page with its highlights.
        Legend is now generated based only on highlights present on this page.

        Args:
            page_index: The 0-based index of the page to render.
            scale: Scale factor for rendering highlights.
            labels: Whether to include a legend for highlights.
            legend_position: Position of the legend.
            render_ocr: Whether to render OCR text on the image.
            resolution: Optional resolution (DPI) for the base page image.
                       Defaults to scale * 72.
            kwargs: Additional keyword arguments for pdfplumber's page.to_image.

        Returns:
            A PIL Image object of the rendered page, or None if rendering fails.
        """
        if page_index < 0 or page_index >= len(self._pdf.pages):
            logger.error(f"Invalid page index {page_index} for rendering.")
            return None

        page = self._pdf[page_index]
        highlights_on_page = self.get_highlights_for_page(page_index) # This list will be empty if clear_page was called

        # --- Get Base Image --- 
        try:
            render_resolution = resolution if resolution is not None else scale * 72
            img_object = page._page.to_image(resolution=render_resolution, **kwargs)
            base_image = img_object.annotated
            if not isinstance(base_image, Image.Image):
                 png_data = img_object._repr_png_()
                 if png_data:
                      base_image = Image.open(io.BytesIO(png_data)).convert('RGB')
                 else:
                      raise ValueError("Could not extract base PIL image from pdfplumber.")
            base_image = base_image.convert('RGBA')
            logger.debug(f"Base image for page {page_index} rendered with resolution {render_resolution}.")
        except Exception as e:
            logger.error(f"Failed to render base image for page {page_index}: {e}", exc_info=True)
            return None

        # --- Render Highlights --- 
        rendered_image: Image.Image
        if highlights_on_page:
            renderer = HighlightRenderer(
                page=page,
                base_image=base_image,
                highlights=highlights_on_page,
                scale=scale,
                render_ocr=render_ocr,
            )
            rendered_image = renderer.render()
        else:
             if render_ocr:
                  # Still render OCR even if no highlights
                  renderer = HighlightRenderer(page, base_image, [], scale, True)
                  rendered_image = renderer.render()
             else:
                  rendered_image = base_image # No highlights, no OCR requested

        # --- Add Legend (Based ONLY on this page's highlights) --- 
        if labels:
            # CHANGE: Create label_colors map only from highlights_on_page
            labels_colors_on_page: Dict[str, Tuple[int, int, int, int]] = {}
            for hl in highlights_on_page:
                if hl.label and hl.label not in labels_colors_on_page:
                    labels_colors_on_page[hl.label] = hl.color
            
            if labels_colors_on_page: # Only add legend if there are labels on this page
                legend = create_legend(labels_colors_on_page)
                if legend: # Ensure create_legend didn't return None
                     rendered_image = merge_images_with_legend(rendered_image, legend, legend_position)
                     logger.debug(f"Added legend with {len(labels_colors_on_page)} labels for page {page_index}.")
                else:
                     logger.debug(f"Legend creation returned None for page {page_index}.")
            else:
                 logger.debug(f"No labels found on page {page_index}, skipping legend.")
        
        return rendered_image 

    def render_preview(
        self,
        page_index: int,
        temporary_highlights: List[Dict],
        scale: float = 2.0,
        labels: bool = True,
        legend_position: str = 'right',
        render_ocr: bool = False,
        resolution: Optional[float] = None,
        **kwargs
    ) -> Optional[Image.Image]:
        """
        Renders a preview image for a specific page containing only the
        provided temporary highlights. Does not affect persistent state.

        Args:
            page_index: Index of the page to render.
            temporary_highlights: List of highlight data dicts (from ElementCollection._prepare).
            scale: Scale factor for rendering.
            labels: Whether to include a legend.
            legend_position: Position of the legend.
            render_ocr: Whether to render OCR text.
            resolution: Resolution for base page image rendering.
            **kwargs: Additional args for pdfplumber's to_image.

        Returns:
            PIL Image of the preview, or None if rendering fails.
        """
        if page_index < 0 or page_index >= len(self._pdf.pages):
            logger.error(f"Invalid page index {page_index} for render_preview.")
            return None

        page = self._pdf.pages[page_index]
        render_resolution = resolution if resolution is not None else scale * 72

        try:
            # Get base image from pdfplumber using the Page object's underlying _page
            img_object = page._page.to_image(resolution=render_resolution, **kwargs)
            base_image = img_object.annotated if hasattr(img_object, 'annotated') else img_object._repr_png_()
            if isinstance(base_image, bytes):
                 from io import BytesIO
                 base_image = Image.open(BytesIO(base_image))
            base_image = base_image.convert("RGB") # Ensure consistent format

            # Convert temporary highlight dicts to Highlight objects
            # Note: Colors/labels should be determined *here* for temporary preview
            preview_highlights = []
            for hl_data in temporary_highlights:
                # Determine the final color using the service logic
                final_color = self._determine_highlight_color(
                    color_input=hl_data.get('color'),
                    label=hl_data.get('label'),
                    use_color_cycling=hl_data.get('use_color_cycling', False)
                )

                # Extract potential attributes to draw
                attrs_to_draw = {}
                element = hl_data.get('element')
                include_attrs = hl_data.get('include_attrs')
                if element and include_attrs:
                    for attr_name in include_attrs:
                        try:
                            attr_value = getattr(element, attr_name, None)
                            if attr_value is not None:
                                attrs_to_draw[attr_name] = attr_value
                        except AttributeError:
                            logger.warning(f"Attribute '{attr_name}' not found on element {element}")

                # Add highlight if geometry exists
                if hl_data.get('bbox') or hl_data.get('polygon'):
                    preview_highlights.append(Highlight(
                        page_index=hl_data['page_index'],
                        bbox=hl_data.get('bbox'),
                        polygon=hl_data.get('polygon'),
                        color=final_color, # Use the determined color
                        label=hl_data.get('label'),
                        attributes=attrs_to_draw
                    ))

            # Render only these highlights
            renderer = HighlightRenderer(page, base_image, preview_highlights, scale, render_ocr)
            rendered_image = renderer.render()

            # Create legend only from temporary highlights
            legend = None
            if labels:
                preview_labels = {h.label: h.color for h in preview_highlights if h.label}
                if preview_labels:
                    legend = create_legend(preview_labels)
                    final_image = merge_images_with_legend(rendered_image, legend, position=legend_position)
                else:
                    final_image = rendered_image # No legend needed
            else:
                final_image = rendered_image

        except Exception as e:
            logger.error(f"Error rendering preview for page {page_index}: {e}", exc_info=True)
            return None

        return final_image 