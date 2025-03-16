"""
Highlighting utilities for natural-pdf.
"""
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from PIL import Image, ImageDraw, ImageFont
import io
import math
import os
from .visualization import get_next_highlight_color, create_legend, merge_images_with_legend, reset_highlight_colors

class Highlight:
    """
    Represents a single highlight with color and optional label.
    """
    def __init__(self, 
                bbox: Tuple[float, float, float, float], 
                color: Tuple[int, int, int, int],
                label: Optional[str] = None,
                polygon: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize a highlight.
        
        Args:
            bbox: Bounding box (x0, top, x1, bottom)
            color: RGBA color tuple (0-255 integers)
            label: Optional label for this highlight
            polygon: Optional polygon points for non-rectangular highlights
        """
        self.bbox = bbox
        self.polygon = polygon
        
        # Ensure color values are integers in 0-255 range
        if isinstance(color, tuple):
            # Convert values to integers in 0-255 range
            processed_color = []
            for i, c in enumerate(color):
                if isinstance(c, float):
                    # 0.0-1.0 float format
                    if c <= 1.0:
                        processed_color.append(int(c * 255))
                    # Already in 0-255 range but as float
                    else:
                        processed_color.append(int(c))
                else:
                    processed_color.append(c)
                    
            # Default alpha value if needed
            if len(processed_color) == 3:
                processed_color.append(100)  # Default alpha
                
            self.color = tuple(processed_color)
        else:
            # Default if invalid color is provided
            self.color = (255, 255, 0, 100)  # Yellow with semi-transparency
            
        self.label = label
        
        # New attributes for displaying element properties
        self.element = None  # Will be set after initialization
        self.include_attrs = None  # Will be set after initialization
        
    @property
    def is_polygon(self) -> bool:
        """Check if this highlight uses polygon coordinates."""
        return self.polygon is not None and len(self.polygon) >= 3
        
    def __repr__(self) -> str:
        """String representation of the highlight."""
        if self.is_polygon:
            prefix = "PolygonHighlight"
        else:
            prefix = "Highlight"
            
        if self.label:
            return f"<{prefix} bbox={self.bbox} label='{self.label}'>"
        else:
            return f"<{prefix} bbox={self.bbox}>"
        
class HighlightManager:
    """
    Manages highlights for a page.
    """
    def __init__(self, page: 'Page'):
        """
        Initialize a highlight manager.
        
        Args:
            page: The page to manage highlights for
        """
        self._page = page
        self._highlights: List[Highlight] = []
        self._labels_colors: Dict[str, Tuple[int, int, int, int]] = {}
    
    def add_polygon_highlight(self,
                             polygon: List[Tuple[float, float]],
                             color: Optional[Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]]] = None,
                             label: Optional[str] = None,
                             cycle_colors: bool = False,
                             element: Optional[Any] = None,
                             include_attrs: Optional[List[str]] = None,
                             existing: str = 'append') -> None:
        """
        Add a polygon highlight to the page.
        
        Args:
            polygon: List of (x, y) coordinate tuples defining the polygon
            color: RGBA color tuple (0-255 or 0.0-1.0), or None to use automatic color
            label: Optional label for this highlight
            cycle_colors: Force color cycling even with no label (default: False)
            element: The original element being highlighted (for attribute access)
            include_attrs: List of attribute names to display on the highlight (e.g., ['confidence', 'type'])
            existing: How to handle existing highlights - 'append' (default) or 'replace'
        """
        # Calculate bounding box from polygon
        if polygon:
            x_coords = [p[0] for p in polygon]
            y_coords = [p[1] for p in polygon]
            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        else:
            # If invalid polygon, use dummy bbox
            bbox = (0, 0, 1, 1)
            
        # Get appropriate color
        final_color = self._get_highlight_color(color, label, cycle_colors)
        
        # Clear existing highlights if replacing
        if existing == 'replace':
            self.clear_highlights()
            
        # Create highlight with polygon
        highlight = Highlight(bbox, final_color, label, polygon)
        
        # Add element and attrs to the highlight
        highlight.element = element
        
        # No automatic display - only show attributes if explicitly requested
        highlight.include_attrs = include_attrs
        
        self._highlights.append(highlight)
        
    def add_highlight(self, 
                     bbox: Tuple[float, float, float, float], 
                     color: Optional[Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]]] = None,
                     label: Optional[str] = None,
                     cycle_colors: bool = False,
                     element: Optional[Any] = None,
                     include_attrs: Optional[List[str]] = None,
                     existing: str = 'append') -> None:
        """
        Add a highlight to the page.
        
        Args:
            bbox: Bounding box (x0, top, x1, bottom)
            color: RGBA color tuple (0-255 or 0.0-1.0), or None to use automatic color
            label: Optional label for this highlight
            cycle_colors: Force color cycling even with no label (default: False)
            element: The original element being highlighted (for attribute access)
            include_attrs: List of attribute names to display on the highlight (e.g., ['confidence', 'type'])
            existing: How to handle existing highlights - 'append' (default) or 'replace'
        """
        # Get appropriate color
        final_color = self._get_highlight_color(color, label, cycle_colors)
        
        # Clear existing highlights if replacing
        if existing == 'replace':
            self.clear_highlights()
            
        # Create highlight
        highlight = Highlight(bbox, final_color, label)
        
        # Add element and attrs to the highlight
        highlight.element = element
        
        # No automatic display - only show attributes if explicitly requested
        highlight.include_attrs = include_attrs
        
        self._highlights.append(highlight)
        
    def _get_highlight_color(self,
                           color: Optional[Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]]] = None,
                           label: Optional[str] = None,
                           cycle_colors: bool = False) -> Tuple[int, int, int, int]:
        """
        Determine the appropriate color for a highlight based on rules.
        
        Args:
            color: RGBA color tuple (0-255 or 0.0-1.0), or None to use automatic color
            label: Optional label for this highlight
            cycle_colors: Force color cycling even with no label (default: False)
            
        Returns:
            RGBA color tuple with values as integers 0-255
        """
        # Color selection logic:
        # 1. If explicit color is provided, use it
        # 2. If label exists in color map, use that color (consistency)
        # 3. If label is provided but new, get a new color and store it
        # 4. If no label & cycle_colors=True, get next color
        # 5. If no label & cycle_colors=False, use default yellow highlight
        
        if color is not None:
            # Explicit color takes precedence
            # Convert from 0.0-1.0 to 0-255 if needed
            if isinstance(color[0], float) and color[0] <= 1.0:
                highlight_color = (
                    int(color[0] * 255),
                    int(color[1] * 255),
                    int(color[2] * 255),
                    int(color[3] * 255) if len(color) > 3 else 100
                )
            else:
                highlight_color = color
        elif label is not None and label in self._labels_colors:
            # Use existing color for this label
            highlight_color = self._labels_colors[label]
        elif label is not None:
            # New label, get a new color and store it
            highlight_color = get_next_highlight_color()
            self._labels_colors[label] = highlight_color
        elif cycle_colors:
            # No label but cycling requested
            highlight_color = get_next_highlight_color()
        else:
            # Default case: no label, no cycling - use yellow
            highlight_color = (255, 255, 0, 100)  # Default yellow highlight
            
        return highlight_color
            
    def clear_highlights(self) -> None:
        """Clear all highlights."""
        self._highlights = []
        self._labels_colors = {}
        reset_highlight_colors()
        
    def get_highlighted_image(self, 
                             scale: float = 2.0,
                             labels: bool = True,
                             legend_position: str = 'right',
                             render_ocr: bool = False) -> Image.Image:
        """
        Get an image of the page with highlights.
        
        Args:
            scale: Scale factor for rendering
            labels: Whether to include a legend for labels
            legend_position: Position of the legend ('right', 'bottom', 'top', 'left')
            render_ocr: Whether to render OCR text with white background boxes
            
        Returns:
            PIL Image with the highlighted page
        """
        # Get the raw page image with higher resolution for clearer results
        page_image = self._page._page.to_image(resolution=72 * scale)
        
        # Convert to PIL Image in RGBA mode for transparency
        img_bytes = io.BytesIO()
        page_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        pil_image = Image.open(img_bytes).convert('RGBA')
        
        # Create a transparent overlay
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw each highlight
        for highlight in self._highlights:
            if highlight.is_polygon:
                # Scale polygon coordinates
                scaled_polygon = [(p[0] * scale, p[1] * scale) for p in highlight.polygon]
                
                # Draw polygon with the highlight color
                draw.polygon(scaled_polygon, fill=highlight.color)
                
                # Draw attribute text if requested
                if highlight.element and highlight.include_attrs:
                    # Calculate bounding box from polygon for text positioning
                    x_coords = [p[0] for p in scaled_polygon]
                    y_coords = [p[1] for p in scaled_polygon]
                    bbox_scaled = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    
                    self._draw_element_attributes(
                        draw, 
                        highlight.element, 
                        highlight.include_attrs,
                        bbox_scaled, 
                        scale
                    )
            else:
                # Regular rectangle highlight
                x0, top, x1, bottom = highlight.bbox
                
                # Scale the bbox to match the rendered image
                x0_scaled = x0 * scale
                top_scaled = top * scale
                x1_scaled = x1 * scale
                bottom_scaled = bottom * scale
                
                # Draw the rectangle
                draw.rectangle([x0_scaled, top_scaled, x1_scaled, bottom_scaled], 
                              fill=highlight.color)
                
                # Draw attribute text if requested
                if highlight.element and highlight.include_attrs:
                    self._draw_element_attributes(
                        draw, 
                        highlight.element, 
                        highlight.include_attrs,
                        [x0_scaled, top_scaled, x1_scaled, bottom_scaled], 
                        scale
                    )
        
        # Combine the original image with the highlight overlay
        highlighted_image = Image.alpha_composite(pil_image, overlay)
        
        # Add OCR text rendering if requested
        if render_ocr:
            highlighted_image = self._render_ocr_text(highlighted_image, scale)
        
        # Add legend if requested and there are labeled highlights
        if labels and self._labels_colors:
            legend = create_legend(self._labels_colors)
            highlighted_image = merge_images_with_legend(highlighted_image, legend, legend_position)
        
        return highlighted_image
        
    def _render_ocr_text(self, image: Image.Image, scale: float = 2.0) -> Image.Image:
        """
        Render OCR text on the image with white background boxes.
        
        Args:
            image: Base image to render text on
            scale: Scale factor for rendering
            
        Returns:
            PIL Image with OCR text rendered
        """
        # First check for OCR text elements from the selector approach
        ocr_elements = self._page.find_all('text[source=ocr]')
        
        # If that doesn't work, try checking the _elements dict directly
        if not ocr_elements or len(ocr_elements) == 0:
            # Check if page has elements loaded
            if hasattr(self._page, '_elements') and self._page._elements is not None:
                # Look for OCR elements in various possible locations
                if 'ocr_text' in self._page._elements and self._page._elements['ocr_text']:
                    ocr_elements = self._page._elements['ocr_text']
                elif 'ocr' in self._page._elements and self._page._elements['ocr']:
                    ocr_elements = self._page._elements['ocr']
        
        # If still no elements, try to run extract_ocr_elements first
        if not ocr_elements or len(ocr_elements) == 0:
            try:
                ocr_elements = self._page.extract_ocr_elements()
            except Exception as e:
                print(f"Error extracting OCR elements: {e}")
        
        # Final check if we have OCR elements
        if not ocr_elements or len(ocr_elements) == 0:
            raise ValueError("No OCR elements found. Run OCR on the page before rendering.")
        
        # Create a new overlay for OCR text
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Try to use a common sans-serif font
        try:
            font_path = None
            for path in ["Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf", "FreeSans.ttf"]:
                if os.path.exists(path):
                    font_path = path
                    break
                    
            if not font_path:
                # Use the default font
                font = ImageFont.load_default()
            else:
                # Use the found font
                font = ImageFont.truetype(font_path, 12)  # Default size, will be scaled
        except Exception:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Process each OCR element
        for element in ocr_elements:
            # Get the element's bounding box
            x0, top, x1, bottom = element.bbox
            
            # Scale the bbox to match the rendered image
            x0_scaled = x0 * scale
            top_scaled = top * scale
            x1_scaled = x1 * scale
            bottom_scaled = bottom * scale
            
            # Calculate text size for optimal rendering
            box_width = x1_scaled - x0_scaled
            box_height = bottom_scaled - top_scaled
            
            # Calculate font size based on box height (approx 90% of box height)
            # Use a higher percentage to make text larger
            font_size = int(box_height * 0.9)
            if font_size < 9:  # Higher minimum readable size
                font_size = 9
                
            # Create a font of the appropriate size
            try:
                if font_path:
                    sized_font = ImageFont.truetype(font_path, font_size)
                else:
                    # If no truetype font, use default and scale as best as possible
                    sized_font = font
            except Exception:
                sized_font = font
            
            # Measure text to check for overflow
            try:
                # Get text width with the sized font
                text_width = draw.textlength(element.text, font=sized_font)
                
                # If text is too wide, scale down font size - but only if significantly too wide
                # Allow more overflow (1.25 instead of 1.1)
                if text_width > box_width * 1.25:
                    # Less aggressive reduction
                    reduction_factor = box_width / text_width
                    # Don't shrink below 75% of original calculated size
                    font_size = max(10, max(int(font_size * 0.75), int(font_size * reduction_factor)))
                    try:
                        if font_path:
                            sized_font = ImageFont.truetype(font_path, font_size)
                    except Exception:
                        pass  # Keep current font if failed
            except Exception:
                # If text measurement fails, continue with current font
                pass
                
            # Add padding to the white background box
            padding = max(2, int(font_size * 0.1))
            
            # Draw white background (slightly larger than text area)
            draw.rectangle(
                [x0_scaled - padding, 
                 top_scaled - padding, 
                 x1_scaled + padding, 
                 bottom_scaled + padding],
                fill=(255, 255, 255, 240)  # Slightly transparent white
            )
            
            # Draw black text - centered horizontally and vertically
            try:
                # Calculate text dimensions for centering
                if hasattr(sized_font, "getbbox"):
                    # PIL 9.2.0+ method
                    text_bbox = sized_font.getbbox(element.text)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                else:
                    # Fallback for older PIL versions
                    text_width = draw.textlength(element.text, font=sized_font)
                    text_height = font_size  # Approximate
                
                # Center the text both horizontally and vertically
                text_x = x0_scaled + (box_width - text_width) / 2
                text_y = top_scaled + (box_height - text_height) / 2
                
                # Don't let text go out of bounds
                text_x = max(x0_scaled, text_x)
                text_y = max(top_scaled, text_y)
                
            except Exception:
                # Fallback if calculation fails
                text_x = x0_scaled
                text_y = top_scaled
                
            draw.text(
                (text_x, text_y),
                element.text,
                fill=(0, 0, 0, 255),
                font=sized_font,
            )
            
        # Combine the original image with the OCR text overlay
        result_image = Image.alpha_composite(image, overlay)
        return result_image
        
    def _draw_element_attributes(self, 
                               draw: ImageDraw.Draw, 
                               element: Any, 
                               attr_names: List[str],
                               bbox_scaled: List[float], 
                               scale: float) -> None:
        """
        Draw element attributes as text on the highlight.
        
        Args:
            draw: PIL ImageDraw object to draw on
            element: Element being highlighted
            attr_names: List of attribute names to display
            bbox_scaled: Scaled bounding box [x0, top, x1, bottom]
            scale: Scale factor for rendering
        """
        # Try to load the font
        try:
            # Load a monospace font for attribute display with larger size
            try:
                # Try to get a default system font
                # Increase font size for better readability
                font_size = max(16, int(16 * scale))
                font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
            except Exception:
                # Fall back to default PIL font
                font = ImageFont.load_default()
                font_size = 16
        except Exception:
            # If font loading fails entirely
            return
            
        # Center attributes near the top of the highlight for better visibility
        # Calculate dimensions
        width = bbox_scaled[2] - bbox_scaled[0]
        height = bbox_scaled[3] - bbox_scaled[1]
        
        # For confidence only, center the text horizontally
        if attr_names == ['confidence']:
            x = bbox_scaled[0] + (width / 2)  # Will be adjusted after measuring text
        else:
            # Left-align attributes at the top of the highlight
            x = bbox_scaled[0] + 8  # More padding from left edge
            
        y = bbox_scaled[1] + 8  # More padding from top
        
        # White background for better readability
        background_color = (255, 255, 255, 255)  # Solid white
        text_color = (0, 0, 0, 255)  # Black
        
        # Process each requested attribute
        attr_text_lines = []
        for attr_name in attr_names:
            # Try to get the attribute value
            try:
                attr_value = getattr(element, attr_name, None)
                if attr_value is not None:
                    # Special formatting for confidence
                    if attr_name == 'confidence' and isinstance(attr_value, float):
                        # Just display the value without the name for confidence
                        attr_text = f"{attr_value:.2f}"
                    # Format other numeric values
                    elif isinstance(attr_value, float):
                        attr_value = f"{attr_value:.2f}"
                        attr_text = f"{attr_name}: {attr_value}"
                    # Regular formatting for other attributes
                    else:
                        attr_text = f"{attr_name}: {attr_value}"
                    
                    attr_text_lines.append(attr_text)
            except Exception:
                continue
                
        if not attr_text_lines:
            return
            
        # Calculate text height for background
        line_height = font_size + 4  # Add more padding between lines
        total_height = line_height * len(attr_text_lines)
        
        # Special case for confidence only - center it horizontally
        if attr_names == ['confidence'] and len(attr_text_lines) == 1:
            text_width = draw.textlength(attr_text_lines[0], font=font)
            center_x = bbox_scaled[0] + (width / 2)
            x = center_x - (text_width / 2)  # Center the text
            
            # Draw a solid background for the text with more padding
            bg_padding = 8  # Even more padding for confidence
            bg_width = text_width + (bg_padding * 2)
            
            # Make background box 
            draw.rectangle(
                [x - bg_padding, 
                 y - bg_padding, 
                 x + text_width + bg_padding, 
                 y + font_size + bg_padding],
                fill=background_color,
                outline=(0, 0, 0, 255),  # Add a black outline
                width=1  # 1px outline
            )
            
            # Draw centered confidence value
            draw.text((x, y), attr_text_lines[0], fill=text_color, font=font)
        else:
            # Draw a solid background for the text with more padding
            bg_padding = 6  # Increased padding
            bg_width = max(draw.textlength(line, font=font) for line in attr_text_lines) + (bg_padding * 2)
            draw.rectangle(
                [x - bg_padding, 
                 y - bg_padding, 
                 x + bg_width, 
                 y + total_height + bg_padding],
                fill=background_color,
                outline=(0, 0, 0, 255),  # Add a black outline
                width=1  # 1px outline
            )
            
            # Draw each attribute line
            current_y = y
            for line in attr_text_lines:
                draw.text((x, current_y), line, fill=text_color, font=font)
                current_y += line_height