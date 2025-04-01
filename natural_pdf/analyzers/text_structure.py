"""
Text structure analyzer for natural-pdf.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
from collections import defaultdict

# Import ElementCollection and TextStyleOptions
from natural_pdf.elements.collections import ElementCollection
from natural_pdf.analyzers.text_options import TextStyleOptions

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element
    # Remove ElementCollection from here if imported above

logger = logging.getLogger(__name__)

# Simple regex to remove common PDF font prefixes like "ABCDEF+"
FONT_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")

# Common font weight/style keywords
FONT_WEIGHTS = {"bold": "Bold", "black": "Bold", "heavy": "Bold", "medium": "", "light": "Light", "thin": "Thin"}
FONT_STYLES = {"italic": "Italic", "oblique": "Italic"}

class TextStyleAnalyzer:
    """
    Analyzes and groups text elements by their style properties based on configuration.
    
    This analyzer groups text elements based on specified font properties
    (controlled by TextStyleOptions) and adds 'style_label', 'style_key',
    and 'style_properties' attributes to each processed text element.
    """
    
    def __init__(self, options: Optional[TextStyleOptions] = None):
        """
        Initialize the text style analyzer.
        
        Args:
            options: Configuration options for the analysis. Uses default if None.
        """
        self.options = options or TextStyleOptions()
        logger.debug(f"Initialized TextStyleAnalyzer with options: {self.options}")

    def analyze(self, page: 'Page', options: Optional[TextStyleOptions] = None) -> 'ElementCollection':
        """
        Analyze text styles on a page, group elements, and add style attributes.
        
        Args:
            page: The Page object to analyze.
            options: Override the analyzer's default TextStyleOptions for this run.
            
        Returns:
            ElementCollection containing all processed text elements (typically words)
            with added 'style_label', 'style_key', and 'style_properties' attributes.
        """
        current_options = options or self.options
        logger.info(f"Starting text style analysis for page {page.number} with options: {current_options}")

        # Use page.words for better granularity
        text_elements = page.words
        # Fallback if words are somehow empty/not generated
        if not text_elements:
             text_elements = page.find_all('text').elements # Get list from collection

        # Skip empty pages or pages with no text elements
        if not text_elements:
            logger.warning(f"Page {page.number} has no text elements to analyze.")
            return ElementCollection([])

        style_cache: Dict[Tuple, Dict[str, Any]] = {} # Maps style_key_tuple -> {'label': str, 'properties': dict}
        processed_elements: List['Element'] = []

        # Ensure consistent ordering for style key creation
        group_by_keys = sorted(current_options.group_by)

        for element in text_elements:
            # Skip elements without necessary attributes (e.g., non-text elements if find_all was used)
            if not hasattr(element, 'text') or not hasattr(element, 'size'):
                logger.debug(f"Skipping element without text/size: {element}")
                continue

            try:
                style_properties = self._extract_style_properties(element, current_options)
                style_key = self._create_style_key(style_properties, group_by_keys)

                if style_key not in style_cache:
                    label = self._generate_style_label(style_properties, current_options, len(style_cache) + 1)
                    style_cache[style_key] = {'label': label, 'properties': style_properties}
                    logger.debug(f"New style detected (Key: {style_key}): Label='{label}', Props={style_properties}")

                # Add attributes to the element
                element.style_label = style_cache[style_key]['label']
                element.style_key = style_key
                # Add the full properties dict for potential detailed inspection
                element.style_properties = style_cache[style_key]['properties']

                processed_elements.append(element)

            except Exception as e:
                 logger.warning(f"Error processing element {element} for text style: {e}", exc_info=True)
                 # Optionally add element without style info or skip it
                 # processed_elements.append(element) # Add anyway?

        # Optionally store a summary on the page
        page._text_styles_summary = style_cache
        logger.info(f"Finished text style analysis for page {page.number}. Found {len(style_cache)} unique styles.")

        return ElementCollection(processed_elements)
    
    def _extract_style_properties(self, element: 'Element', options: TextStyleOptions) -> Dict[str, Any]:
        """
        Extract style properties from a text element based on options.
        
        Args:
            element: Text element.
            options: TextStyleOptions driving the extraction.
            
        Returns:
            Dictionary of extracted style properties.
        """
        properties = {}

        # Font size
        font_size = None
        if hasattr(element, 'size') and element.size is not None:
            # Round based on tolerance
            rounding_factor = 1.0 / options.size_tolerance
            font_size = round(element.size * rounding_factor) / rounding_factor
        properties['size'] = font_size

        # Font name
        font_name = None
        normalized_font_name = None
        if hasattr(element, 'fontname') and element.fontname is not None:
            font_name = element.fontname
            normalized_font_name = self._normalize_font_name(font_name, options)
        properties['fontname'] = normalized_font_name if options.normalize_fontname else font_name

        # Font characteristics (derived from normalized name if available)
        name_to_check = normalized_font_name or font_name or ""
        name_lower = name_to_check.lower()
        is_bold = ('bold' in name_lower or 'black' in name_lower or 'heavy' in name_lower or name_to_check.endswith('-B'))
        is_italic = ('italic' in name_lower or 'oblique' in name_lower or name_to_check.endswith('-I'))

        properties['is_bold'] = is_bold
        properties['is_italic'] = is_italic

        # Text color
        color = None
        if not options.ignore_color and hasattr(element, 'non_stroking_color') and element.non_stroking_color is not None:
            raw_color = element.non_stroking_color
            # Convert color to a hashable form (tuple)
            if isinstance(raw_color, (list, tuple)):
                color = tuple(round(c, 3) for c in raw_color) # Round color components
            else:
                # Handle simple grayscale or other non-list representations if needed
                 try:
                     color = round(float(raw_color), 3)
                 except (ValueError, TypeError):
                     color = str(raw_color) # Fallback to string if cannot convert
            # Normalize common colors (optional, could be complex)
            # Example: (0.0, 0.0, 0.0) -> 'black', (1.0, 1.0, 1.0) -> 'white'
            if color == (0.0, 0.0, 0.0) or color == 0.0: color = 'black'
            if color == (1.0, 1.0, 1.0) or color == 1.0: color = 'white'
        properties['color'] = color

        return properties
    
    def _normalize_font_name(self, fontname: str, options: TextStyleOptions) -> str:
        """ Basic normalization of font names. """
        if not options.normalize_fontname:
            return fontname
        # Remove common subset prefixes like "ABCDEF+"
        name = FONT_PREFIX_RE.sub("", fontname)
        # Could add more rules here, e.g., removing version numbers, standardizing separators
        return name
    
    def _parse_font_name(self, normalized_fontname: str) -> Dict[str, str]:
        """ Attempt to parse family, weight, and style from a font name. Very heuristic. """
        if not normalized_fontname:
            return {'family': 'Unknown', 'weight': '', 'style': ''}

        parts = re.split(r'[-,_ ]', normalized_fontname)
        family_parts = []
        weight = ''
        style = ''

        for part in parts:
            part_lower = part.lower()
            found = False
            # Check weights
            for key, val in FONT_WEIGHTS.items():
                if key in part_lower:
                    weight = val
                    found = True
                    break
            if found: continue # Skip part if it was a weight

            # Check styles
            for key, val in FONT_STYLES.items():
                if key in part_lower:
                    style = val
                    found = True
                    break
            if found: continue # Skip part if it was a style

            # If not weight or style, assume it's part of the family name
            if part: # Avoid empty strings from multiple delimiters
                 family_parts.append(part)

        family = "".join(family_parts) or "Unknown" # Join remaining parts
        # Simple cleanup: Remove "MT" often appended? Maybe too aggressive.
        # if family.endswith("MT"): family = family[:-2]

        return {'family': family, 'weight': weight, 'style': style}
    
    def _create_style_key(self, properties: Dict[str, Any], group_by_keys: List[str]) -> Tuple:
        """ Create a hashable tuple key based on selected properties. """
        key_parts = []
        for key in group_by_keys: # Use the pre-sorted list
            value = properties.get(key)
            # Ensure hashable - colors should already be tuples or basic types
            if isinstance(value, list): # Should not happen if _extract handled color correctly
                value = tuple(value)
            key_parts.append(value)
        return tuple(key_parts)
    
    def _generate_style_label(self, properties: Dict[str, Any], options: TextStyleOptions, style_index: int) -> str:
        """ Generate a style label based on properties and options. """
        if not options.descriptive_labels:
            return f"{options.label_prefix} {style_index}"

        try:
            font_details = self._parse_font_name(properties.get('fontname', ''))

            label_data = {
                'size': properties.get('size', '?'),
                'fontname': properties.get('fontname', 'Unknown'),
                'is_bold': properties.get('is_bold', False),
                'is_italic': properties.get('is_italic', False),
                'color': properties.get('color', ''),
                'family': font_details['family'],
                # Use parsed weight/style if available, otherwise fallback to is_bold/is_italic flags
                'weight': font_details['weight'] or ('Bold' if properties.get('is_bold') else ''),
                'style': font_details['style'] or ('Italic' if properties.get('is_italic') else ''),
            }
            # Ensure style has a space separator if both weight and style exist
            if label_data['weight'] and label_data['style']:
                label_data['style'] = " " + label_data['style']

            # Handle color formatting for label
            color_val = label_data['color']
            if isinstance(color_val, tuple):
                 color_str = f"rgb{color_val}" # Basic tuple representation
            elif isinstance(color_val, str):
                 color_str = color_val # Already string ('black', 'white', or fallback)
            else:
                 color_str = str(color_val) # Other types
            label_data['color_str'] = color_str

            # Format the label, handle potential missing keys in format string gracefully
            label = options.label_format.format_map(defaultdict(str, label_data))
            return label.strip().replace("  ", " ") # Cleanup extra spaces

        except Exception as e:
            logger.warning(f"Error generating descriptive label for style {properties}: {e}. Falling back to numeric label.")
            # Fallback to numeric label on error
            return f"{options.label_prefix} {style_index}"