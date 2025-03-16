"""
Text structure analyzer for natural-pdf.
"""
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.collections import ElementCollection

class TextStyleAnalyzer:
    """
    Analyzes and groups text elements by their style properties.
    
    This analyzer groups text elements based on their font properties
    (size, fontname, etc.) to identify different text styles in a document.
    """
    
    def __init__(self):
        """Initialize the text style analyzer."""
        pass
        
    def analyze(self, page: 'Page') -> Dict[str, 'ElementCollection']:
        """
        Analyze the text styles on a page.
        
        Args:
            page: Page to analyze
            
        Returns:
            Dictionary mapping style labels to element collections
        """
        # Get all text elements
        text_elements = page.find_all('text')
        
        # Skip empty pages
        if not text_elements:
            return {}
        
        # Group elements by their style properties
        style_groups = self._group_by_style(text_elements)
        
        return style_groups
    
    def _group_by_style(self, elements: 'ElementCollection') -> Dict[str, 'ElementCollection']:
        """
        Group text elements by their style properties.
        
        Args:
            elements: Text elements to group
            
        Returns:
            Dictionary mapping style labels to element collections
        """
        from natural_pdf.elements.collections import ElementCollection
        
        # Extract style properties for each element
        element_styles = []
        for element in elements:
            style = self._extract_style_properties(element)
            element_styles.append((element, style))
        
        # Group elements by their style properties
        style_groups = defaultdict(list)
        style_mapping = {}  # Maps style tuple to style number
        
        for element, style in element_styles:
            # Get or create style number
            if style not in style_mapping:
                style_mapping[style] = len(style_mapping)
            
            style_num = style_mapping[style]
            style_groups[f"Text Style {style_num+1}"].append(element)
        
        # Convert to ElementCollections
        return {
            label: ElementCollection(elements) 
            for label, elements in style_groups.items()
        }
    
    def _extract_style_properties(self, element: 'Element') -> Tuple:
        """
        Extract style properties from a text element.
        
        Args:
            element: Text element
            
        Returns:
            Tuple of style properties (hashable)
        """
        # Extract properties that define the style
        properties = []
        
        # Font size (rounded to nearest 0.5 to handle small variations)
        if hasattr(element, 'size') and element.size is not None:
            font_size = round(element.size * 2) / 2  # Round to nearest 0.5
            properties.append(font_size)
        else:
            properties.append(None)
        
        # Font name
        if hasattr(element, 'fontname') and element.fontname is not None:
            properties.append(element.fontname)
        else:
            properties.append(None)
        
        # Font characteristics (derived from name)
        is_bold = False
        is_italic = False
        if hasattr(element, 'fontname') and element.fontname is not None:
            font_lower = element.fontname.lower()
            is_bold = ('bold' in font_lower or 'black' in font_lower or element.fontname.endswith('-B'))
            is_italic = ('italic' in font_lower or 'oblique' in font_lower or element.fontname.endswith('-I'))
        
        properties.append(is_bold)
        properties.append(is_italic)
        
        # Text color
        if hasattr(element, 'non_stroking_color') and element.non_stroking_color is not None:
            # Convert color to a hashable form (tuple)
            if isinstance(element.non_stroking_color, (list, tuple)):
                color = tuple(element.non_stroking_color)
            else:
                color = element.non_stroking_color
            properties.append(color)
        else:
            properties.append(None)
        
        return tuple(properties)
        
    def analyze_and_label(self, page: 'Page') -> 'Page':
        """
        Analyze the page text styles and add style labels to elements.
        
        Args:
            page: Page to analyze
            
        Returns:
            Page with style labels added
        """
        # Analyze the styles
        styles = self.analyze(page)
        
        # Add style as an attribute to each element
        for label, elements in styles.items():
            for element in elements:
                element._style_label = label
                
        # Store the styles on the page
        page._text_styles = styles
        
        return page