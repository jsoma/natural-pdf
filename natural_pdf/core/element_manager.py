"""
Element Manager for natural-pdf.

This class handles the loading, creation, and management of PDF elements like
characters, words, rectangles, and lines extracted from a page.
"""
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from itertools import groupby
import re

from natural_pdf.elements.text import TextElement
from natural_pdf.elements.rect import RectangleElement
from natural_pdf.elements.line import LineElement

logger = logging.getLogger(__name__)

class ElementManager:
    """
    Manages the loading, creation, and retrieval of elements from a PDF page.
    
    This class centralizes the element management functionality previously
    contained in the Page class, providing better separation of concerns.
    """
    
    def __init__(self, page, font_attrs=None):
        """
        Initialize the ElementManager.
        
        Args:
            page: The parent Page object
            font_attrs: Font attributes to consider when grouping characters into words.
                       Default: ['fontname', 'size'] (Group by font name and size)
                       None: Only consider spatial relationships
                       List: Custom attributes to consider (e.g., ['fontname', 'size', 'color'])
        """
        self._page = page
        self._elements = None  # Lazy-loaded
        # Default to grouping by fontname and size if not specified
        self._font_attrs = ['fontname', 'size'] if font_attrs is None else font_attrs
        
    def load_elements(self):
        """
        Load all elements from the page (lazy loading).
        """
        if self._elements is None:
            # Create character elements with font information
            chars = self._create_char_elements()
            
            # Get keep_spaces setting from PDF config or default to True
            keep_spaces = self._page._parent._config.get('keep_spaces', True)
            
            # Group characters into words
            words = self._group_chars_into_words(keep_spaces, self._font_attrs)
            
            # Create the elements dictionary with all element types
            self._elements = {
                'chars': chars,
                'words': words,
                'rects': [RectangleElement(r, self._page) for r in self._page._page.rects],
                'lines': [LineElement(l, self._page) for l in self._page._page.lines],
                # Add other element types as needed
            }
            
            # Add regions if they exist
            if hasattr(self._page, '_regions') and ('detected' in self._page._regions or 'named' in self._page._regions):
                regions = []
                if 'detected' in self._page._regions:
                    regions.extend(self._page._regions['detected'])
                if 'named' in self._page._regions:
                    regions.extend(self._page._regions['named'].values())
                self._elements['regions'] = regions
    
    def _create_char_elements(self):
        """
        Create TextElement objects from page characters with enhanced font information.
        
        Returns:
            List of TextElement objects for characters
        """
        chars = []
        for c in self._page._page.chars:
            # Process font reference information
            self._process_font_information(c)
            
            # Add source attribute for native text elements
            c['source'] = 'native'
            chars.append(TextElement(c, self._page))
            
        return chars
    
    def _process_font_information(self, char_dict):
        """
        Process font information for a character dict, adding real_fontname when possible.
        
        Args:
            char_dict: Character dictionary to process
        """
        # Check for font references (F0, F1, etc.) and map to actual fonts
        if char_dict.get('fontname', '').startswith('F') and len(char_dict['fontname']) <= 3:
            # Access the PDF resource info to get actual font name
            font_ref = char_dict['fontname']
            try:
                # Try to get font info from resources
                if self._page._page.page_obj.get('Resources', {}).get('Font', {}):
                    fonts = self._page._page.page_obj['Resources']['Font']
                    if font_ref in fonts:
                        font_obj = fonts[font_ref]
                        if font_obj.get('BaseFont'):
                            char_dict['real_fontname'] = font_obj['BaseFont']
            except (KeyError, AttributeError, TypeError):
                pass
    
    def _group_chars_into_words(self, keep_spaces=True, font_attrs=None):
        """
        Group characters into words based on font attributes and spatial relationships.
        
        Args:
            keep_spaces: Whether to keep spaces in words or use them as word separators
            font_attrs: Font attributes to consider when grouping characters
            
        Returns:
            List of TextElement word objects
        """
        # Sort chars by y-position (line) and then x-position
        sorted_chars = sorted(self._page._page.chars, key=lambda c: (round(c['top']), c['x0']))
        
        # Group chars by line (similar y-position)
        line_groups = []
        for _, line_chars in groupby(sorted_chars, key=lambda c: round(c['top'])):
            line_chars = list(line_chars)
            
            # Process each line of characters into words
            words = self._process_line_into_words(line_chars, keep_spaces, font_attrs)
            line_groups.extend(words)
        
        return line_groups
    
    def _process_line_into_words(self, line_chars, keep_spaces, font_attrs):
        """
        Process a single line of characters into words.
        
        Args:
            line_chars: List of characters in the line
            keep_spaces: Whether to keep spaces in words
            font_attrs: Font attributes to consider for word breaks
            
        Returns:
            List of TextElement word objects for this line
        """
        words = []
        current_word = []
        
        for i, char in enumerate(line_chars):
            # Handle whitespace characters differently based on keep_spaces setting
            if char['text'].isspace():
                if keep_spaces:
                    # Include spaces in words when keep_spaces is enabled
                    if current_word:
                        current_word.append(char)
                    else:
                        # Skip leading spaces at the start of a line
                        continue
                else:
                    # Original behavior: Skip whitespace and close current word
                    if current_word:
                        # Create word and add to words list
                        word = self._create_word_element(current_word, font_attrs)
                        words.append(word)
                        current_word = []
                    continue
            
            # If this is a new word, start it
            if not current_word:
                current_word.append(char)
            else:
                # Check if this char is part of the current word or a new word
                prev_char = current_word[-1]
                
                # Check if font attributes match for this character
                font_attrs_match = self._check_font_attributes_match(char, prev_char, font_attrs)
                
                # If font attributes don't match, it's a new word
                if not font_attrs_match:
                    # Complete current word
                    word = self._create_word_element(current_word, font_attrs)
                    words.append(word)
                    current_word = [char]
                # If the gap between chars is larger than a threshold, it's a new word
                # Use a wider threshold when keep_spaces is enabled to allow for natural spaces
                elif char['x0'] - prev_char['x1'] > prev_char['width'] * (1.5 if keep_spaces else 0.5):
                    # Complete current word
                    word = self._create_word_element(current_word, font_attrs)
                    words.append(word)
                    current_word = [char]
                else:
                    # Continue current word
                    current_word.append(char)
        
        # Handle the last word if there is one
        if current_word:
            word = self._create_word_element(current_word, font_attrs)
            words.append(word)
        
        return words
    
    def _check_font_attributes_match(self, char, prev_char, font_attrs):
        """
        Check if two characters have matching font attributes.
        
        Args:
            char: Current character
            prev_char: Previous character
            font_attrs: List of font attributes to check
            
        Returns:
            Boolean indicating whether font attributes match
        """
        # Default to match if no font attributes specified
        if not font_attrs:
            return True
            
        # Check each font attribute
        for attr in font_attrs:
            # If attribute doesn't match or isn't present in both chars, they don't match
            if attr not in char or attr not in prev_char or char[attr] != prev_char[attr]:
                return False
                
        return True
    
    def _create_word_element(self, chars, font_attrs):
        """
        Create a word element from a list of character dictionaries.
        
        Args:
            chars: List of character dictionaries
            font_attrs: Font attributes to copy to the word
            
        Returns:
            TextElement representing the word
        """
        # Combine text from characters and normalize spaces
        text = ''.join(c['text'] for c in chars)
        
        # Collapse multiple consecutive spaces into a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Create a combined word object
        word_obj = {
            'text': text,
            'x0': min(c['x0'] for c in chars),
            'x1': max(c['x1'] for c in chars),
            'top': min(c['top'] for c in chars),
            'bottom': max(c['bottom'] for c in chars),
            'fontname': chars[0].get('fontname', ''),
            'size': chars[0].get('size', 0),
            'object_type': 'word',
            'page_number': chars[0]['page_number']
        }
        
        # Handle real fontname if available
        if 'real_fontname' in chars[0]:
            word_obj['real_fontname'] = chars[0]['real_fontname']
            
        # Handle color - use the first char's color
        if 'non_stroking_color' in chars[0]:
            word_obj['non_stroking_color'] = chars[0]['non_stroking_color']
        
        # Copy any additional font attributes
        if font_attrs:
            for attr in font_attrs:
                if attr in chars[0]:
                    word_obj[attr] = chars[0][attr]
        
        # Add source attribute for native text elements
        word_obj['source'] = 'native'
        
        return TextElement(word_obj, self._page)
    
    def create_text_elements_from_ocr(self, ocr_results, image_width=None, image_height=None):
        """
        Convert OCR results to TextElement objects.
        
        Args:
            ocr_results: List of OCR results with text, bbox, and confidence
            image_width: Width of the source image (for coordinate scaling)
            image_height: Height of the source image (for coordinate scaling)
            
        Returns:
            List of created TextElement objects
        """
        elements = []
        
        # Calculate scale factors to convert from image coordinates to PDF coordinates
        # Default to 1.0 if not provided (assume coordinates are already in PDF space)
        scale_x = 1.0
        scale_y = 1.0
        
        if image_width and image_height:
            scale_x = self._page.width / image_width
            scale_y = self._page.height / image_height
        
        for result in ocr_results:
            # Convert numpy int32 to float if needed and scale to PDF coordinates
            x0 = float(result['bbox'][0]) * scale_x
            top = float(result['bbox'][1]) * scale_y
            x1 = float(result['bbox'][2]) * scale_x
            bottom = float(result['bbox'][3]) * scale_y
            
            # Create a TextElement object with additional required fields for highlighting
            element_data = {
                'text': result['text'],
                'x0': x0,
                'top': top,
                'x1': x1,
                'bottom': bottom,
                'width': x1 - x0,
                'height': bottom - top,
                'object_type': 'text',
                'source': 'ocr',
                'confidence': result['confidence'],
                # Add default font information to work with existing expectations
                'fontname': 'OCR-detected',
                'size': 10.0,
                'page_number': self._page.number
            }
            
            elem = TextElement(element_data, self._page)
            elements.append(elem)
            
            # Add to page's elements
            if self._elements is not None:
                # Add to words list to make it accessible via standard API
                if 'words' in self._elements:
                    self._elements['words'].append(elem)
                else:
                    self._elements['words'] = [elem]
                
        return elements
    
    def add_element(self, element, element_type='words'):
        """
        Add an element to the managed elements.
        
        Args:
            element: The element to add
            element_type: The type of element ('words', 'chars', etc.)
            
        Returns:
            True if added successfully, False otherwise
        """
        # Load elements if not already loaded
        self.load_elements()
        
        # Add to the appropriate list
        if element_type in self._elements:
            self._elements[element_type].append(element)
            return True
        
        return False
    
    def add_region(self, region, name=None):
        """
        Add a region to the managed elements.
        
        Args:
            region: The region to add
            name: Optional name for the region
            
        Returns:
            True if added successfully, False otherwise
        """
        # Load elements if not already loaded
        self.load_elements()
        
        # Make sure regions is in _elements
        if 'regions' not in self._elements:
            self._elements['regions'] = []
            
        # Add to elements for selector queries
        if region not in self._elements['regions']:
            self._elements['regions'].append(region)
            return True
            
        return False
    
    def get_elements(self, element_type=None):
        """
        Get all elements of the specified type, or all elements if type is None.
        
        Args:
            element_type: Optional element type ('words', 'chars', 'rects', 'lines', etc.)
            
        Returns:
            List of elements
        """
        # Load elements if not already loaded
        self.load_elements()
        
        if element_type:
            return self._elements.get(element_type, [])
        
        # Combine all element types
        all_elements = []
        for elements in self._elements.values():
            all_elements.extend(elements)
        
        return all_elements
    
    def get_all_elements(self):
        """
        Get all elements from all types.
        
        Returns:
            List of all elements
        """
        # Load elements if not already loaded
        self.load_elements()
        
        # Combine all element types
        all_elements = []
        for elements in self._elements.values():
            all_elements.extend(elements)
        
        return all_elements
        
    @property
    def chars(self):
        """Get all character elements."""
        self.load_elements()
        return self._elements['chars']
    
    @property
    def words(self):
        """Get all word elements."""
        self.load_elements()
        return self._elements['words']
    
    @property
    def rects(self):
        """Get all rectangle elements."""
        self.load_elements()
        return self._elements['rects']
    
    @property
    def lines(self):
        """Get all line elements."""
        self.load_elements()
        return self._elements['lines']
    
    @property
    def regions(self):
        """Get all region elements."""
        self.load_elements()
        if 'regions' not in self._elements:
            self._elements['regions'] = []
        return self._elements['regions'] 