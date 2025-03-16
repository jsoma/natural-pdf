"""
Rectangle element class for natural-pdf.
"""
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING

from natural_pdf.elements.base import Element

if TYPE_CHECKING:
    from natural_pdf.core.page import Page


class RectangleElement(Element):
    """
    Represents a rectangle element in a PDF.
    
    This class is a wrapper around pdfplumber's rectangle objects,
    providing additional functionality for analysis and extraction.
    """
    
    def __init__(self, obj: Dict[str, Any], page: 'Page'):
        """
        Initialize a rectangle element.
        
        Args:
            obj: The underlying pdfplumber object
            page: The parent Page object
        """
        super().__init__(obj, page)
    
    @property
    def type(self) -> str:
        """Element type."""
        return 'rect'
    
    @property
    def fill(self) -> Tuple:
        """Get the fill color of the rectangle (RGB tuple)."""
        # PDFs often use non-RGB values, so we handle different formats
        color = self._obj.get('non_stroking_color', (0, 0, 0))
        
        # If it's a single value, treat as grayscale
        if isinstance(color, (int, float)):
            return (color, color, color)
        
        # If it's a tuple of 3 values, treat as RGB
        if isinstance(color, tuple) and len(color) == 3:
            return color
        
        # If it's a tuple of 4 values, treat as CMYK and convert to approximate RGB
        if isinstance(color, tuple) and len(color) == 4:
            c, m, y, k = color
            r = 1 - min(1, c + k)
            g = 1 - min(1, m + k)
            b = 1 - min(1, y + k)
            return (r, g, b)
        
        # Default to black
        return (0, 0, 0)
    
    @property
    def stroke(self) -> Tuple:
        """Get the stroke color of the rectangle (RGB tuple)."""
        # PDFs often use non-RGB values, so we handle different formats
        color = self._obj.get('stroking_color', (0, 0, 0))
        
        # If it's a single value, treat as grayscale
        if isinstance(color, (int, float)):
            return (color, color, color)
        
        # If it's a tuple of 3 values, treat as RGB
        if isinstance(color, tuple) and len(color) == 3:
            return color
        
        # If it's a tuple of 4 values, treat as CMYK and convert to approximate RGB
        if isinstance(color, tuple) and len(color) == 4:
            c, m, y, k = color
            r = 1 - min(1, c + k)
            g = 1 - min(1, m + k)
            b = 1 - min(1, y + k)
            return (r, g, b)
        
        # Default to black
        return (0, 0, 0)
    
    @property
    def stroke_width(self) -> float:
        """Get the stroke width of the rectangle."""
        return self._obj.get('linewidth', 0)
    
    def text_inside(self, **kwargs) -> Any:
        """
        Get text elements inside this rectangle.
        
        Args:
            **kwargs: Additional filter parameters
            
        Returns:
            ElementCollection of text elements inside this rectangle
        """
        from natural_pdf.elements.collections import ElementCollection
        
        # TODO: Implement proper filtering of elements inside this rectangle
        return ElementCollection([])  # Placeholder
    
    def extract_text(self, **kwargs) -> str:
        """
        Extract text from inside this rectangle.
        
        Args:
            **kwargs: Additional extraction parameters
            
        Returns:
            Extracted text as string
        """
        # Use the region to extract text
        from natural_pdf.elements.region import Region
        region = Region(self.page, self.bbox)
        return region.extract_text(**kwargs)
    
    def __repr__(self) -> str:
        """String representation of the rectangle element."""
        return f"<RectangleElement fill={self.fill} stroke={self.stroke} bbox={self.bbox}>"