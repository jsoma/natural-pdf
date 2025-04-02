"""
CSS-like selector parser for natural-pdf.
"""
import re
import ast
from typing import Dict, Any, List, Optional, Union, Tuple
from colour import Color


def safe_parse_value(value_str: str) -> Any:
    """
    Safely parse a value string without using eval().
    
    Args:
        value_str: String representation of a value (number, tuple, string, etc.)
        
    Returns:
        Parsed value
    """
    # Strip quotes first if it's a quoted string
    value_str = value_str.strip()
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]
    
    # Try parsing as a Python literal (numbers, tuples, lists)
    try:
        return ast.literal_eval(value_str)
    except (SyntaxError, ValueError):
        # If it's not a valid Python literal, return as is
        return value_str


def safe_parse_color(value_str: str) -> tuple:
    """
    Parse a color value which could be an RGB tuple, color name, or hex code.
    
    Args:
        value_str: String representation of a color (e.g., "red", "#ff0000", "(1,0,0)")
        
    Returns:
        RGB tuple (r, g, b) with values from 0 to 1
    """
    value_str = value_str.strip()
    
    # Try parsing as a Python literal (for RGB tuples)
    try:
        # If it's already a valid tuple or list, parse it
        color_tuple = ast.literal_eval(value_str)
        if isinstance(color_tuple, (list, tuple)) and len(color_tuple) >= 3:
            # Return just the RGB components as a tuple
            return tuple(color_tuple[:3])
    except (SyntaxError, ValueError):
        # Not a valid tuple/list, try as a color name or hex
        try:
            # Use colour library to parse color names, hex values, etc.
            color = Color(value_str)
            # Convert to RGB tuple with values between 0 and 1
            return (color.red, color.green, color.blue)
        except (ValueError, AttributeError):
            # If color parsing fails, return a default (black)
            return (0, 0, 0)
    
    # If we got here with a non-tuple, return default
    return (0, 0, 0)


def parse_selector(selector: str) -> Dict[str, Any]:
    """
    Parse a CSS-like selector string into a structured selector object.
    
    Examples:
    - 'text:contains("Revenue")'
    - 'table:below("Financial Data")'
    - 'rect[fill=(1,0,0)]'
    
    Args:
        selector: CSS-like selector string
        
    Returns:
        Dict representing the parsed selector
    """
    # Basic structure for result
    result = {
        'type': 'any',  # Default to any element type
        'filters': [],
        'attributes': {},
        'pseudo_classes': [],
    }
    
    # Check if empty or None
    if not selector or not isinstance(selector, str):
        return result
    
    # Parse element type
    type_match = re.match(r'^([a-zA-Z_]+)', selector)
    if type_match:
        result['type'] = type_match.group(1).lower()
        selector = selector[len(type_match.group(0)):]
    
    # Parse attributes (e.g., [color=(1,0,0)])
    attr_pattern = r'\[([a-zA-Z_]+)(>=|<=|>|<|[*~]?=)([^\]]+)\]'
    attr_matches = re.findall(attr_pattern, selector)
    for name, op, value in attr_matches:
        # Handle special parsing for color attributes
        if name in ['color', 'non_stroking_color', 'fill', 'stroke', 'strokeColor', 'fillColor']:
            value = safe_parse_color(value)
        else:
            # Safe parsing for other attributes
            value = safe_parse_value(value)
        
        # Store attribute with operator
        result['attributes'][name] = {
            'op': op,
            'value': value
        }
    
    # Parse pseudo-classes (e.g., :contains("text"))
    pseudo_pattern = r':([a-zA-Z_]+)(?:\(([^)]+)\))?'
    pseudo_matches = re.findall(pseudo_pattern, selector)
    for name, args in pseudo_matches:
        # Process arguments
        processed_args = args
        if args:
            if name in ['color', 'background']:
                processed_args = safe_parse_color(args)
            else:
                processed_args = safe_parse_value(args)
        
        result['pseudo_classes'].append({
            'name': name,
            'args': processed_args
        })
    
    return result


def selector_to_filter_func(selector: Dict[str, Any], **kwargs) -> callable:
    """
    Convert a parsed selector to a filter function.
    
    Args:
        selector: Parsed selector dictionary
        **kwargs: Additional filter parameters including:
                 - regex: Whether to use regex for text search
                 - case: Whether to do case-sensitive text search
        
    Returns:
        Function that takes an element and returns True if it matches
    """
    def filter_func(element):
        # Check element type
        if selector['type'] != 'any':
            # Special handling for 'text' type to match both 'text', 'char', and 'word'
            if selector['type'] == 'text':
                if element.type not in ['text', 'char', 'word']:
                    return False
            # Special handling for 'region' type to check for detected layout regions
            elif selector['type'] == 'region':
                # Check if this is a Region with region_type property
                if not hasattr(element, 'region_type'):
                    return False
                    
                # If 'type' attribute specified, it will be checked in the attributes section
            # Check for Docling-specific types (section-header, etc.)
            elif hasattr(element, 'normalized_type') and element.normalized_type == selector['type']:
                # This is a direct match with a Docling region type
                pass
            # Otherwise, require exact match with the element's type attribute
            elif not hasattr(element, 'type') or element.type != selector['type']:
                return False
        
        # Check attributes
        for name, attr_info in selector['attributes'].items():
            op = attr_info['op']
            value = attr_info['value']
            
            # Special case for fontname attribute - allow matching part of the name
            if name == 'fontname' and op == '*=':
                element_value = getattr(element, name, None)
                if element_value is None or value.lower() not in element_value.lower():
                    return False
                continue
            
            # Convert hyphenated attribute names to underscore for Python properties
            python_name = name.replace('-', '_')
            
            # Special case for region attributes
            if selector['type'] == 'region':
                if name == 'type':
                    # Use normalized_type for comparison if available
                    if hasattr(element, 'normalized_type') and element.normalized_type:
                        element_value = element.normalized_type
                    else:
                        # Convert spaces to hyphens for consistency with the normalized format
                        element_value = getattr(element, 'region_type', '').lower().replace(' ', '-')
                elif name == 'model':
                    # Special handling for model attribute in regions
                    element_value = getattr(element, 'model', None)
                else:
                    # Get the attribute value from the element normally
                    element_value = getattr(element, python_name, None)
            else:
                # Get the attribute value from the element normally for non-region elements
                element_value = getattr(element, python_name, None)
                
            if element_value is None:
                return False
            
            # Apply operator
            if op == '=':
                if element_value != value:
                    return False
            elif op == '~=':
                # Approximate match (e.g., for colors)
                if not _is_approximate_match(element_value, value):
                    return False
            elif op == '>=':
                # Greater than or equal (element value must be >= specified value)
                if not (isinstance(element_value, (int, float)) and 
                        isinstance(value, (int, float)) and 
                        element_value >= value):
                    return False
            elif op == '<=':
                # Less than or equal (element value must be <= specified value)
                if not (isinstance(element_value, (int, float)) and 
                        isinstance(value, (int, float)) and 
                        element_value <= value):
                    return False
            elif op == '>':
                # Greater than (element value must be > specified value)
                if not (isinstance(element_value, (int, float)) and 
                        isinstance(value, (int, float)) and 
                        element_value > value):
                    return False
            elif op == '<':
                # Less than (element value must be < specified value)
                if not (isinstance(element_value, (int, float)) and 
                        isinstance(value, (int, float)) and 
                        element_value < value):
                    return False
        
        # Check pseudo-classes
        for pseudo in selector['pseudo_classes']:
            name = pseudo['name']
            args = pseudo['args']
            
            # Handle various pseudo-classes
            if name == 'contains' and hasattr(element, 'text'):
                use_regex = kwargs.get('regex', False)
                ignore_case = not kwargs.get('case', True)
                
                if use_regex:
                    import re
                    if not element.text:
                        return False
                    try:
                        pattern = re.compile(args, re.IGNORECASE if ignore_case else 0)
                        if not pattern.search(element.text):
                            return False
                    except re.error:
                        # If regex is invalid, fall back to literal text search
                        element_text = element.text
                        search_text = args
                        
                        if ignore_case:
                            element_text = element_text.lower()
                            search_text = search_text.lower()
                            
                        if search_text not in element_text:
                            return False
                else:
                    # String comparison with case sensitivity option
                    if not element.text:
                        return False
                        
                    element_text = element.text
                    search_text = args
                    
                    if ignore_case:
                        element_text = element_text.lower()
                        search_text = search_text.lower()
                        
                    if search_text not in element_text:
                        return False
            elif name == 'starts-with' and hasattr(element, 'text'):
                if not element.text or not element.text.startswith(args):
                    return False
            elif name == 'ends-with' and hasattr(element, 'text'):
                if not element.text or not element.text.endswith(args):
                    return False
            elif name == 'bold':
                if not (hasattr(element, 'bold') and element.bold):
                    return False
            elif name == 'italic':
                if not (hasattr(element, 'italic') and element.italic):
                    return False
            elif name == 'horizontal':
                if not (hasattr(element, 'is_horizontal') and element.is_horizontal):
                    return False
            elif name == 'vertical':
                if not (hasattr(element, 'is_vertical') and element.is_vertical):
                    return False
            else:
                # Potentially unsupported pseudo-class, or one handled elsewhere (like :not)
                pass
        
        # If we get here, all checks passed
        return True
    
    return filter_func


def _is_approximate_match(value1, value2, tolerance: float = 0.1) -> bool:
    """
    Check if two values approximately match.
    
    This is mainly used for color comparisons with some tolerance.
    
    Args:
        value1: First value
        value2: Second value
        tolerance: Maximum difference allowed
        
    Returns:
        True if the values approximately match
    """
    # Handle string colors by converting them to RGB tuples
    if isinstance(value1, str):
        try:
            value1 = tuple(Color(value1).rgb)
        except:
            pass
    
    if isinstance(value2, str):
        try:
            value2 = tuple(Color(value2).rgb)
        except:
            pass
            
    # If both are tuples/lists with the same length (e.g., colors)
    if (isinstance(value1, (list, tuple)) and 
        isinstance(value2, (list, tuple)) and
        len(value1) == len(value2)):
        
        # Check if all components are within tolerance
        return all(abs(a - b) <= tolerance for a, b in zip(value1, value2))
    
    # If both are numbers
    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
        return abs(value1 - value2) <= tolerance
    
    # Default to exact match for other types
    return value1 == value2


PSEUDO_CLASS_FUNCTIONS = {
    'bold': lambda el: hasattr(el, 'bold') and el.bold,
    'italic': lambda el: hasattr(el, 'italic') and el.italic,
    'first-child': lambda el: hasattr(el, 'parent') and el.parent and el.parent.children[0] == el, # Example placeholder
    'last-child': lambda el: hasattr(el, 'parent') and el.parent and el.parent.children[-1] == el, # Example placeholder
    # Add the new pseudo-classes for negation
    'not-bold': lambda el: hasattr(el, 'bold') and not el.bold,
    'not-italic': lambda el: hasattr(el, 'italic') and not el.italic,
}


def selector_to_filter_func(selector: Dict[str, Any], **kwargs) -> callable:
    """
    Convert a parsed selector to a filter function.
    
    Args:
        selector: Parsed selector dictionary
        **kwargs: Additional filter parameters including:
                 - regex: Whether to use regex for text search
                 - case: Whether to do case-sensitive text search
        
    Returns:
        Function that takes an element and returns True if it matches
    """
    def filter_func(element):
        # Check element type
        if selector['type'] != 'any':
            # Special handling for 'text' type to match both 'text', 'char', and 'word'
            if selector['type'] == 'text':
                if element.type not in ['text', 'char', 'word']:
                    return False
            # Special handling for 'region' type to check for detected layout regions
            elif selector['type'] == 'region':
                # Check if this is a Region with region_type property
                if not hasattr(element, 'region_type'):
                    return False
                    
                # If 'type' attribute specified, it will be checked in the attributes section
            # Check for Docling-specific types (section-header, etc.)
            elif hasattr(element, 'normalized_type') and element.normalized_type == selector['type']:
                # This is a direct match with a Docling region type
                pass
            # Otherwise, require exact match with the element's type attribute
            elif not hasattr(element, 'type') or element.type != selector['type']:
                return False
        
        # Check attributes
        for name, attr_info in selector['attributes'].items():
            op = attr_info['op']
            value = attr_info['value']
            
            # Special case for fontname attribute - allow matching part of the name
            if name == 'fontname' and op == '*=':
                element_value = getattr(element, name, None)
                if element_value is None or value.lower() not in element_value.lower():
                    return False
                continue
            
            # Convert hyphenated attribute names to underscore for Python properties
            python_name = name.replace('-', '_')
            
            # Special case for region attributes
            if selector['type'] == 'region':
                if name == 'type':
                    # Use normalized_type for comparison if available
                    if hasattr(element, 'normalized_type') and element.normalized_type:
                        element_value = element.normalized_type
                    else:
                        # Convert spaces to hyphens for consistency with the normalized format
                        element_value = getattr(element, 'region_type', '').lower().replace(' ', '-')
                elif name == 'model':
                    # Special handling for model attribute in regions
                    element_value = getattr(element, 'model', None)
                else:
                    # Get the attribute value from the element normally
                    element_value = getattr(element, python_name, None)
            else:
                # Get the attribute value from the element normally for non-region elements
                element_value = getattr(element, python_name, None)
                
            if element_value is None:
                return False
            
            # Apply operator
            if op == '=':
                if element_value != value:
                    return False
            elif op == '~=':
                # Approximate match (e.g., for colors)
                if not _is_approximate_match(element_value, value):
                    return False
            elif op == '>=':
                # Greater than or equal (element value must be >= specified value)
                if not (isinstance(element_value, (int, float)) and 
                        isinstance(value, (int, float)) and 
                        element_value >= value):
                    return False
            elif op == '<=':
                # Less than or equal (element value must be <= specified value)
                if not (isinstance(element_value, (int, float)) and 
                        isinstance(value, (int, float)) and 
                        element_value <= value):
                    return False
            elif op == '>':
                # Greater than (element value must be > specified value)
                if not (isinstance(element_value, (int, float)) and 
                        isinstance(value, (int, float)) and 
                        element_value > value):
                    return False
            elif op == '<':
                # Less than (element value must be < specified value)
                if not (isinstance(element_value, (int, float)) and 
                        isinstance(value, (int, float)) and 
                        element_value < value):
                    return False
        
        # Check pseudo-classes
        for pseudo in selector['pseudo_classes']:
            name = pseudo['name']
            args = pseudo['args']
            
            # Handle various pseudo-classes
            if name == 'contains' and hasattr(element, 'text'):
                use_regex = kwargs.get('regex', False)
                ignore_case = not kwargs.get('case', True)
                
                if use_regex:
                    import re
                    if not element.text:
                        return False
                    try:
                        pattern = re.compile(args, re.IGNORECASE if ignore_case else 0)
                        if not pattern.search(element.text):
                            return False
                    except re.error:
                        # If regex is invalid, fall back to literal text search
                        element_text = element.text
                        search_text = args
                        
                        if ignore_case:
                            element_text = element_text.lower()
                            search_text = search_text.lower()
                            
                        if search_text not in element_text:
                            return False
                else:
                    # String comparison with case sensitivity option
                    if not element.text:
                        return False
                        
                    element_text = element.text
                    search_text = args
                    
                    if ignore_case:
                        element_text = element_text.lower()
                        search_text = search_text.lower()
                        
                    if search_text not in element_text:
                        return False
            elif name == 'starts-with' and hasattr(element, 'text'):
                if not element.text or not element.text.startswith(args):
                    return False
            elif name == 'ends-with' and hasattr(element, 'text'):
                if not element.text or not element.text.endswith(args):
                    return False
            elif name == 'bold':
                if not (hasattr(element, 'bold') and element.bold):
                    return False
            elif name == 'italic':
                if not (hasattr(element, 'italic') and element.italic):
                    return False
            elif name == 'horizontal':
                if not (hasattr(element, 'is_horizontal') and element.is_horizontal):
                    return False
            elif name == 'vertical':
                if not (hasattr(element, 'is_vertical') and element.is_vertical):
                    return False
            else:
                # Check pseudo-classes (basic ones like :bold, :italic)
                if name in PSEUDO_CLASS_FUNCTIONS:
                    if not PSEUDO_CLASS_FUNCTIONS[name](element):
                        return False
                elif name == 'contains':
                    if not hasattr(element, 'text') or not element.text:
                        return False
                    text_to_check = element.text
                    search_term = args
                    if not kwargs.get('case', True): # Check case flag from kwargs
                        text_to_check = text_to_check.lower()
                        search_term = search_term.lower()
                        
                    if kwargs.get('regex', False): # Check regex flag from kwargs
                        try:
                            if not re.search(search_term, text_to_check):
                                return False
                        except re.error as e:
                             logger.warning(f"Invalid regex in :contains selector '{search_term}': {e}")
                             return False # Invalid regex cannot match
                    else:
                        if search_term not in text_to_check:
                            return False
                # Skip complex pseudo-classes like :near, :above here, handled later
                elif name in ('above', 'below', 'near', 'left-of', 'right-of'):
                    pass # Handled separately after initial filtering
                else:
                     # Optionally log unknown pseudo-classes
                     # logger.warning(f"Unknown pseudo-class: {name}")
                     pass
        
        return True # Element passes all attribute and simple pseudo-class filters

    return filter_func