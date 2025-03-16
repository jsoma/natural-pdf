"""
Reading order utilities for natural-pdf.
"""
from typing import List, Dict, Any, Callable, Optional


def establish_reading_order(elements: List[Dict[str, Any]], 
                           algorithm: str = 'basic') -> List[Dict[str, Any]]:
    """
    Establish reading order for a collection of elements.
    
    Args:
        elements: List of elements to order
        algorithm: Algorithm to use ('basic', 'column', 'complex')
        
    Returns:
        List of elements in reading order
    """
    if algorithm == 'basic':
        return _basic_reading_order(elements)
    elif algorithm == 'column':
        return _column_reading_order(elements)
    elif algorithm == 'complex':
        return _complex_reading_order(elements)
    else:
        # Default to basic
        return _basic_reading_order(elements)


def _basic_reading_order(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Basic top-to-bottom, left-to-right reading order.
    
    Args:
        elements: List of elements to order
        
    Returns:
        List of elements in reading order
    """
    # Simple sort by y0 (top), then by x0 (left)
    return sorted(elements, key=lambda e: (
        e.get('top', e.get('y0', 0)), 
        e.get('x0', 0)
    ))


def _column_reading_order(elements: List[Dict[str, Any]], 
                         column_threshold: float = 0.2,
                         x_tolerance: float = 10.0) -> List[Dict[str, Any]]:
    """
    Reading order that accounts for columns.
    
    This is more complex as it needs to detect columns first,
    then read each column in order.
    
    Args:
        elements: List of elements to order
        column_threshold: Percentage overlap threshold for column detection (0.0 to 1.0)
        x_tolerance: Horizontal tolerance for determining column edges
        
    Returns:
        List of elements in reading order
    """
    if not elements:
        return []
    
    # 1. Group elements by line
    lines = group_elements_by_line(elements)
    
    # 2. For each line, find the x-coordinate ranges (potential column boundaries)
    line_x_ranges = []
    for line in lines:
        for el in line:
            x0 = el.get('x0', 0)
            x1 = el.get('x1', 0)
            line_x_ranges.append((x0, x1))
    
    # If we don't have enough ranges to detect columns, just use basic ordering
    if len(line_x_ranges) < 3:
        return _basic_reading_order(elements)
    
    # 3. Detect columns by clustering x-coordinate ranges
    def overlaps(range1, range2, threshold=column_threshold):
        """Determine if two ranges overlap by more than threshold percentage."""
        # Calculate overlap
        overlap_start = max(range1[0], range2[0])
        overlap_end = min(range1[1], range2[1])
        overlap = max(0, overlap_end - overlap_start)
        
        # Calculate lengths
        len1 = range1[1] - range1[0]
        len2 = range2[1] - range2[0]
        
        # Calculate overlap as percentage of the shorter range
        shorter_len = min(len1, len2)
        if shorter_len == 0:
            return False
            
        return overlap / shorter_len >= threshold
    
    # Cluster x-ranges into columns
    columns = []
    for x_range in line_x_ranges:
        # Skip zero-width ranges
        if x_range[1] - x_range[0] <= 0:
            continue
            
        # Try to find an existing column to add to
        added = False
        for col in columns:
            if any(overlaps(x_range, r) for r in col):
                col.append(x_range)
                added = True
                break
                
        # If not added to an existing column, create a new one
        if not added:
            columns.append([x_range])
    
    # 4. Get column boundaries by averaging x-ranges in each column
    column_bounds = []
    for col in columns:
        left = sum(r[0] for r in col) / len(col)
        right = sum(r[1] for r in col) / len(col)
        column_bounds.append((left, right))
    
    # Sort columns by x-coordinate (left to right)
    column_bounds.sort(key=lambda b: b[0])
    
    # 5. Assign each element to a column
    element_columns = {}
    for el in elements:
        # Get element x-coordinates
        el_x0 = el.get('x0', 0)
        el_x1 = el.get('x1', 0)
        el_center = (el_x0 + el_x1) / 2
        
        # Find the column this element belongs to
        for i, (left, right) in enumerate(column_bounds):
            # Extend bounds by tolerance
            extended_left = left - x_tolerance
            extended_right = right + x_tolerance
            
            # Check if center point is within extended column bounds
            if extended_left <= el_center <= extended_right:
                element_columns[el] = i
                break
        else:
            # If no column found, assign to nearest column
            distances = [(i, min(abs(el_center - left), abs(el_center - right)))
                         for i, (left, right) in enumerate(column_bounds)]
            nearest_col = min(distances, key=lambda d: d[1])[0]
            element_columns[el] = nearest_col
    
    # 6. Sort elements by column, then by vertical position
    sorted_elements = []
    for col_idx, _ in enumerate(column_bounds):
        # Get elements in this column
        col_elements = [el for el in elements if element_columns.get(el) == col_idx]
        # Sort by top coordinate
        col_elements.sort(key=lambda e: e.get('top', e.get('y0', 0)))
        # Add to final list
        sorted_elements.extend(col_elements)
    
    return sorted_elements


def _complex_reading_order(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Complex reading order that accounts for various document structures.
    
    This considers columns, text flow around images, tables, etc.
    
    Args:
        elements: List of elements to order
        
    Returns:
        List of elements in reading order
    """
    # TODO: Implement complex layout analysis
    # For now, fall back to column-aware reading order
    return _column_reading_order(elements)


def group_elements_by_line(elements: List[Dict[str, Any]], 
                          tolerance: float = 3.0) -> List[List[Dict[str, Any]]]:
    """
    Group elements into lines based on vertical position.
    
    Args:
        elements: List of elements to group
        tolerance: Maximum vertical distance for elements to be considered on the same line
        
    Returns:
        List of lists, where each sublist contains elements on the same line
    """
    if not elements:
        return []
    
    # Sort by top coordinate
    sorted_elements = sorted(elements, key=lambda e: e.get('top', e.get('y0', 0)))
    
    lines = []
    current_line = [sorted_elements[0]]
    current_top = sorted_elements[0].get('top', sorted_elements[0].get('y0', 0))
    
    for element in sorted_elements[1:]:
        element_top = element.get('top', element.get('y0', 0))
        
        # If element is close enough to current line's top, add to current line
        if abs(element_top - current_top) <= tolerance:
            current_line.append(element)
        else:
            # Otherwise, start a new line
            lines.append(current_line)
            current_line = [element]
            current_top = element_top
    
    # Add the last line
    if current_line:
        lines.append(current_line)
    
    # Sort elements within each line by x0
    for line in lines:
        line.sort(key=lambda e: e.get('x0', 0))
    
    return lines