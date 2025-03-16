"""
Test script to verify color conversion in the highlight system.
"""
import os
import sys
from typing import List, Dict, Tuple, Optional, Union, Any, Set

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_color_conversion():
    """Test the color conversion logic directly without relying on the PDF."""
    print("Testing color conversion logic...")
    
    # Test the same logic we added to highlighting.py
    def normalize_color(color) -> Tuple[int, int, int, int]:
        """Normalize color tuple to 0-255 integer format."""
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
                
            return tuple(processed_color)
        else:
            # Default if invalid color is provided
            return (255, 255, 0, 100)  # Yellow with semi-transparency
    
    # Test various color formats
    test_cases = [
        ((255, 0, 0, 128), "Integer RGB with alpha"),
        ((255, 0, 0), "Integer RGB without alpha"),
        ((0.0, 1.0, 0.0, 0.5), "Float RGB with alpha (0-1)"),
        ((0.0, 1.0, 0.0), "Float RGB without alpha (0-1)"),
        ((0.5, 0.5, 255, 0.7), "Mixed float and integer"),
        ((0.5, 0.5, 255), "Mixed without alpha"),
        ((128.5, 64.3, 200.7, 50.9), "Float values > 1"),
        (None, "None case")
    ]
    
    for color, desc in test_cases:
        print(f"\nTesting: {desc}")
        print(f"Input:  {color}")
        result = normalize_color(color)
        print(f"Output: {result}")
    
    print("\nTest complete!")
    
if __name__ == "__main__":
    test_color_conversion()