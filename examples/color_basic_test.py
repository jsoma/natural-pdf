"""
Simple test for color conversion.
"""

# Test color conversion
def normalize_color(color):
    """Test function that normalizes colors from various formats to RGB(A) integers."""
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

# Test cases
print("Testing color conversion:")
print("-----------------------")

test_cases = [
    ((255, 0, 0, 128), "Integer RGB with alpha"),
    ((255, 0, 0), "Integer RGB without alpha"),
    ((0.0, 1.0, 0.0, 0.5), "Float RGB with alpha (0-1)"),
    ((0.0, 1.0, 0.0), "Float RGB without alpha (0-1)"),
    ((0.5, 0.5, 255, 0.7), "Mixed float and integer"),
    ((0.5, 0.5, 255), "Mixed without alpha"),
    ((128.5, 64.3, 200.7, 50.9), "Float values > 1"),
]

for color, desc in test_cases:
    result = normalize_color(color)
    print(f"{desc}: {color} -> {result}")

print("\nAll tests completed!")