"""
Spatial navigation example.

This example demonstrates how to navigate between elements using
spatial navigation methods: next(), prev(), and nearest().
"""
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from natural_pdf import PDF

# Get the current directory of this script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (project root)
root_dir = os.path.dirname(script_dir)
# Default PDF path
default_pdf_path = os.path.join(root_dir, "pdfs", "01-practice.pdf")
# Output directory
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Get PDF path from command line or use default
pdf_path = sys.argv[1] if len(sys.argv) > 1 else default_pdf_path
print(f"Using PDF: {pdf_path}")

# Load the PDF
pdf = PDF(pdf_path)
page = pdf.pages[0]

print("\n=== Spatial Navigation Examples ===")

# First, find a heading or title to start with
title = page.find('text[size>=12]')
if title:
    print(f"\nStarting with: '{title.text}'")
    
    # 1. Find the next element in reading order
    print("\n--- Next Element ---")
    next_element = title.next()
    if next_element:
        print(f"Next element: '{next_element.text if hasattr(next_element, 'text') else next_element.type}'")
    
    # 2. Find the next element matching a selector
    print("\n--- Next Matching Element ---")
    next_bold = title.next('text:bold', limit=20)
    if next_bold:
        print(f"Next bold text: '{next_bold.text}'")
    
    # 3. Find the previous element in reading order
    print("\n--- Previous Element ---")
    prev_element = title.prev()
    if prev_element:
        print(f"Previous element: '{prev_element.text if hasattr(prev_element, 'text') else prev_element.type}'")
    
    # 4. Find the previous element matching a selector
    print("\n--- Previous Matching Element ---")
    # Find a element further down first 
    middle_element = page.find_all('text')[len(page.find_all('text'))//2]
    if middle_element:
        print(f"Middle element: '{middle_element.text}'")
        prev_large = middle_element.prev('text[size>=12]')
        if prev_large:
            print(f"Previous large text: '{prev_large.text}'")
    
    # 5. Find the nearest element matching a selector
    print("\n--- Nearest Element ---")
    nearest_rect = title.nearest('rect')
    if nearest_rect:
        print(f"Nearest rectangle: {nearest_rect.bbox}")
    
    # 6. Find the nearest element with max distance
    print("\n--- Nearest Element with Max Distance ---")
    nearest_small = title.nearest('text[size<10]', max_distance=100)
    if nearest_small:
        print(f"Nearest small text within 100 points: '{nearest_small.text}'")
    else:
        print("No small text within 100 points")
    
    # Visualize the navigation
    print("\n--- Visualizing Navigation ---")
    page.clear_highlights()
    
    # Highlight the starting element
    title.highlight(label="Starting Element")
    
    # Find and highlight the next few elements
    current = title
    for i in range(5):
        next_elem = current.next()
        if next_elem:
            next_elem.highlight(label=f"Next {i+1}")
            current = next_elem
        else:
            break
    
    # Find and highlight the nearest elements
    title.nearest('rect').highlight(label="Nearest Rectangle")
    title.nearest('line').highlight(label="Nearest Line")
    
    # Save the visualization
    output_path = os.path.join(output_dir, "spatial_navigation.png")
    page.to_image(path=output_path, show_labels=True)
    print(f"Saved visualization to {output_path}")
    
else:
    print("Could not find a title to start with.")