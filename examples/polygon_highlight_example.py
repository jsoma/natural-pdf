"""
Example showing the polygon highlighting capabilities for handling non-rectangular regions.

This example demonstrates how polygon-based OCR results are handled and visualized,
which is especially useful for skewed or rotated text in scanned documents.
"""
import os
import sys
from natural_pdf import PDF
from natural_pdf.elements.region import Region
from PIL import Image

# Get the current directory of this script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (project root)
root_dir = os.path.dirname(script_dir)
# Default PDF path (using a document that needs OCR)
default_pdf = os.path.join(root_dir, "pdfs", "needs-ocr.pdf")

# Check for command line arguments
pdf_path = sys.argv[1] if len(sys.argv) > 1 else default_pdf
page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0

print(f"Loading PDF: {pdf_path}")
print(f"Using page: {page_num}")

# Load the PDF with OCR enabled
pdf = PDF(pdf_path, ocr=True)
page = pdf.pages[page_num]

# Create a simulated polygon region to show polygon highlighting
print("Creating polygon region...")
polygon_points = [
    (100, 100),
    (300, 150),
    (250, 250),
    (120, 200)
]

# Create a region with the polygon points
region = Region(page, (100, 100, 300, 250), polygon=polygon_points)
region.highlight(color=(1, 0, 0, 0.5), label="Polygon Region")

# Also extract and highlight text using OCR, which will use polygon detection
print("Running OCR on the page...")
ocr_elements = page.apply_ocr()
print(f"Found {len(ocr_elements)} OCR text elements")

# Highlight OCR elements with different colors based on confidence
print("Highlighting OCR elements...")
for elem in ocr_elements:
    if elem.confidence > 0.8:
        color = (0, 0.8, 0, 0.3)  # Green for high confidence
    elif elem.confidence > 0.5:
        color = (1, 0.8, 0, 0.3)  # Yellow for medium confidence
    else:
        color = (0.8, 0, 0, 0.3)  # Red for low confidence
        
    elem.highlight(color=color)

# Save the result
output_path = os.path.join(root_dir, "output", "polygon_highlight_example.png")
print(f"Saving highlighted image to {output_path}")
page.to_image(path=output_path, show_labels=True)

# Print some information about the elements
print("\nPolygon support details:")

# Check if any OCR elements have polygon data
polygon_elements = [elem for elem in ocr_elements if hasattr(elem, 'has_polygon') and elem.has_polygon]
print(f"- Found {len(polygon_elements)} elements with polygon data")

# Display details of the first few polygon elements
if polygon_elements:
    for i, elem in enumerate(polygon_elements[:3]):
        print(f"\nElement {i+1}:")
        print(f"- Text: '{elem.text}'")
        print(f"- Confidence: {elem.confidence:.2f}")
        print(f"- Bounding box: {elem.bbox}")
        print(f"- Polygon points: {elem.polygon[:2]}... ({len(elem.polygon)} points)")
    
    if len(polygon_elements) > 3:
        print(f"... and {len(polygon_elements) - 3} more")