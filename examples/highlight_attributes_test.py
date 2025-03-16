"""
Demonstrate highlighting with attributes displayed.

This example shows how to display element attributes like confidence scores
directly on the highlighting, using the include_attrs parameter.
"""
import os
import sys
import argparse
from typing import List

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from natural_pdf import PDF

# Get the current directory of this script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (project root)
root_dir = os.path.dirname(script_dir)
# Default PDF path
default_pdf = os.path.join(root_dir, "pdfs", "01-practice.pdf")

# Set up argument parser
parser = argparse.ArgumentParser(description="Highlight attributes example")
parser.add_argument("pdf_path", nargs="?", default=default_pdf, help="Path to a PDF file")
parser.add_argument("--page", type=int, default=0, help="Page number to analyze (0-based)")
args = parser.parse_args()

print(f"Testing attribute display on: {args.pdf_path}")
print(f"Page: {args.page}")

# Load the PDF
pdf = PDF(args.pdf_path)
page = pdf.pages[args.page]

# Test 1: Standard highlight without attributes
print("\nTest 1: Standard layout highlighting (no attributes)")
page.clear_highlights()
page.analyze_layout(model="yolo", confidence=0.2)
page.analyze_layout(model="tatr", confidence=0.2, existing="append")
page.highlight_layout()
output_path = os.path.join(root_dir, "output", "highlight_no_attrs.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Test 2: Highlight with confidence and model attributes
print("\nTest 2: Layout highlighting with explicit confidence and model attributes")
page.clear_highlights()
for region in page.detected_layout_regions:
    # Use a simplified label since details will be shown on the highlight
    label = f"{region.region_type}"
    # Explicitly show confidence and model directly on the highlight
    region.highlight(
        label=label,
        include_attrs=['confidence', 'model']
    )
output_path = os.path.join(root_dir, "output", "highlight_with_attrs.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Test 3: Use highlight_all with include_layout_regions=True (no attributes by default)
print("\nTest 3: Using highlight_all with include_layout_regions=True (no attributes)")
page.clear_highlights()
page.highlight_all(
    include_layout_regions=True,
    include_types=['text'],
    layout_confidence=0.2
)
output_path = os.path.join(root_dir, "output", "highlight_all_with_attrs.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Test 4: Create a collection of regions and highlight with custom attributes
print("\nTest 4: Highlight a collection with custom attributes")
page.clear_highlights()

# Create collections by region type
from natural_pdf.elements.collections import ElementCollection

# Get high confidence regions
high_conf_regions = [r for r in page.detected_layout_regions if hasattr(r, 'confidence') and r.confidence >= 0.8]
if high_conf_regions:
    high_conf_collection = ElementCollection(high_conf_regions)
    high_conf_collection.highlight(
        label="High Confidence",
        color=(0, 1, 0, 0.3),  # Green for high confidence
        include_attrs=['region_type', 'confidence', 'model']
    )

# Get medium confidence regions
med_conf_regions = [r for r in page.detected_layout_regions if hasattr(r, 'confidence') and 0.5 <= r.confidence < 0.8]
if med_conf_regions:
    med_conf_collection = ElementCollection(med_conf_regions)
    med_conf_collection.highlight(
        label="Medium Confidence",
        color=(1, 1, 0, 0.3),  # Yellow for medium confidence
        include_attrs=['region_type', 'confidence', 'model']
    )

# Get low confidence regions
low_conf_regions = [r for r in page.detected_layout_regions if hasattr(r, 'confidence') and r.confidence < 0.5]
if low_conf_regions:
    low_conf_collection = ElementCollection(low_conf_regions)
    low_conf_collection.highlight(
        label="Low Confidence",
        color=(1, 0, 0, 0.3),  # Red for low confidence
        include_attrs=['region_type', 'confidence', 'model']
    )

output_path = os.path.join(root_dir, "output", "highlight_by_confidence.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

print("\nDone!")