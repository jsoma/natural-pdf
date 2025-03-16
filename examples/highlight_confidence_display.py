"""
Demonstrate the enhanced confidence display feature.

This example shows how confidence scores are displayed by default
and also demonstrates customizing the attributes displayed.
"""
import os
import sys
import argparse

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
parser = argparse.ArgumentParser(description="Confidence display example")
parser.add_argument("pdf_path", nargs="?", default=default_pdf, help="Path to a PDF file")
parser.add_argument("--page", type=int, default=0, help="Page number to analyze (0-based)")
args = parser.parse_args()

print(f"Demonstrating confidence display on: {args.pdf_path}")
print(f"Page: {args.page}")

# Load the PDF
pdf = PDF(args.pdf_path)
page = pdf.pages[args.page]

# Run layout analysis
print("\nRunning layout analysis...")
page.analyze_layout(model="yolo", confidence=0.1)  # Use low confidence to show a range of values
regions = page.detected_layout_regions
print(f"Found {len(regions)} layout regions")

# Example 1: Basic highlighting without attributes
print("\nExample 1: Basic highlighting (no attributes)")
page.clear_highlights()
# Regular highlighting without showing confidence
for region in regions:
    region.highlight(label=region.region_type)

output_path = os.path.join(root_dir, "output", "basic_highlighting.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Example 2: Explicitly adding confidence
print("\nExample 2: Explicitly showing confidence")
page.clear_highlights()
for region in regions:
    region.highlight(
        label=region.region_type,
        include_attrs=['confidence']
    )
output_path = os.path.join(root_dir, "output", "explicit_confidence_display.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Example 3: Show confidence values with different colors based on confidence level
print("\nExample 3: Color-coded by confidence level")
page.clear_highlights()

# Group regions by confidence
high_conf = [r for r in regions if r.confidence >= 0.8]
med_conf = [r for r in regions if 0.5 <= r.confidence < 0.8]
low_conf = [r for r in regions if 0.2 <= r.confidence < 0.5]
very_low_conf = [r for r in regions if r.confidence < 0.2]

print(f"  High confidence (>=0.8): {len(high_conf)} regions")
print(f"  Medium confidence (0.5-0.8): {len(med_conf)} regions")
print(f"  Low confidence (0.2-0.5): {len(low_conf)} regions")
print(f"  Very low confidence (<0.2): {len(very_low_conf)} regions")

# Highlight each group with appropriate color
from natural_pdf.elements.collections import ElementCollection
if high_conf:
    ElementCollection(high_conf).highlight(
        label="High Confidence",
        color=(0, 0.8, 0, 0.3),  # Green
        include_attrs=['confidence']  # Show the confidence values
    )
if med_conf:
    ElementCollection(med_conf).highlight(
        label="Medium Confidence",
        color=(0.8, 0.8, 0, 0.3),  # Yellow
        include_attrs=['confidence']  # Show the confidence values
    )
if low_conf:
    ElementCollection(low_conf).highlight(
        label="Low Confidence",
        color=(0.8, 0.4, 0, 0.3),  # Orange
        include_attrs=['confidence']  # Show the confidence values
    )
if very_low_conf:
    ElementCollection(very_low_conf).highlight(
        label="Very Low Confidence",
        color=(0.8, 0, 0, 0.3),  # Red
        include_attrs=['confidence']  # Show the confidence values
    )

output_path = os.path.join(root_dir, "output", "confidence_color_coded.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Example 4: Show multiple attributes (confidence + type)
print("\nExample 4: Showing multiple attributes (confidence, region_type)")
page.clear_highlights()
for region in regions:
    region.highlight(
        include_attrs=['confidence', 'region_type'],
        color=(0, 0.5, 0.8, 0.3)  # Blue
    )
output_path = os.path.join(root_dir, "output", "multiple_attributes_display.png")
page.to_image(path=output_path, show_labels=False)  # No legend needed
print(f"Saved to {output_path}")

print("\nDone!")