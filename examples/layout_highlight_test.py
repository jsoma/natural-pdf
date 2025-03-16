"""
Test the improved highlight_all behavior with layout regions.

This example demonstrates how the updated highlight_all method properly 
highlights layout regions by model and type.
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
parser = argparse.ArgumentParser(description="Layout highlighting test")
parser.add_argument("pdf_path", nargs="?", default=default_pdf, help="Path to a PDF file")
parser.add_argument("--page", type=int, default=0, help="Page number to analyze (0-based)")
args = parser.parse_args()

print(f"Testing layout highlighting on: {args.pdf_path}")
print(f"Page: {args.page}")

# Load the PDF
pdf = PDF(args.pdf_path)
page = pdf.pages[args.page]

# First, let's show the regular highlight_all without layout regions
print("Creating image with standard highlight_all (no layout regions)...")
page.clear_highlights()
page.highlight_all()
output_path = os.path.join(root_dir, "output", "standard_highlight_all.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Now run layout analysis with YOLO
print("\nRunning YOLO layout analysis...")
page.analyze_layout(model="yolo", confidence=0.2)
print(f"Found {len(page.detected_layout_regions)} YOLO layout regions")

# Create an image with highlight_all including layout regions
print("Creating image with highlight_all including YOLO layout regions...")
page.clear_highlights()
page.highlight_all(include_layout_regions=True)
output_path = os.path.join(root_dir, "output", "highlight_all_with_yolo.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Now run table structure analysis with TATR and append to existing regions
print("\nRunning TATR table structure analysis...")
page.analyze_layout(model="tatr", confidence=0.3, existing="append")
print(f"Found {len(page.detected_layout_regions)} total layout regions (YOLO + TATR)")

# Create an image with highlight_all including all layout regions
print("Creating image with highlight_all including all layout regions...")
page.clear_highlights()
page.highlight_all(include_layout_regions=True)
output_path = os.path.join(root_dir, "output", "highlight_all_with_all_layouts.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Compare with the original highlight_layout method
print("\nCreating image with highlight_layout method for comparison...")
page.clear_highlights()
page.highlight_layout()
output_path = os.path.join(root_dir, "output", "highlight_layout_method.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

print("\nDone!")