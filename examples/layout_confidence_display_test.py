"""
Test displaying confidence scores in layout highlighting.

This example demonstrates how confidence scores are displayed next to 
each layout region in both highlight_layout and highlight_all methods.
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
parser = argparse.ArgumentParser(description="Layout confidence display test")
parser.add_argument("pdf_path", nargs="?", default=default_pdf, help="Path to a PDF file")
parser.add_argument("--page", type=int, default=0, help="Page number to analyze (0-based)")
args = parser.parse_args()

print(f"Testing confidence display on: {args.pdf_path}")
print(f"Page: {args.page}")

# Load the PDF
pdf = PDF(args.pdf_path)
page = pdf.pages[args.page]

# Run layout analysis with different models
print("Running layout analysis...")
page.analyze_layout(model="yolo", confidence=0.1)  # Use low confidence to get more regions
page.analyze_layout(model="tatr", confidence=0.1, existing="append")  # Low confidence for TATR too
print(f"Found {len(page.detected_layout_regions)} total layout regions")

# Test 1: highlight_layout with default format
print("\nTest 1: Using highlight_layout with default format")
page.clear_highlights()
page.highlight_layout()
output_path = os.path.join(root_dir, "output", "conf_display_highlight_layout.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Test 2: highlight_all with include_layout_regions=True
print("\nTest 2: Using highlight_all with include_layout_regions=True")
page.clear_highlights()
page.highlight_all(include_layout_regions=True, layout_confidence=0.1)
output_path = os.path.join(root_dir, "output", "conf_display_highlight_all.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Test 3: highlight_all with only layout regions
print("\nTest 3: Using highlight_all with only layout regions")
page.clear_highlights()
page.highlight_all(include_layout_regions=True, include_types=[], layout_confidence=0.1)
output_path = os.path.join(root_dir, "output", "conf_display_layout_only.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

print("\nDone!")