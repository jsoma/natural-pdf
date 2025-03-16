"""
Test the layout_confidence=True behavior in highlight_all method.

This example demonstrates that when layout_confidence=True is passed,
all layout regions are included regardless of their confidence score.
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
parser = argparse.ArgumentParser(description="Layout confidence test")
parser.add_argument("pdf_path", nargs="?", default=default_pdf, help="Path to a PDF file")
parser.add_argument("--page", type=int, default=0, help="Page number to analyze (0-based)")
args = parser.parse_args()

print(f"Testing layout_confidence=True on: {args.pdf_path}")
print(f"Page: {args.page}")

# Load the PDF
pdf = PDF(args.pdf_path)
page = pdf.pages[args.page]

# Run layout analysis with YOLO and TATR
print("Running layout analysis...")
page.analyze_layout(model="yolo", confidence=0.1)  # Use low confidence to get more regions
page.analyze_layout(model="tatr", confidence=0.1, existing="append")  # Low confidence for TATR too
print(f"Found {len(page.detected_layout_regions)} total layout regions")

# Count regions by confidence thresholds
high_conf = [r for r in page.detected_layout_regions if r.confidence >= 0.5]
med_conf = [r for r in page.detected_layout_regions if 0.2 <= r.confidence < 0.5]
low_conf = [r for r in page.detected_layout_regions if r.confidence < 0.2]

print(f"High confidence (>=0.5): {len(high_conf)} regions")
print(f"Medium confidence (0.2-0.5): {len(med_conf)} regions")
print(f"Low confidence (<0.2): {len(low_conf)} regions")

# Test 1: highlight_all with default layout_confidence=0.2
print("\nTest 1: Using default layout_confidence=0.2")
page.clear_highlights()
page.highlight_all(include_layout_regions=True)
output_path = os.path.join(root_dir, "output", "layout_conf_default.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Test 2: highlight_all with layout_confidence=0.5 (high threshold)
print("\nTest 2: Using layout_confidence=0.5 (high threshold)")
page.clear_highlights()
page.highlight_all(include_layout_regions=True, layout_confidence=0.5)
output_path = os.path.join(root_dir, "output", "layout_conf_high.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Test 3: highlight_all with layout_confidence=True (include all)
print("\nTest 3: Using layout_confidence=True (include all)")
page.clear_highlights()
page.highlight_all(include_layout_regions=True, layout_confidence=True)
output_path = os.path.join(root_dir, "output", "layout_conf_all.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

# Test 4: highlight_all with layout_confidence=0.0 (include all)
print("\nTest 4: Using layout_confidence=0.0 (include all)")
page.clear_highlights()
page.highlight_all(include_layout_regions=True, layout_confidence=0.0)
output_path = os.path.join(root_dir, "output", "layout_conf_zero.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved to {output_path}")

print("\nDone!")