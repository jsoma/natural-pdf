"""
Example demonstrating the chainable analyze_layout method.

This example shows how to use the chainable analyze_layout method
to create more concise code by chaining method calls together.
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
default_pdf = os.path.join(root_dir, "pdfs", "2019 Statistics.pdf")

# Set up argument parser
parser = argparse.ArgumentParser(description="Chainable layout analysis example")
parser.add_argument("pdf_path", nargs="?", default=default_pdf, help="Path to a PDF file")
parser.add_argument("--page", type=int, default=0, help="Page number to analyze (0-based)")
parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold for detections")
parser.add_argument("--output", type=str, default=None, help="Output file path for highlighted image")
args = parser.parse_args()

print(f"Analyzing PDF: {args.pdf_path}")
print(f"Page: {args.page}")
print(f"Confidence threshold: {args.conf}")

# Load the PDF
pdf = PDF(args.pdf_path)
page = pdf.pages[args.page]

print("Running document layout analysis with method chaining...")

# Example 1: Chain analyze_layout with highlight_all
page.analyze_layout(confidence=args.conf)\
    .highlight_all(include_layout_regions=True)

print(f"Found {len(page.detected_layout_regions)} regions with confidence >= {args.conf}")

# Example 2: Save a highlighted image with labels
output_path = args.output or os.path.join(root_dir, "output", "chainable_layout.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Chain the whole sequence: clear highlights, analyze layout, highlight all, save image
page.clear_highlights()\
    .analyze_layout(model="yolo", confidence=args.conf)\
    .highlight_all(include_layout_regions=True)\
    .to_image(path=output_path, show_labels=True)

print(f"Saved highlighted image to {output_path}")

# Example 3: Chain with specialized highlighting
if page.find_all('region[type=title]'):
    result_path = os.path.join(os.path.dirname(output_path), "titles_only.png")
    
    page.clear_highlights()\
        .analyze_layout(confidence=args.conf)\
        .find_all('region[type=title]')\
        .highlight(label="Document Titles", color=(1, 0, 0, 0.4))
    
    page.to_image(path=result_path, show_labels=True)
    print(f"Saved titles-only highlighted image to {result_path}")

print("Done!")