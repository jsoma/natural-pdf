"""
Document layout analysis example using YOLO model.

This example demonstrates how to use the document layout analysis
functionality to detect and extract content from different regions
of a PDF document.
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
parser = argparse.ArgumentParser(description="Document layout analysis example")
parser.add_argument("pdf_path", nargs="?", default=default_pdf, help="Path to a PDF file")
parser.add_argument("--page", type=int, default=0, help="Page number to analyze (0-based)")
parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold for detections")
parser.add_argument("--model-path", type=str, default=None, help="Path to custom YOLO model")
parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on ('cpu' or 'cuda:0')")
parser.add_argument("--output", type=str, default=None, help="Output file path for highlighted image")
args = parser.parse_args()

print(f"Analyzing PDF: {args.pdf_path}")
print(f"Page: {args.page}")
print(f"Confidence threshold: {args.conf}")

# Load the PDF
pdf = PDF(args.pdf_path)
page = pdf.pages[args.page]

print(f"Running document layout analysis...")

# Run document layout analysis
# The analyze_layout method now returns self for method chaining
page.analyze_layout(
    confidence=args.conf,
    model_path=args.model_path,
    device=args.device
)

print(f"Found {len(page.detected_layout_regions)} regions with confidence >= {args.conf}")

# Group regions by type
regions_by_type = {}
for region in page.detected_layout_regions:
    region_type = region.region_type
    if region_type not in regions_by_type:
        regions_by_type[region_type] = []
    regions_by_type[region_type].append(region)
    
# Print a summary of detected regions by type
for region_type, type_regions in regions_by_type.items():
    print(f"  - {region_type}: {len(type_regions)} regions")
    
# You can highlight layout regions in two ways:
# 1. Using the dedicated highlight_layout method
# page.highlight_layout(regions, confidence=args.conf)

# 2. Using highlight_all with include_layout_regions=True
page.highlight_all(include_layout_regions=True, layout_confidence=args.conf)

# Demonstrate using selectors to find regions by type
print("\nSelecting regions by type:")
for region_type in regions_by_type.keys():
    # Convert spaces to hyphens for selector syntax
    selector_type = region_type.lower().replace(' ', '-')
    selector = f"region[type={selector_type}]"
    
    found_regions = page.find_all(selector)
    print(f"  - {selector}: {len(found_regions)} regions")
    
    # Extract text from the first region if available
    if found_regions:
        text = found_regions[0].extract_text()
        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"    First region text: {preview}")

# Finding high-confidence titles
high_conf_titles = page.find_all('region[type=title][confidence>=0.8]')
if high_conf_titles:
    print(f"\nFound {len(high_conf_titles)} high-confidence titles:")
    for i, title in enumerate(high_conf_titles):
        text = title.extract_text().strip()
        print(f"  {i+1}. {text} (conf: {title.confidence:.2f})")

# Save the highlighted image
output_path = args.output or os.path.join(root_dir, "output", "layout_detection.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
print(f"\nSaving highlighted layout to {output_path}")
page.to_image(path=output_path, show_labels=True)
print(f"Done!")

# Show an example of using a detected region for further analysis
if "table" in regions_by_type and regions_by_type["table"]:
    print("\nExample: Working with a detected table region")
    table_region = regions_by_type["table"][0]
    
    # Highlight the table region with a specific color
    table_region.highlight(label="Selected Table", color=(0, 1, 0, 0.3))
    
    # Find text elements within the table region
    table_text = table_region.find_all('text')
    print(f"  Found {len(table_text)} text elements in the table")
    
    # Extract the table text
    table_content = table_region.extract_text()
    preview = table_content[:100] + "..." if len(table_content) > 100 else table_content
    print(f"  Table content: {preview}")
    
    # Save the highlighted table
    table_output = os.path.join(os.path.dirname(output_path), "detected_table.png")
    page.to_image(path=table_output, show_labels=True)
    print(f"  Table highlighted image saved to {table_output}")