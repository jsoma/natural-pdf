"""
Document layout analysis example using PaddlePaddle's PP-Structure model.

This example demonstrates how to use PaddlePaddle for document layout analysis
to detect and extract content from different regions of a PDF document.

Features:
- Standard layout detection using PaddlePaddle's PP-Structure
- Enhanced text detection by combining PP-Structure with direct OCR
- Visualization of different region types and sources
- Comparison mode to evaluate performance with and without text detection
- Support for polygon-based text regions from OCR
"""
import os
import sys
import logging
from pathlib import Path
import argparse

# Import the library with its logging utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from natural_pdf import configure_logging, PDF

# Get the current directory of this script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (project root)
root_dir = os.path.dirname(script_dir)
# Default PDF path
default_pdf = os.path.join(root_dir, "pdfs", "HARRY ROQUE_redacted.pdf")

# Set up argument parser
parser = argparse.ArgumentParser(description="PaddlePaddle layout analysis example")
parser.add_argument("pdf_path", nargs="?", default=default_pdf, help="Path to a PDF file")
parser.add_argument("--page", type=int, default=0, help="Page number to analyze (0-based)")
parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold for detections")
parser.add_argument("--lang", type=str, default="en", help="Language code (en, ch, etc.)")
parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on ('cpu' or 'gpu')")
parser.add_argument("--output", type=str, default=None, help="Output file path for highlighted image")
parser.add_argument("--disable-table", action="store_true", help="Disable table detection")
parser.add_argument("--text-detection", action="store_true", help="Enable direct text detection")
parser.add_argument("--compare", action="store_true", help="Compare with and without text detection")
parser.add_argument("--verbose", action="store_true", help="Show detailed debug output")
parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                    default="INFO", help="Set logging level")
args = parser.parse_args()

# Configure logging based on command-line arguments
log_level = getattr(logging, args.log_level)
configure_logging(level=log_level)

# Further adjust logging for verbose mode
if args.verbose:
    configure_logging(level=logging.DEBUG)

print(f"Analyzing PDF: {args.pdf_path}")
print(f"Page: {args.page}")
print(f"Confidence threshold: {args.conf}")

# Load the PDF
pdf = PDF(args.pdf_path)
page = pdf.pages[args.page]

print(f"Running PaddlePaddle layout analysis...")

# Enable debugging output
print("PDF page dimensions:", page.width, "x", page.height)

# Check if we should run comparison
if args.compare:
    print("\n=== Comparing Layout Detection With and Without Text Detection ===")
    
    # First run without text detection
    print("\nRunning WITHOUT text detection...")
    import time
    start = time.time()
    regions_without_text = page.analyze_layout(
        model="paddle",
        confidence=args.conf,
        device=args.device,
        model_params={
            "lang": args.lang,
            "show_log": args.verbose,
            "detect_text": False,
            "verbose": args.verbose
        }
    )
    time_without = time.time() - start
    
    # Highlight without text detection
    page.highlight_layout()
    
    # Save the highlighted image
    output_without = os.path.join(
        os.path.dirname(args.output or os.path.join(root_dir, "output", "paddle_layout_detection.png")), 
        "paddle_layout_without_text.png"
    )
    page.to_image(path=output_without, show_labels=True)
    print(f"Found {len(regions_without_text)} regions WITHOUT text detection in {time_without:.2f} seconds")
    print(f"Saved image to {output_without}")
    
    # Clear highlights
    page.clear_highlights()
    
    # Then run with text detection
    print("\nRunning WITH text detection...")
    start = time.time()
    regions_with_text = page.analyze_layout(
        model="paddle",
        confidence=args.conf,
        device=args.device,
        model_params={
            "lang": args.lang,
            "show_log": args.verbose,
            "detect_text": True,
            "verbose": args.verbose
        }
    )
    time_with = time.time() - start
    
    # Highlight with text detection
    page.highlight_layout()
    
    # Save the highlighted image
    output_with = os.path.join(
        os.path.dirname(args.output or os.path.join(root_dir, "output", "paddle_layout_detection.png")), 
        "paddle_layout_with_text.png"
    )
    page.to_image(path=output_with, show_labels=True)
    print(f"Found {len(regions_with_text)} regions WITH text detection in {time_with:.2f} seconds")
    print(f"Saved image to {output_with}")
    
    # Comparison
    print("\nComparison results:")
    print(f"  - WITHOUT text detection: {len(regions_without_text)} regions in {time_without:.2f} seconds")
    print(f"  - WITH text detection: {len(regions_with_text)} regions in {time_with:.2f} seconds")
    print(f"  - Additional regions: {len(regions_with_text) - len(regions_without_text)}")
    print(f"  - Speed difference: {time_with / time_without:.2f}x longer with text detection")
    
    # Continue with the regions from the requested mode
    regions = regions_with_text if args.text_detection else regions_without_text
    
else:
    # Run regular layout analysis
    regions = page.analyze_layout(
        model="paddle",
        confidence=args.conf,
        device=args.device,
        model_params={
            "lang": args.lang,
            "show_log": args.verbose,
            "detect_text": args.text_detection,
            "verbose": args.verbose
        }
    )

print(f"Found {len(regions)} regions with confidence >= {args.conf}")

# Group regions by type and source
regions_by_type = {}
sources = {"layout": 0, "ocr": 0, "unknown": 0}

for region in regions:
    region_type = region.region_type
    if region_type not in regions_by_type:
        regions_by_type[region_type] = []
    regions_by_type[region_type].append(region)
    
    # Count sources
    source = getattr(region, "source", "unknown")
    sources[source] = sources.get(source, 0) + 1
    
# Print a summary of detected regions by type
for region_type, type_regions in regions_by_type.items():
    print(f"  - {region_type}: {len(type_regions)} regions")

# Print source information
print("\nRegion sources:")
for source, count in sources.items():
    print(f"  - {source}: {count} regions")

# If the user enabled text detection, show source-specific highlighting
if args.text_detection:
    print("\nHighlighting regions by source...")
    
    # Clear any existing highlights
    page.clear_highlights()
    
    # Get text regions separately using normalized_type
    text_regions = page.find_all('region[normalized_type=plain-text][model=paddle]')
    figure_regions = page.find_all('region[normalized_type=figure][model=paddle]')
    
    # Highlight figure regions in blue
    for region in figure_regions:
        region.highlight(color=(0, 0, 1, 0.3), label=f"Figure: {region.region_type}")
    
    # Highlight text regions in green
    for region in text_regions:
        region.highlight(color=(0, 1, 0, 0.3), label=f"Text: {region.region_type}")
    
    # Save the source-highlighted image
    sources_output = os.path.join(
        os.path.dirname(args.output or os.path.join(root_dir, "output", "paddle_layout_detection.png")), 
        "paddle_layout_sources.png"
    )
    page.to_image(path=sources_output, show_labels=True)
    print(f"Saved source-highlighted layout to {sources_output}")
    
    # Show polygon visualizations if any OCR regions have polygons
    regions_with_polygons = [r for r in regions if hasattr(r, "polygon")]
    if regions_with_polygons:
        print(f"\nVisualizing {len(regions_with_polygons)} regions with polygon points...")
        page.clear_highlights()
        
        # Highlight regions with polygons in red
        for region in regions_with_polygons:
            region.highlight(color=(1, 0, 0, 0.3), label="Polygon Region")
            
        # Save the polygon-highlighted image
        polygon_output = os.path.join(
            os.path.dirname(args.output or os.path.join(root_dir, "output", "paddle_layout_detection.png")), 
            "paddle_layout_polygons.png"
        )
        page.to_image(path=polygon_output, show_labels=True)
        print(f"Saved polygon visualization to {polygon_output}")
    
    # Clear highlights for standard view
    page.clear_highlights()
    
# Highlight all detected regions normally
page.highlight_all(include_layout_regions=True, layout_confidence=args.conf)

# Demonstrate using selectors to find regions by type and model
print("\nSelecting regions by type and model:")
for region_type in regions_by_type.keys():
    # Convert spaces to hyphens for selector syntax
    selector_type = region_type.lower().replace(' ', '-')
    
    # Use model-specific selector
    # Use either type or normalized_type in selector
    if region_type.lower() == 'text':
        selector = f"region[normalized_type=plain-text][model=paddle]"
    else:
        selector = f"region[normalized_type={selector_type}][model=paddle]"
    
    found_regions = page.find_all(selector)
    print(f"  - {selector}: {len(found_regions)} regions")
    
    # Try different selectors to debug the issue
    model_regions = page.find_all(f"region[type={selector_type}]")
    paddle_regions = page.find_all(f"region[model=paddle]")
    layout_regions = page.find_all(f"region[source=layout]")
    ocr_regions = page.find_all(f"region[source=ocr]")
    detected_regions = page.find_all(f"region[source=detected]")
    
    print(f"    - With type only: {len(model_regions)} regions")
    print(f"    - With model=paddle: {len(paddle_regions)} regions")
    print(f"    - With source=layout: {len(layout_regions)} regions")
    print(f"    - With source=ocr: {len(ocr_regions)} regions")
    print(f"    - With source=detected: {len(detected_regions)} regions")
    
    # Debug a sample region
    if model_regions:
        region = model_regions[0]
        print(f"    - Sample region attributes: type={region.region_type}, normalized_type={getattr(region, 'normalized_type', 'N/A')}, " +
              f"source={getattr(region, 'source', 'N/A')}, model={getattr(region, 'model', 'N/A')}")
    
    # For text regions, find a sample to debug
    if region_type.lower() == 'text' and detected_regions:
        text_sample = None
        for i, r in enumerate(detected_regions[:10]):
            print(f"    - Detected region {i}: type={r.region_type}, normalized_type={getattr(r, 'normalized_type', 'N/A')}")
    
    # Extract text from the first region if available
    if found_regions:
        text = found_regions[0].extract_text()
        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"    First region text: {preview}")

# Save the highlighted image
output_path = args.output or os.path.join(root_dir, "output", "paddle_layout_detection.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
print(f"\nSaving highlighted layout to {output_path}")
page.to_image(path=output_path, show_labels=True)
print(f"Done!")

# Show an example of working with a table region
if "table" in regions_by_type and regions_by_type["table"]:
    print("\nExample: Working with a detected table region")
    table_region = regions_by_type["table"][0]
    
    # Extract table data
    try:
        # Try using the extract_table method on the region
        table_data = table_region.extract_table()
        print(f"  Extracted {len(table_data)} rows from table")
        
        # Show some table data
        for i, row in enumerate(table_data[:2]):  # Show first 2 rows
            print(f"    Row {i}: {row}")
            
        # Check for cells
        cells = page.find_all('region[type=table_cell][model=paddle]')
        if cells:
            print(f"\n  Found {len(cells)} table cells")
            cell = cells[0]
            print(f"    First cell text: {cell.extract_text()}")
            print(f"    Row index: {getattr(cell, 'row_idx', 'N/A')}, Column index: {getattr(cell, 'col_idx', 'N/A')}")
    except Exception as e:
        print(f"  Error extracting table data: {e}")
    
    # Save the highlighted table
    table_output = os.path.join(os.path.dirname(output_path), "paddle_detected_table.png")
    table_region.highlight(color=(0, 1, 0, 0.3), label="PaddlePaddle Table")
    page.to_image(path=table_output, show_labels=True)
    print(f"  Table highlighted image saved to {table_output}")