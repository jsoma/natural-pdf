"""
Table structure detection example using Table Transformer.

This example demonstrates how to use the Table Transformer (TATR)
to detect tables and their structure in PDF documents.

Note: This example requires additional dependencies:
    - torch
    - torchvision
    - transformers
    
These will be automatically installed when you install natural-pdf.
"""
import os
from natural_pdf import PDF

# Get the current directory of this script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (project root)
root_dir = os.path.dirname(script_dir)
# Setup paths
pdf_path = os.path.join(root_dir, "pdfs", "01-practice.pdf")
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

print(f"Analyzing table structure in: {pdf_path}")

# Load the PDF - this file has a single page with a table
pdf = PDF(pdf_path)
page = pdf.pages[0]  # Get the first page

print("Running YOLO layout analysis first (excluding tables)...")
# First run YOLO detector but exclude tables
page.analyze_layout(
    model="yolo",
    confidence=0.3,
    exclude_classes=["table", "table_caption", "table_footnote"]
)

print(f"Found {len(page.detected_layout_regions)} general layout regions")

print("Now running Table Transformer detection...")
# Then run Table Transformer detection and add to existing regions
page.analyze_layout(
    model="tatr",
    confidence=0.4,  # Table detection confidence threshold
    existing="append"
)

print(f"Found {len(page.detected_layout_regions)} total regions (including table structure)")

# Example of method chaining
print("\nDemonstrating method chaining for layout analysis and highlighting:")
# Create a highlighted image with a single method chain
page.clear_highlights()\
    .analyze_layout(model="tatr", confidence=0.3)\
    .highlight_layout()\
    .to_image(path=os.path.join(output_dir, "chained_analysis.png"), show_labels=True)
print("Created highlighted image with method chaining")

# Group regions by type and model
regions_by_type = {}
for region in page.detected_layout_regions:
    region_type = region.region_type
    if region_type not in regions_by_type:
        regions_by_type[region_type] = []
    regions_by_type[region_type].append(region)

# Print a summary of all detected regions by type
print("\nAll detected regions:")
for region_type, type_regions in regions_by_type.items():
    model_name = type_regions[0].model if hasattr(type_regions[0], 'model') else "unknown"
    print(f"  - {region_type} ({model_name}): {len(type_regions)} regions")

# Highlight all regions using method chaining
output_path = os.path.join(output_dir, "all_detected_regions.png")
page.clear_highlights()\
    .highlight_layout()\
    .to_image(path=output_path, show_labels=True)
print(f"\nSaved combined layout visualization to {output_path}")

# Highlight only YOLO regions using selector and chaining
output_path = os.path.join(output_dir, "yolo_regions.png")
page.clear_highlights()\
    .find_all('region[model=yolo]')\
    .highlight(label="YOLO Regions")
page.to_image(path=output_path, show_labels=True)
print(f"Saved YOLO layout visualization to {output_path}")

# Highlight only Table Transformer regions using selector and chaining
output_path = os.path.join(output_dir, "table_structure.png")
page.clear_highlights()\
    .find_all('region[model=tatr]')\
    .highlight(label="Table Structure")
page.to_image(path=output_path, show_labels=True)
print(f"Saved Table Transformer visualization to {output_path}")

# Find tables and process their content
tables = page.find_all('region[type=table]')
if tables:
    print(f"\nFound {len(tables)} tables")
    
    # Get the first table
    table = tables[0]
    print(f"Table details:")
    print(f"  Confidence: {table.confidence:.2f}")
    print(f"  Bounding box: {table.bbox}")
    
    # Find rows, columns, and headers within this table
    # Note: Original class names with spaces are converted to hyphenated format in selectors
    rows = page.find_all('region[type=table-row]')
    columns = page.find_all('region[type=table-column]')
    headers = page.find_all('region[type=table-column-header]')
    
    print(f"  Structure: {len(rows)} rows, {len(columns)} columns, {len(headers)} headers")
    
    # Extract text from the table
    table_text = table.extract_text()
    print(f"  Content preview: {table_text[:150]}..." if len(table_text) > 150 else table_text)
    
    # Highlight the table structure with distinct colors
    page.clear_highlights()
    
    # First highlight the table
    table.highlight(label="Table", color=(1, 0, 0, 0.3))
    
    # Then highlight the structure elements
    for row in rows:
        row.highlight(label="Row", color=(0, 1, 0, 0.3))
    for column in columns:
        column.highlight(label="Column", color=(0, 0, 1, 0.3))
    for header in headers:
        header.highlight(label="Header", color=(0, 1, 1, 0.3))
        
    # Save the highlighted table structure
    output_path = os.path.join(output_dir, "table_structure_detail.png")
    page.to_image(path=output_path, show_labels=True)
    print(f"  Saved detailed table structure visualization to {output_path}")
    
    # Now find text elements within the table
    print("\nExtracting text from table cells:")
    table_text_elements = table.find_all('text')
    print(f"  Found {len(table_text_elements)} text elements in the table")
    
    # Show the first few text elements
    for i, elem in enumerate(table_text_elements[:5]):
        print(f"  Text {i+1}: '{elem.text}'")
    
    # You can also extract text just from table headers
    if headers:
        header = headers[0]
        header_text = header.extract_text()
        print(f"\nHeader text: {header_text}")
else:
    print("\nNo tables detected on this page")