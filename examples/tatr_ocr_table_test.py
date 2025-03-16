import os
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF

# Get absolute path for the PDF
script_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(script_dir)
pdf_path = os.path.join(root_dir, "pdfs", "HARRY ROQUE_redacted.pdf")

print(f"Loading PDF: {pdf_path}")
pdf = PDF(pdf_path)

# Create output directory if it doesn't exist
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Use a specific page
page = pdf.pages[3]  # Try page 3 (this should be correct - pages are indexed from 0)

# Run document layout analysis to find tables
print("\n-- Running layout analysis to find tables --")
regions = page.analyze_layout(model='tatr')

# Find the first table
table = page.find('region[type=table][model=tatr]')
if not table:
    print("No tables found.")
    sys.exit(1)

print(f"Found table at coordinates: {table.bbox}")

# Find table structure elements
rows = page.find_all(f'region[type=table-row][model=tatr]')
columns = page.find_all(f'region[type=table-column][model=tatr]')
headers = page.find_all(f'region[type=table-column-header][model=tatr]')

# Filter to elements that are part of this table
def is_in_table(region, table):
    region_center_x = (region.x0 + region.x1) / 2
    region_center_y = (region.top + region.bottom) / 2
    return (table.x0 <= region_center_x <= table.x1 and
            table.top <= region_center_y <= table.bottom)

table_rows = [r for r in rows if is_in_table(r, table)]
table_columns = [c for c in columns if is_in_table(c, table)]
table_headers = [h for h in headers if is_in_table(h, table)]

# Print structure info
print(f"Table has {len(table_rows)} rows, {len(table_columns)} columns, and {len(table_headers)} headers")

# Create cells and check OCR on some of them
cells = table.create_cells()
print(f"Created {len(cells)} cells")

# Try OCR on a few individual cells to debug
print("\n-- Testing OCR on individual cells --")
if cells:
    sample_cells = cells[:50]  # First 50 cells
    
    for i, cell in enumerate(sample_cells):
        # print(f"Cell {i+1}:", cell.bbox)
        
        # Try OCR with very low confidence
        ocr_config = {
            "enabled": True,
            "min_confidence": 0.01,
            "detection_params": {
                "text_threshold": 0.001,  # Lower threshold to detect more text (default is 0.7)
                "mag_ratio": 4.0,  # Double the magnification during detectio
                "link_threshold": 1
            },
            "recognition_params": {
                "min_size": 6
            }
        }
        
        ocr_elements = cell.apply_ocr(**ocr_config)
        if ocr_elements:
            print(f"  OCR detected {len(ocr_elements)} text elements:")
            for elem in ocr_elements:
                print(f"    '{elem.text}' (conf: {elem.confidence:.2f})")
            
        # Get regular text
        text = cell.extract_text().strip()
        if text:
            print(f"  Regular extraction: '{text}'")

print("\nTest completed successfully!")