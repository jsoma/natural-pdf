"""
Table extraction example using both TATR and pdfplumber methods.

This example demonstrates how to extract tables from PDF documents
using both the Table Transformer (TATR) structure detection and
pdfplumber's table extraction methods.

Note: This example requires additional dependencies:
    - torch
    - torchvision
    - transformers
    
These will be automatically installed when you install natural-pdf.
"""
import os
from natural_pdf import PDF
import pprint

# Get the current directory of this script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (project root)
root_dir = os.path.dirname(script_dir)
# Setup paths
pdf_path = os.path.join(root_dir, "pdfs", "01-practice.pdf")
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

print(f"Extracting tables from: {pdf_path}")

# Load the PDF
pdf = PDF(pdf_path)
page = pdf.pages[0]  # This PDF has a single page with a table

# First, let's try the traditional pdfplumber method
print("\n== TRADITIONAL TABLE EXTRACTION ==")
table_plumber = page.extract_table()  # Uses pdfplumber's table extraction
print("PDFPlumber extracted table:")
pprint.pprint(table_plumber)

# Now, let's detect and extract using TATR
print("\n== TABLE TRANSFORMER (TATR) EXTRACTION ==")

# Run table structure detection
print("Running Table Transformer detection...")
tatr_regions = page.analyze_layout(
    model="tatr",
    confidence=0.4  # Table detection confidence threshold
)

# Find the detected table
tables = page.find_all('region[type=table][model=tatr]')

if tables:
    print(f"Found {len(tables)} tables")
    
    # Get the first table
    table = tables[0]
    
    # Now extract the table using TATR structure (auto-detected)
    tatr_table_data = table.extract_table()  # Automatically uses TATR because it's a TATR region
    print("\nExtracted table data (TATR auto-detection):")
    pprint.pprint(tatr_table_data)
    
    # You can also explicitly specify which method to use
    plumber_table_data = table.extract_table(method='plumber')
    print("\nExtracted table data (explicit pdfplumber method):")
    pprint.pprint(plumber_table_data)
    
    # Compare the results
    print("\n== EXTRACTION METHOD COMPARISON ==")
    print(f"TATR rows: {len(tatr_table_data)}, cols in first row: {len(tatr_table_data[0]) if tatr_table_data else 0}")
    print(f"Plumber rows: {len(plumber_table_data)}, cols in first row: {len(plumber_table_data[0]) if plumber_table_data else 0}")
    
    # Visualize the table structure
    page.clear_highlights()
    
    # First highlight the table
    table.highlight(label="Table", color=(1, 0, 0, 0.3))
    
    # Then highlight the structure elements
    rows = page.find_all('region[type=table-row][model=tatr]')
    columns = page.find_all('region[type=table-column][model=tatr]')
    headers = page.find_all('region[type=table-column-header][model=tatr]')
    
    for row in rows:
        row.highlight(label="Row", color=(0, 1, 0, 0.3))
    for column in columns:
        column.highlight(label="Column", color=(0, 0, 1, 0.3))
    for header in headers:
        header.highlight(label="Header", color=(0, 1, 1, 0.3))
    
    # Save the highlighted table structure
    output_path = os.path.join(output_dir, "table_extraction.png")
    page.to_image(path=output_path, show_labels=True)
    print(f"\nSaved table structure visualization to {output_path}")
    
    # Demonstrate working with individual cells
    if rows and columns:
        print("\n== EXTRACTING INDIVIDUAL CELLS ==")
        # Create a cell at the intersection of first row and first column
        from natural_pdf.elements.region import Region
        
        row = rows[0]
        col = columns[0]
        
        cell_bbox = (col.x0, row.top, col.x1, row.bottom)
        cell = Region(page, cell_bbox)
        
        cell_text = cell.extract_text().strip()
        print(f"Text in first cell: '{cell_text}'")
    
    # When working with tables with headers, you might want to create a dictionary
    if headers and rows and columns:
        print("\n== CREATING A DICTIONARY FROM TABLE ==")
        header_texts = [header.extract_text().strip() for header in headers]
        
        table_dict = []
        for row in rows:
            row_dict = {}
            for i, col in enumerate(columns):
                if i < len(header_texts):
                    # Create cell region
                    cell_bbox = (col.x0, row.top, col.x1, row.bottom)
                    cell = Region(page, cell_bbox)
                    
                    # Extract text and add to dictionary
                    row_dict[header_texts[i]] = cell.extract_text().strip()
            
            if row_dict:
                table_dict.append(row_dict)
        
        print("Table as dictionary:")
        pprint.pprint(table_dict)
else:
    print("No tables detected with TATR.")