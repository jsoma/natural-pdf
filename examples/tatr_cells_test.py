import os
import sys
from pathlib import Path

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
page = pdf.pages[6]

# Test 1: Analyze layout with create_cells=True
print("\n-- Testing layout detection with cell creation --")
regions = page.analyze_layout(model='tatr', create_cells=True)

# Count tables and cells
tables = page.find_all('region[type=table][model=tatr]')
cells = page.find_all('region[type=table-cell][model=tatr]')

print(f"Found {len(tables)} tables")
print(f"Found {len(cells)} table cells")

# Test 2: Create cells explicitly from a table
if tables:
    print("\n-- Testing explicit cell creation from a table --")
    table = tables[0]
    # Create cells if not already created
    explicit_cells = table.create_cells()
    print(f"Created {len(explicit_cells)} cells explicitly")
    
    # Highlight the first few cells
    for i, cell in enumerate(explicit_cells[:5]):
        cell.highlight(label=f"Cell {i+1}", color=(255, 0, 0, 50))
    
    # Highlight the table
    table.highlight(label="Table", color=(0, 0, 255, 50))

# Save the highlighted image
output_path = os.path.join(output_dir, "tatr_cells_test.png")
print(f"\nSaving highlighted image to: {output_path}")
page.to_image(path=output_path, show_labels=True)

print("\nTest completed successfully!")