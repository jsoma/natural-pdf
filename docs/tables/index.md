# Table Extraction

PDFs with tables can be challenging to work with, but Natural PDF provides tools to make table extraction easier and more accurate.

## Basic Table Extraction

The simplest way to extract a table is with the `extract_table()` method:

```python
from natural_pdf import PDF

pdf = PDF('document.pdf')
page = pdf.pages[0]

# Extract the first table found on the page
table = page.extract_table()
if table:
    for row in table:
        print(row)
```

## Working with Detected Tables

For better results, you can use layout analysis to find tables first:

```python
# Detect tables using layout analysis
page.analyze_layout()  # Uses YOLO model by default
tables = page.find_all('region[type=table]')

if tables:
    print(f"Found {len(tables)} tables")
    
    # Extract the first table
    first_table = tables[0]
    
    # Highlight the table
    first_table.highlight(color=(0, 0, 1, 0.2), label="Table")
    page.save_image("detected_table.png")
    
    # Extract the table data
    table_data = first_table.extract_table()  # Uses pdfplumber by default
    
    # Print the table data
    for row in table_data:
        print(row)
```

### Table Detection Models

Natural PDF offers two different approaches to table detection:

1. **YOLO**: The default model detects tables as single regions, identifying their boundaries on the page. This makes it easier for pdfplumber to extract the table data from the correct region, but doesn't provide information about the internal table structure.

2. **TATR** (Table Transformer): A specialized table detection model that not only identifies tables but also analyzes their internal structure (rows, columns, headers). This is particularly useful for complex tables where pdfplumber might struggle.

```python
# Simple table detection with YOLO
page.analyze_layout()  # Uses YOLO by default
tables = page.find_all('region[type=table]')

# Comprehensive table structure detection with TATR
page.analyze_layout(model="tatr") 
tables = page.find_all('region[type=table]')
rows = page.find_all('region[type=table-row]')
columns = page.find_all('region[type=table-column]')
headers = page.find_all('region[type=table-column-header]')
```

When you detect tables with TATR, the library can automatically generate cell regions at the intersections of rows and columns, providing more accurate extraction compared to pdfplumber's algorithm.

## Controlling Table Extraction Method

Natural PDF automatically selects the appropriate extraction method based on the detection model used, but you can also explicitly specify which method to use:

```python
# Basic approach - use pdfplumber's table extraction
table_plumber = page.extract_table(method='plumber')

# Advanced approach for complex tables 
page.analyze_layout(model="tatr")  # Detect detailed table structure
table_region = page.find('region[type=table]')
if table_region:
    # The TATR method uses detected rows and columns to create cells
    # This often handles complex tables better than pdfplumber
    table_tatr = table_region.extract_table(method='tatr')
```

When to use each approach:

- **pdfplumber method**: Works well for simple tables with clear borders or good spacing
- **TATR method**: Better for complex tables, especially when:
  - Borders are missing or inconsistent
  - Cells span multiple rows or columns
  - Tables have irregular structures
  - Text alignment is complex

If you run `analyze_layout(model="tatr")` and then call `extract_table()` on a table region without specifying a method, Natural PDF will automatically use the TATR method since it has the detailed structure information available.

## Table Settings

You can customize table extraction settings:

```python
# With pdfplumber method
table_settings = {
    "vertical_strategy": "text",
    "horizontal_strategy": "lines",
    "intersection_x_tolerance": 10,
    "intersection_y_tolerance": 10
}

table = page.extract_table(table_settings=table_settings)
```

## Saving Tables

You can save extracted tables to files:

```python
import csv

# Extract a table
table = page.extract_table()

# Save as CSV
if table:
    with open("table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table)
        
# Save as JSON
import json
if table:
    # Convert to a list of dictionaries (assuming first row is header)
    header = table[0]
    data = []
    for row in table[1:]:
        row_dict = {header[i]: cell for i, cell in enumerate(row)}
        data.append(row_dict)
        
    with open("table.json", "w") as f:
        json.dump(data, f, indent=2)
```

## Working with Table Structure

For precise table extraction, the TATR model provides detailed structural analysis:

```python
# Analyze table structure using TATR model
page.analyze_layout(model="tatr")

# Get table structure elements
table = page.find('region[type=table]')
rows = page.find_all('region[type=table-row]')
columns = page.find_all('region[type=table-column]')
headers = page.find_all('region[type=table-column-header]')

if table:
    # TATR automatically creates cells at row/column intersections
    cells = table.create_cells()
    print(f"Found {len(cells)} cells in the table")
    
    # Extract data from individual cells
    for cell in cells:
        print(f"Cell at {cell.x0},{cell.top} to {cell.x1},{cell.bottom}: {cell.extract_text().strip()}")
    
    # Or extract the entire table directly
    table_data = table.extract_table(method='tatr')
    for row in table_data:
        print(row)
```

### YOLO vs TATR for Table Extraction

Here's a comparison of the two approaches:

| Feature | YOLO + pdfplumber | TATR |
|---------|-------------------|------|
| Detection speed | Faster | Slower but more thorough |
| Simple tables | Good results | Good results |
| Complex tables | May struggle | Usually better |
| Cell detection | Uses line detection algorithm | Uses machine learning to identify rows/columns |
| Structure awareness | None | Detects rows, columns, headers |
| Memory usage | Lower | Higher |
| Best use case | Simple tables with clear lines | Complex tables with irregular structure |

When dealing with a document containing many tables, a good strategy is:

1. Start with YOLO detection to identify all tables
2. For any tables that aren't extracted correctly, use TATR for those specific tables
3. For very complex tables, use TATR with manual cell creation

For more manual control, you can build a structured table from detected elements:

```python
# Get header texts
header_texts = []
if headers:
    for header in headers:
        header_texts.append(header.extract_text().strip())

# Process each row
structured_table = []
for row in rows:
    row_data = {}
    for i, col in enumerate(columns):
        # Create a cell at the intersection of row and column
        cell_region = page.create_region(
            col.x0, row.top, col.x1, row.bottom
        )
        
        # Extract cell text
        cell_text = cell_region.extract_text().strip()
        
        # Add to row data
        if i < len(header_texts):
            row_data[header_texts[i]] = cell_text
        else:
            row_data[f"Column {i+1}"] = cell_text
            
    structured_table.append(row_data)
    
# Print structured table
import json
print(json.dumps(structured_table, indent=2))
```

## OCR for Tables

For scanned documents or images with tables:

```python
# Enable OCR
pdf = PDF('scanned_document.pdf', ocr=True)
page = pdf.pages[0]

# Apply OCR first
page.apply_ocr()

# Then detect and extract tables
page.analyze_layout(model="tatr")
tables = page.find_all('region[type=table]')

if tables:
    # Extract table with OCR text
    table_data = tables[0].extract_table(use_ocr=True)
    
    # Print the table
    for row in table_data:
        print(row)
```

## Visualizing Tables

You can visualize detected tables:

```python
# Detect tables
page.analyze_layout()
tables = page.find_all('region[type=table]')

# Highlight tables
tables.highlight(color=(0, 0, 1, 0.2), label="Tables")

# With table structure detection
page.analyze_layout(model="tatr")
tables = page.find_all('region[type=table]')
rows = page.find_all('region[type=table-row]')
columns = page.find_all('region[type=table-column]')
headers = page.find_all('region[type=table-column-header]')

# Color-code table elements
tables.highlight(color=(0, 0, 1, 0.2), label="Tables")
rows.highlight(color=(1, 0, 0, 0.2), label="Rows")
columns.highlight(color=(0, 1, 0, 0.2), label="Columns")
headers.highlight(color=(0.5, 0, 0.5, 0.2), label="Headers")

# Save the visualization
page.save_image("table_structure.png", labels=True)
```

## Creating Table Images

You can create images of just the tables:

```python
# Find a table
table = page.find('region[type=table]')
if table:
    # Generate an image of just the table
    table_image = table.to_image(resolution=300)
    table.save_image("table.png")
```

## A Complete Table Extraction Example

Here's a complete example that demonstrates table extraction using the TATR model:

```python
from natural_pdf import PDF
import csv

# Open a PDF with tables
pdf = PDF("document_with_tables.pdf")
page = pdf.pages[0]  # Assuming table is on the first page

# Detect tables using Table Transformer
page.analyze_layout(model="tatr", confidence=0.4)

# Find table regions
tables = page.find_all('region[type=table]')
if not tables:
    print("No tables found.")
    exit()

print(f"Found {len(tables)} tables")

# Process the first table
table = tables[0]

# Highlight the table and its structure
table.highlight(color=(0, 0, 1, 0.2), label="Table")
page.find_all('region[type=table-row]').highlight(color=(1, 0, 0, 0.2), label="Rows")
page.find_all('region[type=table-column]').highlight(color=(0, 1, 0, 0.2), label="Columns")
page.find_all('region[type=table-column-header]').highlight(color=(0.5, 0, 0.5, 0.2), label="Headers")

# Save visualization
page.save_image("table_structure_detail.png", labels=True)

# Extract the table data using TATR
table_data = table.extract_table(method='tatr')

# Print the extracted data
for row in table_data:
    print(row)

# Save to CSV
with open("table_with_tatr.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(table_data)

print(f"Table extracted and saved to table_with_tatr.csv")
```

## Coming Soon: Tables Across Pages

Support for extracting tables that span across multiple pages is under development and will be available in a future release.

## Next Steps

Now that you know how to work with tables, you might want to explore:

- [Layout Analysis](../layout-analysis/index.md) for detecting document structure including tables
- [Table Structure Detection](../layout-analysis/index.md#table-structure-detection) for detailed TATR table analysis
- [Document QA](../document-qa/index.md) for asking questions about your tables
- [Visual Debugging](../visual-debugging/index.md) for visualizing extraction results
- [PDF Extraction Challenges](../explanations/pdf-extraction-challenges.md#3-tables-lose-their-structure) for handling difficult tables