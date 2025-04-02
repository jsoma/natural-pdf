# Table Extraction

Extracting tables from PDFs can range from straightforward to complex. Natural PDF provides several tools and methods to handle different scenarios, leveraging both rule-based (`pdfplumber`) and model-based (`TATR`) approaches.

## Setup

Let's load a PDF containing tables. `02-practice.pdf` has a few examples.

```python
from natural_pdf import PDF
from pathlib import Path

# Ensure table-specific models are installed if needed:
# !pip install natural-pdf[layout_yolo]  # Default layout model
# !pip install natural-pdf[layout_tatr]   # Advanced table structure model

# Path to your sample PDF
pdf_path = Path("../tutorials/pdfs/02-practice.pdf") # Contains tables

# Load the PDF
pdf = PDF(pdf_path)

# Select the first page
page = pdf.pages[0]

# Display the page
page.show()
```

## Basic Table Extraction (No Detection)

If you know a table exists, you can try `extract_table()` directly on the page or a region. This uses `pdfplumber` behind the scenes.

```python
# Extract the first table found on the page using pdfplumber
# This works best for simple tables with clear lines
table_data = page.extract_table() # Returns a list of lists
table_data
```

*This might fail or give poor results if there are multiple tables or the table structure is complex.*

## Layout Analysis for Table Detection

A more robust approach is to first *detect* the table boundaries using layout analysis.

### Using YOLO (Default)

The default YOLO model finds the overall bounding box of tables.

```python
# Detect layout elements using YOLO (default)
page.analyze_layout(engine='yolo')

# Find regions detected as tables
table_regions_yolo = page.find_all('region[type=table][model=yolo]')
len(table_regions_yolo)
```

```python
# Highlight the detected table regions (YOLO)
if table_regions_yolo:
    page.highlight_all(table_regions_yolo, color="blue", label="YOLO Table Region").show()
```

```python
# Extract data from the first detected YOLO table region
# This still uses pdfplumber, but confined to the detected region
if table_regions_yolo:
    table_data_yolo = table_regions_yolo[0].extract_table()
    table_data_yolo
```

### Using TATR (Table Transformer)

The TATR model provides detailed table structure (rows, columns, headers).

```python
# Detect layout using TATR (requires natural-pdf[layout_tatr])
page.clear_regions() # Clear previous YOLO regions for clarity
try:
    page.analyze_layout(engine='tatr')
    "TATR analysis complete."
except Exception as e:
    f"TATR analysis failed or not installed: {e}"
```

```python
# Find the main table region(s) detected by TATR
table_regions_tatr = page.find_all('region[type=table][model=tatr]')
len(table_regions_tatr)
```

```python
# Find rows, columns, headers detected by TATR
rows = page.find_all('region[type=table-row][model=tatr]')
cols = page.find_all('region[type=table-column][model=tatr]')
hdrs = page.find_all('region[type=table-column-header][model=tatr]')
f"TATR found: {len(rows)} rows, {len(cols)} columns, {len(hdrs)} headers"
```

```python
# Visualize the detailed TATR structure
img = page.copy()
if table_regions_tatr: img.highlight_all(table_regions_tatr, color=(0,0,1,0.1), label="Table")
if rows: img.highlight_all(rows, color=(1,0,0,0.1), label="Row")
if cols: img.highlight_all(cols, color=(0,1,0,0.1), label="Column")
if hdrs: img.highlight_all(hdrs, color=(1,0,1,0.2), label="Header")
img.show()
```

## Controlling Extraction Method (`plumber` vs `tatr`)

When you call `extract_table()` on a region:
- If the region was detected by **YOLO** (or not detected at all), it uses the `plumber` method.
- If the region was detected by **TATR**, it defaults to the `tatr` method, which uses the detected row/column structure.

You can override this using the `method` argument.

```python
# Extract using TATR structure (default for TATR regions)
if table_regions_tatr:
    table_data_tatr_method = table_regions_tatr[0].extract_table(method='tatr')
    table_data_tatr_method
```

```python
# Force using pdfplumber even on a TATR-detected region
# (Might be useful for comparison or if TATR structure is flawed)
if table_regions_tatr:
    table_data_plumber_on_tatr = table_regions_tatr[0].extract_table(method='plumber')
    table_data_plumber_on_tatr
```

### When to Use Which Method?

- **`plumber`**: Good for simple tables with clear grid lines. Faster.
- **`tatr`**: Better for tables without clear lines, complex cell merging, or irregular layouts. Leverages the model's understanding of rows and columns.

## Customizing `pdfplumber` Settings

If using the `plumber` method (explicitly or implicitly), you can pass `pdfplumber` settings via `table_settings`.

```python
# Example: Use text alignment for vertical lines, explicit lines for horizontal
# See pdfplumber documentation for all settings
plumber_settings = {
    "vertical_strategy": "text",
    "horizontal_strategy": "lines",
    "intersection_x_tolerance": 5, # Increase tolerance for intersections
}

if table_regions_yolo: # Using YOLO region as an example
    custom_plumber_table = table_regions_yolo[0].extract_table(
        method='plumber',
        table_settings=plumber_settings
    )
    custom_plumber_table
```

## Saving Extracted Tables

You can easily save the extracted data (list of lists) to common formats.

```python
import csv
import json

# Use the table extracted via TATR method
if 'table_data_tatr_method' in locals() and table_data_tatr_method:
    # Save as CSV
    with open("output_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(table_data_tatr_method)
    print("Saved table to output_table.csv")

    # Save as JSON (list of lists)
    with open("output_table_lol.json", "w", encoding="utf-8") as f:
        json.dump(table_data_tatr_method, f, indent=2)
    print("Saved table to output_table_lol.json")

    # Save as JSON (list of dicts, assuming first row is header)
    if len(table_data_tatr_method) > 1:
        header = table_data_tatr_method[0]
        data_dicts = [
            {header[i]: cell for i, cell in enumerate(row)}
            for row in table_data_tatr_method[1:]
        ]
        with open("output_table_lod.json", "w", encoding="utf-8") as f:
            json.dump(data_dicts, f, indent=2)
        print("Saved table to output_table_lod.json")
else:
    print("TATR table data not available for saving.")
```

## Working Directly with TATR Cells

The TATR engine implicitly creates cell regions at the intersection of detected rows and columns. You can access these for fine-grained control.

```python
# Get the TATR table region again
if table_regions_tatr:
    tatr_table = table_regions_tatr[0]

    # Access the generated cells (may not exist if structure detection failed)
    # .cells is a property that runs create_cells() if needed
    cells = tatr_table.cells
    if cells:
        print(f"TATR generated {len(cells)} cell regions.")
        # Highlight the first few cells
        img = page.copy()
        img.highlight_all(cells[:5], color="random", alpha=0.5).show()

        # Extract text from the first cell
        # print(f"First cell text: '{cells[0].text}'")
    else:
        print("TATR cell generation failed or yielded no cells.")

```

## OCR for Tables in Scanned Documents

If your PDF is scanned, combine OCR with table extraction.

```python
# Example using a scanned PDF (if available)
# pdf_scanned_path = Path("../path/to/scanned_table.pdf")
# if pdf_scanned_path.exists():
#     pdf_scan = PDF(pdf_scanned_path, ocr=True) # Enable OCR
#     scan_page = pdf_scan.pages[0]
#     scan_page.apply_ocr()
#     scan_page.analyze_layout(engine="tatr") # Use TATR for structure
#     scan_tables = scan_page.find_all('region[type=table][model=tatr]')
#     if scan_tables:
#         # Extract using TATR method, which will use OCR text within cells
#         ocr_table_data = scan_tables[0].extract_table(method='tatr')
#         print("Extracted OCR Table Data:")
#         print(ocr_table_data)
# else:
#     "Scanned PDF table example skipped."

"Example for scanned table extraction commented out."
```

## Next Steps

- [Enhanced Table Processing](../enhanced-table-processing/index.md): Techniques for cleaning and structuring complex tables.
- [Layout Analysis](../layout-analysis/index.md): Understand how table detection fits into overall document structure analysis.
- [Working with Regions](../regions/index.md): Manually define table areas if detection fails.