# Table Extraction

Extracting tables from PDFs can range from straightforward to complex. Natural PDF provides several tools and methods to handle different scenarios, leveraging both rule-based (`pdfplumber`) and model-based (`TATR`) approaches.

## Setup

Let's load a PDF containing tables.

```python
from natural_pdf import PDF

# Load the PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")

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

A more robust approach can be to first *detect* the table boundaries using layout analysis.

### Using YOLO (Default)

The default YOLO model finds the overall bounding box of tables.

```python
# Detect layout elements using YOLO (default)
page.analyze_layout(engine='yolo')

# Find regions detected as tables
table_regions_yolo = page.find_all('region[type=table][model=yolo]')
table_regions_yolo.show()
```

```python
table_regions_yolo[0].extract_table()
```

### Using TATR (Table Transformer)

The TATR model provides detailed table structure (rows, columns, headers).

```python
page.clear_detected_layout_regions() # Clear previous YOLO regions for clarity
page.analyze_layout(engine='tatr')
```

```python
# Find the main table region(s) detected by TATR
tatr_table = page.find('region[type=table][model=tatr]')
tatr_table.show()
```

```python
# Find rows, columns, headers detected by TATR
rows = page.find_all('region[type=table-row][model=tatr]')
cols = page.find_all('region[type=table-column][model=tatr]')
hdrs = page.find_all('region[type=table-column-header][model=tatr]')
f"TATR found: {len(rows)} rows, {len(cols)} columns, {len(hdrs)} headers"
```

## Controlling Extraction Method (`plumber` vs `tatr`)

When you call `extract_table()` on a region:
- If the region was detected by **YOLO** (or not detected at all), it uses the `plumber` method.
- If the region was detected by **TATR**, it defaults to the `tatr` method, which uses the detected row/column structure.

You can override this using the `method` argument.

```python
tatr_table = page.find('region[type=table][model=tatr]')
tatr_table.extract_table(method='tatr')
```

```python
# Force using pdfplumber even on a TATR-detected region
# (Might be useful for comparison or if TATR structure is flawed)
tatr_table = page.find('region[type=table][model=tatr]')
tatr_table.extract_table(method='pdfplumber')
```

### When to Use Which Method?

- **`pdfplumber`**: Good for simple tables with clear grid lines. Faster.
- **`tatr`**: Better for tables without clear lines, complex cell merging, or irregular layouts. Leverages the model's understanding of rows and columns.

## Customizing `pdfplumber` Settings

If using the `pdfplumber` method (explicitly or implicitly), you can pass `pdfplumber` settings via `table_settings`.

```python
# Example: Use text alignment for vertical lines, explicit lines for horizontal
# See pdfplumber documentation for all settings
table_settings = {
    "vertical_strategy": "text",
    "horizontal_strategy": "lines",
    "intersection_x_tolerance": 5, # Increase tolerance for intersections
}

results = page.extract_table(
    table_settings=table_settings
)
```

## Saving Extracted Tables

You can easily save the extracted data (list of lists) to common formats.

```python
import pandas as pd

pd.DataFrame(page.extract_table())
```

## Working Directly with TATR Cells

The TATR engine implicitly creates cell regions at the intersection of detected rows and columns. You can access these for fine-grained control.

```python
# This doesn't work! I forget why, I should troubleshoot later.
# tatr_table.cells
```

## Next Steps

- [Layout Analysis](../layout-analysis/index.ipynb): Understand how table detection fits into overall document structure analysis.
- [Working with Regions](../regions/index.ipynb): Manually define table areas if detection fails.