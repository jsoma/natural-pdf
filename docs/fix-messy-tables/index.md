# Fix Messy Table Extraction

Your PDF has tables, but when you try to extract them, you get garbled data, missing rows, or columns that don't line up. Here's how to fix the most common table extraction problems.

## The Problem

PDF tables come in many flavors, and each one breaks in its own special way:

- **No visible borders**: Text that looks like a table but has no lines
- **Merged cells**: Headers that span multiple columns 
- **Inconsistent spacing**: Columns that don't line up perfectly
- **Multiple tables**: Several tables crammed onto one page
- **Rotated tables**: Tables turned sideways or upside down
- **Image tables**: Tables that are actually pictures

Simple extraction methods often fail on these cases.

## Quick Fix: Try Layout Detection First

Instead of extracting tables blind, detect them first:

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Detect table regions using AI
page.analyze_layout(engine='yolo')

# Find and extract detected tables
table_regions = page.find_all('region[type=table]')
for i, table in enumerate(table_regions):
    data = table.extract_table()
    print(f"Table {i+1}: {len(data)} rows")
```

## When Basic Detection Fails: Use TATR

For tables without clear borders or with complex structure:

```python
# Use Table Transformer - understands table structure better
page.clear_detected_layout_regions()  # Clear previous attempts
page.analyze_layout(engine='tatr')

# TATR understands rows, columns, and headers
table = page.find('region[type=table][model=tatr]')
rows = page.find_all('region[type=table-row][model=tatr]')
cols = page.find_all('region[type=table-column][model=tatr]') 
headers = page.find_all('region[type=table-column-header][model=tatr]')

print(f"Found: {len(rows)} rows, {len(cols)} columns, {len(headers)} headers")

# Extract using the detected structure
data = table.extract_table(method='tatr')
```

## Manual Table Definition

When detection fails completely, define the table area yourself:

```python
# Visually inspect the page to find table boundaries
page.show()

# Manually define the table region
table_area = page.create_region(
    x0=50,      # Left edge of table
    top=200,    # Top edge of table  
    x1=550,     # Right edge of table
    bottom=400  # Bottom edge of table
)

# Highlight to verify
table_area.highlight(color="blue", label="Table area")
page.show()

# Extract from the defined area
data = table_area.extract_table()
```

## Define Table Boundaries Using Header Labels (pdfplumber)

Sometimes the easiest trick is to let the **column headers** tell you exactly where the table starts and ends.  
Here's a quick, self-contained approach that works great on the violations table inside `01-practice.pdf`:

```python
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Find all header labels
headers = page.find_all(
    'text:contains("Statute|Description|Level|Repeat")',
    regex=True,
)
col_edges = headers.apply(lambda header: header.x0) + [page.width]
print(col_edges)

headers.show(crop=True)
```

```python
table_area = (
    headers[0]
    .below(
        until="text[size<10]",
        include_endpoint=False
    )
)
table_area.show()
```

```python
row_items = (
    headers[0]
    .below(width='element')
    .find_all('text')
)

row_borders = row_items.apply(lambda entry: entry.top) + [row_items[-1].bottom]
print(row_borders)
row_items.show()
```

```python
options = {
    "vertical_strategy": "explicit", 
    "horizontal_strategy": "explicit",
    "explicit_vertical_lines": col_edges,
    "explicit_horizontal_lines": row_borders,
}
data = table_area.extract_table('pdfplumber', table_settings=options)
data
```

## Fix Alignment Issues

When columns don't line up properly:

```python
# Customize pdfplumber settings for better alignment
table_settings = {
    "vertical_strategy": "text",        # Use text alignment instead of lines
    "horizontal_strategy": "lines",     # Still use lines for rows
    "intersection_x_tolerance": 10,     # More forgiving column alignment
    "intersection_y_tolerance": 5,      # More forgiving row alignment
    "edge_min_length": 3,              # Shorter minimum line length
}

data = table_area.extract_table(table_settings=table_settings)
```

## Handle Multiple Tables on One Page

Separate and extract each table individually:

```python
# Detect all tables
page.analyze_layout(engine='yolo')
tables = page.find_all('region[type=table]')

print(f"Found {len(tables)} tables")

# Extract each table separately
all_data = []
for i, table in enumerate(tables):
    # Show which table we're working on
    table.highlight(color="red", label=f"Table {i+1}")
    
    data = table.extract_table()
    all_data.append(data)
    
page.show()  # See all highlighted tables

# Save each table separately
import pandas as pd
for i, data in enumerate(all_data):
    df = pd.DataFrame(data)
    df.to_csv(f"table_{i+1}.csv", index=False)
```

## Deal with Rotated Tables

Some PDFs have sideways tables:

```python
# For rotated tables, you might need to work with coordinates differently
# First, see if layout detection picks it up
page.analyze_layout(engine='yolo')
tables = page.find_all('region[type=table]')

# If the table appears rotated in the highlights, 
# the extraction might still work normally
for table in tables:
    table.show()  # Check if this looks right
    data = table.extract_table()
    if data and len(data) > 1:  # If we got reasonable data
        print("Table extracted successfully despite rotation")
```

## Extract Tables from Images

When tables are embedded as images:

```python
# Apply OCR first to convert image text to searchable text
page.apply_ocr(engine='easyocr', languages=['en'])

# Now try table detection on the OCR'd content
page.analyze_layout(engine='tatr')
tables = page.find_all('region[type=table]')

# Extract using OCR text
for table in tables:
    data = table.extract_table()
```

## Debug Table Extraction Step by Step

When nothing works, debug systematically:

```python
# Step 1: See what elements exist in the table area
table_area = page.create_region(50, 200, 550, 400)
elements = table_area.find_all('text')

print(f"Found {len(elements)} text elements in table area")

# Step 2: See the actual text content
text_content = table_area.extract_text(layout=True)
print("Raw text content:")
print(text_content)

# Step 3: Check for lines that might define the table
lines = table_area.find_all('line')
print(f"Found {len(lines)} lines in table area")

# Step 4: Visualize everything
elements.highlight(color="blue", label="Text")
lines.highlight(color="red", label="Lines")
page.show()
```

## Troubleshooting Checklist

**No tables detected?**
- Try different engines: `yolo`, `tatr`, `surya`
- Apply OCR first if it's a scanned document
- Manually define the table area

**Columns misaligned?**
- Adjust `intersection_x_tolerance` in table settings
- Try `vertical_strategy="text"` instead of `"lines"`
- Check if there are actual lines vs. just text alignment

**Missing rows?**
- Increase `intersection_y_tolerance`
- Try `horizontal_strategy="text"`
- Look for merged cells that might be confusing the detector

**Garbled data?**
- Extract manually row by row
- Check OCR quality if text looks wrong
- Verify the table region boundaries

**Multiple tables mixed together?**
- Use layout detection to separate them first
- Define manual regions for each table
- Process tables in reading order 