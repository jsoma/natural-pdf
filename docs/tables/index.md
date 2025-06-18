# Getting Tables Out of PDFs

Tables in PDFs can be a real pain. Sometimes they're perfectly formatted with nice lines, other times they're just text floating around that vaguely looks like a table. Natural PDF gives you several different approaches to tackle whatever table nightmare you're dealing with.

## Setup

Let's start with a PDF that has some tables to work with.

```python
from natural_pdf import PDF

# Load the PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")

# Select the first page
page = pdf.pages[0]

# Display the page
page.show()
```

## The Quick and Dirty Approach

If you know there's a table somewhere and just want to try extracting it, start simple:

```python
# Try to extract the first table found on the page
# This uses pdfplumber behind the scenes
table_data = page.extract_table() # Returns a list of lists
table_data
```

*This might work great, or it might give you garbage. Tables are tricky.*

## The Smart Way: Detect First, Then Extract

A better approach is to first find where the tables actually are, then extract them properly.

### Finding Tables with YOLO (Fast and Pretty Good)

The YOLO model is good at spotting table-shaped areas on a page.

```python
# Use YOLO to find table regions
page.analyze_layout(engine='yolo')

# Find what it thinks are tables
table_regions_yolo = page.find_all('region[type=table][model=yolo]')
table_regions_yolo.show()
```

```python
# Extract data from the detected table
table_regions_yolo[0].extract_table()
```

### Finding Tables with TATR (Slow but Very Smart)

The TATR model actually understands table structure - it can tell you where rows, columns, and headers are.

```python
# Clear previous results and try TATR
page.clear_detected_layout_regions() 
page.analyze_layout(engine='tatr')
```

```python
# Find the table that TATR detected
tatr_table = page.find('region[type=table][model=tatr]')
tatr_table.show()
```

```python
# TATR finds the internal structure too
rows = page.find_all('region[type=table-row][model=tatr]')
cols = page.find_all('region[type=table-column][model=tatr]')
hdrs = page.find_all('region[type=table-column-header][model=tatr]')
f"TATR found: {len(rows)} rows, {len(cols)} columns, {len(hdrs)} headers"
```

## Choosing Your Extraction Method

When you call `extract_table()` on a detected region, Natural PDF picks the extraction method automatically:
- **YOLO-detected regions** → uses `pdfplumber` (looks for lines and text alignment)
- **TATR-detected regions** → uses the smart `tatr` method (uses the detected structure)

You can override this if needed:

```python
tatr_table = page.find('region[type=table][model=tatr]')
# Use TATR's smart extraction
tatr_table.extract_table(method='tatr')
```

```python
# Or force it to use pdfplumber instead (maybe for comparison)
tatr_table.extract_table(method='pdfplumber')
```

### When to Use Which?

- **`pdfplumber`**: Great for clean tables with visible grid lines. Fast and reliable.
- **`tatr`**: Better for messy tables, tables without lines, or tables with merged cells. Slower but smarter.

## When Tables Don't Cooperate

Sometimes the automatic detection doesn't work well. You can tweak pdfplumber's settings:

```python
# Custom settings for tricky tables
table_settings = {
    "vertical_strategy": "text",      # Use text alignment instead of lines
    "horizontal_strategy": "lines",   # Still use lines for rows
    "intersection_x_tolerance": 5,    # Be more forgiving about line intersections
}

results = page.extract_table(table_settings=table_settings)
```

## Saving Your Results

Once you've got your table data, you'll probably want to do something useful with it:

```python
import pandas as pd

# Convert to a pandas DataFrame for easy manipulation
df = pd.DataFrame(page.extract_table())
df
```

## Working with TATR Cell Structure

TATR is smart enough to create individual cell regions, but accessing them directly is still a work in progress:

```python
# This should work but doesn't quite yet - we're working on it!
# tatr_table.cells
```

## Next Steps

Tables are just one part of document structure. Once you've got table extraction working:

- [Layout Analysis](../layout-analysis/index.ipynb): See how table detection fits into understanding the whole document
- [Working with Regions](../regions/index.ipynb): Manually define table areas when automatic detection fails