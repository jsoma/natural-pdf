# Basic Table Extraction

PDFs often contain tables, and `natural-pdf` provides methods to extract their data. The key is to first triangulate where your table is on the page, then use powerful extraction tools on that specific region.

Let's extract the "Violations" table from our practice PDF.

```python
#%pip install natural-pdf  # core install already includes pdfplumber
```

## Method 1 – pdfplumber (default)

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# For a single table, extract_table returns list-of-lists
table = page.extract_table(method="pdfplumber")
table  # List-of-lists of cell text
```

`extract_table()` defaults to the **plumber** backend, so the explicit `method` is optional—but it clarifies what's happening.

## Method 2 – TATR-based extraction

When you do a TATR layout analysis, it detects tables, rows and cells with a LayoutLM model. Once a region has `source="detected"` and `type="table"`, calling `extract_table()` on that region uses the **tatr** backend automatically.

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Detect layout with Table Transformer
page.analyze_layout(engine="tatr")

# Grab the first detected table region
table_region = page.find('region[type=table]')

table_region.show(label="TATR Table", color="purple")
```

```python
tatr_rows = table_region.extract_table()  # Uses TATR backend implicitly
```

## Method 3 – PaddleOCR Layout

You can also try PaddleOCR's layout detector to locate tables:

```python
page.clear_detected_layout_regions()
page.analyze_layout(engine="paddle", confidence=0.3)

paddle_table = page.find('region[type=table]')
if paddle_table:
    paddle_table.show(color="green", label="Paddle Table")
    paddle_rows = paddle_table.extract_table(method="pdfplumber")  # fall back to ruling-line extraction inside the region
```

---

### Choosing the right backend

* **plumber** – fastest; needs rule lines or tidy whitespace.
* **tatr** – robust to missing lines; slower; requires AI extra.
* **text** – whitespace clustering; fallback when lines + models fail.

You can call `page.extract_table(method="text")` or on a `Region` as well.

The general workflow is: try different layout analyzers to locate your table, then extract from the specific region. Keep trying options until one works for your particular PDF!

For complex grids where even models struggle, see Tutorial 11 (enhanced table processing) for a lines-first workflow.

## Using Guides for Borderless Tables

When tables lack visible borders, use the `Guides` class to define structure based on content:

```python
from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides

pdf = PDF("document.pdf")
page = pdf.pages[0]

# Create guides for the page
guides = Guides(page)

# Define columns from header text positions
headers = (
    page
    .find(text="NUMBER")
    .right(include_source=True)
    .expand(top=3, bottom=3)
    .find_all('text')
)
guides.vertical.from_content(headers, align='left')

# Define rows from zebra stripes or content patterns
guides.horizontal.from_stripes()

# Preview the grid
guides.show()

# Extract the table
df = guides.extract_table(include_outer_boundaries=True).to_df()
```

## Guides from Content Patterns

For tables where rows start with a specific pattern:

```python
from natural_pdf.analyzers.guides import Guides

guides = Guides(page)

# Define columns from header names
columns = ['Number', 'Date', 'Location', 'Description', 'Disposition']
guides.vertical.from_content(columns, outer="last")

# Define rows based on content that starts each row
guides.horizontal.from_content(
    lambda p: p.find_all('text:starts-with(NF-)')
)

# Extract with first row as header
table_result = guides.extract_table(header="first")
df = table_result.to_df()
```

## Multi-Page Table Extraction

Extract tables that span multiple pages:

```python
# Find headers on first page
headers = page.find_all('text[y0=min()]')

# Create guides from headers
guides = Guides(page)
guides.vertical.from_headers(headers)

# Extract across all pages
df = guides.extract_table(pdf.pages).to_df()
print(f"Found {len(df)} rows across all pages")
```

## Converting Tables to pandas DataFrames

Once you have a table extracted, convert it to a pandas DataFrame for analysis:

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Extract the table
table = page.extract_table()

# Convert to DataFrame with first row as headers
df = table.to_df(header="first")
df
```

The `to_df()` method accepts these options:
- `header="first"` – Use the first row as column headers
- `header="none"` – No headers, use numeric indices
- `header=["Col A", "Col B", ...]` – Provide custom column names

```python
# Example: Post-process numeric columns
df['Amount'] = df['Amount'].str.replace('$', '').str.replace(',', '').astype(float)
df['Percentage'] = df['Percentage'].str.rstrip('%').astype(float) / 100
```

## Related Tutorials

- **[Spatial Navigation](08-spatial-navigation.md)** – Use `.below()` to extract just the table region
- **[Layout Analysis](07-layout-analysis.md)** – Detect tables automatically with AI models

## TODO

* Compare accuracy/time of the three methods on the sample PDF.
* Show how to call `page.extract_table(method="text")` as a no-dependency fallback.
* Demonstrate cell post-processing (strip %, cast numbers).
