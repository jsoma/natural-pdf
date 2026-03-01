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

# extract_table returns a TableResult (supports iteration and .to_df())
table = page.extract_table(method="pdfplumber")
table
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
* **tatr** – handles tables without visible lines; slower; requires `pip install torch transformers`.
* **text** – whitespace clustering; fallback when lines + models fail.

You can call `page.extract_table(method="text")` or on a `Region` as well.

The general workflow is: try different layout analyzers to locate your table, then extract from the specific region. If one backend doesn't work, try another — `plumber` and `tatr` handle different table styles, so comparing their output on your PDF is the fastest way to find what works.

## Using Guides for Borderless Tables

When tables lack visible borders, use guides to define column and row structure manually. See the [Guides cookbook](../cookbook/guides.md) for full details.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]

# Create guides and detect structure
guides = page.guides()
guides.vertical.from_content(['Number', 'Date', 'Location', 'Description'])
guides.horizontal.from_stripes()

# Preview the grid
guides.show()

# Extract the table
df = guides.extract_table(include_outer_boundaries=True).to_df()
```

For tables where rows start with a specific pattern:

```python
guides = page.guides()
guides.vertical.from_content(['Number', 'Date', 'Location', 'Description', 'Disposition'], outer="last")
guides.horizontal.from_content(
    lambda p: p.find_all('text:starts-with(NF-)')
)

df = guides.extract_table(header="first").to_df()
```

For multi-page tables, create guides on one page and pass all pages to `extract_table()`:

```python
guides = page.guides()
guides.vertical.from_headers(page.find_all('text:bold[size>=11]'))

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

## Tables with Checkboxes

If a table contains checkboxes, run `detect_checkboxes()` before extracting. Each detected checkbox gets `alt_text` (`[CHECKED]` or `[UNCHECKED]`) that flows into `extract_text()`, so checkbox cells appear in the extracted table instead of being empty.

```python
page.detect_checkboxes()
df = guides.extract_table().to_df()
```

See the [Guides cookbook](../cookbook/guides.md#tables-with-checkboxes) for a full example. Requires `pip install onnxruntime huggingface_hub`.

## Related Tutorials

- **[Guides](../cookbook/guides.md)** – Define table structure manually for borderless or complex tables
- **[Spatial Navigation](08-spatial-navigation.md)** – Use `.below()` to extract just the table region
- **[Layout Analysis](07-layout-analysis.md)** – Detect tables automatically with AI models
