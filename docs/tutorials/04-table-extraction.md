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

## TODO

* Compare accuracy/time of the three methods on the sample PDF.
* Show how to call `page.extract_table(method="text")` as a no-dependency fallback.
* Add snippet exporting `rows` to pandas DataFrame.
* Demonstrate cell post-processing (strip %, cast numbers).
