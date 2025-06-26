# Enhanced Table Processing

Tables can appear in PDFs in wildly different ways—cleanly tagged in the PDF structure, drawn with ruling lines, or simply implied by visual spacing.  `natural-pdf` exposes several back-ends under the single method `extract_table()` so you can choose the strategy that matches your document.

Below we walk through the three main options, when to reach for each one, and sample code you can adapt (replace the example PDF URLs with your own files).

## 1. `method="pdfplumber"`  (default)

* **How it works** – delegates to pdfplumber's ruling-line heuristics; looks for vertical/horizontal lines and whitespace gutters.
* **Best for** – digitally-born PDFs where the table grid is drawn or where columns have consistent whitespace.

### Example A – Grid-based (line) detection

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Optional fine-tuning for pdfplumber.  Typical tweaks are vertical/horizontal strategies.
settings = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "intersection_tolerance": 3,
}

rows = page.extract_table(method="pdfplumber", table_settings=settings)
rows  # ▶︎ returns a list of lists
```

Expected output: a small list of rows containing the text exactly as it appears in the digital table.

### Example B – Whitespace-driven detection

Sometimes a table is **drawn without ruling lines** (or the PDF stores them as thick rectangles so the line detector ignores them).  In that case you can switch both strategies to `"text"` so pdfplumber clusters by the gaps between words rather than relying on graphics commands:

```python
settings_text = {
    "vertical_strategy": "text",   # look for whitespace gutters
    "horizontal_strategy": "text", # group into rows by vertical gaps
    "text_x_tolerance": 2,          # tune for narrow columns
    "text_y_tolerance": 2,
}

rows_text = page.extract_table(method="pdfplumber", table_settings=settings_text)
```

Compare `rows_text` with the earlier `rows` list—if your PDF omits the grid, the whitespace strategy will usually outperform line-based detection.

---

## 2. `method="tatr"`  (Table Transformer)

* **How it works** – runs Microsoft's Table Transformer (LayoutLM-based) to detect tables, rows and cells visually, then reads the text inside each cell.
* **Best for** – scanned or camera-based documents, or born-digital files where ruling lines are missing/irregular.
* **Dependencies** – requires the **AI** extra (`pip install "natural-pdf[ai]"`) because it needs `torch`, `transformers`, and `torchvision`.

### Example

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/needs-ocr.pdf")
page = pdf.pages[0]

# If the page is scanned, run OCR first so each cell has text
page.apply_ocr(engine="easyocr", languages=["en"], resolution=200)

# Table Transformer needs the layout model; specify device if you have GPU
rows = page.extract_table(method="tatr")
rows
```

Expected output: the table rows—even when the grid is just implied—arrive with text already OCR-corrected.

---

## 3. `method="text"`  (Whitespace heuristic)

* **How it works** – groups words into lines, then uses whitespace clustering (Jenks breaks) to infer columns; no layout model.
* **Best for** – simple, left-aligned tables with consistent columns but no ruling lines; fastest option.

### Example

```python
# from natural_pdf import PDF

# pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/whitespace-table.pdf")
# page = pdf.pages[0]

# rows = page.extract_table(method="text", table_settings={"min_words_horizontal": 2})
# for row in rows:
#     print(row)
```

Expected output: printed rows that roughly match the visual columns; best effort on ragged layouts.

---

## 4. Lines-first workflow (when pdfplumber misses rows/cols)

If `method="pdfplumber"` cannot find the grid, detect lines explicitly and build the table structure yourself.

```python
from natural_pdf.analyzers import Guides

page.detect_lines(resolution=200, source_label="detected", horizontal=True, vertical=True)

# (Optional) visual check
page.find_all("line[source=detected]").show(group_by="orientation")

# Convert lines → regions using Guides
guides = Guides.from_lines(page, source_label="detected")
guides.build_grid(source="detected", cell_padding=0.5)

table = page.find("region[type='table']")
```

---

## TODO

* Provide a benchmark matrix of speed vs. accuracy for the three methods.
* Add a snippet showing how to export cell regions directly to a pandas **DataFrame**.
* Document edge-cases: rotated tables, merged cells, or header repetition across pages.
* Include guidance on mixing methods—e.g., run `detect_lines` first, fall back to `text` for cells lacking grid.
