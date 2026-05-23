# Guides

Guides are manual column and row markers you place on a page or region to define table structure. They work like rulers in a design tool — you set vertical lines for columns and horizontal lines for rows, then extract a table from the grid they form.

Use guides when:

- The table has no visible borders or ruling lines
- Automatic table extraction (`extract_table()`) merges or splits columns incorrectly
- You need precise control over where column and row boundaries fall
- The table spans multiple pages with consistent column layout

## Creating Guides

Guides need a context — a page, region, or flow region that defines the coordinate space. All coordinates are in PDF points (1/72 inch), measured from the top-left corner of the context object.

**From a page or region method:**

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]

guides = page.guides()
```

**From the Guides class directly:**

```python
from natural_pdf.analyzers.guides import Guides

guides = Guides(page)
```

Both are equivalent. The `.guides()` method is available on pages, regions, and flow regions.

## Detecting Columns and Rows

Guides has two axis lists — `guides.vertical` (columns) and `guides.horizontal` (rows). Each has detection methods that analyze the content and place guides automatically.

### From Lines

Detect guides from ruling lines in the PDF (drawn borders, grid lines, separators):

```python
guides = page.guides()
guides.vertical.from_lines()
guides.horizontal.from_lines()
guides.show()
```

The `threshold` parameter controls sensitivity for pixel-based line detection. Lower values detect fainter lines:

```python
guides = page.guides()
guides.vertical.from_lines(threshold=0.4)
guides.horizontal.from_lines(threshold=0.3)
```

If `from_lines()` finds too few guides, try lowering `threshold` (e.g., `0.3` or `0.2`). If it finds nothing at all, the PDF may not have drawn lines — use `from_content()` or `from_whitespace()` instead.

### From Content

Place guides aligned to text elements. Pass a list of strings to search for, a selector, an `ElementCollection`, or a callable:

```python
guides = page.guides()

# From header text — guides snap to the left edge of each match
guides.vertical.from_content(['Name', 'Date', 'Amount', 'Status'])

# From a selector
guides.vertical.from_content('text:bold[size>=10]')

# Align to right edges instead
guides.vertical.from_content(['Name', 'Date'], align='right')

# Align between elements (midpoints between adjacent matches)
guides.vertical.from_content(['Name', 'Date', 'Amount'], align='between')
```

The `outer` parameter controls whether boundary guides are added at the edges of the context region. Set `outer=True` (default) to add both edges, `outer="first"` or `outer="last"` for just one side, or `outer=False` for none:

```python
# Only add a boundary after the last column, not before the first
guides.vertical.from_content(['Number', 'Date', 'Description'], outer="last")
```

For horizontal guides, you can use callables to find row markers dynamically:

```python
# Rows start with case numbers like "NF-2024-001"
guides.horizontal.from_content(
    lambda p: p.find_all('text:starts-with(NF-)')
)
```

### From Whitespace

Detect gaps between text — places guides in empty space where no content exists:

```python
guides = page.guides()
guides.vertical.from_whitespace(min_gap=15)
guides.horizontal.from_whitespace(min_gap=8)
```

`min_gap` sets the minimum gap width (in points) to consider as a separator. If too few guides appear, try lowering `min_gap`. If too many appear, raise it.

### From Headers

Detect column boundaries by analyzing the whitespace valleys between header elements. Works only for vertical guides:

```python
# Find the header row
headers = page.find_all('text:bold[size>=11]')

guides = page.guides()
guides.vertical.from_headers(headers)
```

Two detection methods are available:

- `'min_crossings'` (default) — fast, finds the x-position in each gap that crosses the fewest text bounding boxes. Works well for most tables.
- `'seam_carving'` — slower, but better when columns aren't strictly vertical (e.g., text that drifts left/right at different rows).

```python
guides.vertical.from_headers(headers, method='seam_carving')
```

### From Headers Plus Row Anchors

For crowded tables without ruling lines, a reliable row ID is often better than
whitespace for horizontal guides. Use the header row to seed column boundaries,
snap those boundaries into nearby whitespace, then use first-column IDs, case
numbers, or other stable row markers for the row boundaries:

```python
import re

case_number = re.compile(r"\d{2}-\d{4}-\d+")

header = page.find('text:contains("CASE #")')
texts = list(page.find_all("text"))
header_row = [el for el in texts if abs(el.top - header.top) <= 2]
row_anchors = [
    el
    for el in texts
    if (
        case_number.fullmatch(el.extract_text().strip())
        and el.top > header.top
        and abs(el.x0 - header.x0) <= 8
    )
]

guides = page.guides().from_headers_and_row_anchors(
    header_row,
    row_anchors,
    header_anchor=header,
)

df = guides.extract_table(
    header="first",
    cell_extract="words",
    cell_overlap="center",
    include_outer_boundaries=True,
).to_df()
```

This pattern works well when headers, dates, numbers, and names use different
alignment rules. The helper builds vertical guides from the headers, snaps those
guides into nearby text whitespace, and builds horizontal guides from the header
row plus the row anchors. The first guide pass only needs to be close;
`snap_to_whitespace()` moves the vertical guides into the gaps between columns.

For many multi-page native-text tables, header-derived vertical guides are
enough: `extract_table()` can infer the row bands while reusing the page-1
column geometry across the page collection.

```python
headers = page.find_all('text[y0=min()]')

guides = page.guides()
guides.vertical.from_headers(headers)

df = guides.extract_table(pdf.pages).to_df()
```

`from_headers()` only creates vertical column guides. If row segmentation is
ambiguous, or if you want explicit review evidence for row boundaries, add
horizontal guides from row IDs on each page:

```python
def page_row_markers(target_page):
    markers = [header] if target_page.number == page.number else []
    markers.extend(
        el
        for el in target_page.find_all("text")
        if (
            case_number.fullmatch(el.extract_text().strip())
            and abs(el.x0 - header.x0) <= 8
        )
    )
    return markers

guides = page.guides()
guides.vertical.from_headers(header_row)
guides.vertical.snap_to_whitespace(min_gap=2, detection_method="text")
guides.horizontal.from_content(page_row_markers, align="between", outer=True)

df = guides.extract_table(
    pdf.pages,
    header="first",
    cell_extract="words",
    cell_overlap="center",
    include_outer_boundaries=True,
).to_df()
```

If one tiny text object spans two adjacent cells, `cell_overlap="partial"` may
recover it but can duplicate the same text into both cells. Prefer snapped
guides with `cell_overlap="center"` first, then split known merged values by a
value pattern when the PDF text layer really merged adjacent cells.

If you want the usual settings in one call, use `extract_table_guided()`:

```python
df = page.extract_table_guided(
    header_row,
    row_anchors,
).to_df()
```

By default this uses `cell_extract="words"`, `cell_overlap="center"`,
`header="first"`, and outer boundaries, which are often the right defaults for
small or crowded native-text tables. Header elements are the most reliable
input; header text strings can work when labels are unique, but selected
elements avoid ambiguity from repeated labels or body text.

### From Stripes

For zebra-striped tables with alternating row colors:

```python
guides = page.guides()
guides.horizontal.from_stripes()
```

Auto-detects the most common stripe color. To target a specific color:

```python
guides.horizontal.from_stripes(color='#e8f4f8')
```

### Dividing Evenly

Split the context bounds into equal parts:

```python
guides = page.guides()
guides.vertical.divide(4)    # 3 vertical guides → 4 columns
guides.horizontal.divide(10) # 9 horizontal guides → 10 rows
```

When called on a region, `divide()` splits that region's bounds, not the full page.

## Chaining

Every detection and manipulation method returns the parent `Guides` object, so you can chain across axes:

```python
(table_area.guides()
    .vertical.from_lines(threshold=0.4)
    .horizontal.from_lines()
    .show())
```

A more involved example — detect columns from content, rows from stripes, then refine:

```python
(table_area.guides()
    .vertical.from_content(['Name', 'Date', 'Amount'])
    .vertical.snap_to_whitespace()
    .horizontal.from_stripes()
    .show())
```

## Refining Guides

After initial detection, you can adjust individual guides.

### Snapping to Whitespace

Move each guide to the nearest whitespace gap. Useful after `from_content()` or `from_lines()` when guides land slightly inside text. Works on both axes:

```python
guides.vertical.snap_to_whitespace()
guides.horizontal.snap_to_whitespace()

# With options
guides.vertical.snap_to_whitespace(min_gap=5, detection_method='text')
```

### Snapping to Content

Move each guide to the nearest text element edge:

```python
guides.vertical.snap_to_content(align='left')
```

### Manual Adjustments

Coordinates are in the context's coordinate space — page coordinates for a page, region coordinates for a region. Use `guides.show()` to see where existing guides land before adding or shifting.

```python
# Add a guide at a specific coordinate
guides.vertical.add(150.0)

# Add multiple guides
guides.vertical.add([100.0, 200.0, 300.0])

# Shift the third guide 5 points to the right
guides.vertical.shift(2, 5.0)

# Remove the first guide
guides.vertical.remove_at(0)

# Clear all horizontal guides and start over
guides.horizontal.clear_all()
```

## Previewing

`show()` returns a PIL Image with guides drawn over the page — blue for vertical, red for horizontal:

```python
img = guides.show()
img.save("guides_preview.png")
```

Check the guide count and positions:

```python
print(f"{len(guides.vertical)} vertical guides, {len(guides.horizontal)} horizontal guides")
print("Vertical:", list(guides.vertical))
print("Horizontal:", list(guides.horizontal))
```

Note: the number of columns is `len(guides.vertical) - 1` (guides are boundaries between columns, not columns themselves). If your guides don't include outer edges, use `include_outer_boundaries=True` when extracting.

## Extracting Tables

Once guides define the grid, extract a table:

```python
result = guides.extract_table()
df = result.to_df()
```

`extract_table()` returns a `TableResult`. Call `.to_df()` to get a pandas DataFrame.

### Header Options

```python
# First row becomes column headers (default)
df = guides.extract_table(header="first").to_df()

# No headers
df = guides.extract_table(header=None).to_df()

# Custom headers
df = guides.extract_table(header=["ID", "Name", "Value"]).to_df()
```

### Outer Boundaries

If your guides don't include the outer edges of the table, `include_outer_boundaries=True` adds them from the context bounds:

```python
df = guides.extract_table(include_outer_boundaries=True).to_df()
```

## Working with Regions

Guides work best when scoped to the table area rather than the full page:

```python
# Find the table area using spatial navigation
header = page.find('text:contains("Item")')
table_area = header.below(until='text:contains("Total")', include_source=True)

# Create guides scoped to this region
guides = table_area.guides()
guides.vertical.from_content(['Item', 'Qty', 'Price'])
guides.horizontal.from_whitespace()

df = guides.extract_table().to_df()
```

## Multi-Page Tables

For tables that continue across pages, create guides on one page and apply them to a collection:

```python
# Set up guides from the first page
page = pdf.pages[0]
guides = page.guides()
guides.vertical.from_content(['Name', 'Date', 'Amount'])
guides.horizontal.from_whitespace()

# Extract across all pages
df = guides.extract_table(pdf.pages).to_df()
```

For multi-page flow regions (created with spatial navigation and `multipage=True`), guides work across the combined region automatically:

```python
header = pdf.pages[0].find('text:contains("Item")')
flow = header.below(multipage=True)

guides = flow.guides()
guides.vertical.from_content(['Item', 'Qty', 'Price'])
guides.horizontal.from_whitespace()

df = guides.extract_table().to_df()
```

## Tables with Checkboxes

When a table contains checkboxes, `detect_checkboxes()` finds them and sets `alt_text` on each region (`[CHECKED]` or `[UNCHECKED]`). This text flows into `extract_text()` automatically, so checkbox cells show up in table extraction instead of appearing empty.

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/needs-ocr.pdf")
page = pdf.pages[0]

# OCR first (this is a scanned PDF)
page.apply_ocr()

# Find the table area
table_area = (
    page
    .find('text:contains("Violations")')
    .below(
        until='text:contains("Jungle")',
        include_endpoint=False
    )
)

# Detect guides from ruling lines
guides = table_area.guides()
guides.vertical.from_lines(threshold=0.4)
guides.horizontal.from_lines()

# Detect checkboxes — their alt_text fills in during extraction
page.detect_checkboxes()

df = guides.extract_table().to_df()
```

Checkbox cells that would otherwise be empty now contain `[CHECKED]` or `[UNCHECKED]`. To change these labels:

```python
from natural_pdf import options

options.alt_text.checkbox_checked = "Yes"
options.alt_text.checkbox_unchecked = "No"
```

Requires `pip install onnxruntime huggingface_hub` for the checkbox detection model.

## Passing Guides to pdfplumber

If you want to use guides as explicit line hints for pdfplumber's table extraction instead of the built-in grid extractor:

```python
settings = guides.to_dict()
# Returns: {'explicit_vertical_lines': [...], 'explicit_horizontal_lines': [...]}

df = page.extract_table(method="pdfplumber", table_settings=settings).to_df()
```

## Full Example

```python
from natural_pdf import PDF

pdf = PDF("report.pdf")
page = pdf.pages[0]

# Scope to the table area
table_header = page.find('text:contains("Account")')
table_area = table_header.below(
    until='text:contains("Total")',
    include_source=True
)

# Detect columns from headers, rows from whitespace
guides = table_area.guides()
guides.vertical.from_content(['Account', 'Description', 'Debit', 'Credit'])
guides.vertical.snap_to_whitespace()
guides.horizontal.from_whitespace(min_gap=5)

# Preview
guides.show().save("temp/table_guides.png")

# Extract
df = guides.extract_table(header="first").to_df()
print(df)

pdf.close()
```

The same example with chaining:

```python
from natural_pdf import PDF

pdf = PDF("report.pdf")
page = pdf.pages[0]

table_header = page.find('text:contains("Account")')
table_area = table_header.below(
    until='text:contains("Total")',
    include_source=True
)

df = (table_area.guides()
    .vertical.from_content(['Account', 'Description', 'Debit', 'Credit'])
    .vertical.snap_to_whitespace()
    .horizontal.from_whitespace(min_gap=5)
    .extract_table(header="first")
    .to_df())

print(df)
pdf.close()
```
