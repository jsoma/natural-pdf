# Natural-PDF Cheat Sheet for LLMs

This document is **optimised to be embedded in an LLM prompt**.  Keep lines short, avoid waffle, and show canonical code.

---
## 1. Core imports
```python
from natural_pdf import PDF              # main entry-point
import pandas as pd                      # for DataFrame helpers
```

## 2. Opening files
```python
pdf = PDF("local/or/remote/file.pdf")
page = pdf.pages[0]                      # 1-based in UI, 0-based in code
```

## 3. Rendering / preview (debug only)
```python
page.show()                              # display image in notebook
```

## 4. Text extraction
```python
plain = page.extract_text()              # str
layout = page.extract_text(layout=True)  # str, roughly preserves horizontal/vertical spacing
```

## 5. Element queries (CSS-style selectors)
```python
page.find('text:contains("Total")')     # first match
page.find('text:contains(\d\d-\d\d\d)', regex=True)     # regular expressions text search
page.find_all('text[size<10]')           # list of matches
page.find(text="Date")                  # exact match shortcut
page.find("rect[color~=green]")         # green rectangle
page.find("line[width>3]|text:bold")         # thick line OR bold text
page.find_all("blob[width>100][color~=red]") # chunks of red page
```

### Attribute filters
```
text              rect / line / table / image / blob
[color~=#ff0000]  hexadecimal colour (approx)
[font_family~=Helv]
[size>=12]
:strike           strikeout
:underline        underlined text
```

### Pseudo-classes
```
:contains("Revenue")
:starts-with("INS-")
```

## 6. Spatial navigation  *(anchor-first pattern)*
```python
anchor = page.find(text="Date")
right_box = anchor.right(height='element', until='text')
txt = right_box.extract_text()

below_tbl = anchor.below().nearest('table')

# parameterised variants
region = anchor.below(                    # move down
    top=10, bottom=200,                  # pad region (px)
    include_source=False,                # exclude anchor
    until='text[size<8]',                # stop at small text
    include_endpoint=False               # don't include stop element
)

### Find the enclosing container – `parent()`
```python
cell  = page.find(text="Qty")
table = cell.parent('table')               # smallest table that contains the cell
panel = heading.parent('rect', mode='overlap')  # coloured background box
```
`mode` chooses how containment is judged:
* `contains` (default) – container bbox fully covers the element
* `center` – container contains the element's centroid
* `overlap` – any intersection is enough
```

**Tip – per-element behaviour**
`find_all(...).below()` is applied *for every element in the collection*.
If you `page.find_all('text:bold').below(until='text:bold')` you will get *n* distinct
regions – one for each bold heading – not one giant block.

**Prefer anchors over magic numbers**
Whenever possible use `until=...` to mark the *end* of a region instead of guessing
a pixel height.
```python
# Good – resilient to different font sizes / scans
body = header.below(until="text:contains('Total')", include_endpoint=False)

# Fragile – breaks if table grows/shrinks
body = header.below(height=120)
```

## 7. Layout models
```python
page.analyze_layout()        # YOLOv8 – tables, figures, etc.
page.analyze_layout('tatr')  # TATR – high-precision tables
# NOTE: `analyze_layout()` **does not return regions**. It enriches the page and
# immediately returns the *same* page object so you can continue chaining:
#
#   page.analyze_layout('tatr').find('table')
#
# or two-step:
#   page.analyze_layout('tatr')
#   tbl = page.find('table')
#
# -- Batch helper --
# You can process **all pages at once**:
#
pdf.analyze_layout('tatr')   # returns the same PDF, every page now tagged

Tables become selectable via `'table'` (or `'region[type=table]'`).
```

## 8. Tables → pandas
```python
# Method 1: Guides API (PREFERRED for most cases)
from natural_pdf.analyzers import Guides

# Smart table extraction without layout models
guides = Guides.from_content(page, axis='vertical', markers=['Column1', 'Column2'])
guides.vertical.snap_to_whitespace()  # Auto-adjust to natural gaps
guides.horizontal.from_content(markers='text[size>=10]')
guides.build_grid()
df = page.find('table').extract_table().df

# Method 2: Direct extraction from regions (simple tables)
region = page.find('text:contains("Header")').below(until='text:bold')
df = region.extract_table().df  # Works without any layout model!

# Method 3: TATR (only for complex multi-table pages)
page.analyze_layout('tatr')
first_tbl = page.find('table')
df = first_tbl.extract_table().df  # Note: extract_table, not extract_tables
```

### Why Guides over TATR?
- **Faster**: No ML model inference needed
- **More control**: Snap to whitespace, content, or manual positions
- **Robust**: Works even when TATR fails on unusual layouts
- **Flexible**: Combine multiple detection methods

### Custom header options:
```python
# No header row
df = table.extract_table().to_df(header=None)

# Skip first N rows
df = table.extract_table().to_df(skiprows=2)

# Multi-row headers
result = table.extract_table()
df = result.to_df(header=[0, 1])  # Combine first two rows as header
```

## 9. OCR when native text is absent
```python
# light & fast (default)
page.apply_ocr('easyocr')
# high-quality table/handwriting / supports rotation
page.apply_ocr('surya')
# robust Chinese/Korean/… (heavy)
page.apply_ocr('paddleocr')
# deep-learning Doctr (slow, needs GPU)
page.apply_ocr('doctr')

# -- Discard corrupted text layer --
# Option 1: Ignore text layer when opening PDF
pdf = PDF("corrupted.pdf", text_layer=False)

# Option 2: Remove text layer from existing page
page.remove_text_layer()
page.apply_ocr('surya')  # Fresh OCR without interference

# -- Batch helper --
# Run OCR on **all pages in a single call** (faster & keeps progress bar tidy):

pdf.apply_ocr('surya')        # same parameters as page-level
text = pdf.pages[0].extract_text()
```

> **When to discard text layer:**
> - Native text shows as "(cid:xxx)" gibberish
> - Text extraction returns garbled/incorrect characters
> - OCR quality would be better than native layer
> - Scanned PDF with poor automatic text recognition

> Engines: **easyocr** (default), **surya** (recommended), **paddleocr**, **doctr**.
  Choose by performance vs language support – no extra code changes needed.

## 10. Colour blobs
```python
# Detect shapes on image-based PDFs
blobs = page.detect_blobs()               # graphical fills
highlight = page.find('blob[color~=#dcefc4]')
related = highlight.find('text').extract_text()
```

## 11. Expand / contract regions
```python
box = anchor.below(height=50)
big = box.expand(left=-10, right=20, top=-5, bottom=5)
wide = box.expand(50)                 # uniformly in all directions
```

## 12. Line detection on scanned tables
```python
page.apply_ocr('surya', resolution=200)

area = page.find('text:contains("Violations")').below()
# preview peaks to pick thresholds
area.detect_lines(source_label='manual', peak_threshold_h=0.4, peak_threshold_v=0.25)
area.detect_table_structure_from_lines(source_label='manual')
table = area.extract_table()
```

## 13. Manual table structure with Guides API
```python
from natural_pdf.analyzers import Guides

# PREFERRED APPROACH for tables - no TATR needed!
guides = Guides(page)

# Smart content-based guides with flexible markers
guides.vertical.from_content(markers=['Name', 'Age'], align='between')
guides.horizontal.from_content(markers='text[size>=10]', align='center')

# Snap to natural boundaries (NEW - very powerful!)
guides.vertical.snap_to_whitespace(min_gap=10)
guides.horizontal.snap_to_whitespace()

# From detected lines (pixels or vectors)
guides.vertical.from_lines(detection_method='pixels', threshold='auto')

# Build and extract
guides.build_grid()
table = page.find('table')
df = table.extract_table().df  # Note: .extract_table() not .extract_tables()
```

### Recent Improvements
- **Tiny text support**: Fonts <7pt now extracted reliably
- **RTL languages**: Arabic, Hebrew handled automatically with proper BiDi
- **Better .extract_table()**: Single method that returns TableResult with .df property

## 14. Vision classification
```python
# page-level
pdf.classify(['diagram', 'text', 'invoice'], using='vision')

# region-level (e.g., checkbox)
rect = page.find('rect')[0]
rect.classify(['checked', 'unchecked'], using='vision').category
```

## 15. Deskew crooked scans
```python
# page-level preview (returns PIL.Image)
fixed_img = page.deskew()              # auto-detects skew angle

# deskew whole document → new PDF object
clean = pdf.deskew()                   # optional angle=..., resolution=300
clean.save_pdf('deskewed.pdf', original=True)
```

## 16. Split repeating sections
```python
sections = page.get_sections(
    start_elements='text:bold',
    include_boundaries='start'
)
for sec in sections:
    print(sec.extract_text()[:100])
```

## 17. Extractive QA helpers
```python
answer = page.ask("What date was the inspection?")
answer.show()                         # visual context

# batch
fields = ["site", "violation count", "summary"]
page.extract(fields)                  # returns dict-like

# Pydantic schema
class Info(BaseModel):
    site: str
    date: str
page.extract(schema=Info)
```

### E. Colour-coded legend extraction
```python
page = pdf.pages[1]
page.detect_blobs()
legend_box = page.find('blob[color~=#dcefc4]').expand(20)
legend_text = legend_box.find_all('text').extract_each_text()
```

Add more examples as the library evolves – keep snippets short, **describe page quality** (e.g. "skewed book scan"), anchor-first, and avoid pixel magic.

### RTL scripts handled automatically
```python
# Arabic search – works with normal logical order
page.find("text:contains('الجريدة الرسمية')")

# Disable Unicode BiDi pass if you prefer raw PDF order
raw = page.extract_text(bidi=False)
```

Parentheses/brackets are mirrored and mixed Western digits stay left-to-right –
no string reversal needed in your queries.

---
## Example Workflows (few-shot)

### A. Extract **second** table on page 64
```
