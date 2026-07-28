---
fixture: pdfs/needs-ocr.pdf
thumbnail: 4
tier: nightly
---

# Guides: build the grid from evidence

`extract_table()` on a scan has nothing to work with — OCR gives you words, but the ruling lines are just dark pixels. Guides make the grid explicit: you build the column and row boundaries from whatever evidence exists, look at them, and only then extract.

The document is the scanned inspection report from [the OCR page](02-ocr.md). OCR it and anchor the table region, same as there:

```python
from natural_pdf import PDF

scan = PDF("https://github.com/jsoma/natural-pdf/raw/main/pdfs/needs-ocr.pdf")
spage = scan.pages[0]
spage.apply_ocr()

table_area = (
    spage
    .find('text:contains("Violations")')
    .below(until='text:contains("Jungle Health")', include_endpoint=False)
    .trim()
)
table_area.show(crop=True)
```

**Columns from the headers.** The header words are evenly spread, one per column — tell the guides to split there:

```python
guides = table_area.guides()
guides.vertical.from_headers(["Statute", "Description", "Level", "Repeat?"])
guides.show()
```

Look before extracting — that's the whole point of guides. The boundaries landed *on the text*: one runs through the middle of "Statute", another clips the longest description line, a third cuts through the Level column.

**Snap to whitespace.** Migrate each boundary to the nearest empty vertical channel:

```python
guides.snap_to_whitespace()
guides.show()
```

**Rows from content.** No horizontal lines to detect, but the first column is dependable — one statute number per row. Use those elements as row anchors:

```python
anchors = guides.column(0).find_all('text')
guides.horizontal.from_content(anchors)
guides.show()
```

A full grid, built from headers, whitespace, and a reliable column — three kinds of evidence, none of them ruling lines. One last gap before extracting: the Repeat? column is drawn checkboxes, which OCR can't read. This is the gap left open on the very first Learn page, and this is where it closes:

:::caution[This downloads a model on first use]
`detect_checkboxes()` downloads a YOLO12n ONNX model (`wendys-llc/checkbox-detector`, ~10 MB) on the first call.
:::

```python
checkboxes = spage.detect_checkboxes()
len(checkboxes)
```

```python
guides.extract_table().to_df()
```

Statutes, descriptions, levels, and real `[CHECKED]` / `[UNCHECKED]` states — from a page that started as pixels.

**The punchline.** That ladder was the general method, and it works when a table has no lines at all. This table, though, *does* have visible lines — they're just pixels, not vector graphics. `from_lines(detection_method='pixels')` scans the rendered image for dark runs, and the whole grid takes three lines:

```python
guides = table_area.guides()
guides.vertical.from_lines(detection_method='pixels')
guides.horizontal.from_lines(detection_method='pixels')
guides.show()
```

```python
guides.extract_table().to_df()
```

Same DataFrame. The craft is matching the tool to the evidence: pixel lines when the rulings are visible, headers + whitespace + content anchors when they aren't, and `guides.show()` at every step, because a wrong grid produces a *plausible* wrong table.

## What you can do now

Build table grids from headers, whitespace, content anchors, or pixel lines — checking each step with `guides.show()`, because a wrong grid produces a *plausible* wrong table — and read drawn checkboxes with `detect_checkboxes()`. The last page puts the whole toolkit on one real document, start to finish.
