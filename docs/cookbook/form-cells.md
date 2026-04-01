# Form Cell Detection

Detect rectangular cells in structured forms — arrest reports, tax forms, government applications — using image-based morphological processing. Works on scanned documents and image-based PDFs where the grid lines are visible pixels, not PDF drawing commands.

**When to use this:**

- Government forms with printed grid cells (arrest reports, tax forms, permit applications)
- Scanned documents where `extract_table()` doesn't find the structure
- Forms where you need to navigate between labeled cells ("find the cell labeled Jurisdiction, get the value below it")

**Prerequisites:**

```bash
pip install opencv-python-headless numpy
```

## Basic Usage

```python
from natural_pdf import PDF

pdf = PDF("pdfs/cookbook/arrest_report.pdf")
page = pdf.pages[0]

cells = page.detect_form_cells()
print(f"Found {len(cells)} cells")
```

Each detected cell is a `Region` with `type="form_cell"`. You can find them later with selectors:

```python
form_cells = page.find_all('region[type=form_cell]')
```

## How It Works

The pipeline renders the page to an image, then:

1. **Adaptive threshold** converts to binary (black/white)
2. **Morphological opening** with horizontal and vertical kernels isolates form lines from text
3. **Line extension** bridges small gaps to form a watertight grid
4. **Contour detection** finds enclosed rectangular regions (the cells)
5. **OCR merge** combines cells where a text element spans multiple cells on the same row (fixes spurious vertical line splits in banner labels)

## Merging Cells

By default, `merge=True` combines cells that a single text element spans. This fixes banner labels like "GENERAL CASE INFORMATION" that get split into fragments by incidental vertical lines.

```python
# Merge is on by default
cells = page.detect_form_cells(merge=True)

# Disable merging to see raw detected cells
raw_cells = page.detect_form_cells(merge=False)
```

### Protecting Cells with Labels

The merge can incorrectly combine adjacent cells when a text bounding box is slightly oversized and bleeds past a real divider. For example, a "Jurisdiction" label bbox might extend a few pixels into the adjacent "Beat" cell, causing them to merge.

Use `cell_labels` to mark cells that should never merge with each other. Pass selectors (strings) or elements:

```python
cells = page.detect_form_cells(
    merge=True,
    cell_labels=[
        'text:contains("Jurisdiction")',
        'text:contains("Beat")',
        'text:contains("Call Source")',
    ]
)
```

The logic: if two cells both contain a label element (by center point), that merge is skipped. A labeled cell can still merge with an empty fragment — only two labeled cells are prevented from merging.

**Selectors work the same as `page.find()`**, so you can use any match type:

```python
cell_labels=[
    'text:contains("Jurisdiction")',   # substring match
    'text:regex("Case\\s+No")',        # regex for OCR variation
    'text:closest("Beat")',            # fuzzy match for OCR typos
]
```

You can also pass elements directly, which is useful when OCR garbles the text:

```python
# Find elements first, then protect them
juris = page.find('text:contains("Jradadon")')  # OCR'd "Jurisdiction"
beat = page.find('text:contains("521")')

cells = page.detect_form_cells(
    merge=True,
    cell_labels=[juris, beat]
)
```

For bulk processing (100k+ forms of the same template), you define the label list once and reuse it.

## Navigating Between Cells

Once cells are detected, use `.parent()` to find which cell contains a text element:

```python
# Find text, then find its containing cell
text = page.find('text:contains("East Village")')
if text:
    cell = text.parent('region[type=form_cell]', mode='center')
    if cell:
        print(cell.extract_text())
```

## Debugging

Pass `debug_dir` to save step-by-step images of the detection pipeline:

```python
cells = page.detect_form_cells(
    merge=True,
    debug_dir="temp/form_cell_debug"
)
```

This saves images for each stage:

| File | Contents |
|------|----------|
| `00_rendered.png` | Page rendered at detection resolution |
| `01_gray.png` | Grayscale conversion |
| `02_binary.png` | Adaptive threshold result |
| `03a-e` | Horizontal line detection steps |
| `04a-e` | Vertical line detection steps |
| `05_line_bboxes.png` | Detected line bounding boxes (red=horizontal, blue=vertical) |
| `06_grid_mask.png` | Extended lines forming the grid |
| `07_inverse.png` | Inverted grid showing cell candidates |
| `08_cells_pre_merge.png` | Cells before OCR merge (colored) |
| `08b_cells_with_text.png` | Cells + text bounding boxes (green=cells, red=text) |
| `09_cells_final.png` | Final cells after merge (colored) |

Use `08b_cells_with_text.png` to understand merge behavior — it shows which text bboxes cross cell boundaries.

## Parameters

Override detection parameters via keyword arguments:

```python
cells = page.detect_form_cells(
    resolution=2200,      # target image width in pixels (default: 2200)
    merge=True,           # merge cells spanning same text (default: True)
    adaptive_c=5,         # adaptive threshold constant (default: 5, lower = more aggressive)
    min_cell_w=15,        # minimum cell width in pixels
    min_cell_h=15,        # minimum cell height in pixels
    max_aspect=50,        # maximum width/height ratio
)
```

## Working with Regions and Collections

`detect_form_cells()` is available on pages, regions, and collections:

```python
# Single page
cells = page.detect_form_cells()

# Region — detects on the full page, filters to cells inside the region
region = page.find('text:contains("VICTIM")').below()
victim_cells = region.detect_form_cells()

# All pages
all_cells = pdf.pages.detect_form_cells()
```
