---
fixture: https://github.com/jsoma/natural-pdf/raw/main/pdfs/multicolumn.pdf
tier: nightly
thumbnail: 1
---

# Multi-column pages that read in the wrong order

Sometimes you have data that flows over multiple columns, or pages, or just isn't arranged in a "normal" top-to-bottom way. This page has three columns of text with two tables buried in them — and a plain `extract_text()` walks straight across all three columns at once, interleaving sentences that have nothing to do with each other.

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/main/pdfs/multicolumn.pdf")
page = pdf.pages[0]
page.show()
```

Look at the first few hundred characters of a naive extraction — each line jumps from column to column:

```python
print(page.extract_text(layout=True)[:500])
```

The fix is to *reflow* the page: cut out each column as a region, then paste the pieces back together end-to-end so "below" means "further along in reading order."

## Cut the page into columns

`page.region()` cuts a rectangle by coordinates. Three vertical slices, each a third of the page wide:

```python
left = page.region(left=0, right=page.width/3, top=0, bottom=page.height)
mid = page.region(left=page.width/3, right=page.width/3*2, top=0, bottom=page.height)
right = page.region(left=page.width/3*2, right=page.width, top=0, bottom=page.height)
page.highlight(left, mid, right)
```

## Stack them with a Flow

A `Flow` connects separate regions (or whole pages) vertically or horizontally. Stacking the three columns vertically turns the page into one long single-column document:

```python
from natural_pdf.flows import Flow

stacked = [left, mid, right]
flow = Flow(segments=stacked, arrangement="vertical")
flow.show()
```

Now any spatial operation — "find something below this" — follows reading order, even when that means jumping from the bottom of one column to the top of the next:

```python
region = (
    flow
    .find('text:contains("Table one")')
    .below(
        until='text:contains("Table two")',
        include_endpoint=False
    )
)
region.show()
```

## Pull out every table

The tables sit under bold headers. Find the headers first — the `width > 10` filter skips some tiny empty boxes that also register as bold:

```python
flow.find_all('text[width>10]:bold').show()
```

Then take each header and sweep down until the *next* bold header or the closing paragraph, whichever comes first. The `|` in the selector means "either of these":

```python
regions = (
    flow
    .find_all('text[width>10]:bold')
    .below(
        until='text[width>10]:bold|text:contains("Here is a bit")',
        include_endpoint=False
    )
)
regions.show()
```

Each region is one table, so `extract_table()` on a region can't grab the wrong one:

```python
regions[0].extract_table().to_df()
```

And if the tables share a structure, combine them into one DataFrame:

```python
import pandas as pd

dfs = regions.apply(lambda region: region.extract_table().to_df())
merged = pd.concat(dfs, ignore_index=True)
merged
```

## If your columns aren't this tidy

This page splits into exact thirds, so hardcoded coordinates work. When column widths vary — or you can't hand-measure every document in a stack — `page.analyze_layout()` can detect text blocks for you (it downloads YOLO weights on first use), and the detected regions feed into a `Flow` the same way. The same `Flow` trick also spans *pages*: pass `pdf.pages` as the segments and a table that breaks across a page boundary becomes one table.
