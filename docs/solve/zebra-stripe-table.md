---
fixture: https://github.com/jsoma/natural-pdf/raw/main/pdfs/m27.pdf
tier: nightly
thumbnail: 1
---

# Zebra-stripe tables with no ruling lines

This PDF lists alcohol licensees in Oklahoma. The table has no drawn cell borders at all — rows are separated by alternating background colors ("zebra stripes"), cells wrap onto multiple lines, and the columns sit close enough together that the automatic table detector merges them. A bare `page.extract_table()` comes back scrambled.

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/main/pdfs/m27.pdf")
page = pdf.pages[0]
page.show()
```

## Exclude the header and footer

First, think about what you *don't* want: the letterhead above the column headers, and the "Page N of M" stamp at the bottom.

```python
header = page.find(text="PREMISE").above()
footer = page.find(r"text:regex(Page \d+ of)")
(header + footer).show()
```

Registering exclusions on the *PDF* with a `lambda page:` makes them apply to every page, not just this one:

```python
print("Before exclusions:", page.extract_text()[:200])

pdf.add_exclusion(lambda page: page.find(text="PREMISE").above())
pdf.add_exclusion(lambda page: page.find(r"text:regex(Page \d+ of)").expand())

print("After exclusions:", page.extract_text()[:200])

page.show(exclusions='red')
```

## Build the grid with guides

Since there are no lines to detect, we draw our own with `Guides` — explicit column and row boundaries that `extract_table()` will respect.

```python
from natural_pdf.guides import Guides

guides = Guides(page)
```

### Columns from the headers

The column headers are all left-aligned, which is exactly what a guide can lock onto. Grab the "NUMBER" header and everything to its right:

```python
region = (
    page
    .find(text="NUMBER")
    .right(include_source=True)
)
region.show(crop=100)
```

The headers aren't *perfectly* aligned on one baseline, so give the row 3px of wiggle room before collecting the text elements:

```python
headers = (
    page
    .find(text="NUMBER")
    .right(include_source=True)
    .expand(top=3, bottom=3)
    .find_all('text')
)
headers.show(crop=100)
```

Every header is left-aligned, so drop a vertical guide at the left edge of each one:

```python
guides.vertical.from_content(headers, align='left')
guides.show()
```

### Rows — two ways

Now the rows. Two options, and the zebra stripes that made this PDF annoying turn out to be the first one.

/// tab | By zebra stripes

Rows have alternating blue bands behind them. `horizontal.from_stripes()` runs a two-step process:

1. Find the most common color of rectangle on the page
2. Add guides at the top and bottom of each one

You can pass `color=` or hand it the rectangles yourself, but here the defaults work:

```python
guides.horizontal.from_stripes()
guides.show()
```

///

/// tab | By license number

If the stripes were unreliable — scanned pages lose them all the time — anchor on something that appears exactly once per row: the license number. Draw down from the **NUMBER** header:

```python
(
    page
    .find(text="NUMBER")
    .below(width='element')
).show(crop=100, width=700)
```

Collect the numbers underneath it. `overlap='partial'` is needed because the header column doesn't fully cover each license number:

```python
rows = (
    page
    .find(text="NUMBER")
    .below(
      width='element',
      include_source=True
    )
    .find_all('text', overlap='partial')
)
rows.show(crop=100, width=700)
```

Feed each row to `from_content` and draw a boundary at the bottom of each one:

```python
guides.horizontal.from_content(rows, align='bottom')
guides.show()
```

///

## Extract

`include_outer_boundaries=True` closes the grid at the page edges so both approaches above produce a complete table (one of them gives you an extra column of margin, which is easy to drop):

```python
df = (
  guides
  .extract_table(include_outer_boundaries=True)
  .to_df()
)
df.head()
```

The multi-line cells come through intact because the row guides — not line breaks — decide where one record ends and the next begins.
