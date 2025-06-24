# Restructuring page content

Flows are a way to restructure pages that are not in normal one-page reading order. This might be columnal data, tables than span pages, etc.

## A multi-column PDF

Here is a multi column PDF.

```python
from natural_pdf import PDF
from natural_pdf.flows import Flow

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/multicolumn.pdf")
page = pdf.pages[0]
page.to_image(width=500)
```

We can grab individual columns from it.

```python
left = page.region(right=page.width/3)
mid = page.region(left=page.width/3, right=page.width/3*2)
right = page.region(left=page.width/3*2)

mid.show(width=500)
```

## Restructuring

We can use Flows to stack the three columns on top of each other.

```python
stacked = [left, mid, right]
flow = Flow(segments=stacked, arrangement="vertical")
```

As a result, I can find text in the first column and ask it to grab what's "below" until it hits content in the second column.

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

While you can't easily extract tables yet, you can at least extract text!

```python
print(region.extract_text())
```

## find_all and reflows

Let's say we have a few headers...

```python
(
    flow
    .find_all('text[width>10]:bold')
    .show()
)
```

...it's easy to extract each table that's betwen them.

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

## Merging tables that span pages

Flows can also stitch **multi-page** content together. One pattern is a table that repeats
its header on every page – you want one continuous DataFrame.

```python
from natural_pdf import PDF
from natural_pdf.flows import Flow
import pandas as pd

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/long_table_across_pages.pdf")

# 1️⃣ Build one region per page that contains only the body of the table
def body_region(p):
    header = p.find('text:bold').first()               # first bold header row
    return header.below(until='rect[height<=1]')       # until thin footer rule

segments = pdf.pages.apply(body_region)

# 2️⃣ Stack them vertically so they behave like *one* long page
flow = Flow(segments, arrangement="vertical")

# 3️⃣ Extract once – header row from the first segment, everything else is data
raw = flow.extract_table()
header, *rows = raw
df = pd.DataFrame(rows, columns=header)
print(df.shape)
```

The key trick is that **exclusions are inherited**: if you called
`pdf.add_exclusion(...)` to wipe headers/footers they do not appear in any
segment, so the Flow sees only clean table rows.  No extra coordinates needed.