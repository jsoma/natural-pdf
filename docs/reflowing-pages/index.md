# Restructuring page content

Flows are a way to restructure pages that are not in normal one-page reading order. This might be columnal data, tables than span pages, etc.

## A multi-column PDF

Here is a multi column PDF.

```python
from natural_pdf import PDF
from natural_pdf.flows import Flow

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/multicolumn.pdf")
page = pdf.pages[0]
page.show(width=500)
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

TK
