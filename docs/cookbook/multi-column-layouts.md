# Multi-Column Layouts with Flow

Academic papers, newspapers, and many reports use multi-column layouts. The `Flow` class helps you read content in the correct order by stacking columns vertically.

```python
#%pip install natural-pdf
```

## The Problem

When a document has multiple columns, `extract_text()` reads across the page width, mixing content from different columns:

```python
from natural_pdf import PDF

pdf = PDF("academic_paper.pdf")
page = pdf.pages[0]

# This reads left-to-right, mixing columns!
text = page.extract_text()
print(text[:500])  # Jumbled content
```

## Solution: Define Columns and Stack Them

Split the page into column regions and combine them with `Flow`:

```python
from natural_pdf import PDF
from natural_pdf.flows import Flow

pdf = PDF("academic_paper.pdf")
page = pdf.pages[0]

# Define the three columns
left = page.region(left=0, right=page.width/3, top=0, bottom=page.height)
mid = page.region(left=page.width/3, right=page.width/3*2, top=0, bottom=page.height)
right = page.region(left=page.width/3*2, right=page.width, top=0, bottom=page.height)

# Preview the column divisions
page.highlight(left, mid, right)
```

```python
# Stack columns into a vertical flow
stacked = [left, mid, right]
flow = Flow(segments=stacked, arrangement="vertical")

# Now text extraction reads in the correct order
flow.show()
```

## Finding Content Within Flows

Use `.find()` and `.find_all()` on flows just like pages:

```python
# Find a section header in the flow
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

## Extracting Multiple Tables

Find all tables within a multi-column document:

```python
# Find bold headers and get content below each
regions = (
    flow
    .find_all('text[width>10]:bold')
    .below(
        until='text[width>10]:bold|text:contains("Here is a bit")',
        include_endpoint=False
    )
)
regions.show()

# Extract the first table
regions[0].extract_table().to_df()
```

## Combining Data from Multiple Regions

Use `.apply()` to process each region and combine results:

```python
import pandas as pd

# Apply a function to each region in the collection
elements = flow.find_all('text:bold')
texts = elements.apply(lambda el: el.extract_text())

# Extract table from each region
dfs = regions.apply(lambda r: r.extract_table().to_df())

# Merge all tables into one DataFrame
merged = pd.concat(dfs, ignore_index=True)
merged
```

## Two-Column Layouts

For simpler two-column documents:

```python
from natural_pdf import PDF
from natural_pdf.flows import Flow

pdf = PDF("newsletter.pdf")
page = pdf.pages[0]

# Split into left and right columns
left = page.region(left=0, right=page.width/2)
right = page.region(left=page.width/2, right=page.width)

# Create flow
flow = Flow(segments=[left, right], arrangement="vertical")

# Extract text in reading order
text = flow.extract_text()
print(text)
```

## Detecting Columns Automatically

For documents where column boundaries aren't fixed, use whitespace detection:

```python
# Find vertical gaps that might indicate column boundaries
lines = page.find_all('line:vertical')
if lines:
    # Use detected lines as column dividers
    boundaries = [line.x0 for line in lines]
```

## Related Tutorials

- **[Multipage Content](multipage-content.md)** - Handle content spanning pages
- **[Spatial Navigation](../tutorials/08-spatial-navigation.ipynb)** - Navigate within regions
- **[Table Extraction](../tutorials/04-table-extraction.ipynb)** - Extract tables from flow regions
