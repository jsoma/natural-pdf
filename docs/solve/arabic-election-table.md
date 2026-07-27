---
fixture: https://pub-4e99d31d19cb404d8d4f5f7efa51ef6e.r2.dev/pdfs/mednine/mednine.pdf
tier: nightly
thumbnail: 1
---

# Arabic election results across four pages

This PDF holds election results from the Tunisian region of Mednine: one long table split over four pages, with spanning header cells, rotated headers, and — the real test — Arabic script, which reads right-to-left. Natural PDF handles RTL text ordering during extraction, so the words come out in logical order rather than reversed.

```python
from io import BytesIO

import requests
from natural_pdf import PDF

url = "https://pub-4e99d31d19cb404d8d4f5f7efa51ef6e.r2.dev/pdfs/mednine/mednine.pdf"
pdf = PDF(BytesIO(requests.get(url).content))
pdf.show(cols=3)
```

One wrinkle before the PDF itself: this host rejects downloads from Python's built-in `urllib` (a plain `PDF(url)` gets a 403), so fetch the bytes with `requests` and hand them over — `PDF()` accepts a path, a URL, or any file-like object.

Since the PDF is all one big table with nothing else in the way, the whole job is: treat four pages as one page, extract one table, clean up in pandas.

/// tab | Multi-page flows

## Build a flow

A `Flow` connects pages (or regions) vertically or horizontally. Stack all four pages on top of each other:

```python
from natural_pdf.flows import Flow

flow = Flow(pdf.pages, arrangement='vertical')
flow.show(width=300)
```

## Extract the table

With a flow, `extract_table()` automatically combines the table across every segment:

```python
df = flow.extract_table().to_df(header=None)
df
```

///

/// tab | Manually combining dataframes

If you'd rather not use a Flow, go page by page. `.apply` keeps it shorter than a `for` loop (a list comprehension works too):

```python
import pandas as pd

dataframes = pdf.pages.apply(
    lambda page: page.extract_table().to_df(header=None)
)
print("Found", len(dataframes), "tables")

df = pd.concat(dataframes, ignore_index=True)
df
```

///

Take as much as possible, then clean it up later. We *could* spend time wrangling the spanning column headers on the first page, but it's faster to grab everything and sort it out in pandas.

## Clean up the data

The header rows landed in the data — rows 2 and 3 hold the two levels of the spanning headers. Merge them into column names, drop the header rows, and de-space the numbers:

```python
# Use row 3 as header, filling gaps from row 2
df.columns = df.iloc[3].fillna(df.iloc[2]).str.replace("\n", " ")

# Drop the header rows
df = df.iloc[4:].reset_index(drop=True)

# Remove spaces from numbers and convert to int
numeric_cols = df.columns[0:4]
df[numeric_cols] = df[numeric_cols].replace(r"\s+", "", regex=True).astype(int)
df
```

This kind of cleanup is an exercise in data wrangling more than PDF work — if the row indices differ on your document, print `df.head(6)` first and adjust which rows become the header.
