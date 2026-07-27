---
fixture: https://github.com/jsoma/natural-pdf/raw/main/pdfs/use-of-force-raw.pdf
tier: nightly
thumbnail: 1
---

# Use-of-force records in six-point type

This PDF contains Vancouver Police use-of-force incident records, released to journalists after a public records request. The catch: the font is tiny, the rows are mostly empty space, and there isn't a single ruling line — so table detectors see a haze of sparse text instead of columns.

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/main/pdfs/use-of-force-raw.pdf")
page = pdf.pages[0]
page.show()
```

## Find the column headers

The headers are the text at the very top of the page, which means they have the smallest `y0`. `min()` inside an attribute filter finds them without hardcoding a coordinate:

```python
headers = page.find_all('text[y0=min()]')
headers.extract_each_text()
```

## Fit guides between the columns

`from_headers()` drops a vertical boundary between each pair of headers, nudging each divider to where it crosses the least text below:

```python
from natural_pdf.guides import Guides

guides = Guides(page)
guides.vertical.from_headers(headers)
guides.show()
```

With the columns established, extract. Rows are worked out from the text layout behind the scenes:

```python
guides.extract_table().to_df()
```

## Combine the results for every page

Give `extract_table()` a list of pages and the same guides are applied to each one. Repeated column headers on later pages are removed automatically:

```python
df = guides.extract_table(pdf.pages).to_df()
print("You found", len(df), "rows")

df.tail()
```

Four pages, one DataFrame. If a later page shifted its columns (it happens with re-run report exports), build a fresh `Guides` per page from that page's own header row instead of reusing one set of x-positions.
