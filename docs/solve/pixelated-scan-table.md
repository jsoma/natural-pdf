---
fixture: https://pub-4e99d31d19cb404d8d4f5f7efa51ef6e.r2.dev/pdfs/statecallcenterdata_redacted/statecallcenterdata_redacted.pdf
tier: nightly
thumbnail: 1
---

# A pixelated FOIA scan with an invisible table

This PDF holds wait-time data for a state agency call center, released through a public records request. It's a heavily pixelated scan — no text layer at all — and the table on page one has ruling lines the scanner half-dissolved. Reading the numbers is hard for a human; for a machine it takes OCR plus hand-drawn column boundaries.

One wrinkle before the PDF itself: this host rejects downloads from Python's built-in `urllib` (a plain `PDF(url)` gets a 403), so fetch the bytes with `requests` and hand them over — `PDF()` accepts a path, a URL, or any file-like object.

```python
from io import BytesIO

import requests
from natural_pdf import PDF

url = "https://pub-4e99d31d19cb404d8d4f5f7efa51ef6e.r2.dev/pdfs/statecallcenterdata_redacted/statecallcenterdata_redacted.pdf"
pdf = PDF(BytesIO(requests.get(url).content))
page = pdf.pages[0]
page.show()
```

The pages are images, so there's no text to extract — always worth double-checking before blaming your selectors:

```python
# Empty? Needs OCR.
print(repr(page.extract_text()))
```

## Apply OCR

`apply_ocr()` uses RapidOCR by default — its models ship inside the package, so there's no download on first use. Two ways to check the results: look at *where* it found text, and look at *what* the text says.

```python
page.apply_ocr()
page.find_all('text').show(crop=True)
```

```python
print(page.extract_text(layout=True))
```

Legible, with the usual scan-quality noise (`interriew.n.minutes`). The structure survived, and that's what the extraction below leans on.

## Isolate the table area

The table runs from the "Figure" header down to the "Please use the comments field" instruction:

```python
table_area = (
    page
    .find('text:contains(Figure)')
    .below(
        until='text:contains(Please use the comments)',
        include_endpoint=False
    )
)
table_area.show(crop='wide')
```

That's the right rows but the full page width. Cut in from the right (the table only occupies the left 40% of the page), trim the left margin, and add a hair at the bottom. These are hand-picked values — with OCR boxes there's often no cleaner anchor to snap to:

```python
table_area = (
    page
    .find('text:contains(Figure)')
    .below(
        until='text:contains(Please use the comments)',
        include_endpoint=False
    )
    .expand(
        right=-(page.width * 0.58),
        left=-30,
        bottom=3
    )
)
table_area.show(crop='wide')
```

Confirm the area holds the right text elements:

```python
table_area.find_all('text').show(crop=True)
```

## Draw the grid

`extract_table()` alone can't split these columns — the gaps between them are too inconsistent after OCR. So drop three vertical dividers, then shuffle them into the whitespace so they don't cut through any text. The rows are easier: the scan still has enough of its ruling lines for pixel detection to find them.

```python
from natural_pdf.guides import Guides

guide = Guides(table_area)
guide.vertical.divide(3)
guide.vertical.snap_to_whitespace(detection_method='text')
guide.horizontal.from_lines(detection_method='pixels')
guide.show()
```

And now the table comes out as data:

```python
df = (
  guide
  .extract_table()
  .to_df(
    header=['value', 'amount', 'comments']
  )
)
df
```

The numbers carry OCR artifacts (`19Minutes`, merged words) — that's cleanup work for pandas, not a reason to re-run extraction. If a *value* is unreadable, `page.compare_ocr(engines=[...])` shows what other engines make of the same crop.

The later pages of this PDF are a different beast — multi-year pivot grids at the same scan quality:

```python
pdf.pages[1].show()
```

That one needs its own strategy (and possibly a better copy of the document).
