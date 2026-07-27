---
fixture: https://github.com/jsoma/natural-pdf/raw/main/pdfs/24480polcompleted.pdf
tier: nightly
thumbnail: 7
---

# A 43-page borderless call log (with animals)

This PDF is a service call report covering 911 incidents at the Rainforest Cafe in Niagara Falls, NY — we're hunting for the animal calls. The data is a spreadsheet printed to PDF: wildly varied column widths, no cell borders anywhere, whole columns that sit empty for pages at a time, and a stray "2770 Records Found" line waiting to land in the middle of your data.

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/main/pdfs/24480polcompleted.pdf")
pdf.show(cols=3, limit=9)
```

## Select the spreadsheet pages

The first four pages are cover letters. The table starts on page 5:

```python
pages = pdf.pages[4:]
pages.show(cols=6)
```

## Exclude the record count

The last page ends with "2770 Records Found" — a line that would otherwise be swallowed into the table as a bogus row:

```python
pages[-1].show(crop=200)
```

Instead of matching the exact string, match the *shape* of it with a regex — then the same code works next year when the count is different:

```python
(
  pages[-1]
  .find_all(r'text:regex(\d+ Records Found)')
  .show(crop=100)
)
```

Anything you can find, you can exclude. Two options, depending on how much you trust the document:

/// tab | Exclude on the last page

The count is always on the last page, so a selector exclusion on that one page is enough:

```python
pages[-1].add_exclusion(r'text:regex(\d+ Records Found)')
```

///

/// tab | Exclude on all possible pages

If you're not sure the count only appears at the end — or you'll be feeding in other documents like this one — register it on the PDF, and every page filters it:

```python
pdf.add_exclusion(
  lambda page: page.find_all(r'text:regex(\d+ Records Found)')
)
```

///

Record counts: excluded.

```python
pages[-1].show(exclusions='red')
```

## Look at the table structure

```python
pages[0].show()
```

No borders, so we'll draw our own with **guides**:

- Vertical boundaries at the start of each column header, reused on every page.
- Horizontal boundaries wherever text starts with `NF-` — every row begins with a case number like `NF-00051026-24`, so multi-line rows can't split into two records.

Why not just `extract_table()` per page and concatenate? Because if an entire column is empty on some page, plain extraction has no way to know a column is missing there — a grid built from the headers does.

/// tab | Reusable guides

Vertical guides come from the header text. The last column ("Main Officer") has no boundary to its right, so `outer="last"` counts the space after the final guide as a column:

```python
from natural_pdf.guides import Guides

guide = Guides(pages[0])
columns = ['Number', 'Date Occurred', 'Time Occurred', 'Location', 'Call Type', 'Description', 'Disposition', 'Main Officer']
guide.vertical.from_content(columns, outer="last")
guide.horizontal.from_content(
  lambda p: p.find_all('text:starts-with(NF-)')
)
guide.show()
```

The **lambda** matters: it means "find the `NF-` rows on *whatever page this guide is applied to*," not just the first page.

Now apply the guide to every page in one call. The headers only exist on page one, so `header="first"`:

```python
table_result = guide.extract_table(pages, header="first")
df = table_result.to_df()
df.head()
```

///

/// tab | Guides with loops

The same thing, spelled out. Build the base grid from the first page:

```python
from natural_pdf.guides import Guides

base = Guides(pages[0])
columns = ['Number', 'Date Occurred', 'Time Occurred', 'Location', 'Call Type', 'Description', 'Disposition', 'Main Officer']
base.vertical.from_content(columns, outer="last")
base.horizontal.from_content(pages[0].find_all('text:starts-with(NF-)'))
base.show()
```

Extract the first page — the only one with headers:

```python
first_table = base.extract_table().to_df()
first_table.head()
```

Then walk the remaining pages, reusing the column positions but recomputing the rows from each page's own `NF-` anchors. `to_df(header=columns)` sets the headers manually so the frames can stack:

```python
dataframes = [first_table]

for page in pages[1:]:
    guides = Guides(page)
    guides.vertical = base.vertical
    guides.horizontal.from_content(page.find_all('text:starts-with(NF-)'))
    single_df = guides.extract_table().to_df(header=columns)
    dataframes.append(single_df)
print("We made", len(dataframes), "dataframes")
```

```python
import pandas as pd

df = pd.concat(dataframes, ignore_index=True)
df.head()
```

///

From here the animal hunt is a pandas filter on `Call Type` — the PDF part is done.
