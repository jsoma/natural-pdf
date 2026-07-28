---
fixture: pdfs/Atlanta_Public_Schools_GA_sample.pdf
thumbnail: 4
tier: fast
---

# Putting it together

This is the capstone: one real document, no models, and every move comes from the earlier pages. The document is Atlanta Public Schools' library weeding log — the running record of every book pulled from a school shelf, with the title, school, barcode, and dates. Look at the fine print at the bottom of the page: this sample is 5 pages of a report whose footer says *Total pages: 23,133*. Nobody reads that by hand. The goal is a DataFrame: one row per removed book.

```python
from natural_pdf import PDF

pdf = PDF("pdfs/Atlanta_Public_Schools_GA_sample.pdf")
page = pdf.pages[0]
page.show()
```

The layout, from the top: a letterhead, then date headers ("6/12/2023 - Copies Removed: 2"), and under each date a stack of book records — bold title line, an Author/ISBN/Published line, then a little Site/Barcode/Price/Acquired/Removed-By block.

## Clear the furniture first

The letterhead repeats on every page, and so does a footer ("Report generated on...") — pure noise for extraction, and worse than noise: the *last* record on each page runs to the bottom of the page, so the footer would end up inside it. Register both zones on the **PDF**, as functions — every page evaluates them for itself, exactly like the end of the first Learn page:

```python
pdf.add_exclusion(lambda page: page.find('line[width>=2]').above())
pdf.add_exclusion(lambda page: page.find_all('line')[-1].below())
page.show(exclusions='red')
```

Everything above the thick rule and below the last horizontal line is now invisible to every extraction that follows, on every page.

## Find the spine of the document

Each record starts with a bold title. What selector isolates them? Ask the page:

```python
page.find_all('text').inspect(limit=8)
```

Read the columns: the date header is 12 pt, titles are 10 pt, but the Author/ISBN lines are *also* 10 pt. Size alone won't cut it. The tell is `font_variant` — titles and labels are variant `AAAAAB` (the bold face), field text is `AAAAAD`. Titles are the only 10 pt text in the bold variant:

```python
(len(page.find_all('text[size=10]')),
 len(page.find_all('text[font_variant=AAAAAB][size=10]')))
```

76 candidates by size, 7 by size plus variant — and this page visibly has 7 books. `font_variant` codes are meaningless strings (subset font tags), but they're *consistent* within a document, which is all a selector needs:

```python
TITLE = 'text[font_variant=AAAAAB][size=10]'

titles = page.find_all(TITLE)
titles.show()
```

## One region per record

Each record is "everything from its title down to the next title." That's `.below(until=...)` — applied to the whole collection at once, producing one region per title. `include_source=True` keeps the title line inside its own region:

```python
books = titles.below(until=TITLE, include_endpoint=False, include_source=True)
books.show()
```

Seven bands, one per book. This is the segmentation that makes everything else easy: any `find` on `books` runs once *per region* and gives back one result per record, in order.

## Harvest the fields

**Whole-line fields.** The author line is inside each record and starts with "Author:" — `find` on the collection returns one element per record:

```python
books.find('text:contains("Author")').extract_each_text()
```

**The school name — and a trap.** The site value sits *below* the "Site" label. Try the obvious move and look closely at the first result:

```python
naive = (
    books
    .find('text:contains("Site")')
    .below()
    .apply(lambda region: region.find_all('text[x0<47][size=10]').extract_text())
)
naive[0][:120]
```

The first "school name" contains every school and title on the rest of the page. `.below()` doesn't know where the record ends — it ran to the bottom of the page, straight through the next six records. `.clip(books)` cuts each region back to its own record's band:

```python
sites = (
    books
    .find('text:contains("Site")')
    .below()
    .clip(books)
    .apply(lambda region: region.find_all('text[x0<47][size=10]').extract_text())
)
sites
```

(The `x0<47` filter is from `inspect()` too: within the site block, the school name is the only 10 pt text starting left of the "Site" label, which sits at x0=47. School names wrap to two lines, which is why this one is an `apply` + `extract_text()` — it folds the wrapped lines into one string.)

**Column values under a label.** Barcode values sit directly under the "Barcode" label but stick out wider than it. Take a thin region below each label and widen it:

```python
barcodes = (
    books
    .find('text:contains("Barcode")')
    .below(width='element', height=12)
    .expand(left=10, right=50)
    .extract_each_text()
)
barcodes
```

Without the `expand`, the first digit falls outside the label's width and every barcode comes back one digit short — off-by-a-few-points errors like that are exactly why you spot-check extracted values against the page image.

**Looking *up* instead of down.** Which date batch does each book belong to? The date header is above the record — possibly far above, with other records in between. `.above(until=...)` climbs from each record to the first 12 pt-or-larger text, and `.endpoints` collects the elements it stopped at:

```python
dates = books.above(until='text[size>10]').endpoints
dates.show()
```

```python
dates.extract_each_text()
```

Three date headers on the page, and each of the seven records found its own — records 1–2 map to 6/12, 3–4 to 6/7, the rest to 6/6.

## The DataFrame

Every column is one chained expression. Line them up:

```python
import pandas as pd

df = pd.DataFrame({
    'title': titles.extract_each_text(),
    'author': books.find('text:contains("Author")').extract_each_text(),
    'site': sites,
    'barcode': barcodes,
    'removed': dates.extract_each_text(),
})
df
```

One row per book, straight from selectors and spatial navigation. From here it's ordinary pandas, so ordinary cleanup applies — strip the label prefixes and the "(Removed: 1)" suffixes, and split the batch date off its header:

```python
df['title'] = df['title'].str.replace(r'\s*\(Removed: \d+\)$', '', regex=True)
df['author'] = df['author'].str.replace('Author: ', '', regex=False)
df['removed_on'] = df['removed'].str.split(' - ').str[0]
df.drop(columns='removed')
```

## All five pages

Nothing above referenced page 1 specifically — the selectors describe *any* page of this report. Wrap the recipe in a function and concatenate:

```python
def records_from(page):
    titles = page.find_all(TITLE)
    if not titles:
        return None
    books = titles.below(until=TITLE, include_endpoint=False, include_source=True)
    return pd.DataFrame({
        'title': titles.extract_each_text(),
        'author': books.find('text:contains("Author")').extract_each_text(),
        'site': (books.find('text:contains("Site")').below().clip(books)
                 .apply(lambda r: r.find_all('text[x0<47][size=10]').extract_text())),
        'barcode': (books.find('text:contains("Barcode")')
                    .below(width='element', height=12).expand(left=10, right=50)
                    .extract_each_text()),
        'removed': books.above(until='text[size>10]').endpoints.extract_each_text(),
    })

frames = [records_from(p) for p in pdf.pages]
all_books = pd.concat([f for f in frames if f is not None], ignore_index=True)
len(all_books)
```

```python
all_books['site'].value_counts()
```

37 books across 5 pages, and the site counts give the story its first shape — which schools were shedding books that week. Notice the blank entry with a count of 2: two records came back with no site at all. Before blaming the code, check the page — and in fact those two records (one on page 2, one on page 4) genuinely have no Site block in the PDF. Missing on the page became empty in the DataFrame, which is the behavior you want; an extraction that quietly *invented* a school would be far worse. This check — every oddity in the output traced back to the page image — is the habit that makes extraction trustworthy.

## The batch loop

The real version of this project is many PDFs — the next FOIA batch, the next district, the next year. The pattern: open, register exclusions, extract, **close**. `close()` releases the file handle; skip it in a loop over two hundred PDFs and you'll find out why it matters:

```python
sources = {
    'atlanta': "pdfs/Atlanta_Public_Schools_GA_sample.pdf",
    # next district's log goes here
}

collected = []
for district, path in sources.items():
    pdf = PDF(path)
    try:
        pdf.add_exclusion(lambda page: page.find('line[width>=2]').above())
        pdf.add_exclusion(lambda page: page.find_all('line')[-1].below())
        frames = [records_from(p) for p in pdf.pages]
        district_df = pd.concat([f for f in frames if f is not None], ignore_index=True)
        district_df['district'] = district
        collected.append(district_df)
    finally:
        pdf.close()

combined = pd.concat(collected, ignore_index=True)
combined['district'].value_counts()
```

`try`/`finally` guarantees the close even when one malformed PDF throws halfway through — and in a two-hundred-file batch, one always does. From here, `combined.to_csv("weeded_books.csv")` and you're in spreadsheet land.

## Where to go from here

You've now seen the whole toolkit run end to end: selectors and `inspect()` to find anchors, spatial navigation to turn anchors into fields, exclusions to silence page furniture, collections to do it per-record, and pandas to finish the job. Two directions from here:

- **Solve** — the task-oriented section of these docs: short recipes for specific jobs (redactions, checkboxes, multi-page tables, searchable PDFs) that assume you know everything this track just covered.
- **Reference** — the lookup pages: [every selector and pseudo-class](../reference/selectors.md), [engines and their trade-offs](../reference/engines.md), [install extras](../reference/installation-extras.md), and [OCR options](../reference/ocr-options.md).

And when a document fights back — garbled text, no text, tables that won't line up — you now know which page of this track to reread.
