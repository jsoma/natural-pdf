---
fixture: pdfs/01-practice.pdf
thumbnail: 1
---

# Text and tables

You have a stack of inspection reports and you need what's inside them: who got inspected, when, how many violations, and the table listing each one. This page works through a single report — a fake slaughterhouse inspection borrowed from *The Jungle* — but every move on it is the same move you'd make on a real filing.

## Open the PDF

`PDF(...)` takes a file path, a URL, or raw bytes. `.pages` is the list of pages; grab the first one and look at it.

```python
from natural_pdf import PDF

pdf = PDF("pdfs/01-practice.pdf")
page = pdf.pages[0]
page.show()
```

`.show()` returns a PIL image, so `page.show().save("page.png")` writes it to disk if you need the file.

## Grab the text

Most of the time you want the words. `extract_text()` gives you all of them.

```python
text = page.extract_text()
print(text)
```

Everything is there, but it's a stream — labels, values, table cells, and the footnote all run together. If you want a version that keeps the visual arrangement, pass `layout=True`:

```python
print(page.extract_text(layout=True))
```

That's readable, but "readable" isn't "extracted." To pull out *specific* pieces — the date, the site, the violation count — you need a way to point at them.

## Describing what you want

Look at the inspection ID, **INS-UP70N51NCL41R** (squint: it spells UPTON SINCLAIR). How would you describe it to someone over the phone?

- "It's in a box"
- "It's the second piece of text on the page"
- "It's the red text"
- "It starts with INS-"

Each of those descriptions is a selector. All four roads lead to the same element.

### "It's in a box"

The box is a rectangle element. `page.find()` takes a CSS-like selector and returns the first match — or `None` if nothing matches, so check before chaining onto the result.

```python
page.find('rect').show(crop=50)
```

`crop=50` renders just the element plus a margin of context instead of the whole page — bigger number, more surroundings.

### "It's the second piece of text"

`find_all()` returns every match as a collection.

```python
page.find_all('text').show()
```

`extract_each_text()` gives you one string per element, in reading order:

```python
texts = page.find_all('text').extract_each_text()
texts[:5]
```

So the second piece of text is:

```python
texts[1]
```

Counting positions works until a report adds a line and everything shifts by one. The next two descriptions hold up better.

### "It's the red text"

Square brackets filter by attributes. `color~=` is an approximate match — it tolerates the almost-but-not-quite-red that PDFs actually use.

```python
ins_id = page.find('text[color~=red]')
ins_id.show(crop=50)
```

Same box, different road — this time the highlight is on the text element itself, which knows its own content:

```python
ins_id.extract_text()
```

The same trick works for other colors — "Chicago, Ill." is the grey text:

```python
page.find('text[color~=grey]')
```

That's the element's repr, not its text — call `.extract_text()` when you want the string.

### "It starts with INS-"

Pseudo-classes match on content: `:contains("INS-")` matches anywhere in the text, `:starts-with("INS-")` anchors to the front.

```python
code = page.find('text:starts-with("INS-")')
code.show(crop=20)
```

```python
code.extract_text()
```

## Learning what's on the page

Those selectors assumed you already knew the ID was red and 8 pt. On an unfamiliar PDF, ask the page to describe itself:

```python
page.describe()
```

44 words, 21 lines, 8 rects, one font. For element-by-element detail, `inspect()` prints a table of every element and its attributes — `limit` caps the rows:

```python
page.find_all('text').inspect(limit=8)
```

Those `size`, `font_family`, and `color` columns are exactly what goes inside the square brackets. `max()` and `min()` work in attribute filters too — here's the largest text on the page, no hardcoded size:

```python
page.find_all('text[size=max()]').show(crop=50)
```

That's the bold 12 pt "Violations" heading. Hold that thought — it becomes the anchor for the table later.

## Spatial navigation

Labels like "Date:" and "Site:" are easy to find, but you want the value *next to* the label. Every element can look `.right()`, `.left()`, `.above()`, and `.below()`, and each returns a Region — a rectangle of page you can show, search, or extract from.

### The date

```python
page.find(text="Date").right().show(crop=50)
```

`.right()` defaults to the height of the element itself, so the region is just that row. Extract it:

```python
page.find(text="Date").right().extract_text()
```

### The site

Same move on "Site:" — but this row has two things in it:

```python
page.find(text="Site").right().extract_text()
```

The site name *and* the grey city ran together, because `.right()` sweeps everything to the edge of the page. To stop early, pass `until=` — the region runs up to and including the first element matching that selector, and `.endpoint` is the element it stopped at:

```python
site = page.find(text="Site").right(until='text')
site.show(crop=50)
```

```python
site.endpoint.extract_text()
```

### The violation count

Same move as the date — no surprises this time:

```python
page.find(text="Violation Count").right().extract_text()
```

### The summary

Try the same move on "Summary:" and watch it go wrong:

```python
page.find(text="Summary").right().extract_text()
```

That's one line of a paragraph that runs seven. `.right()` stayed in the label's row, but the summary wraps. What you actually want is everything *below* the label, down to the horizontal rule that closes the section. `.below()` defaults to the full page width, `until='line'` stops at the rule, and `include_source=True` keeps the label's own row in the region:

```python
summary = page.find(text="Summary").below(until='line', include_source=True)
summary.show(crop=True)
```

```python
summary.extract_text(newlines=False)
```

`newlines=False` folds the wrapped lines back into one string.

## Tables

`page.extract_table()` finds the table on the page and returns a `TableResult`:

```python
page.extract_table()
```

This page has exactly one table, so grabbing it page-wide works. On a page with several tables — or with headers that confuse the detector — the reliable move is to describe the region the table lives in and extract from *that*.

/// tab | Whole page

`TableResult.to_df()` hands you a pandas DataFrame:

```python
page.extract_table().to_df()
```

///

/// tab | Scoped to a region

Anchor on the "Violations" heading (the 12 pt bold text from earlier), take everything below it until the fine print, and trim the whitespace:

```python
violations = (
    page
    .find('text[size=max()]:bold:contains("Violations")')
    .below(until='text[size=min()]', include_endpoint=False)
    .trim()
)
violations.show(crop=True)
```

The region *is* the table — so extracting from it can't pick up anything else:

```python
violations.extract_table().to_df()
```

///

If `extract_table()` comes back empty or scrambled, scoping to a region like this is the first thing to try.

One honest gap in both versions: the Repeat? column is all `<NA>`. Look at the page — those cells are drawn checkboxes, not text, so text extraction has nothing to read there. Getting checkbox states out is a separate, model-backed step (`detect_checkboxes()`), which is out of scope for this page.

## Ignoring content with exclusion zones

Now imagine two hundred of these reports, and all you want is the text-y top half — no letterhead, no table, no footnote. Instead of describing what you want, describe what you *don't* want.

`page.region()` cuts a rectangle by coordinates; spatial navigation builds the other zone from the thick rule above the table:

```python
letterhead = page.region(top=0, left=0, height=80)
below_the_rule = page.find('line[width>=2]').below()
(letterhead + below_the_rule).show()
```

Register both as exclusions. `page.show(exclusions='red')` confirms what's being blocked:

```python
page.add_exclusion(letterhead)
page.add_exclusion(below_the_rule)
page.show(exclusions='red')
```

Exclusions don't delete anything — they filter what read operations return. The same `extract_text()` from the top of this page now skips both zones:

```python
print(page.extract_text())
```

For a whole stack of reports, register the exclusions once on the PDF as functions — each page evaluates them when it loads, and `add_exclusion` hands the PDF back so registrations can chain:

```python
pdf.add_exclusion(lambda page: page.region(top=0, left=0, height=80))
pdf.add_exclusion(lambda page: page.find('line[width>=2]').below())
```

Headers, footers, page-number stamps, "DRAFT" watermarks — anything that repeats across a filing is a candidate for an exclusion instead of a workaround in every extraction.

## What you can do now

Open a PDF, dump its text, select elements by type, attribute, and content, navigate from labels to values, pull a table into a DataFrame, and blank out the parts you never want to see again. All of it assumed the PDF has real text underneath. When it doesn't — when the page is a scan — that's what OCR is for, and that's the next page.
