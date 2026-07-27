---
fixture: https://pub-4e99d31d19cb404d8d4f5f7efa51ef6e.r2.dev/pdfs/k046682-111320-opa-lea-database-install_1/k046682-111320-opa-lea-database-install_1.pdf
tier: nightly
thumbnail: 1
---

# A relational database printed to PDF, with redactions

This PDF is a set of complaint records from a local law enforcement agency — a database report printed to paper. Each complaint is a form-like block holding one-to-many tables of complaints and officers, the layout repeats with fields that are sometimes empty, and black redaction boxes break the automatic column detection right where you need it most.

This host rejects downloads from Python's built-in `urllib` (a plain `PDF(url)` gets a 403), so fetch the bytes with `requests` and hand them to `PDF()` — it accepts any file-like object:

```python
from io import BytesIO

import requests
from natural_pdf import PDF

url = "https://pub-4e99d31d19cb404d8d4f5f7efa51ef6e.r2.dev/pdfs/k046682-111320-opa-lea-database-install_1/k046682-111320-opa-lea-database-install_1.pdf"
pdf = PDF(BytesIO(requests.get(url).content))
pdf.show(cols=3)
```

```python
page = pdf.pages[0]
page.show()
```

## Exclude the report chrome

Every page repeats the vendor footer and the report title. Register both as PDF-level exclusions so nothing downstream ever sees them:

```python
pdf.add_exclusion(lambda page: page.find(text='L.E.A. Data Technologies').below(include_source=True))
pdf.add_exclusion(lambda page: page.find(text='Complaints By Date').above(include_source=True))

page.show(exclusions='black')
```

## Break the document into one section per complaint

The colored bars look like the obvious anchors, but text is usually the sturdier choice. Every record has a "Recorded On Camera" line, so cut the whole PDF into sections there. `include_boundaries='start'` keeps the anchor line inside its section:

```python
sections = pdf.get_sections(
  'text:contains(Recorded)',
  include_boundaries='start'
)
sections.show(cols=3)
```

```python
section = sections[3]
section.show(crop=True)
```

Every section has the same skeleton, even when fields are blank — which turns the rest of this into "solve one section, then loop."

## Label-value pairs

For the fields up top, find the label and walk right until the next piece of text:

```python
complainant = (
  section
  .find("text:contains(Complainant)")
  .right(until='text')
)
print("Complainant is", complainant.extract_text())
complainant.show(crop=100)
```

Date of birth is missing in many records — but this report helpfully prints *empty text elements* in the blank slots, so `until='text'` still stops in the right place instead of running into the next column:

```python
dob = (
  section
  .find("text:contains(DOB)")
  .right(until='text')
)
print("DOB is", dob.extract_text())
dob.show(crop=100)
```

For labels whose value sits *underneath*, `.below(until='text')` stops at the first text it touches — but touching isn't containing. The region only overlaps the start of the case number, and extracting it would clip the value:

```python
number = (
    section
    .find("text:contains(Number)")
    .below(until='text', width='element')
)
print("Number is", number.extract_text())
number.show(crop=100)
```

Ask for the element that *partially* overlaps the region instead — that expands the grab to the whole number:

```python
number = (
    section
    .find("text:contains(Number)")
    .below(until='text', width='element')
    .find('text', overlap='partial')
)
print("Number is", number.extract_text())
number.show(crop=100)
```

"Date Assigned" needs the opposite discipline: the value you want is *fully* underneath the label, and the neighboring sergeant's name is close enough that `until='text'` would stop on it:

```python
(
  section
  .find('text:contains(Date Assigned)')
  .below(width='element')
  .show(crop=100)
)
```

`.find('text')` inside the region defaults to full containment — only elements entirely inside count:

```python
(
  section
  .find('text:contains(Date Assigned)')
  .below(width='element')
  .find('text')
  .extract_text()
)
```

Same three moves — right-until, below-with-partial, below-with-containment — cover every field on the form.

## The complaint table

The tables look like the hard part, but it's just: describe the area, extract. The complaint rows all start with "Complaint #", so the table is everything to the right of those labels:

```python
(
    section
    .find_all('text:contains(Complaint #)')
    .right(include_source=True)
    .show(crop=section)
)
```

`.merge()` fuses the row-strips into one region, and a small expand catches the borders:

```python
(
    section
    .find_all('text:contains(Complaint #)')
    .right(include_source=True)
    .merge()
    .expand(top=5, bottom=7)
    .show(crop=section)
)
```

Typing three header names is faster than scraping them:

```python
(
    section
    .find_all('text:contains(Complaint #)')
    .right(include_source=True)
    .merge()
    .expand(top=5, bottom=7)
    .extract_table()
    .to_df(header=['Type of Complaint', 'Description', 'Complaint Disposition'])
)
```

That works — *here*. On sections with redactions, the black boxes fool the column detector. The vertical rules are still visible, though — they're painted rather than stored as vector lines, so detect them from the rendered pixels and demand exactly four of them:

```python
from natural_pdf.guides import Guides

table = (
    section
    .find_all('text:contains(Complaint #)')
    .right(include_source=True)
    .merge()
    .expand(top=5, bottom=7)
)

guides = Guides(table)
guides.vertical.from_lines(n=4, detection_method='pixels')

(
  table
  .extract_table(verticals=guides.vertical)
  .to_df(header=['Type of Complaint', 'Description', 'Complaint Disposition'])
)
```

## The officers table

Same recipe, different anchor and column count:

```python
table = (
    section
    .find_all('text:contains(Officer #)')
    .right(include_source=True)
    .merge()
    .expand(top=5, bottom=7)
)

guides = Guides(table)
guides.vertical.from_lines(n=8, detection_method='pixels')

(
  table
  .extract_table(verticals=guides.vertical)
  .to_df(header=['Name', 'ID No.', 'Rank', 'Division', 'Officer Disposition', 'Action Taken', 'Body Cam'])
)
```

## Loop it: one row per complaint

Everything above, applied to every section. The Date Assigned/Completed regions get a few pixels of side expansion because the dates run slightly wider than their labels:

```python
rows = []
for section in sections:
    complainant = section.find("text:contains(Complainant)").right(until='text')
    dob = section.find("text:contains(DOB)").right(until='text')
    address = section.find("text:contains(Address)").right(until='text')
    gender = section.find("text:contains(Gender)").right(until='text')
    phone = section.find("text:contains(H Phone)").right(until='text')
    investigator = (
        section
        .find("text:contains(Investigator)")
        .below(until='text', width='element')
        .find('text', overlap='partial')
    )
    number = (
        section
        .find("text:contains(Number)")
        .below(until='text', width='element')
        .find('text', overlap='partial')
    )
    date_assigned = (
      section
      .find('text:contains(Date Assigned)')
      .below(width='element')
      .expand(left=5, right=5)
      .find('text')
    )
    completed = (
      section
      .find('text:contains(Completed)')
      .below(width='element')
      .expand(left=5, right=5)
      .find('text')
    )
    recorded = (
      section
      .find('text:contains(Recorded)')
      .below(until='text', width='element')
      .expand(left=5, right=5)
    )

    row = {}
    row['complainant'] = complainant.extract_text()
    row['investigator'] = investigator.extract_text()
    row['number'] = number.extract_text()
    row['dob'] = dob.extract_text()
    row['address'] = address.extract_text()
    row['gender'] = gender.extract_text()
    row['phone'] = phone.extract_text()
    row['date_assigned'] = date_assigned.extract_text()
    row['completed'] = completed.extract_text()
    row['recorded'] = recorded.extract_text()
    rows.append(row)

print("We found", len(rows), "rows")
```

```python
import pandas as pd

df = pd.DataFrame(rows)
df
```

## And one combined CSV for the officer tables

The one-to-many side works the same way, with the case number carried along so the tables stay joinable:

```python
officer_dfs = []
for section in sections:
    # Not every section has officers — skip the ones without
    if 'Officer #' not in section.extract_text():
      continue

    case_number = (
        section
        .find("text:contains(Number)")
        .below(until='text', width='element')
        .find('text', overlap='partial')
        .extract_text()
    )

    table = (
        section
        .find_all('text:contains(Officer #)')
        .right(include_source=True)
        .merge()
        .expand(top=3, bottom=6)
    )

    guides = Guides(table)
    guides.vertical.from_lines(n=8, detection_method='pixels')
    columns = ['Name', 'ID No.', 'Rank', 'Division', 'Officer Disposition', 'Action Taken', 'Body Cam']
    officer_df = (
      table
      .extract_table(verticals=guides.vertical)
      .to_df(header=columns)
    )

    officer_df['case_number'] = case_number
    officer_dfs.append(officer_df)

print("Combining", len(officer_dfs), "officer dataframes")
df = pd.concat(officer_dfs, ignore_index=True)
df.head()
```

Repeat with the "Complaint #" anchor and `n=4` for the complaints table, and the relational database this printout came from is a database again.
