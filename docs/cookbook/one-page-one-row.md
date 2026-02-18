# One Page = One Row

Extract data from multi-page documents where each page is a repeating form. Output one row per page to a CSV or DataFrame.

**When to use this pattern:**

- Inspection reports (each page = one facility)
- Disclosure forms (each page = one filing)
- Intake forms (each page = one applicant)
- Any document where the same form template repeats

## The Problem

You have a PDF with 100 pages. Each page is the same form filled out differently. You need to extract specific fields from every page and produce a spreadsheet.

## Sample PDF

This tutorial uses `pdfs/cookbook/facility_inspections.pdf` - a 3-page document where each page is an inspection report.

```python
import natural_pdf as npdf

pdf = npdf.PDF("pdfs/cookbook/facility_inspections.pdf")
pdf.pages[0].show()  # View the first page
```

## Basic Pattern

The core loop is simple: iterate pages, find labels, extract values.

```python
import natural_pdf as npdf
import pandas as pd

pdf = npdf.PDF("pdfs/cookbook/facility_inspections.pdf")

rows = []
for page in pdf.pages:
    row = {}

    # Find label, then get the value to its right
    facility_id = page.find('text:contains("Facility ID:")')
    if facility_id:
        row['facility_id'] = facility_id.right().extract_text().strip()

    date = page.find('text:contains("Inspection Date:")')
    if date:
        row['date'] = date.right().extract_text().strip()

    inspector = page.find('text:contains("Inspector:")')
    if inspector:
        row['inspector'] = inspector.right().extract_text().strip()

    status = page.find('text:contains("Status:")')
    if status:
        row['status'] = status.right().extract_text().strip()

    rows.append(row)

pdf.close()

df = pd.DataFrame(rows)
print(df)
```

**Output:**

```
  facility_id        date    inspector status
0     FAC-001  2024-01-15     J. Smith   PASS
1     FAC-002  2024-01-16   M. Johnson   FAIL
2     FAC-003  2024-01-17     J. Smith   PASS
```

## Handling Missing Data

Real-world forms have missing fields. Always check for `None`:

```python
for page in pdf.pages:
    row = {}

    facility_id = page.find('text:contains("Facility ID:")')
    row['facility_id'] = facility_id.right().extract_text().strip() if facility_id else None

    # Continue for other fields...
    rows.append(row)
```

## Extracting Tables Within Forms

Some forms contain embedded tables (like violation lists). Extract them per-page:

```python
for page in pdf.pages:
    row = {}

    # Extract metadata
    facility_id = page.find('text:contains("Facility ID:")')
    row['facility_id'] = facility_id.right().extract_text().strip() if facility_id else None

    # Find the violations section
    violations_header = page.find('text:contains("Violations Found:")')
    if violations_header:
        # Get the region below the header
        table_region = violations_header.below()
        table = table_region.extract_table()
        if table:
            row['violation_count'] = len(table.to_df())
        else:
            row['violation_count'] = 0

    rows.append(row)
```

## Using a Field Map for Cleaner Code

When extracting many fields, use a dictionary to reduce repetition:

```python
FIELD_MAP = {
    'facility_id': 'Facility ID:',
    'facility_name': 'Facility Name:',
    'date': 'Inspection Date:',
    'inspector': 'Inspector:',
    'status': 'Status:',
    'score': 'Score:',
}

def extract_page(page):
    """Extract all fields from a single page."""
    row = {}
    for field_name, label_text in FIELD_MAP.items():
        label = page.find(f'text:contains("{label_text}")')
        if label:
            row[field_name] = label.right().extract_text().strip()
        else:
            row[field_name] = None
    return row

# Main extraction loop
rows = [extract_page(page) for page in pdf.pages]
df = pd.DataFrame(rows)
```

## Handling Different Label-Value Layouts

Labels and values aren't always side-by-side. Use the appropriate directional method:

```python
# Value is to the RIGHT of label (most common)
label.right().extract_text()

# Value is BELOW the label
label.below().extract_text()

# Value is in a specific column position
# Useful when labels are in column 1, values in column 2
label.right(width=150).extract_text()  # Fixed width region
```

## Grouping Pages by Category

If your PDF contains different form types, use `groupby()` to process them separately:

```python
# Group by a status field on each page
def get_status(page):
    status = page.find('text:contains("Status:")')
    if status:
        return status.right().extract_text().strip()
    return None

grouped = pdf.pages.groupby(get_status)

# Process each group
for status, pages in grouped:
    print(f"\n{status}: {len(pages)} pages")
    for page in pages:
        # Process pages of this status...
        pass
```

## Saving Results

```python
# To CSV
df.to_csv("inspections.csv", index=False)

# To Excel
df.to_excel("inspections.xlsx", index=False)

# To JSON (records format)
df.to_json("inspections.json", orient="records", indent=2)
```

## Complete Example

```python
import natural_pdf as npdf
import pandas as pd

FIELD_MAP = {
    'facility_id': 'Facility ID:',
    'facility_name': 'Facility Name:',
    'date': 'Inspection Date:',
    'inspector': 'Inspector:',
    'status': 'Status:',
    'score': 'Score:',
}

def extract_inspection(page):
    """Extract data from a single inspection page."""
    row = {}

    # Extract labeled fields
    for field_name, label_text in FIELD_MAP.items():
        label = page.find(f'text:contains("{label_text}")')
        if label:
            row[field_name] = label.right().extract_text().strip()
        else:
            row[field_name] = None

    # Count violations from embedded table
    violations = page.find('text:contains("Violations Found:")')
    if violations:
        table_region = violations.below()
        table = table_region.extract_table()
        row['violation_count'] = len(table.to_df()) if table else 0
    else:
        row['violation_count'] = 0

    return row

# Main
pdf = npdf.PDF("pdfs/cookbook/facility_inspections.pdf")
rows = [extract_inspection(page) for page in pdf.pages]
pdf.close()

df = pd.DataFrame(rows)
df.to_csv("inspections.csv", index=False)
print(df)
```

## Next Steps

- [Label-Value Extraction](label-value-extraction.md) - Dive deeper into finding labels and extracting values
- [Batch Processing](batch-processing.md) - Process hundreds of PDFs using this pattern
- [Messy Tables](messy-tables.md) - Handle embedded tables with formatting issues
