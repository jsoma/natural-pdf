# Messy Tables

Extract and clean tables with multi-line cells, continuation rows, merged data, and missing values.

**When to use this pattern:**

- Police/incident logs with multi-line descriptions
- Government data dumps with inconsistent formatting
- Legacy database exports
- Any table where one logical row spans multiple visual rows

## The Problem

You have a table where:

- Descriptions wrap across multiple lines
- Some rows are "continuations" of the previous row
- Multiple values appear in one cell (e.g., multiple officers)
- Some cells are empty or marked with placeholders

## Sample PDF

This tutorial uses `pdfs/cookbook/incident_log.pdf` - an incident log with all these issues.

```python
import natural_pdf as npdf

pdf = npdf.PDF("pdfs/cookbook/incident_log.pdf")
pdf.pages[0].show()
```

## Step 1: Extract the Raw Table

```python
import natural_pdf as npdf
import pandas as pd

pdf = npdf.PDF("pdfs/cookbook/incident_log.pdf")
page = pdf.pages[0]

# Extract the table
table = page.extract_table()
df = table.to_df()

print(df)
pdf.close()
```

**Raw output (before cleaning):**

```
    ID    Date       Officer(s)      Location                             Description
0  001   01/15        Smith, J.    100 Main St   Traffic stop - vehicle observed running red...
1  002   01/15  Smith, J.\nJohnson, M.  250 Oak Ave   Disturbance call at residential address....
2        (continued)                              Additional notes: Neighbor reported ongoing...
3  003   01/16        Davis, R.  500 Park Blvd   Welfare check requested by family member....
4  004   01/16  Davis, R.\nSmith, J.\nWilson, K.  75 Commerce Dr   Shoplifting report at retail store....
5        (continued)                              Suspect identified as J. Doe (DOB: 03/15/1990)...
6  005   01/17      Johnson, M.    300 River Rd   Vehicle accident - two cars, minor damage....
7  006   01/17                  425 Industrial Way   Anonymous tip received. Area checked,...
```

## Step 2: Identify Continuation Rows

Continuation rows usually have empty ID fields or contain markers like "(continued)":

```python
def is_continuation(row):
    """Check if this row is a continuation of the previous."""
    # Empty ID indicates continuation
    if pd.isna(row['ID']) or row['ID'] == '':
        return True
    # Explicit continuation marker
    if '(continued)' in str(row.get('Date', '')).lower():
        return True
    return False
```

## Step 3: Merge Continuation Rows

```python
def merge_continuations(df):
    """Merge continuation rows into their parent rows."""
    merged_rows = []
    current_row = None

    for _, row in df.iterrows():
        if is_continuation(row):
            if current_row is not None:
                # Append description to current row
                current_desc = current_row.get('Description', '')
                new_desc = row.get('Description', '')
                if new_desc:
                    current_row['Description'] = f"{current_desc} {new_desc}".strip()
        else:
            # Save previous row and start new one
            if current_row is not None:
                merged_rows.append(current_row)
            current_row = row.to_dict()

    # Don't forget the last row
    if current_row is not None:
        merged_rows.append(current_row)

    return pd.DataFrame(merged_rows)

df_merged = merge_continuations(df)
print(df_merged)
```

## Step 4: Handle Multi-Value Cells

Some cells contain multiple values (like multiple officers). Split them into lists:

```python
def parse_officers(officer_str):
    """Parse officer string into list of names."""
    if pd.isna(officer_str) or officer_str == '':
        return []

    # Split on newlines or commas
    officers = []
    for part in str(officer_str).replace('\n', ',').split(','):
        name = part.strip()
        if name:
            officers.append(name)
    return officers

df_merged['officers_list'] = df_merged['Officer(s)'].apply(parse_officers)
df_merged['officer_count'] = df_merged['officers_list'].apply(len)
```

## Step 5: Clean Empty/Placeholder Values

```python
def clean_value(val, empty_markers=None):
    """Clean a value, treating certain markers as empty."""
    if empty_markers is None:
        empty_markers = ['', 'N/A', 'n/a', '-', '--', 'None', 'null']

    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if val_str in empty_markers:
        return None
    return val_str

# Apply to all columns
for col in df_merged.columns:
    if col not in ['officers_list', 'officer_count']:
        df_merged[col] = df_merged[col].apply(clean_value)
```

## Step 6: Validate and Flag Issues

```python
def validate_row(row):
    """Check for data quality issues."""
    issues = []

    if not row.get('ID'):
        issues.append('missing_id')
    if not row.get('Date'):
        issues.append('missing_date')
    if row.get('officer_count', 0) == 0:
        issues.append('no_officer')

    return issues if issues else None

df_merged['data_issues'] = df_merged.apply(validate_row, axis=1)

# Show rows with issues
issues_df = df_merged[df_merged['data_issues'].notna()]
print(f"Rows with issues: {len(issues_df)}")
```

## Complete Example

```python
import natural_pdf as npdf
import pandas as pd

def is_continuation(row):
    """Check if this row is a continuation of the previous."""
    if pd.isna(row['ID']) or row['ID'] == '':
        return True
    if '(continued)' in str(row.get('Date', '')).lower():
        return True
    return False

def merge_continuations(df, text_column='Description'):
    """Merge continuation rows into their parent rows."""
    merged_rows = []
    current_row = None

    for _, row in df.iterrows():
        if is_continuation(row):
            if current_row is not None:
                current_desc = current_row.get(text_column, '')
                new_desc = row.get(text_column, '')
                if new_desc:
                    current_row[text_column] = f"{current_desc} {new_desc}".strip()
        else:
            if current_row is not None:
                merged_rows.append(current_row)
            current_row = row.to_dict()

    if current_row is not None:
        merged_rows.append(current_row)

    return pd.DataFrame(merged_rows)

def parse_multi_value(value, separators=None):
    """Split a cell with multiple values into a list."""
    if separators is None:
        separators = ['\n', ';', ' and ']

    if pd.isna(value) or value == '':
        return []

    result = str(value)
    for sep in separators:
        result = result.replace(sep, ',')

    items = [item.strip() for item in result.split(',')]
    return [item for item in items if item]

def clean_incident_log(pdf_path):
    """Extract and clean an incident log table."""
    pdf = npdf.PDF(pdf_path)
    page = pdf.pages[0]

    # Extract raw table
    table = page.extract_table()
    df = table.to_df()

    # Merge continuation rows
    df = merge_continuations(df, text_column='Description')

    # Parse multi-value officer column
    df['officers'] = df['Officer(s)'].apply(parse_multi_value)
    df['officer_count'] = df['officers'].apply(len)

    # Clean up
    df = df.drop(columns=['Officer(s)'])
    df = df.rename(columns={'officers': 'Officer(s)'})

    # Reset index
    df = df.reset_index(drop=True)

    pdf.close()
    return df

# Usage
df = clean_incident_log("pdfs/cookbook/incident_log.pdf")
print(df[['ID', 'Date', 'Officer(s)', 'officer_count']])
```

## Common Messy Table Patterns

### Pattern: Detecting Headers Mid-Table

Some tables repeat headers on each page:

```python
def is_header_row(row, expected_headers):
    """Check if row contains header values."""
    row_values = [str(v).strip().lower() for v in row.values if pd.notna(v)]
    header_values = [h.lower() for h in expected_headers]
    return len(set(row_values) & set(header_values)) > len(expected_headers) / 2

# Filter out repeated headers
headers = ['ID', 'Date', 'Officer(s)', 'Location', 'Description']
df = df[~df.apply(lambda r: is_header_row(r, headers), axis=1)]
```

### Pattern: One-to-Many Records

When one incident has multiple related sub-records:

```python
# Explode the officers list into separate rows
df_exploded = df.explode('Officer(s)')
# Now each officer-incident combination is a row
```

### Pattern: Handling Footnotes and Annotations

```python
def remove_footnotes(text):
    """Remove footnote markers like [1], *, etc."""
    import re
    if pd.isna(text):
        return text
    # Remove [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove standalone asterisks
    text = re.sub(r'\s*\*+\s*', ' ', text)
    return text.strip()

df['Description'] = df['Description'].apply(remove_footnotes)
```

## Troubleshooting

### "Table extraction misses columns"

Try extracting with explicit settings:

```python
# Use layout analysis first
page.analyze_layout()
tables = page.find_all('region[type=table]')
for t in tables:
    df = t.extract_table().to_df()
```

### "Cells are merged incorrectly"

Check if the PDF uses visual alignment vs actual table structure:

```python
# View the raw elements
page.find_all('text').show()

# If it's whitespace-aligned, not a real table,
# you may need to extract by position
```

### "Text wrapping creates garbage"

Normalize whitespace after extraction:

```python
def normalize_whitespace(text):
    """Clean up wrapped text."""
    if pd.isna(text):
        return text
    # Replace multiple whitespace with single space
    import re
    return re.sub(r'\s+', ' ', text).strip()
```

## Next Steps

- [Simple Table Extraction](../tutorials/04-table-extraction.ipynb) - Basics of table extraction
- [One Page = One Row](one-page-one-row.md) - Handle forms with embedded tables
- [Multipage Content](multipage-content.md) - Tables spanning multiple pages
