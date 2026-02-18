# Multipage Content

Extract content that spans multiple pages - tables that continue, sections that flow across page breaks.

**When to use this pattern:**

- Tables that continue on the next page
- Sections that span page boundaries
- Long-form content split across pages
- Any "continued on next page" scenarios

## The Problem

You find a table header on page 1, but the table data continues through pages 2 and 3. Or you need content between "Chapter 1" on page 5 and "Chapter 2" on page 8.

## Sample PDF

This tutorial uses `pdfs/cookbook/budget_items.pdf` - a budget table spanning two pages.

```python
import natural_pdf as npdf

pdf = npdf.PDF("pdfs/cookbook/budget_items.pdf")
print(f"Pages: {len(pdf.pages)}")
pdf.pages[0].show()
```

## Using `multipage=True`

The `multipage` parameter tells directional methods to continue across page boundaries:

```python
import natural_pdf as npdf

pdf = npdf.PDF("pdfs/cookbook/budget_items.pdf")
page1 = pdf.pages[0]

# Find the table header
header = page1.find('text:contains("DEPARTMENT BUDGET")')

# Get content below, spanning multiple pages
content_region = header.below(multipage=True)

# This returns a FlowRegion that spans both pages
print(f"Region type: {type(content_region).__name__}")
text = content_region.extract_text()
print(text[:500])

pdf.close()
```

## Extracting Multipage Tables

### Method 1: Extract Tables Per Page, Then Concatenate

```python
import natural_pdf as npdf
import pandas as pd

pdf = npdf.PDF("pdfs/cookbook/budget_items.pdf")

# Extract table from each page
all_dfs = []
for page in pdf.pages:
    table = page.extract_table()
    if table:
        df = table.to_df()
        all_dfs.append(df)

# Concatenate, handling repeated headers
combined = pd.concat(all_dfs, ignore_index=True)

# Remove duplicate header rows (if headers repeat on each page)
header_mask = combined.iloc[:, 0] == 'Line'  # First column header value
combined = combined[~header_mask].reset_index(drop=True)

print(combined)
pdf.close()
```

### Method 2: Use Bounded Multipage Regions

```python
# Find start and end markers
start = pdf.pages[0].find('text:contains("Line")')  # Table header
end = pdf.pages[-1].find('text:contains("TOTAL")')  # Table footer

if start and end:
    # Get region from start to end, across pages
    table_region = start.below(until='text:contains("TOTAL")', multipage=True)
    text = table_region.extract_text()
```

## Extracting Sections Across Pages

### Find Section Boundaries

```python
import natural_pdf as npdf

pdf = npdf.PDF("pdfs/cookbook/annual_report.pdf")

# Find section headers across all pages
def find_section(pdf, section_name):
    """Find a section header across all pages."""
    for page in pdf.pages:
        header = page.find(f'text:contains("{section_name}")')
        if header:
            return header, page
    return None, None

# Get content between two sections
start_header, start_page = find_section(pdf, "FINANCIAL HIGHLIGHTS")
end_header, end_page = find_section(pdf, "RISKS AND CHALLENGES")

if start_header:
    # Extract from this header to the next section
    content = start_header.below(
        until='text:bold[size>=13]',  # Stop at next bold header
        multipage=True
    )
    print(content.extract_text())

pdf.close()
```

## Enabling Multipage by Default

If you're doing a lot of multipage navigation, set it globally:

```python
import natural_pdf as npdf

# Enable multipage for all directional navigation
npdf.set_option('layout.auto_multipage', True)

pdf = npdf.PDF("pdfs/cookbook/budget_items.pdf")
page1 = pdf.pages[0]

# Now .below() automatically spans pages
header = page1.find('text:contains("DEPARTMENT BUDGET")')
content = header.below()  # Automatically multipage

# Override for single-page when needed
single_page_content = header.below(multipage=False)

pdf.close()
```

## FlowRegion vs Region

When content spans multiple pages, you get a `FlowRegion` instead of a `Region`:

```python
region = header.below(multipage=True)

if hasattr(region, 'pages'):
    # It's a FlowRegion
    print(f"Spans {len(region.pages)} pages")
else:
    # It's a regular Region (single page)
    print("Single page region")
```

`FlowRegion` supports the same operations:

- `extract_text()` - Returns combined text
- `find()` / `find_all()` - Search within the region
- `show()` - Visualize the region

## Processing Page-by-Page with Context

Sometimes you need to process each page but track cross-page context:

```python
import natural_pdf as npdf

pdf = npdf.PDF("pdfs/cookbook/budget_items.pdf")

current_category = None
all_items = []

for page in pdf.pages:
    # Check for category headers
    categories = page.find_all('text:bold:contains("Category")')
    if categories:
        current_category = categories[0].extract_text()

    # Extract table rows
    table = page.extract_table()
    if table:
        df = table.to_df()
        # Skip header rows
        df = df[df.iloc[:, 0] != 'Line']
        # Add category context
        df['category_context'] = current_category
        all_items.append(df)

combined = pd.concat(all_items, ignore_index=True)
print(combined)
pdf.close()
```

## Complete Example: Budget Table Extraction

```python
import natural_pdf as npdf
import pandas as pd

def extract_multipage_budget(pdf_path):
    """Extract a budget table that spans multiple pages."""
    pdf = npdf.PDF(pdf_path)

    # Method: Extract per page and combine
    page_dfs = []

    for i, page in enumerate(pdf.pages):
        table = page.extract_table()
        if table:
            df = table.to_df()

            # First page has headers
            if i == 0:
                # Store headers for validation
                headers = df.iloc[0].tolist()
                df = df.iloc[1:]  # Remove header row from data

            # Subsequent pages might repeat headers
            else:
                # Remove rows that match header
                df = df[df.iloc[:, 0] != headers[0]]

            page_dfs.append(df)

    # Combine all pages
    combined = pd.concat(page_dfs, ignore_index=True)

    # Clean up
    # Remove empty rows
    combined = combined.dropna(how='all')

    # Remove total row for data processing (but capture the value)
    total_row = combined[combined.iloc[:, 2].str.contains('TOTAL', na=False)]
    if not total_row.empty:
        total_value = total_row.iloc[0, 3]
        combined = combined[~combined.iloc[:, 2].str.contains('TOTAL', na=False)]

    # Set proper column names
    combined.columns = ['Line', 'Category', 'Description', 'Amount']

    pdf.close()
    return combined, total_value

# Usage
df, total = extract_multipage_budget("pdfs/cookbook/budget_items.pdf")
print(df)
print(f"\nTotal: {total}")
```

## Troubleshooting

### "Content is duplicated"

Headers often repeat on each page. Filter them:

```python
# Identify header text
header_text = df.iloc[0, 0]

# Remove rows matching header
df = df[df.iloc[:, 0] != header_text]
```

### "Missing content between pages"

Check that `multipage=True` is set:

```python
# Explicit multipage
region = header.below(multipage=True)

# Or enable globally
npdf.set_option('layout.auto_multipage', True)
```

### "FlowRegion extract_table() doesn't work"

Table extraction works best per-page. Extract tables from each page and combine:

```python
# Don't do this
flow_region.extract_table()  # May not work as expected

# Do this instead
for page in pdf.pages:
    table = page.extract_table()
    # ... combine tables
```

## Next Steps

- [Finding Sections](finding-sections.md) - Extract bounded sections
- [Messy Tables](messy-tables.md) - Handle table cleaning after extraction
- [One Page = One Row](one-page-one-row.md) - Process repeating forms
