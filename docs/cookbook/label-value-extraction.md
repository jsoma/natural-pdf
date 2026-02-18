# Label-Value Extraction

Find a label element and extract the value next to it. The fundamental pattern for form data extraction.

**When to use this pattern:**

- Invoices ("Total:" -> "$500")
- Forms ("Name:" -> "John Smith")
- Any document with label-value pairs

## The Problem

You need to find "Invoice Number:" on a page and get "INV-2024-00789" from next to it. The position varies between documents, so you can't use fixed coordinates.

## Sample PDF

This tutorial uses `pdfs/cookbook/vendor_invoice.pdf` - an invoice with header fields and line items.

```python
import natural_pdf as npdf

pdf = npdf.PDF("pdfs/cookbook/vendor_invoice.pdf")
pdf.pages[0].show()
```

## Basic Pattern

```python
import natural_pdf as npdf

pdf = npdf.PDF("pdfs/cookbook/vendor_invoice.pdf")
page = pdf.pages[0]

# Find the label
label = page.find('text:contains("Invoice Number:")')

# Get the value to its right
if label:
    value_region = label.right()
    value = value_region.extract_text().strip()
    print(f"Invoice Number: {value}")

pdf.close()
```

**Output:**
```
Invoice Number: INV-2024-00789
```

## Directional Methods

Natural PDF provides four directional methods. Each creates a region extending from the source element:

| Method | Default Behavior | Use When |
|--------|------------------|----------|
| `.right()` | Same height as element | Value is beside the label |
| `.left()` | Same height as element | Label is to the right of value |
| `.below()` | Full page width | Value is under the label |
| `.above()` | Full page width | Value is above the label |

```python
# Value is beside the label (most common)
label.right().extract_text()

# Value is under the label (stacked layout)
label.below().extract_text()

# Combine for complex layouts
# Get value that's below AND within a column
label.below(width='element').extract_text()
```

## Controlling Region Size

By default, `.right()` and `.left()` match the element's height, while `.below()` and `.above()` span the full page width. Override these defaults:

```python
# Fixed width region (150 pixels)
label.right(width=150).extract_text()

# Match the label's width
label.below(width='element').extract_text()

# Full page width
label.right(height='full').extract_text()
```

## Using `until` to Bound Regions

Stop a region at another element instead of extending to the page edge:

```python
# Get text between "Description:" and the next bold text
desc = page.find('text:contains("Description:")')
content = desc.below(until='text:bold').extract_text()

# Get text until a specific label
start = page.find('text:contains("Summary:")')
end_label = 'text:contains("Total:")'
section = start.below(until=end_label).extract_text()
```

## Extracting Multiple Fields

Loop through a list of labels:

```python
fields_to_extract = [
    'Invoice Number:',
    'Invoice Date:',
    'Due Date:',
    'Vendor:',
    'PO Number:',
]

data = {}
for field in fields_to_extract:
    label = page.find(f'text:contains("{field}")')
    if label:
        data[field] = label.right().extract_text().strip()
    else:
        data[field] = None

print(data)
```

**Output:**
```python
{
    'Invoice Number:': 'INV-2024-00789',
    'Invoice Date:': '2024-03-15',
    'Due Date:': '2024-04-15',
    'Vendor:': 'Acme Corporation',
    'PO Number:': 'PO-2024-456'
}
```

## Handling Variations

### Case-Insensitive Matching

```python
# Matches "total:", "Total:", "TOTAL:"
label = page.find('text:contains("total")', case=False)
```

### Partial Matches

The `contains()` selector matches substrings:

```python
# Matches "Invoice Number:", "Invoice No:", "Invoice #:"
label = page.find('text:contains("Invoice")')
```

### Multiple Possible Labels

Try several labels until one matches:

```python
possible_labels = ['Total:', 'Grand Total:', 'Amount Due:', 'Balance:']

total_value = None
for label_text in possible_labels:
    label = page.find(f'text:contains("{label_text}")')
    if label:
        total_value = label.right().extract_text().strip()
        break

print(f"Total: {total_value}")
```

## Working with Tables of Label-Value Pairs

Some forms use two-column tables for metadata:

```python
# Extract the entire metadata table
metadata_table = page.extract_table()
if metadata_table:
    df = metadata_table.to_df()

    # Convert two-column table to dictionary
    if len(df.columns) == 2:
        data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        print(data)
```

## Extracting Monetary Values

```python
import re

label = page.find('text:contains("TOTAL:")')
if label:
    raw = label.right().extract_text().strip()
    # Remove currency symbols and commas
    amount = re.sub(r'[^\d.]', '', raw)
    total = float(amount)
    print(f"Total: ${total:,.2f}")
```

## Complete Invoice Extraction Example

```python
import natural_pdf as npdf
import re

def extract_invoice(pdf_path):
    """Extract key fields from an invoice PDF."""
    pdf = npdf.PDF(pdf_path)
    page = pdf.pages[0]

    data = {}

    # Header fields
    header_fields = {
        'invoice_number': 'Invoice Number:',
        'invoice_date': 'Invoice Date:',
        'due_date': 'Due Date:',
        'vendor': 'Vendor:',
        'po_number': 'PO Number:',
    }

    for key, label_text in header_fields.items():
        label = page.find(f'text:contains("{label_text}")')
        data[key] = label.right().extract_text().strip() if label else None

    # Total (with currency parsing)
    total_label = page.find('text:contains("TOTAL:")')
    if total_label:
        raw = total_label.right().extract_text().strip()
        data['total'] = float(re.sub(r'[^\d.]', '', raw))

    # Line items table
    line_items_header = page.find('text:contains("Line Items")')
    if line_items_header:
        table_region = line_items_header.below(until='text:contains("Subtotal")')
        table = table_region.extract_table()
        if table:
            data['line_items'] = table.to_df().to_dict('records')

    pdf.close()
    return data

# Usage
invoice_data = extract_invoice("pdfs/cookbook/vendor_invoice.pdf")
print(f"Invoice: {invoice_data['invoice_number']}")
print(f"Vendor: {invoice_data['vendor']}")
print(f"Total: ${invoice_data['total']:,.2f}")
```

## Troubleshooting

### "Value region is empty"

The label might be part of a larger text element. Try finding the specific text:

```python
# Instead of finding just "Total:"
label = page.find('text:contains("Total:")')

# You might need to find the element that STARTS with "Total:"
labels = page.find_all('text:contains("Total")')
for l in labels:
    if l.extract_text().strip().startswith("Total:"):
        value = l.right().extract_text()
        break
```

### "Getting wrong value (from different row)"

Constrain the region height:

```python
# Use element height to stay on the same line
label.right(height='element').extract_text()
```

### "Label and value are in the same element"

Sometimes PDFs store "Label: Value" as one text element:

```python
element = page.find('text:contains("Invoice Number:")')
full_text = element.extract_text()  # "Invoice Number: INV-2024-00789"

# Parse the value from the text
if ': ' in full_text:
    label, value = full_text.split(': ', 1)
```

## Next Steps

- [One Page = One Row](one-page-one-row.md) - Apply this pattern to multi-page forms
- [Finding Sections](finding-sections.md) - Extract content between section headers
- [Simple Table Extraction](../tutorials/04-table-extraction.ipynb) - Extract structured tables
