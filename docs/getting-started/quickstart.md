# Quickstart

Get up and running with Natural PDF in under 5 minutes.

## Installation

```bash
pip install natural-pdf
```

## Load a PDF

```python
from natural_pdf import PDF

pdf = PDF("pdfs/01-practice.pdf")
page = pdf.pages[0]
```

## Extract All Text

```python
text = page.extract_text()
print(text)
```

## Find a Specific Element

Natural PDF uses **selectors** to find elements - a simple, CSS-like syntax:

```
'text'                      # All text elements
'text:bold'                 # Bold text (colon for pseudo-classes)
'text:contains("Invoice")'  # Text containing "Invoice"
'text[size>12]'             # Text with size > 12 (brackets for attributes)
```

!!! tip "Learn More"
    See [Selectors 101](selectors.md) for the complete syntax reference.

```python
# Find text containing "Violations"
element = page.find('text:contains("Violations")')
if element:
    print(element.extract_text())
```

```python
# Find bold text
bold_text = page.find('text:bold')
if bold_text:
    print(bold_text.extract_text())
```

## Extract a Table

```python
table = page.extract_table()
for row in table:
    print(row)
```

## Visualize What You Found

```python
# Highlight the element on the page
element = page.find('text:contains("Violations")')
element.show()
```

## Next Steps

- **[Selectors 101](selectors.md)** - Master the selector syntax
- **[Core Concepts](concepts.md)** - Understand how Natural PDF thinks about PDFs
- **[Finding Elements](../tutorials/02-finding-elements.ipynb)** - Practice finding elements
- **[Spatial Navigation](../tutorials/08-spatial-navigation.ipynb)** - Navigate with `.below()`, `.above()`
- **[Table Extraction](../tutorials/04-table-extraction.ipynb)** - Extract complex tables
