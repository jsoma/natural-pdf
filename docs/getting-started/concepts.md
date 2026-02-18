# Core Concepts

Natural PDF provides a hierarchy of objects for working with PDF documents. Understanding these concepts will help you navigate and extract content effectively.

```
PDF
 |
 +-- Page (one or more)
      |
      +-- Element (text, line, rect, image)
      |
      +-- Region (spatial area on the page)
```

## PDF

The `PDF` object is your entry point. It represents the entire document and provides access to pages and document-level operations.

```python
from natural_pdf import PDF

# Load from a local file
pdf = PDF("pdfs/01-practice.pdf")

# Access metadata
print(f"Pages: {len(pdf.pages)}")
```

You can also load from URLs:

```python skip
# Load from a URL (example - not runnable)
pdf = PDF("https://example.com/document.pdf")
```

## Page

A `Page` represents a single page in the document. Pages are accessed by index (zero-based).

```python
# Get the first page
page = pdf.pages[0]

# Get the last page
last_page = pdf.pages[-1]

# Iterate through all pages
for page in pdf.pages:
    print(f"Page {page.number}: {len(page.find_all('text'))} text elements")
```

Pages have dimensions and serve as the primary workspace for finding and extracting content.

```python
print(f"Page size: {page.width} x {page.height}")
```

## Element

Elements are the atomic units in a PDF: text characters, lines, rectangles, and images. You find elements using CSS-like selectors.

```python
# Find a single element
title = page.find('text:bold')

# Find all matching elements (returns ElementCollection)
all_text = page.find_all('text')
```

### Element Types

| Type | Description | Example Selector |
|------|-------------|------------------|
| `text` | Text content | `text:contains("Total")` |
| `line` | Lines and rules | `line[width>=2]` |
| `rect` | Rectangles | `rect[fill]` |
| `image` | Embedded images | `image` |

### Element Properties

Each element has properties you can access or filter on:

```python
element = page.find('text:contains("Summary")')

# Access properties
print(element.extract_text())  # The text content
print(element.x0)              # Left edge position
print(element.top)             # Top edge position
print(element.size)            # Font size (for text)
```

## Region

A `Region` is a rectangular area on a page. Regions are created through spatial navigation methods and allow you to scope operations to a specific part of the page.

```
+------------------------------------------+
|  Page                                    |
|                                          |
|  +------------------------------------+  |
|  | Region (created from .below())    |  |
|  |                                    |  |
|  |   [text]  [text]  [text]          |  |
|  |                                    |  |
|  +------------------------------------+  |
|                                          |
+------------------------------------------+
```

### Creating Regions

Regions are typically created by navigating from an element:

```python
# Find a heading
heading = page.find('text:contains("Violations")')

# Create a region below it
content = heading.below()

# Create a region below it, stopping at another element
section = heading.below(until='text:contains("Footer")')
```

### Using Regions

Once you have a region, you can extract content from it or search within it:

```python
# Extract text from the region
text = content.extract_text()

# Find elements only within this region
items = content.find_all('text')
```

## ElementCollection

When you use `find_all()`, you get an `ElementCollection` - a list-like object with PDF-specific methods.

```python
# Get all bold text
headings = page.find_all('text:bold')

# Access like a list
first = headings[0]
count = len(headings)

# Extract text from all elements
all_text = headings.extract_text()

# Filter further
large_headings = headings.filter(lambda e: e.size > 12)
```

## Putting It Together

Here is a typical workflow that uses all these concepts:

```python
from natural_pdf import PDF

# Load the PDF
pdf = PDF("pdfs/01-practice.pdf")
page = pdf.pages[0]

# Find a section header (Element)
header = page.find('text:contains("Violations")')

# Get the region below it (Region)
violations_section = header.below()

# Find all text in that region (ElementCollection)
violation_items = violations_section.find_all('text')

# Extract the content
for item in violation_items[:5]:
    print(item.extract_text())
```

## Summary

| Object | What It Is | How You Get It |
|--------|------------|----------------|
| `PDF` | The document | `PDF("file.pdf")` |
| `Page` | A single page | `pdf.pages[0]` |
| `Element` | Text, line, rect, image | `page.find('selector')` |
| `Region` | Area on a page | `element.below()` |
| `ElementCollection` | Multiple elements | `page.find_all('selector')` |
