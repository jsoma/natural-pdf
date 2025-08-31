# Guide Adjustment for Stream Extraction

## Overview

When using the `stream` extraction method (text-based edge detection) with explicit vertical guides, Natural PDF automatically adjusts guides that fall outside text bounds to ensure proper table extraction.

## The Problem

In pdfplumber's stream method, horizontal edges are only created where text exists. If vertical guides are placed outside the horizontal extent of text (e.g., at x=0 when text starts at x=51.6), these guides won't intersect with horizontal edges, causing missing columns in the extracted table.

## The Solution

Natural PDF automatically clips vertical guides to text bounds when:
1. Using `method="stream"` or `horizontal_strategy="text"`
2. Explicit vertical lines are provided
3. Text elements exist in the region

## Example

```python
from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides

# Load PDF and find headers
pdf = PDF("document.pdf")
page = pdf[0]
headers = page.find_all("text[y<100]")  # Find header row

# Create guides from headers
guides = Guides(page)
guides.vertical.from_headers(headers, margin=0)

# Guides might include page boundaries (0, page.width)
# which could be outside text bounds

# Extract table - guides are automatically adjusted
table = page.extract_table(method="stream", verticals=guides.vertical.data)

# All columns including first and last are properly extracted
```

## How It Works

1. **Detection**: When stream method is used with explicit vertical guides
2. **Text Bounds**: The system finds all text elements and determines their bounding box
3. **Adjustment**:
   - Guides left of text bounds are moved to the left edge of text
   - Guides right of text bounds are moved to the right edge of text
   - Guides within text bounds remain unchanged
4. **Extraction**: The adjusted guides are used for table extraction

## When This Applies

Guide adjustment happens when ALL of these conditions are met:
- Extraction method is `pdfplumber` (or its aliases `stream`)
- `horizontal_strategy` is `"text"` (text-based edge detection)
- `vertical_strategy` is `"explicit"` (using provided guides)
- `explicit_vertical_lines` are provided in table settings

## Debugging

Enable debug logging to see guide adjustments:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Extract table - will show adjustment messages
table = page.extract_table(method="stream", verticals=guides.vertical.data)
```

Example debug output:
```
Region (0, 0, 1224, 1584): Adjusted left guide from 0.0 to 51.6
Region (0, 0, 1224, 1584): Adjusted right guide from 1224.0 to 1155.7
Region (0, 0, 1224, 1584): Adjusted 26 guides for stream extraction. Text bounds: 51.6-1155.7
```

## Other Methods

This adjustment only applies to stream/text-based extraction. When using:
- `method="lattice"` (line-based): No adjustment, guides used as-is
- `method="tatr"` or `method="text"`: Different extraction methods, guides not used

## Best Practices

1. **Use from_headers()**: This method creates appropriate guides for your content
2. **Set margin=0**: For tables that span the full width of text
3. **Verify with lattice first**: If your PDF has visible lines, lattice method may work better
4. **Check text bounds**: Use `page.find_all("text").merge().bbox` to see text extent
