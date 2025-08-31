# Spatial Navigation

Spatial navigation lets you work with PDF content based on the physical layout of elements on the page. It's perfect for finding elements relative to each other and extracting information in context.

## Smart Defaults

Natural PDF uses intelligent defaults for spatial navigation:
- **`.left()` and `.right()`**: Default to `height='element'` (matches source element height)
- **`.above()` and `.below()`**: Default to `width='full'` (full page width)
- **Directional offset**: 0.01 points by default (configurable via `natural_pdf.options.layout.directional_offset`)
- **Exclusions**: Applied by default (disable with `apply_exclusions=False`)

```python
#%pip install natural-pdf
```

```python
from natural_pdf import PDF

# Load a PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Find the title of the document
title = page.find('text:contains("Jungle Health")')

# Visualize our starting point
title.show(color="red", label="Document Title")

# Display the title text
title.text
```

## Finding Elements Above and Below

The `.above()` and `.below()` methods default to full page width, which is ideal for finding content across the entire page width below headings or sections.

```python
# Create a region below the title
# Default: width='full' (full page width)
region_below = title.below(height=100)

# You can restrict to element width if needed
narrow_region = title.below(height=100, width='element')

# Visualize the regions
region_below.show(color="blue", label="Below Title (full width)")
narrow_region.show(color="cyan", label="Below Title (element width)")

# Find and extract text from this region
text_below = region_below.extract_text()
text_below
```

## Finding Content Between Elements

```python
# Find two labels to serve as boundaries
site_label = page.find('text:contains("Site:")')
date_label = page.find('text:contains("Date:")')

# Get the region between these labels
between_region = site_label.below(
    include_source=True,     # Include starting element
    until='text:contains("Date:")',  # Stop at this element
    include_endpoint=False    # Don't include ending element
)

# Visualize the region between labels
between_region.show(color="green", label="Between")

# Extract text from this bounded area
between_region.extract_text()
```

## Navigating Left and Right

The `.left()` and `.right()` methods now use smart defaults that match common use cases. By default, they maintain the same height as the source element, which is perfect for finding values next to labels.

```python
# Find a field label
site_label = page.find('text:contains("Site:")')

# Get the content to the right (the field value)
# Default: height='element' (matches source element height)
value_region = site_label.right(width=200)

# You can still use full page height if needed
full_height_region = site_label.right(width=200, height='full')

# Visualize the label and value regions
site_label.show(color="red", label="Label")
value_region.show(color="blue", label="Value (element height)")
full_height_region.show(color="green", label="Value (full height)")

# Extract just the value text
value_region.extract_text()
```

## Finding Adjacent Elements

```python
# Start with a label element
label = page.find('text:contains("Site:")')

# Find the next and previous elements in reading order
next_elem = label.next()
prev_elem = label.prev()

# Visualize all three elements
label.show(color="red", label="Current")
next_elem.show(color="green", label="Next") if next_elem else None
prev_elem.show(color="blue", label="Previous") if prev_elem else None

# Show the text of adjacent elements
{
    "current": label.text,
    "next": next_elem.text if next_elem else "None",
    "previous": prev_elem.text if prev_elem else "None"
}
```

## Combining with Element Selectors

```python
# Find a section label
summary = page.find('text:contains("Summary:")')

# Find the next bold text element
next_bold = summary.next('text:bold', limit=20)

# Find the nearest line element
nearest_line = summary.nearest('line')

# Visualize what we found
summary.show(color="red", label="Summary")
next_bold.show(color="blue", label="Next Bold") if next_bold else None
nearest_line.show(color="green", label="Nearest Line") if nearest_line else None

# Show the content we found
{
    "summary": summary.text,
    "next_bold": next_bold.text if next_bold else "None found",
    "nearest_line": nearest_line if nearest_line else "None found"
}
```

## Extracting Table Rows with Spatial Navigation

```python
# Find a table heading
table_heading = page.find('text:contains("Statute")')
table_heading.show(color="purple", label="Table Header")

# Extract table rows using spatial navigation
rows = []
current = table_heading

# Get the next 4 rows
for i in range(4):
    # Find the next row below the current one
    next_row = current.below(height=15)

    if next_row:
        rows.append(next_row)
        current = next_row  # Move to the next row
    else:
        break

# Visualize all found rows
with page.highlights() as h:
    for i, row in enumerate(rows):
        h.add(row, label=f"Row {i+1}")
    h.show()
```

```python
# Extract text from each row
[row.extract_text() for row in rows]
```

## Extracting Key-Value Pairs

```python
# Find all potential field labels (text with a colon)
labels = page.find_all('text:contains(":")')

# Visualize the labels
labels.show(color="blue", label="Labels")

# Extract key-value pairs
field_data = {}

for label in labels:
    # Clean up the label text
    key = label.text.strip().rstrip(':')

    # Skip if not a proper label
    if not key:
        continue

    # Get the value to the right
    value = label.right(width=200).extract_text().strip()

    # Add to our collection
    field_data[key] = value

# Show the extracted data
field_data
```

Spatial navigation mimics how humans read documents, letting you navigate content based on physical relationships between elements. It's especially useful for extracting structured data from forms, tables, and formatted documents.

## Configuring Spatial Navigation

Natural PDF provides fine control over spatial navigation behavior through global configuration and method parameters.

### Directional Offset Configuration

By default, directional methods (.above(), .below(), .left(), .right()) include a small offset of 0.01 points to avoid edge cases with touching elements. You can configure this globally:

```python
from natural_pdf import PDF
import natural_pdf

# View current offset setting
print(natural_pdf.options.layout.directional_offset)  # Default: 0.01

# Change the global offset
natural_pdf.options.layout.directional_offset = 0.1  # Larger gap
# or
natural_pdf.options.layout.directional_offset = 0    # No gap (exact boundaries)

# Load PDF and use directional methods with new offset
pdf = PDF("example.pdf")
element = pdf.pages[0].find('text:contains("Label")')
# This will now use the configured offset
region = element.below(height=50)
```

### Working with Exclusions

When using spatial navigation with pages that have exclusions (headers, footers, etc.), you can control whether exclusions are applied:

```python
# Add a header exclusion
pdf.add_exclusion(lambda page: page.find('text:contains("Header")'))

# Find an element
element = page.find('text:contains("Content")')

# By default, directional methods respect exclusions
region_with_exclusions = element.below(height=200)  # apply_exclusions=True by default

# Disable exclusion filtering if needed
region_all_content = element.below(height=200, apply_exclusions=False)

# The expand() method also supports this parameter
expanded = element.expand(bottom=100, apply_exclusions=False)
```

This is particularly useful when:
- You want to see all content in a region, including normally excluded headers/footers
- You're debugging and need to verify what content exists before exclusions
- You're working with documents where exclusions might interfere with spatial relationships

## TODO

* Add examples for navigating across multiple pages using `pdf.pages` slicing and `below(..., until=...)` that spans pages.
* Show how to chain selectors, e.g., `page.find('text:bold').below().right()` for complex paths.
* Include a sidebar on performance when many spatial calls are chained and how to cache intermediate regions.
* Add examples using `.until()` for one-liner "from here until X" extractions.
* Demonstrate attribute selectors (e.g., `line[width>2]`) and `:not()` pseudo-class for exclusion in spatial chains.

## Chaining Spatial Calls

Spatial helpers like `.below()`, `.right()`, `.nearest()` and friends **return Element or Region objects**, so you can keep chaining operations just like you would with jQuery or BeautifulSoup.

1. Start with a selector (string or Element).
2. Apply a spatial function.
3. Optionally, add another selector to narrow the result.
4. Repeat!

### Example 1 – Heading → next bold word → value to its right

```python
# Step 1 – find the heading text
heading = page.find('text:contains("Summary:")')

# Step 2 – get the first bold word after that heading (skip up to 30 elements)
value_label = heading.next('text:bold', limit=30)

# Step 3 – grab the value region to the right of that bold word
value_region = value_label.right(until='line')  # Extend until the boundary line

value_region.show(color="orange", label="Summary Value")
value_region.extract_text()
```

### Example 2 – Find a label anywhere on the document and walk to its value in one chain

```python
inspection_date_value = (
    page.find('text:startswith("Date:")')
        .right(width=500, height='element')            # Move right to get the date value region
        .find('text')                # Narrow to text elements only
)
```

Because each call returns an element, **you never lose the spatial context** – you can always add another `.below()` or `.nearest()` later.

## Enhanced Region Expansion with .expand()

The `.expand()` method provides a flexible way to create regions by expanding from an element in multiple directions at once. It supports various expansion modes that can be mixed and matched:

### Expansion Modes

1. **Boolean (True)**: Expand to the page edge
2. **Number**: Expand by a fixed number of pixels
3. **String (selector)**: Expand until finding an element (exclude by default)
4. **String with '+' prefix**: Expand until finding an element and include it

Like directional methods, `.expand()` also respects exclusions by default but can be configured to include all content:

### Basic Examples

```python
# Find a label element as our starting point
statute = page.find('text:contains("Statute")')

# Expand to page edges
full_width = statute.expand(left=True, right=True)
full_width.show(color="blue", label="Full Width")

# Fixed pixel expansion
padded = statute.expand(10)  # 10px in all directions
padded_custom = statute.expand(left=20, right=50, top=5, bottom=5)

# Expand until selectors
# This expands right until finding "Repeat?" but doesn't include it
field_region = statute.expand(right='text:contains("Repeat?")')
field_region.show(color="green", label="Until Repeat (excluded)")

# Use '+' prefix to include the endpoint
field_with_end = statute.expand(right='+text:contains("Repeat?")')
field_with_end.show(color="purple", label="Through Repeat (included)")

# Expand without respecting exclusions (e.g., to include headers/footers)
all_content = statute.expand(bottom=True, apply_exclusions=False)
all_content.show(color="red", label="All content (including exclusions)")
```

### Mixed Mode Example

The real power comes from combining different expansion modes:

```python
# Find a form field label
label = page.find('text:contains("Site:")')

# Create a complex region using multiple modes
form_field = label.expand(
    left='+text:contains("Site:")',     # Include the label itself
    right='text:contains("Date:")',     # Until next field (exclude)
    top=5,                              # 5px padding above
    bottom=True                         # Extend to page bottom
)

form_field.show(color="orange", label="Complex Region")
form_field.extract_text()
```

### Extracting Table Rows

The enhanced expand() is particularly useful for extracting table data:

```python
# Find table headers
statute_header = page.find('text:contains("Statute")')
repeat_header = page.find('text:contains("Repeat?")')

# Extract the entire first row
first_row = statute_header.expand(
    left=True,                          # Full table width
    right=True,
    bottom=20                           # Approximate row height
)

# Or extract just the data between columns
statute_data = statute_header.expand(
    right='text:contains("Repeat?")',   # Stop at next column
    bottom=100                          # Cover multiple rows
)

statute_data.show(color="cyan", label="Statute Column")
statute_data.extract_text()
```

### Form Field Extraction Pattern

A common pattern is to extract labeled form fields:

```python
# Find all field labels (ending with colon)
labels = page.find_all('text:contains(":")')

# Extract each field with its label and value
fields = {}
for label in labels[:5]:  # First 5 fields
    # Expand to include both label and value
    field_region = label.expand(
        left='+text',                    # Include the label
        right='text:contains(":")',      # Until next label
        top=2,                          # Small padding
        bottom=2
    )

    # Extract and clean the text
    text = field_region.extract_text().strip()
    if ':' in text:
        key, value = text.split(':', 1)
        fields[key.strip()] = value.strip()

fields
```

### Comparison with Directional Methods

While `.right()`, `.left()`, etc. are great for navigation (finding regions in a direction), `.expand()` is better when you want to:

1. **Expand in multiple directions at once**
2. **Mix different expansion types** (pixels, selectors, page edges)
3. **Create regions that include the source element**

```python
# Navigation approach (excludes source)
region = element.right(until='text:contains("End")')

# Expansion approach (includes source)
region = element.expand(right='text:contains("End")')

# Multi-directional expansion (not possible with navigation methods)
region = element.expand(
    left=True,                         # To page edge
    right='text:contains("End")',      # Until element
    top=10,                           # Fixed pixels
    bottom='+text:contains("Footer")'  # Through element
)
```
