# Spatial Navigation

Spatial navigation lets you work with PDF content based on the physical layout of elements on the page. It's perfect for finding elements relative to each other and extracting information in context.

```python
#%pip install "natural-pdf[all]"
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

```python
# Create a region below the title
region_below = title.below(height=100)

# Visualize the region
region_below.show(color="blue", label="Below Title")

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
    include_element=True,     # Include starting element
    until='text:contains("Date:")',  # Stop at this element
    include_endpoint=False    # Don't include ending element
)

# Visualize the region between labels
between_region.show(color="green", label="Between")

# Extract text from this bounded area
between_region.extract_text()
```

## Navigating Left and Right

```python
# Find a field label
site_label = page.find('text:contains("Site:")')

# Get the content to the right (the field value)
value_region = site_label.right(width=200)

# Visualize the label and value regions
site_label.show(color="red", label="Label")
value_region.show(color="blue", label="Value")

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
page.clear_highlights()
for i, row in enumerate(rows):
    row.highlight(label=f"Row {i+1}")
page.to_image(width=700)
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