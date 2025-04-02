# Form Field Extraction

Business documents like invoices, forms, and applications contain field-value pairs that need to be extracted. This tutorial shows how to identify and extract these form fields.

```python
#%pip install "natural-pdf[all]"
```

```python
from natural_pdf import PDF

# Load a PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Find fields with labels ending in colon
labels = page.find_all('text:contains(":")')

# Visualize the found labels
labels.show(color="blue", label="Field Labels")

# Count how many potential fields we found
len(labels)
```

## Extracting Field Values

```python
# Extract the value for each field label
form_data = {}

for label in labels:
    # Clean up the label text
    field_name = label.text.strip().rstrip(':')
    
    # Find the value to the right of the label
    value_region = label.right(width=200)
    value = value_region.extract_text().strip()
    
    # Store in our dictionary
    form_data[field_name] = value

# Display the extracted data
form_data
```

## Visualizing Labels and Values

```python
# Clear previous highlights
page.clear_highlights()

# Highlight both labels and their values
for label in labels:
    # Highlight the label in red
    label.show(color="red", label="Label")
    
    # Highlight the value area in blue
    label.right(width=200).show(color="blue", label="Value")

# Show the page image with highlighted elements
page.to_image()
```

## Handling Multi-line Values

```python
# Extract values that might span multiple lines
multi_line_data = {}

for label in labels:
    # Get the field name
    field_name = label.text.strip().rstrip(':')
    
    # Look both to the right and below
    right_value = label.right(width=200).extract_text().strip()
    below_value = label.below(height=50).extract_text().strip()
    
    # Combine the values if they're different
    if right_value in below_value:
        value = below_value
    else:
        value = f"{right_value} {below_value}".strip()
    
    # Add to results
    multi_line_data[field_name] = value

# Show fields with potential multi-line values
multi_line_data
```

## Finding Pattern-Based Fields

```python
import re

# Find dates in the format July 31, YYY
date_pattern = r'\b\w+ \d+, \d\d\d\d\b'

# Search all text elements for dates
text_elements = page.find_all('text')
print([elem.text for elem in text_elements])
dates = text_elements.filter(lambda elem: re.search(date_pattern, elem.text))

# Visualize the date fields
dates.show(color="green", label="Date")

# Extract just the date values
date_texts = [re.search(date_pattern, elem.text).group(0) for elem in dates]
date_texts
```

## Working with Form Tables

```python
# Run layout analysis to find table structures
page.analyze_layout()

# Find possible form tables
tables = page.find_all('region[type=table]')

if tables:
    # Visualize the tables
    tables.show(color="purple", label="Form Table")
    
    # Extract data from the first table
    first_table = tables[0]
    table_data = first_table.extract_table()
    table_data
else:
    # Try to find form-like structure using text alignment
    # Create a region where a form might be
    form_region = page.create_region(50, 200, page.width - 50, 500)
    
    # Group text by vertical position
    rows = {}
    text_elements = form_region.find_all('text')
    
    for elem in text_elements:
        # Round y-position to group elements in the same row
        row_pos = round(elem.top / 5) * 5
        if row_pos not in rows:
            rows[row_pos] = []
        rows[row_pos].append(elem)
    
    # Extract data from rows (first 5 rows)
    row_data = []
    for y in sorted(rows.keys())[:5]:
        # Sort elements by x-position (left to right)
        elements = sorted(rows[y], key=lambda e: e.x0)
        
        # Show the row
        row_box = form_region.create_region(
            min(e.x0 for e in elements), 
            min(e.top for e in elements),
            max(e.x1 for e in elements),
            max(e.bottom for e in elements)
        )
        row_box.show(color=None, use_color_cycling=True)
        
        # Extract text from row
        row_text = [e.text for e in elements]
        row_data.append(row_text)
    
    # Show the extracted rows
    row_data
```

## Combining Different Extraction Techniques

```python
# Combine label-based and pattern-based extraction
all_fields = {}

# 1. First get fields with explicit labels
for label in labels:
    field_name = label.text.strip().rstrip(':')
    value = label.right(width=200).extract_text().strip()
    all_fields[field_name] = value

# 2. Add date fields that we found with pattern matching
for date_elem in dates:
    # Find the nearest label
    nearby_label = date_elem.nearest('text:contains(":")')
    
    if nearby_label:
        # Extract the label text
        label_text = nearby_label.text.strip().rstrip(':')
        
        # Get the date value
        date_value = re.search(date_pattern, date_elem.text).group(0)
        
        # Add to our results if not already present
        if label_text not in all_fields:
            all_fields[label_text] = date_value

# Show all extracted fields
all_fields
```

Form field extraction enables you to automate data entry and document processing. By combining different techniques like label detection, spatial navigation, and pattern matching, you can handle a wide variety of form layouts. 