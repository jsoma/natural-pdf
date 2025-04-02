# Text Extraction Guide

This guide demonstrates various ways to extract text from PDFs using Natural PDF, from simple page dumps to targeted extraction based on elements, regions, and styles.

## Setup

First, let's import necessary libraries and load a sample PDF. We'll use `example.pdf` from the tutorials' `pdfs` directory. *Adjust the path if your setup differs.*

```python
from natural_pdf import PDF
from pathlib import Path

# Path to your sample PDF
pdf_path = Path("../tutorials/pdfs/example.pdf") # Adjust if needed

# Load the PDF
pdf = PDF(pdf_path)

# Select the first page for initial examples
page = pdf.pages[0]

# Display the first page
page.show()
```

## Basic Text Extraction

Get all text from a page or the entire document.

```python
# Extract all text from the first page
# Displaying first 500 characters
page.extract_text()[:500]
```

```python
# Extract text from the entire document (may take time)
# Uncomment to run:
# pdf.extract_text()[:500]
```

## Extracting Text from Specific Elements

Use selectors with `find()` or `find_all()` to target specific elements. *Selectors like `:contains("Summary")` are examples; adapt them to your PDF.*

```python
# Find a single element, e.g., a title containing "Summary"
# Adjust selector as needed
title_element = page.find('text:contains("Summary")')
title_element # Display the found element object
```

```python
# Highlight the found element on the page
if title_element:
    page.highlight(title_element, color="yellow").show()
```

```python
# Extract text just from that element
if title_element:
    title_element.text
```

```python
# Find multiple elements, e.g., bold headings (size >= 14)
# Adjust selector as needed
heading_elements = page.find_all('text[size>=14]:bold')
heading_elements # Shows the list of found elements
```

```python
# Highlight all found headings
if heading_elements:
    page.highlight_all(heading_elements, color="lightblue").show()
```

```python
# Extract their combined text, separated by newlines
if heading_elements:
    heading_elements.extract_text(delimiter="\n")
```

## Multi-Word Searches

Natural PDF handles spaces well, making multi-word searches easy. *Adjust search phrases based on your PDF.*

```python
# Exact phrase (case-sensitive)
page.find('text:contains("Annual Report")')
```

```python
# Case-insensitive search
page.find('text:contains("financial statement")', case=False)
```

```python
# Regular expression (e.g., "YYYY Report")
page.find('text:contains("\\d{4}\\s+Report")', regex=True)
```

```python
# Highlight one of the findings (e.g., case-insensitive search)
statement_phrase = page.find('text:contains("financial statement")', case=False)
if statement_phrase:
    page.highlight(statement_phrase, color="orange").show()
```

*You can control space preservation during loading with `PDF(..., keep_spaces=False)`.*

## Extracting Text from Regions

Define regions geographically or relative to elements.

```python
# Region below an element (e.g., below "Introduction")
# Adjust selector as needed
intro_heading = page.find('text:contains("Introduction")')

if intro_heading:
    content_below = intro_heading.below()
    page.highlight(content_below, color="lightgreen").show()
```

```python
# Extract text from that 'below' region
if intro_heading:
    content_below.extract_text()[:500] # Show sample
```

```python
# Region between two elements (e.g., "Methodology" to "Results")
# Adjust selectors as needed
start_element = page.find('text:contains("Methodology")')
end_element = page.find('text:contains("Results")')

if start_element and end_element:
    method_section = start_element.below(until=end_element, include_until=False)
    page.highlight(method_section, color="cyan").show()
```

```python
# Extract text from the 'between' region
if start_element and end_element:
    method_section.extract_text()[:500] # Show sample
```

```python
# Manually defined region via coordinates (x0, top, x1, bottom)
manual_region = page.create_region(100, 200, 500, 600)
page.highlight(manual_region, color="magenta").show()
```

```python
# Extract text from the manual region
manual_region.extract_text()[:500] # Show sample
```

## Filtering Out Headers and Footers

Use Exclusion Zones to remove unwanted content before extraction. *Adjust selectors for typical header/footer content.*

```python
# Identify potential header/footer areas
header_content = page.find('text:contains("Confidential Document")') # Adjust
footer_content = page.find('text:contains("Page")') # Adjust

# Create regions for exclusion
exclusions_to_add = []
header_zone = header_content.bbox.above() if header_content else None
footer_zone = footer_content.bbox.below() if footer_content else None

if header_zone: exclusions_to_add.append(header_zone)
if footer_zone: exclusions_to_add.append(footer_zone)

# Highlight the potential exclusion zones on a copy
if exclusions_to_add:
    img = page.copy() # Work on a copy for highlighting only
    if header_zone: img.highlight(header_zone, color="red", alpha=0.3)
    if footer_zone: img.highlight(footer_zone, color="red", alpha=0.3)
    img.show()
else:
    "Selectors for header/footer didn't match, skipping highlight."
```

```python
# Add the exclusions to the actual page object
# This modifies the page state for subsequent extractions
if exclusions_to_add:
    page.add_exclusions(exclusions_to_add)
    f"Added {len(page.exclusions)} exclusion zones to the page."
else:
    "No exclusion zones added."
```

```python
# Extract text - exclusions applied by default now
if page.exclusions:
    clean_text = page.extract_text()
    clean_text[:500] # Show sample of cleaned text
```

```python
# Compare with text extracted *without* applying exclusions
if page.exclusions:
    full_text_no_exclusions = page.extract_text(apply_exclusions=False)
    f"Original length: {len(full_text_no_exclusions)}, Excluded length: {len(clean_text)}"
```

```python
# Clean up exclusions if you want to reset the page state
# page.clear_exclusions()
# f"Cleared exclusions. Current count: {len(page.exclusions)}"
```

*Exclusions can also be defined globally at the PDF level using `pdf.add_exclusion()` with a function.*

## Controlling Whitespace

Manage how spaces and blank lines are handled during extraction using `keep_blank_chars`.

```python
# Default (keep_blank_chars=True) - Use repr to see whitespace characters
repr(page.extract_text(keep_blank_chars=True)[:100])
```

```python
# Remove blank characters (keep_blank_chars=False)
repr(page.extract_text(keep_blank_chars=False)[:100])
```

*`preserve_whitespace=True` is an alias for `keep_blank_chars=True`.*

## Font-Aware Text Extraction

Natural PDF uses font attributes (`fontname`, `size` by default) when grouping characters into words. This helps maintain separation between text with different styling. Change this behavior during loading:

```python
# Default loading (already done):
# pdf = PDF(pdf_path) # font_attrs=['fontname', 'size']

# Load grouping only by spatial proximity:
# pdf_spatial_only = PDF(pdf_path, font_attrs=[])

# Load grouping by font, size, and color:
# pdf_custom_font = PDF(pdf_path, font_attrs=['fontname', 'size', 'non_stroking_color'])

"PDF loaded with default font settings. See comments for other options."
```

### Font Information Access

Inspect font details of text elements.

```python
# Find the first text element on the page
first_text = page.find('text')
first_text # Display basic info
```

```python
# Highlight the first text element
if first_text:
    page.highlight(first_text).show()
```

```python
# Get detailed font properties dictionary
if first_text:
    first_text.font_info()
```

```python
# Check specific style properties directly
if first_text:
    f"Is Bold: {first_text.bold}, Is Italic: {first_text.italic}, Font: {first_text.fontname}, Size: {first_text.size}"
```

```python
# Find elements by font attributes (adjust selectors)
# Example: Find Arial fonts
arial_text = page.find_all('text[fontname*=Arial]')
arial_text # Display list of found elements
```

```python
# Highlight Arial text found
if arial_text:
    page.highlight_all(arial_text, color="purple").show()
```

```python
# Example: Find large text (e.g., size >= 16)
large_text = page.find_all('text[size>=16]')
large_text
```

```python
# Highlight large text found
if large_text:
    page.highlight_all(large_text, color="green").show()
```

## Working with Font Styles

Analyze and group text elements by their computed font *style*, which combines attributes like font name, size, boldness, etc., into logical groups.

```python
# Analyze styles on the page
# This returns a dictionary mapping style names to ElementList objects
text_styles = page.analyze_text_styles()
f"Found {len(text_styles)} distinct text styles."
```

```python
# Show the identified style names and the number of elements in each
{name: len(group) for name, group in text_styles.items()}
```

```python
# Highlight elements belonging to the first identified style
if text_styles:
    first_style_name = list(text_styles.keys())[0]
    first_style_group = text_styles[first_style_name]
    page.highlight_all(first_style_group, color="teal").show()
    f"Highlighted style: {first_style_name}"
```

```python
# Extract text just from this style group
if text_styles:
    first_style_group.extract_text(delimiter=" ")[:500] # Show sample
```

```python
# Visualize all text styles with distinct colors automatically assigned
page.highlight_text_styles().show()
```

*Font variants (e.g., `AAAAAB+FontName`) are also accessible via the `font-variant` attribute selector: `page.find_all('text[font-variant="AAAAAB"]')`.*

## Reading Order

Text extraction respects the natural reading order (top-to-bottom, left-to-right by default). `page.find_all('text')` returns elements already sorted this way.

```python
# Get first 5 text elements in reading order
elements_in_order = page.find_all('text')
elements_in_order[:5]
```

```python
# Text extracted via page.extract_text() respects this order automatically
# (Result already shown in Basic Text Extraction section)
page.extract_text()[:100]
```

## Element Navigation

Move between elements sequentially based on reading order using `.next()` and `.previous()`.

```python
# Find an element to start from (adjust selector)
start_nav_element = page.find('text:contains("Results")')

if start_nav_element:
    page.highlight(start_nav_element, color="yellow").show()
    f"Starting navigation from: {start_nav_element.text[:30]}..."
else:
    "Could not find 'Results' element to start navigation demo."

```

```python
# Find the *very next* element (any type)
if start_nav_element:
    next_any = start_nav_element.next()
    if next_any:
        page.highlight(next_any, color="orange").show()
        f"Next element (any type): {next_any}"
```

```python
# Find the next *text* element specifically
if start_nav_element:
    next_text = start_nav_element.next('text')
    if next_text:
        page.highlight(next_text, color="lightblue").show()
        f"Next text element: {next_text.text[:50]}..."
```

```python
# Find the *previous* element (any type)
if start_nav_element:
    prev_any = start_nav_element.prev()
    if prev_any:
        # Highlight on a fresh copy to avoid overlapping highlights
        page.copy().highlight(start_nav_element, color="yellow").highlight(prev_any, color="pink").show()
        f"Previous element (any type): {prev_any}"
```

```python
# Find the previous element matching a selector (e.g., large text)
if start_nav_element:
    prev_large_text = start_nav_element.prev('text[size>=14]') # Adjust selector/size
    if prev_large_text:
        page.copy().highlight(start_nav_element, color="yellow").highlight(prev_large_text, color="lightgreen").show()
        f"Previous large text element: {prev_large_text.text[:50]}..."

```

## Next Steps

Now that you know how to extract text, you might want to explore:

- [Working with regions](../regions/index.md) for more precise extraction
- [OCR capabilities](../ocr/index.md) for scanned documents
- [Document layout analysis](../layout-analysis/index.md) for automatic structure detection
- [Document QA](../document-qa/index.md) for asking questions directly to your documents