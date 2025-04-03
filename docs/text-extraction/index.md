# Text Extraction Guide

This guide demonstrates various ways to extract text from PDFs using Natural PDF, from simple page dumps to targeted extraction based on elements, regions, and styles.

## Setup

First, let's import necessary libraries and load a sample PDF. We'll use `example.pdf` from the tutorials' `pdfs` directory. *Adjust the path if your setup differs.*

```python
from natural_pdf import PDF

# Load the PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")

# Select the first page for initial examples
page = pdf.pages[0]

# Display the first page
page.show(width=700)
```

## Basic Text Extraction

Get all text from a page or the entire document.

```python
# Extract all text from the first page
# Displaying first 500 characters
print(page.extract_text()[:500])
```

You can also preserve layout with `layout=True`.

```python
# Extract text from the entire document (may take time)
# Uncomment to run:
print(page.extract_text(layout=True)[:2000])
```

## Extracting Text from Specific Elements

Use selectors with `find()` or `find_all()` to target specific elements. *Selectors like `:contains("Summary")` are examples; adapt them to your PDF.*

```python
# Find a single element, e.g., a title containing "Summary"
# Adjust selector as needed
date_element = page.find('text:contains("Site")')
date_element # Display the found element object
```

```python
date_element.show()
```

```python
date_element.text
```

```python
# Find multiple elements, e.g., bold headings (size >= 8)
heading_elements = page.find_all('text[size>=8]:bold')
heading_elements 
```

```python
page.find_all('text[size>=8]:bold').show()
```

```python
# Pull out all of their text (why? I don't know!)
print(heading_elements.extract_text())
```

## Advanced text searches

```python
# Exact phrase (case-sensitive)
page.find('text:contains("Hazardous Materials")').text
```

```python
# Exact phrase (case-sensitive)
page.find('text:contains("HAZARDOUS MATERIALS")', case=False).text
```

```python
# Regular expression (e.g., "YYYY Report")
regex = "\d+, \d{4}"
page.find(f'text:contains("{regex}")', regex=True)
```

```python
# Regular expression (e.g., "YYYY Report")
page.find_all('text[fontname="Helvetica"][size=10]')
```

# Regions

```python
# Region below an element (e.g., below "Introduction")
# Adjust selector as needed
page.find('text:contains("Summary")').below(include_element=True).show()
```

```python
(
    page
    .find('text:contains("Summary")')
    .below(include_element=True)
    .extract_text()
    [:500]
)
```

```python
(
    page
    .find('text:contains("Summary")')
    .below(include_element=True, until='line:horizontal')
    .show()
)
```

```python
# Manually defined region via coordinates (x0, top, x1, bottom)
manual_region = page.create_region(30, 60, 600, 300)
manual_region.show()
```

```python
# Extract text from the manual region
manual_region.extract_text()[:500]
```

## Filtering Out Headers and Footers

Use Exclusion Zones to remove unwanted content before extraction. *Adjust selectors for typical header/footer content.*

```python
header_content = page.find('rect')
footer_content = page.find_all('line')[-1].below()

header_content.highlight()
footer_content.highlight()
page.to_image()
```

```python
page.extract_text()[:500]
```

```python
page.add_exclusion(header_content)
page.add_exclusion(footer_content)
```

```python
page.extract_text()[:500]
```

```python
full_text_no_exclusions = page.extract_text(use_exclusions=False)
clean_text = page.extract_text()
f"Original length: {len(full_text_no_exclusions)}, Excluded length: {len(clean_text)}"
```

```python
page.clear_exclusions()
```

*Exclusions can also be defined globally at the PDF level using `pdf.add_exclusion()` with a function.*

## Controlling Whitespace

Manage how spaces and blank lines are handled during extraction using `layout`.

```python
print(page.extract_text())
```

```python
print(page.extract_text(use_exclusions=False, layout=True))
```

### Font Information Access

Inspect font details of text elements.

```python
# Find the first text element on the page
first_text = page.find_all('text')[1]
first_text # Display basic info
```

```python
# Highlight the first text element
first_text.show()
```

```python
# Get detailed font properties dictionary
first_text.font_info()
```

```python
# Check specific style properties directly
f"Is Bold: {first_text.bold}, Is Italic: {first_text.italic}, Font: {first_text.fontname}, Size: {first_text.size}"
```

```python
# Find elements by font attributes (adjust selectors)
# Example: Find Arial fonts
arial_text = page.find_all('text[fontname*=Helvetica]')
arial_text # Display list of found elements
```

```python
# Example: Find large text (e.g., size >= 16)
large_text = page.find_all('text[size>=12]')
large_text
```

```python
# Example: Find large text (e.g., size >= 16)
bold_text = page.find_all('text:bold')
bold_text
```

## Working with Font Styles

Analyze and group text elements by their computed font *style*, which combines attributes like font name, size, boldness, etc., into logical groups.

```python
# Analyze styles on the page
# This returns a dictionary mapping style names to ElementList objects
page.analyze_text_styles()
page.text_style_labels
```

```python
page.find_all('text').highlight(group_by='style_label').to_image()
```

```python
page.find_all('text[style_label="8.0pt Helvetica"]')
```

```python
page.find_all('text[fontname="Helvetica"][size=8]')
```

*Font variants (e.g., `AAAAAB+FontName`) are also accessible via the `font-variant` attribute selector: `page.find_all('text[font-variant="AAAAAB"]')`.*

## Reading Order

Text extraction respects a pathetic attempt at natural reading order (top-to-bottom, left-to-right by default). `page.find_all('text')` returns elements already sorted this way.

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
page.clear_highlights()

start = page.find('text:contains("Date")')
start.highlight(label='Date label')
start.next().highlight(label='Maybe the date', color='green')
start.next('text:contains("\d")', regex=True).highlight(label='Probably the date')

page.to_image()
```

## Next Steps

Now that you know how to extract text, you might want to explore:

- [Working with regions](../regions/index.ipynb) for more precise extraction
- [OCR capabilities](../ocr/index.md) for scanned documents
- [Document layout analysis](../layout-analysis/index.ipynb) for automatic structure detection
- [Document QA](../document-qa/index.ipynb) for asking questions directly to your documents