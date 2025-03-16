# PDF Navigation

This guide covers the basics of working with PDFs in Natural PDF - opening documents, accessing pages, and navigating through content.

## Opening a PDF

The main entry point to Natural PDF is the `PDF` class:

```python
from natural_pdf import PDF

# Open a PDF file
pdf = PDF('document.pdf')

# Use a context manager to automatically close the file
with PDF('document.pdf') as pdf:
    # Work with the PDF here
    pass  # File closes automatically when the block exits
    
# Customize with non-default options when needed
pdf = PDF(
    'document.pdf',
    keep_spaces=False,  # Don't preserve spaces in text elements
    font_attrs=['fontname', 'size', 'color']  # Include color in font grouping
)
```

## Accessing Pages

Once you have a PDF object, you can access its pages:

```python
# Get the total number of pages
num_pages = len(pdf)
print(f"This PDF has {num_pages} pages")

# Get a specific page (0-indexed)
first_page = pdf.pages[0]
last_page = pdf.pages[-1]

# Iterate through all pages
for page in pdf.pages:
    print(f"Page {page.page_number} has dimensions {page.width} x {page.height}")
```

## Page Properties

Each `Page` object has useful properties:

```python
# Page dimensions in points (1/72 inch)
width = page.width
height = page.height

# Page number (1-indexed as shown in PDF viewers)
page_number = page.page_number

# Page index (0-indexed position in the PDF)
page_index = page.page_index
```

## PDF Configuration Options

When opening a PDF, you can configure various behaviors:

```python
# Configure text handling
pdf = PDF(
    'document.pdf',
    
    # Reading order options
    reading_order=True,  # Sort elements in reading order (default: True)
    
    # Text preservation options
    keep_spaces=True,    # Keep spaces in word elements (default: True)
    
    # Font handling
    font_attrs=['fontname', 'size', 'bold'],  # Group text by these font attributes
    
    # OCR configuration
    ocr={
        "enabled": "auto",  # Automatically use OCR when needed
        "languages": ["en"],  # Languages to use for OCR
        "min_confidence": 0.5  # Confidence threshold for OCR
    },
    
    # OCR engine selection
    ocr_engine="easyocr"  # Use EasyOCR engine (default)
)
```

## Working Across Pages

Natural PDF makes it easy to work with content across multiple pages:

```python
# Extract text from all pages
all_text = pdf.extract_text()

# Find elements across all pages
all_headings = pdf.find_all('text[size>=14]:bold')

# Add exclusion zones to all pages (like headers/footers)
pdf.add_exclusion(
    lambda page: page.find('text:contains("CONFIDENTIAL")').above() if page.find('text:contains("CONFIDENTIAL")') else None,
    label="header"
)
```

## The Page Collection

The `pdf.pages` object is a `PageCollection` that allows batch operations on pages:

```python
# Get sections across all pages
sections = pdf.pages.get_sections(
    start_elements='text[size>=14]:bold',
    new_section_on_page_break=True  # Start a new section on page boundaries
)

# Extract text from specific pages
text = pdf.pages[2:5].extract_text()

# Find elements across specific pages
elements = pdf.pages[2:5].find_all('text:contains("Annual Report")')
```

## Working with Multiple Pages

Here are some ways to handle content that spans across multiple pages:

```python
# Extract text from all pages with a consistent format
all_text = pdf.extract_text()

# Find all instances of a phrase across all pages
all_occurrences = pdf.find_all('text:contains("Revenue")')
print(f"Found {len(all_occurrences)} occurrences across the document")

# Group by page number
by_page = {}
for element in all_occurrences:
    page_num = element.page.page_number
    if page_num not in by_page:
        by_page[page_num] = []
    by_page[page_num].append(element)

# Print occurrences by page
for page_num, elements in by_page.items():
    print(f"Page {page_num}: {len(elements)} occurrences")
```

## Document Sections Across Pages

You can extract sections that span across multiple pages:

```python
# Get sections with headings as section starts
sections = pdf.pages.get_sections(
    start_elements='text[size>=14]:bold',
    new_section_on_page_break=True  # Optional: Create new sections at page boundaries
)

# Process each section
for i, section in enumerate(sections):
    print(f"Section {i+1}:")
    if hasattr(section, 'start_element') and section.start_element:
        print(f"  Starts with: {section.start_element.text}")
    print(f"  Content: {section.extract_text()[:50]}...")
    print(f"  Spans pages: {section.page.page_number}")
```

## Using OCR Across Pages

For scanned documents, you can apply OCR across multiple pages:

```python
# Enable OCR for the document
pdf = PDF('scanned_document.pdf', ocr=True)

# Apply OCR to all pages
for page in pdf.pages:
    page.apply_ocr()
    
# Extract text from the entire document
all_text = pdf.extract_text()

# Find OCR text elements across all pages
ocr_elements = pdf.find_all('text[source=ocr]')
```

## Complete Multi-page Navigation Example

Here's a complete example of working with a multi-page document:

```python
from natural_pdf import PDF

# Open a PDF
pdf = PDF('annual_report.pdf')
print(f"Document has {len(pdf)} pages")

# Find all headings across the document
headings = pdf.find_all('text[size>=14]:bold')
print(f"Found {len(headings)} headings across all pages")

# Group headings by page
heading_by_page = {}
for heading in headings:
    page_num = heading.page.page_number
    if page_num not in heading_by_page:
        heading_by_page[page_num] = []
    heading_by_page[page_num].append(heading)

# Print headings by page
for page_num in sorted(heading_by_page.keys()):
    print(f"\nPage {page_num} headings:")
    for heading in heading_by_page[page_num]:
        print(f"  - {heading.text}")

# Find the financial section across all pages
financial_heading = pdf.find('text:contains("Financial")')
if financial_heading:
    print(f"\nFound Financial section on page {financial_heading.page.page_number}")
    
    # Extract the financial section
    financial_section = financial_heading.below()
    financial_text = financial_section.extract_text()
    print(f"Financial section excerpt: {financial_text[:200]}...")
    
    # Look for the next heading in the document (might be on next page)
    next_heading = None
    for heading in headings:
        if heading.page.page_number > financial_heading.page.page_number or (
            heading.page.page_number == financial_heading.page.page_number and 
            heading.top > financial_heading.top
        ):
            next_heading = heading
            break
            
    if next_heading:
        print(f"Next section is '{next_heading.text}' on page {next_heading.page.page_number}")

# Extract text from a page range
pages_3_to_5_text = pdf.pages[2:5].extract_text()  # Pages 3-5 (0-indexed)
print(f"\nText from pages 3-5 (excerpt): {pages_3_to_5_text[:200]}...")
```

## Next Steps

Now that you know how to navigate PDFs, you can:

- [Find elements using selectors](../element-selection/index.md)
- [Extract text from your documents](../text-extraction/index.md)
- [Work with specific regions](../regions/index.md)