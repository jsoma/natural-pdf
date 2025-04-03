# PDF Navigation

This guide covers the basics of working with PDFs in Natural PDF - opening documents, accessing pages, and navigating through content.

## Opening a PDF

The main entry point to Natural PDF is the `PDF` class:

```python
from natural_pdf import PDF

# Open a PDF file
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/0500000US42001.pdf")
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

# Iterate through the first 20 pages
for page in pdf.pages[:20]:
    print(f"Page {page.number} has {len(page.extract_text())} characters")
```

## Page Properties

Each `Page` object has useful properties:

```python
# Page dimensions in points (1/72 inch)
print(page.width, page.height)

# Page number (1-indexed as shown in PDF viewers)
print(page.number)

# Page index (0-indexed position in the PDF)
print(page.index)
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
# Extract text from specific pages
text = pdf.pages[2:5].extract_text()

# Find elements across specific pages
elements = pdf.pages[2:5].find_all('text:contains("Annual Report")')
```

## Document Sections Across Pages

You can extract sections that span across multiple pages:

```python
# Get sections with headings as section starts
sections = pdf.pages.get_sections(
    start_elements='text[size>=14]:bold',
    new_section_on_page_break=False
)
```

## Next Steps

Now that you know how to navigate PDFs, you can:

- [Find elements using selectors](../element-selection/index.ipynb)
- [Extract text from your documents](../text-extraction/index.ipynb)
- [Work with specific regions](../regions/index.ipynb)