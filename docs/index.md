# Natural PDF

A friendly library for working with PDFs, built on top of [pdfplumber](https://github.com/jsvine/pdfplumber).

Natural PDF lets you find and extract content from PDFs using simple code that makes sense.

- [Live demo here](https://colab.research.google.com/github/jsoma/natural-pdf/blob/main/notebooks/Examples.ipynb)

<div style="max-width: 400px; margin: auto"><a href="assets/sample-screen.png"><img src="assets/sample-screen.png"></a></div>

## Installation

```
pip install natural_pdf
# All the extras
pip install "natural_pdf[all]"
```

## Quick Example

```python
from natural_pdf import PDF

pdf = PDF('https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf')
page = pdf.pages[0]

# Find the title and get content below it
title = page.find('text:contains("Summary"):bold')
content = title.below().extract_text()

# Exclude everything above 'CONFIDENTIAL' and below last line on page
page.add_exclusion(page.find('text:contains("CONFIDENTIAL")').above())
page.add_exclusion(page.find_all('line')[-1].below())

# Get the clean text without header/footer
clean_text = page.extract_text()
```

## Getting Started

### New to Natural PDF?
- **[Installation](installation/)** - Get Natural PDF installed and run your first extraction
- **[Quick Reference](quick-reference/)** - Essential commands and patterns in one place
- **[Tutorial Series](tutorials/)** - Step-by-step learning path through all features

### Learning the Basics
Follow the tutorial series to learn Natural PDF systematically:

1. [Loading and Basic Text Extraction](tutorials/01-loading-and-extraction.md)
2. [Finding Specific Elements](tutorials/02-finding-elements.md)
3. [Extracting Content Blocks](tutorials/03-extracting-blocks.md)
4. [Table Extraction](tutorials/04-table-extraction.md)
5. [Excluding Unwanted Content](tutorials/05-excluding-content.md)
6. [Document Question Answering](tutorials/06-document-qa.md)
7. [Layout Analysis](tutorials/07-layout-analysis.md)
8. [Spatial Navigation](tutorials/08-spatial-navigation.md)
9. [Section Extraction](tutorials/09-section-extraction.md)
10. [Form Field Extraction](tutorials/10-form-field-extraction.md)
11. [Enhanced Table Processing](tutorials/11-enhanced-table-processing.md)
12. [OCR Integration](tutorials/12-ocr-integration.md)
13. [Semantic Search](tutorials/13-semantic-search.md)
14. [Categorizing Documents](tutorials/14-categorizing-documents.md)

## Solving Specific Problems

### Text Extraction Issues
- **[Extract Clean Text Without Headers and Footers](extracting-clean-text/)** - Remove repeated content that's cluttering your text extraction
- **[Getting Text from Scanned Documents](ocr/)** - Use OCR to extract text from image-based PDFs

### Table Problems
- **[Fix Messy Table Extraction](fix-messy-tables/)** - Handle tables with no borders, merged cells, or poor alignment
- **[Getting Tables Out of PDFs](tables/)** - Basic to advanced table extraction techniques

### Data Extraction
- **[Extract Data from Forms and Invoices](process-forms-and-invoices/)** - Pull structured information from standardized documents
- **[Pulling Structured Data from PDFs](data-extraction/)** - Use AI to extract specific fields from any document

### Document Analysis
- **[Ask Questions to Your Documents](document-qa/)** - Use natural language to find information
- **[Categorizing Pages and Regions](categorizing-documents/)** - Automatically classify document types and content

### Finding Content
- **[Finding What You Need in PDFs](element-selection/)** - Master selectors to locate any element
- **[PDF Navigation](pdf-navigation/)** - Move around documents and work with multiple pages

### Layout and Structure
- **[Document Layout Analysis](layout-analysis/)** - Automatically detect titles, tables, and document structure
- **[Working with Regions](regions/)** - Define and work with specific areas of pages
- **[Visual Debugging](visual-debugging/)** - See what you're extracting and debug selector issues

## Key Features

### Find Elements with Selectors

Use CSS-like selectors to find text, shapes, and more.

```python
# Find bold text containing "Revenue"
page.find('text:contains("Revenue"):bold').extract_text()

# Find all large text
page.find_all('text[size>=12]').extract_text()
```

### Navigate Spatially

Move around the page relative to elements, not just coordinates.

```python
# Extract text below a specific heading
intro_text = page.find('text:contains("Introduction")').below().extract_text()

# Extract text from one heading to the next
methods_text = page.find('text:contains("Methods")').below(
    until='text:contains("Results")'
).extract_text()
```

### Extract Clean Text

Easily extract text content, automatically handling common page elements like headers and footers (if exclusions are set).

```python
# Extract all text from the page (respecting exclusions)
page_text = page.extract_text()

# Extract text from a specific region
some_region = page.find(...)
region_text = some_region.extract_text()
```

### Apply OCR

Extract text from scanned documents using various OCR engines.

```python
# Apply OCR using the default engine
ocr_elements = page.apply_ocr()

# Extract text (will use OCR results if available)
text = page.extract_text()
```

### Analyze Document Layout

Use AI models to detect document structures like titles, paragraphs, and tables.

```python
# Detect document structure
page.analyze_layout()

# Highlight titles and tables
page.find_all('region[type=title]').show()
page.find_all('region[type=table]').show()

# Extract data from the first table
table_data = page.find('region[type=table]').extract_table()
```

### Document Question Answering

Ask natural language questions directly to your documents.

```python
# Ask a question
result = page.ask("What was the company's revenue in 2022?")
if result.found:
    print(f"Answer: {result.answer}")
    result.show()  # Highlight where the answer was found
```

### Classify Pages and Regions

Categorize pages or specific regions based on their content using text or vision models.

```python
# Classify a page based on text
labels = ["invoice", "scientific article", "presentation"]
page.classify(labels, using="text")
print(f"Page Category: {page.category} (Confidence: {page.category_confidence:.2f})")

# Classify a page based on what it looks like
page.classify(labels, using="vision")
print(f"Page Category: {page.category} (Confidence: {page.category_confidence:.2f})")
```

### Visualize Your Work

Debug and understand your extractions visually.

```python
# Highlight headings
page.find_all('text[size>=14]').show(color="red", label="Headings")

# Launch the interactive viewer (Jupyter)
page.viewer()
```

## Reference Documentation

- **[Quick Reference](quick-reference/)** - Cheat sheet of essential commands and patterns
- **[API Reference](api/)** - Complete library reference

## Understanding Natural PDF

Coming soon: Conceptual guides explaining how Natural PDF thinks about PDFs and when to use different approaches.
