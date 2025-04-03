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

pdf = PDF('document.pdf')
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

## Key Features

Here are a few highlights of what you can do:

### Find Elements with Selectors

Use CSS-like selectors to find text, shapes, and more.

```python
# Find bold text containing "Revenue"
page.find('text:contains("Revenue"):bold').extract_text()

# Find all large text
page.find_all('text[size>=12]').extract_text()
```

[Learn more about selectors →](element-selection/index.ipynb)

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

[Explore more navigation methods →](pdf-navigation/index.ipynb)

### Extract Clean Text

Easily extract text content, automatically handling common page elements like headers and footers (if exclusions are set).

```python
# Extract all text from the page (respecting exclusions)
page_text = page.extract_text()

# Extract text from a specific region
some_region = page.find(...)
region_text = some_region.extract_text()
```

[Learn about text extraction →](text-extraction/index.ipynb)
[Learn about exclusion zones →](regions/index.ipynb#exclusion-zones)

### Apply OCR

Extract text from scanned documents using various OCR engines.

```python
# Apply OCR using the default engine
ocr_elements = page.apply_ocr()

# Extract text (will use OCR results if available)
text = page.extract_text()
```

[Explore OCR options →](ocr/index.md)

### Analyze Document Layout

Use AI models to detect document structures like titles, paragraphs, and tables.

```python
# Detect document structure
page.analyze_layout()

# Highlight titles and tables
page.find_all('region[type=title]').highlight(color="purple")
page.find_all('region[type=table]').highlight(color="blue")

# Extract data from the first table
table_data = page.find('region[type=table]').extract_table()
```

[Learn about layout models →](layout-analysis/index.ipynb)
[Working with tables? →](tables/index.ipynb)

### Document Question Answering

Ask natural language questions directly to your documents.

```python
# Ask a question
result = pdf.ask("What was the company's revenue in 2022?")
if result.get("found", False):
    print(f"Answer: {result['answer']}")
```

[Learn about Document QA →](document-qa/index.ipynb)

### Visualize Your Work

Debug and understand your extractions visually.

```python
# Highlight headings
page.find_all('text[size>=14]').highlight(color="red", label="Headings")

# Launch the interactive viewer (Jupyter)
# Requires: pip install natural-pdf[interactive]
page.viewer()

# Or save an image
# page.save_image("highlighted.png")
```

[See more visualization options →](visual-debugging/index.ipynb)

## Documentation Topics

Choose what you want to learn about:

### Task-based Guides
- [Getting Started](installation/index.md): Install the library and run your first extraction
- [PDF Navigation](pdf-navigation/index.ipynb): Open PDFs and work with pages
- [Element Selection](element-selection/index.ipynb): Find text and other elements using selectors
- [Text Extraction](text-extraction/index.ipynb): Extract clean text from documents
- [Regions](regions/index.ipynb): Work with specific areas of a page
- [Visual Debugging](visual-debugging/index.ipynb): See what you're extracting
- [OCR](ocr/index.md): Extract text from scanned documents
- [Layout Analysis](layout-analysis/index.ipynb): Detect document structure
- [Tables](tables/index.ipynb): Extract tabular data
- [Document QA](document-qa/index.ipynb): Ask questions to your documents

### Reference
- [API Reference](api/index.md): Complete library reference