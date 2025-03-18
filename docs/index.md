# Natural PDF

A friendly library for working with PDFs, built on top of [pdfplumber](https://github.com/jsvine/pdfplumber).

Natural PDF lets you find and extract content from PDFs using simple code that makes sense.

- [Live demo here](https://colab.research.google.com/github/jsoma/natural-pdf/blob/main/notebooks/Examples.ipynb)

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

## What can you do with Natural PDF?

- Find text using CSS-like selectors (like `page.find('text:contains("Revenue"):bold')`)
- Navigate spatially (like `heading.below()` to get content below a heading)
- Extract text while automatically excluding headers and footers
- Visualize what's happening to debug your extraction
- Apply OCR to scanned documents
- Detect tables, headings, and other document structures using AI models
- Ask natural language questions to your documents

## Core Features

### CSS-like Selectors

```python
# Find and extract text from bold elements containing "Revenue"
page.find('text:contains("Revenue"):bold').extract_text()

# Extract all large text
page.find_all('text[size>=12]').extract_text()

# Highlight red rectangles
page.find_all('rect[color~=red]').highlight(color="red")

# Find text with specific font and extract it
page.find_all('text[fontname*=Arial]').extract_text()

# Highlight high-confidence OCR text
page.find_all('text[source=ocr][confidence>=0.8]').highlight(label="High Confidence OCR")
```

Selectors support attribute matching, pseudo-classes, and content searches. [Learn more about selectors →](element-selection/index.md)

Curious about those weird font names like 'AAAAAB+Arial'? [Explore PDF font handling →](explanations/pdf-fonts.md)

### Spatial Navigation

```python
# Extract text below a heading
intro_text = page.find('text:contains("Introduction")').below().extract_text()

# Extract text from one heading to another
methods_text = page.find('text:contains("Methods")').below(
    until='text:contains("Results")',
    include_until=False
).extract_text()

# Extract content above a footer
main_text = page.find('text:contains("Page 1 of 10")').above().extract_text()
```

Navigate PDFs spatially instead of using coordinates. [Explore more navigation methods →](pdf-navigation/index.md)

Working with headers and footers? [Learn about exclusion zones →](regions/index.md#exclusion-zones)

### Document Layout Analysis

```python
# Detect document structure using AI models
page.analyze_layout()

# Highlight titles and tables with different colors
page.find_all('region[type=title]').highlight(color="purple", label="Titles")
page.find_all('region[type=table]').highlight(color="blue", label="Tables")

# Extract text from paragraphs
paragraph_text = page.find_all('region[type=plain-text]').extract_text()

# Extract data from the first table as a list of rows
table_data = page.find('region[type=table]').extract_table()
```

Natural PDF supports multiple layout models including YOLO for general document analysis and Table Transformer (TATR) for detailed table structure. [Learn about layout models →](layout-analysis/index.md)

Working with tables? [See specialized table extraction methods →](tables/index.md)

### Document Question Answering

```python
# Ask questions directly to your documents
result = pdf.ask("What was the company's revenue in 2022?")
if result.get("found", False):
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Highlight where the answer was found
    if "source_elements" in result:
        for element in result["source_elements"]:
            element.highlight(color="orange")
            
    # Display the answer location
    image = pdf.pages[result.get('page_num', 0) - 1].to_image()
    image
```

Document QA uses LayoutLM models that understand both text content and visual layout. Unlike general LLMs, the answers come directly from your document without hallucinations. [Learn about Document QA →](document-qa/index.md)

Having OCR problems? [Understand OCR challenges and solutions →](explanations/ocr-challenges.md)

### OCR Support

```python
# Open a scanned document with OCR enabled
pdf = PDF('scanned_document.pdf', ocr=True)

# Extract text with automatic OCR
text = pdf.pages[0].extract_text()

# Apply OCR explicitly
ocr_elements = page.apply_ocr()
print(f"Found {len(ocr_elements)} OCR text elements")
```

Natural PDF supports both EasyOCR and PaddleOCR engines. PaddleOCR is often more accurate while EasyOCR is simpler to set up. [Explore OCR options →](ocr/index.md)

Got scanned documents or images? [See how to optimize OCR results →](explanations/ocr-challenges.md)

### Visual Debugging

```python
# Highlight different elements
page.find_all('text[size>=14]').highlight(color="red", label="Headings")
page.find_all('rect').highlight(color="green", label="Boxes")
page.find_all('line').highlight(color="blue", label="Lines")

# Highlight layout regions
page.analyze_layout()
page.find_all('region[type=table]').highlight(color="orange", label="Tables")
page.find_all('region[type=title]').highlight(color="purple", label="Titles")

# Get the visualization as an image (great for Jupyter notebooks)
image = page.to_image(labels=True)
image

# Or save the image if needed
# page.save_image("highlighted.png", labels=True)
```

Visualizing elements helps debug extraction issues and understand document structure. Natural PDF provides color-coding and labels to make it clear what's being detected. [See more visualization options →](visual-debugging/index.md)

Having trouble with PDF extraction? [Understand common PDF challenges →](explanations/pdf-extraction-challenges.md)

## Page Sections

Here's how to split pages into logical sections for extracting structured content:

```python
# Simple approach: Get content between headings
intro_text = page.find('text:contains("Introduction")').below(
    until='text:contains("Methods")', include_until=False
).extract_text()

# Get sections based on headings
sections = page.get_sections(start_elements='text[size>=14]:bold')

# Process each section
for section in sections:
    # Extract text from the section
    section_text = section.extract_text()
    print(f"Section text: {section_text[:50]}...")
    
    # Highlight the section
    section.highlight()
    
# Get sections across multiple pages
doc_sections = pdf.pages.get_sections(
    start_elements='text[size>=14]:bold',
    new_section_on_page_break=True
)
```

Sections help break down documents into logical chunks. Use them to extract structured content like chapters, articles, or report sections. [Learn more about sectioning →](regions/index.md#document-sections)

Need to extract specific document components? [See layout analysis for automatic structure detection →](layout-analysis/index.md)

## Advanced Example

Here's a more complex example that uses multiple features:

```python
from natural_pdf import PDF
import os

# Create output directory
os.makedirs("output", exist_ok=True)

# Open a PDF
pdf = PDF("annual_report.pdf")

# Add exclusion zones for header and footer
pdf.add_exclusion(
    lambda page: page.find('text:contains("CONFIDENTIAL")').above() if page.find('text:contains("CONFIDENTIAL")') else None,
    label="header"
)
pdf.add_exclusion(
    lambda page: page.find('text:contains("Page")').below() if page.find('text:contains("Page")') else None,
    label="footer"
)

# Find financial section
financial_heading = pdf.find('text:contains("Financial Results")')
if financial_heading:
    page = financial_heading.page
    
    # Run layout analysis
    page.analyze_layout()
    
    # Get the region below the heading
    financial_section = financial_heading.below(height=300)
    financial_section.highlight(color="yellow", label="Financial Section")
    
    # Find tables in or near this section
    tables = page.find_all('region[type=table]')
    tables_in_section = [table for table in tables if financial_section.intersects(table)]
    
    if tables_in_section:
        # Highlight and extract tables
        for i, table in enumerate(tables_in_section):
            table.highlight(color="teal", label=f"Table {i+1}")
            
            # Extract table data
            data = table.extract_table()
            print(f"\nTable {i+1}:")
            for row in data:
                print(row)
    
    # Ask questions about the financial section
    questions = [
        "What was the total revenue?",
        "What was the net income?",
        "What was the year-over-year growth?"
    ]
    
    print("\nDocument QA Results:")
    for question in questions:
        result = financial_section.ask(question)
        if result.get("found", False):
            print(f"Q: {question}")
            print(f"A: {result['answer']} (confidence: {result['confidence']:.2f})")
            
            # Highlight the answer
            if "source_elements" in result:
                for elem in result["source_elements"]:
                    elem.highlight(color="red", label=f"Answer: {question}")
    
    # Get the highlighted page as an image
    image = page.to_image(labels=True)
    # Just return the image as the last line in a Jupyter cell
    image
    
    # Extract text from the financial section
    financial_text = financial_section.extract_text()
    print(f"\nExtracted text from Financial Section ({len(financial_text)} characters):")
    print(financial_text[:200] + "..." if len(financial_text) > 200 else financial_text)
```

## Documentation Topics

Choose what you want to learn about:

### Task-based Guides
- [Getting Started](installation/index.md): Install the library and run your first extraction
- [PDF Navigation](pdf-navigation/index.md): Open PDFs and work with pages
- [Element Selection](element-selection/index.md): Find text and other elements using selectors
- [Text Extraction](text-extraction/index.md): Extract clean text from documents
- [Regions](regions/index.md): Work with specific areas of a page
- [Visual Debugging](visual-debugging/index.md): See what you're extracting
- [OCR](ocr/index.md): Extract text from scanned documents
- [Layout Analysis](layout-analysis/index.md): Detect document structure
- [Tables](tables/index.md): Extract tabular data
- [Document QA](document-qa/index.md): Ask questions to your documents

### Understanding PDFs
- [PDF Explanations](explanations/index.md): Deep dives into PDF extraction challenges, fonts, OCR, and more

### Reference
- [API Reference](api/index.md): Complete library reference