# Natural PDF

A friendly library for working with PDFs, built on top of [pdfplumber](https://github.com/jsvine/pdfplumber).

Natural PDF lets you find and extract content from PDFs using simple code that makes sense.

- [Complete documentation here](https://jsoma.github.io/natural-pdf)
- [Live demo here](https://colab.research.google.com/github/jsoma/natural-pdf/blob/main/notebooks/Examples.ipynb)

## Features

- **Fluent API** for chaining operations
- **CSS-like selectors** for finding elements
- **Spatial navigation** with intuitive methods like `above()`, `below()`, and `select_until()`
- **Element collections** for batch operations
- **Visual highlighting** for debugging
- **Region visualization** with direct image extraction of specific regions
- **Text style analysis** for document structure
- **Exclusion zones** for headers, footers, and other areas to ignore
- **OCR integration** for extracting text from scanned documents
- **Document layout analysis** for detecting document structure with ML models
- **Table extraction** with multiple detection methods
- **Structured logging** with configurable levels and handlers

## Installation

```bash
pip install natural-pdf
```

or if you're picky...

```bash
# Minimal installation without AI models (faster, smaller)
pip install natural-pdf[core]

# With all OCR engines
pip install natural-pdf[easyocr,paddle]
```

## Quick Start

```python
from natural_pdf import PDF

# Open a local PDF
pdf = PDF('document.pdf')

# Or open a PDF from a URL
pdf = PDF('https://example.com/document.pdf')

# Get the first page
page = pdf.pages[0]

# Find elements using CSS-like selectors
heading = page.find('text:contains("Summary"):bold')

# Extract content below the heading
content = heading.below().extract_text()
print(content)

# Exclude headers and footers
page.add_exclusion(page.find('text:contains("CONFIDENTIAL")').above())
page.add_exclusion(page.find_all('line')[-1].below())

# Extract clean text
clean_text = page.extract_text()
print(clean_text)
```

## Selectors

The library supports CSS-like selectors for finding elements:

```python
# Find text containing a specific string
element = page.find('text:contains("Revenue")')

# Find bold text with a specific font size
headings = page.find_all('text[size>=12]:bold')

# Find thick red lines
lines = page.find_all('line[width>=2][color~=(1,0,0)]')
```

## Spatial Navigation

Navigate through the document with intuitive spatial methods:

```python
# Get content below a heading
heading = page.find('text:contains("Introduction")')
content = heading.below().extract_text()

# Get content from one element to another
start = page.find('text:contains("Start")')
end = page.find('text:contains("End")')
region = start.select_until(end)
content = region.extract_text()
```

## Exclusion Zones

Exclude headers, footers, or other areas from extraction:

```python
# Page-level exclusion
page.add_exclusion(page.find('text:contains("Page")').above())
page.add_exclusion(page.find_all('line')[-1].below())

# PDF-level exclusion with lambdas
pdf.add_exclusion(
    lambda page: page.find('text:contains("Header")').above(),
    label="headers"
)

# Extract text with exclusions applied
text = pdf.extract_text()

# Extract from a specific region with exclusions
summary = page.find('text:contains("Summary")')
conclusion = page.find('text:contains("Conclusion")')
region = page.create_region(summary.x0, summary.top, conclusion.x1, conclusion.bottom)
region_text = region.extract_text(apply_exclusions=True)  # Excludes headers/footers

# Disable exclusions for a specific extraction
full_text = page.extract_text(apply_exclusions=False)
```

Exclusions work efficiently with different region types:
- Regions without intersection with exclusion zones → exclusions ignored entirely
- Rectangular regions with header/footer exclusions → optimized cropping
- Complex regions with partial exclusions → advanced filtering with warning

## OCR Integration

Extract text from scanned documents using OCR with multiple engine options:

```python
# Using the default EasyOCR engine
pdf = PDF('scanned_document.pdf', ocr={
    'enabled': 'auto',  # Only use OCR when necessary
    'languages': ['en'],
    'min_confidence': 0.5
})

# Using PaddleOCR for better Asian language support
pdf = PDF('scanned_document.pdf', 
          ocr_engine='paddleocr',
          ocr={
              'enabled': True,
              'languages': ['zh-cn', 'en'],  # Chinese and English
              'min_confidence': 0.3,
              'model_settings': {
                  'use_angle_cls': False,  # PaddleOCR-specific setting
                  'rec_batch_num': 6
              }
          })

# Extract text, OCR will be used if needed
text = page.extract_text()

# Force OCR regardless of existing text
ocr_text = page.extract_text(ocr=True)

# Find OCR-detected text with high confidence
high_confidence = page.find_all('text[source=ocr][confidence>=0.8]')

# Visualize OCR results with color-coded confidence levels
for elem in page.find_all('text[source=ocr]'):
    if elem.confidence >= 0.8:
        color = (0, 1, 0, 0.3)  # Green for high confidence
    elif elem.confidence >= 0.5:
        color = (1, 1, 0, 0.3)  # Yellow for medium confidence
    else:
        color = (1, 0, 0, 0.3)  # Red for low confidence
        
    elem.highlight(color=color, label=f"OCR ({elem.confidence:.2f})")
page.save_image('ocr_results.png', labels=True)
```

## Logging

The library includes a structured logging system to provide visibility into its operations:

```python
import logging
from natural_pdf import PDF, configure_logging

# Configure logging with INFO level to console
configure_logging(level=logging.INFO)

# Or log to a file with DEBUG level
file_handler = logging.FileHandler("natural_pdf.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
configure_logging(level=logging.DEBUG, handler=file_handler)

# Now operations will generate logs
pdf = PDF("document.pdf")
# Log: natural_pdf.core.pdf - INFO - Initializing PDF from document.pdf

# Run layout detection with verbose logging
regions = pdf.pages[0].analyze_layout(
    model="paddle",
    model_params={"verbose": True}
)
# Log: natural_pdf.analyzers.layout.paddle - INFO - Starting PaddleLayout detection...
# Log: natural_pdf.analyzers.layout.paddle - DEBUG - Parameters: confidence=0.2...
```

Logs follow a hierarchical structure matching the library's module organization:
- `natural_pdf.core` - Core PDF operations
- `natural_pdf.analyzers` - Layout analysis operations
- `natural_pdf.ocr` - OCR engine operations

## Document QA

Ask questions directly to your documents:

```python
# Ask questions about the document content
result = pdf.ask("What was the company's revenue in 2022?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")

# Access more details in the result dictionary
result = pdf.ask("Who is the CEO?")
print(f"Answer: {result['answer']}")
print(f"Found on page: {result['page_num']}")
print(f"Source text: {result.get('source_text', 'N/A')}")
```

## More details

[Complete documentation here](https://jsoma.github.io/natural-pdf)
