# Natural PDF

A friendly library for working with PDFs, built on top of [pdfplumber](https://github.com/jsvine/pdfplumber).

Natural PDF lets you find and extract content from PDFs using simple code that makes sense.

- [Complete documentation here](https://jsoma.github.io/natural-pdf)
- [Live demos here](https://colab.research.google.com/github/jsoma/natural-pdf/)

<div style="max-width: 400px; margin: auto"><a href="sample-screen.png"><img src="sample-screen.png"></a></div>

## Installation

```bash
pip install natural-pdf
```

For optional features like specific OCR engines, layout analysis models, or the interactive Jupyter widget, you can install extras:

```bash
# Example: Install with EasyOCR support
pip install natural-pdf[easyocr]
pip install natural-pdf[surya]
pip install natural-pdf[paddle]

# Example: Install with interactive viewer support
pip install natural-pdf[interactive]

# Example: Install with semantic search support (Haystack)
pip install natural-pdf[haystack]

# Install everything
pip install natural-pdf[all]
```

See the [installation guide](https://jsoma.github.io/natural-pdf/installation/) for more details on extras.

## Quick Start

```python
from natural_pdf import PDF

# Open a PDF
pdf = PDF('document.pdf')
page = pdf.pages[0]

# Find elements using CSS-like selectors
heading = page.find('text:contains("Summary"):bold')

# Extract content below the heading
content = heading.below().extract_text()
print("Content below Summary:", content[:100] + "...")

# Exclude headers/footers automatically (example)
# You might define these based on common text or position
page.add_exclusion(page.find('text:contains("CONFIDENTIAL")').above())
page.add_exclusion(page.find_all('line')[-1].below())

# Extract clean text from the page
clean_text = page.extract_text()
print("\nClean page text:", clean_text[:200] + "...")

# Highlight the heading and view the page
heading.highlight(color='red')
page.to_image()
```

And as a fun bonus, `page.viewer()` will provide an interactive method to explore the PDF.

## Key Features

Natural PDF offers a range of features for working with PDFs:

*   **CSS-like Selectors:** Find elements using intuitive query strings (`page.find('text:bold')`).
*   **Spatial Navigation:** Select content relative to other elements (`heading.below()`, `element.select_until(...)`).
*   **Text & Table Extraction:** Get clean text or structured table data, automatically handling exclusions.
*   **OCR Integration:** Extract text from scanned documents using engines like EasyOCR, PaddleOCR, or Surya.
*   **Layout Analysis:** Detect document structures (titles, paragraphs, tables) using AI models.
*   **Document QA:** Ask natural language questions about your document's content.
*   **Semantic Search:** Index PDFs and find relevant pages or documents based on semantic meaning using Haystack.
*   **Visual Debugging:** Highlight elements and use an interactive viewer or save images to understand your selections.

## Learn More

Dive deeper into the features and explore advanced usage in the [**Complete Documentation**](https://jsoma.github.io/natural-pdf).
