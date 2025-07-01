# Natural PDF

[![CI](https://github.com/jsoma/natural-pdf/actions/workflows/ci.yml/badge.svg)](https://github.com/jsoma/natural-pdf/actions/workflows/ci.yml)

A friendly library for working with PDFs, built on top of [pdfplumber](https://github.com/jsvine/pdfplumber).

Natural PDF lets you find and extract content from PDFs using simple code that makes sense.

- [Complete documentation here](https://jsoma.github.io/natural-pdf)
- [Live demos here](https://colab.research.google.com/github/jsoma/natural-pdf/)

<div style="max-width: 400px; margin: auto"><a href="sample-screen.png"><img src="sample-screen.png"></a></div>

## Installation

```bash
pip install natural-pdf
```

Need OCR engines, layout models, or other heavy add-ons? Install the **core** once, then use the helper `npdf` command to pull in exactly what you need:

```bash
# Everything you need for classification, document-QA, semantic search, etc.
npdf install ai

# Surya OCR and the YOLO Doc-Layout detector in one go
npdf install surya yolo

# add PaddleOCR (+paddlex) after the fact
npdf install paddle

# see what's already on your machine
npdf list
```

Lightweight extras such as `deskew` or `search` can still be added with
classic `pip install`:

```bash
pip install "natural-pdf[deskew]"
pip install "natural-pdf[search]"
pip install "natural-pdf[ai]"
```

More details in the [installation guide](https://jsoma.github.io/natural-pdf/installation/).

## Quick Start

```python
from natural_pdf import PDF

# Open a PDF
pdf = PDF('https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf')
page = pdf.pages[0]

# Extract all of the text on the page
page.extract_text()

# Find elements using CSS-like selectors
heading = page.find('text:contains("Summary"):bold')

# Extract content below the heading
content = heading.below().extract_text()

# Examine all the bold text on the page
page.find_all('text:bold').show()

# Exclude parts of the page from selectors/extractors
header = page.find('text:contains("CONFIDENTIAL")').above()
footer = page.find_all('line')[-1].below()
page.add_exclusion(header)
page.add_exclusion(footer)

# Extract clean text from the page ignoring exclusions
clean_text = page.extract_text()
```

And as a fun bonus, `page.viewer()` will provide an interactive method to explore the PDF.

## Key Features

Natural PDF offers a range of features for working with PDFs:

*   **CSS-like Selectors:** Find elements using intuitive query strings (`page.find('text:bold')`).
*   **Spatial Navigation:** Select content relative to other elements (`heading.below()`, `element.select_until(...)`).
*   **Text & Table Extraction:** Get clean text or structured table data, automatically handling exclusions.
*   **OCR Integration:** Extract text from scanned documents using engines like EasyOCR, PaddleOCR, or Surya.
*   **Layout Analysis:** Detect document structures (titles, paragraphs, tables) using various engines (e.g., YOLO, Paddle, LLM via API).
*   **Document QA:** Ask natural language questions about your document's content.
*   **Semantic Search:** Index PDFs and find relevant pages or documents based on semantic meaning using Haystack.
*   **Visual Debugging:** Highlight elements and use an interactive viewer or save images to understand your selections.

## Learn More

Dive deeper into the features and explore advanced usage in the [**Complete Documentation**](https://jsoma.github.io/natural-pdf).

## Best friends

Natural PDF sits on top of a *lot* of fantastic tools and mdoels, some of which are:

- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [EasyOCR](https://www.jaided.ai/easyocr/)
- [PaddleOCR](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html)
- [Surya](https://github.com/VikParuchuri/surya)
- A specific [YOLO](https://github.com/opendatalab/DocLayout-YOLO)
- [deskew](https://github.com/sbrunner/deskew)
- [doctr](https://github.com/mindee/doctr)
- [docling](https://github.com/docling-project/docling)
- [Hugging Face](https://huggingface.co/models)