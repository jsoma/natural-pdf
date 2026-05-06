# Getting Started with Natural PDF

Let's get Natural PDF installed and run your first extraction.

## Installation

The base install includes the core library for selectors, extraction, and spatial navigation. It does not try to install every OCR, layout, or research backend up front.

```bash
pip install natural-pdf
```

Optional dependencies can be installed individually as needed. The library will tell you what to install if something is missing.

```bash
# Bundles
pip install "natural-pdf[all]"      # Recommended feature-complete runtime install
pip install "natural-pdf[export]"   # PDF export helpers only
pip install "natural-pdf[paddle]"   # PaddleOCR stack (paddlepaddle + paddleocr + paddlex) — includes paddlevl engine

# Individual packages
pip install rapidocr                # Default OCR backend
pip install easyocr                 # EasyOCR engine
pip install "surya-ocr<0.15"        # Surya OCR engine
pip install python-doctr            # Doctr OCR engine
pip install doclayout_yolo          # YOLO layout detection
```

`natural-pdf[all]` means the recommended runtime bundle: the default RapidOCR backend, sentence-transformers-based semantic search, QA/extraction dependencies, YOLO layout detection, and export support. It does not include every optional backend. Engines such as EasyOCR, PaddleOCR, Surya, and Doctr remain opt-in.

If you attempt to use an engine that is missing, the library will raise an
error with the `pip install` command you need.

You can check what's installed at any time:

```bash
npdf doctor
```

`npdf list` is kept as an alias for the same diagnostic output.

## Your First PDF Extraction

Here's a quick example to make sure everything is working:

```python
from natural_pdf import PDF

# Open a PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")

# Get the first page
page = pdf.pages[0]

# Extract all text
text = page.extract_text()
print(text)

# Find something specific
title = page.find('text:bold')
if title:
    print(f"Found title: {title.extract_text()}")
```

## What's Next?

Now that you have Natural PDF installed, you can:

- Follow the [Quickstart](../getting-started/quickstart.md) guide
- Learn to [find elements](../tutorials/02-finding-elements.ipynb) in PDFs
- See how to [extract text](../tutorials/01-loading-and-extraction.ipynb)
