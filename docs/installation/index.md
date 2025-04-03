# Getting Started with Natural PDF

Let's get Natural PDF installed and run your first extraction.

## Installation

The base installation includes the core library and necessary AI dependencies (like PyTorch and Transformers):

```bash
pip install natural-pdf
```

### Optional Dependencies

Natural PDF has modular dependencies for different features. Install them based on your needs:

```bash
# --- OCR Engines ---
# Install support for EasyOCR
pip install natural-pdf[easyocr]

# Install support for PaddleOCR (requires paddlepaddle)
pip install natural-pdf[paddle]

# Install support for Surya OCR
pip install natural-pdf[surya]

# --- Layout Detection ---
# Install support for YOLO layout model
pip install natural-pdf[layout_yolo]

# --- Interactive Widget ---
# Install support for the interactive .viewer() widget in Jupyter
pip install natural-pdf[interactive]

# --- All Features ---
# Install all optional dependencies
pip install natural-pdf[all]
```

## Your First PDF Extraction

Here's a quick example to make sure everything is working:

```python
from natural_pdf import PDF

# Open a PDF
pdf = PDF('your_document.pdf')

# Get the first page
page = pdf.pages[0]

# Extract all text
text = page.extract_text()
print(text)

# Find something specific
title = page.find('text:bold')
print(f"Found title: {title.text}")
```

## What's Next?

Now that you have Natural PDF installed, you can:

- Learn to [navigate PDFs](../pdf-navigation/index.ipynb)
- Explore how to [select elements](../element-selection/index.ipynb)
- See how to [extract text](../text-extraction/index.ipynb)