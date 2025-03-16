# Getting Started with Natural PDF

Let's get Natural PDF installed and run your first extraction.

## Installation

Natural PDF is available on PyPI. The simplest way to install it is with pip:

```bash
pip install natural-pdf
```

You can also install from source:

```bash
git clone https://github.com/jsoma/natural-pdf.git
cd natural-pdf
pip install -e .
```

### Optional Dependencies

Natural PDF has modular dependencies for different features:

```bash
# Install OCR support with EasyOCR
pip install natural-pdf[easyocr]

# Install PaddleOCR support
pip install natural-pdf[paddle]

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
if title:
    print(f"Found title: {title.text}")
```

## What's Next?

Now that you have Natural PDF installed, you can:

- Learn to [navigate PDFs](../pdf-navigation/index.md)
- Explore how to [select elements](../element-selection/index.md)
- See how to [extract text](../text-extraction/index.md)