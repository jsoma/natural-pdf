# Getting Started with Natural PDF

Let's get Natural PDF installed and run your first extraction.

## Installation

The base installation includes the core library which will allow you to select, extract, and use spatial navigation.

```bash
pip install natural-pdf
```

But! If you want to recognize text, do page layout analysis, document q-and-a or other things, you can install optional dependencies.

```bash
# Install deskewing, OCR (surya and easyocr),
# layout analysis (yolo), and interactive browsing
pip install natural-pdf[favorites]

# Install **everything**
pip install natural-pdf[all]
```


### Optional Dependencies

Natural PDF has modular dependencies for different features. Install them based on your needs:

```bash
# Interactive PDF viewer 
pip install natural-pdf[viewer]

# Deskewing
pip install natural-pdf[deskew]

# OCR options
pip install natural-pdf[easyocr]
pip install natural-pdf[surya]
pip install natural-pdf[paddle]
pip install natural-pdf[doctr]

# Layout analysis
pip install natural-pdf[surya]
pip install natural-pdf[docling]
pip install natural-pdf[layout_yolo]
pip install natural-pdf[paddle]

# AI stuff
pip install natural-pdf[core-ml]
pip install natural-pdf[llm]

# Semantic search
pip install natural-pdf[core-ml]

# Install everything
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