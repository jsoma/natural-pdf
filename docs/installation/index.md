# Getting Started with Natural PDF

Let's get Natural PDF installed and run your first extraction.

## Installation

The base installation includes the core library which will allow you to select, extract, and use spatial navigation.

```bash
pip install natural-pdf
```

But! If you want to recognize text, do page layout analysis, document q-and-a or other things, you can install optional dependencies.

Natural PDF has modular dependencies for different features. Install them based on your needs:

```bash
# Deskewing
pip install natural-pdf[deskew]

# LLM features (OpenAI)
pip install natural-pdf[llm]

# Semantic search
pip install natural-pdf[search]

# Install everything in the 'favorites' collection
pip install natural-pdf[favorites]
```

Other OCR and layout analysis engines like `surya`, `easyocr`, `paddle`, `doctr`, and `docling` can be installed via `pip` as needed. The library will provide you with an error message and installation command if you try to use an engine that isn't installed.

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