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
# Document QA + extraction helpers
pip install natural-pdf[qa]

# Zero-shot/vision classification
pip install natural-pdf[classification]

# OCR engines (EasyOCR, Surya, etc.)
pip install natural-pdf[ocr-ai]

# Deskewing
pip install natural-pdf[deskew]

# Semantic search (LanceDB + embeddings)
pip install natural-pdf[search]

# Or install the full ML stack
pip install natural-pdf[ai]
```

Other OCR and layout analysis engines like `surya`, `easyocr`, `paddle`, `doctr`, and `docling` can be installed via `pip` as needed. The library will provide you with an error message and installation command if you try to use an engine that isn't installed.

After the core install you have two ways to add **optional engines**:

### 1 – Helper CLI (recommended)

```bash
# list optional groups and their install-status
npdf list

# everything for classification, QA, semantic search, etc.
npdf install ai

# install PaddleOCR stack
npdf install paddle

# install Surya OCR + YOLO layout detector
npdf install surya yolo
```

The CLI runs each wheel in its own resolver pass, so it avoids strict
version pins like `paddleocr → paddlex==3.0.1` while still upgrading to
`paddlex 3.0.2`.

### 2 – Classic extras (for the light stuff)

```bash
# Full AI/ML stack
pip install "natural-pdf[ai]"

# Document QA
pip install "natural-pdf[qa]"

# Classification helpers
pip install "natural-pdf[classification]"

# OCR engines
pip install "natural-pdf[ocr-ai]"

# Deskewing / semantic search
pip install "natural-pdf[deskew]"
pip install "natural-pdf[search]"
```

If you attempt to use an engine that is missing, the library will raise an
error that tells you which `npdf install …` command to run.

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
print(f"Found title: {title.text}")
```

## What's Next?

Now that you have Natural PDF installed, you can:

- Learn to [navigate PDFs](../pdf-navigation/index.ipynb)
- Explore how to [select elements](../element-selection/index.ipynb)
- See how to [extract text](../text-extraction/index.ipynb)
