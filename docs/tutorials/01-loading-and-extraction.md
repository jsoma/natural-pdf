# Loading and Basic Text Extraction

```python
#%pip install "natural-pdf[all]"
```

In this tutorial, we'll learn how to:

1. Load a PDF document
2. Extract text from pages
3. Extract specific elements

## Loading a PDF

Let's start by loading a PDF file:

```python
from natural_pdf import PDF
import os

# Load a PDF file
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")

# Basic info about the document
{
    "Filename": os.path.basename(pdf.path),
    "Pages": len(pdf.pages),
    "Title": pdf.metadata.get("Title", "N/A"),
    "Author": pdf.metadata.get("Author", "N/A")
}
```

## Extracting Text

Now that we have loaded the PDF, let's extract the text from the first page:

```python
# Get the first page
page = pdf.pages[0]

# Extract text from the page
text = page.extract_text()

# Show the first 200 characters of the text
print(text[:200])
```

## Finding and Extracting Specific Elements

We can find specific elements using spatial queries and text content:

```python
# Find text elements containing specific words
elements = page.find_all('text:contains("Inadequate")')

# Show these elements on the page
elements.show()
```

## Working with Layout Regions

We can analyze the layout of the page to identify different regions:

```python
# Analyze the page layout
page.analyze_layout(engine='yolo')

# Find and highlight all detected regions
page.find_all('region').show(group_by='type')
```

## Working with Multiple Pages

You can also work with multiple pages:

```python
# Process all pages
for page in pdf.pages:
    page_text = page.extract_text()
    print(f"Page {page.number}", page_text[:100])  # First 100 chars of each page
```

## Extracting Text with Layout Preservation

For documents with complex layouts, preserve the visual structure:

```python
# Extract text preserving spatial layout
text = page.extract_text(layout=True)
print(text)
```

## Converting Pages to Markdown

If you have access to a Vision Language Model, you can convert pages to structured markdown. This works well for pages with complex formatting, tables, and mixed content.

```python
from natural_pdf import PDF, set_default_client
from openai import OpenAI

# Configure a VLM client once
set_default_client(OpenAI(), model="gpt-4o")

pdf = PDF("document.pdf")
page = pdf.pages[0]

# Get markdown representation of the page
md = page.to_markdown()
print(md)
```

Without a configured client, `to_markdown()` falls back to `extract_text()`.

## Viewing Multiple Pages at Once

Use `pages.show()` to preview multiple pages in a grid:

```python
from natural_pdf import PDF

pdf = PDF("long_document.pdf")

# Show all pages in a grid (6 columns)
pdf.pages.show(cols=6)

# Show only specific pages
pdf.pages[5:20].show(cols=4)
```

## Discarding Bad Text Layers

Some PDFs have corrupted OCR text. Load without the text layer:

```python
# Load PDF and discard its flawed text layer
pdf = PDF("badly_ocred.pdf", text_layer=False)

# Then apply fresh OCR
pdf.pages.apply_ocr()
text = pdf.pages[0].extract_text()
```

## Closing PDFs

Always close PDFs when you're done to free resources:

```python
# Process and close
pdf = PDF("document.pdf")
text = pdf.pages[0].extract_text()
pdf.close()

# Or use try/finally for safety
pdf = PDF("document.pdf")
try:
    text = pdf.pages[0].extract_text()
finally:
    pdf.close()
```

This tutorial covered the basics of loading PDFs and extracting text. In the next tutorials, we'll explore more advanced features like searching for specific elements, extracting structured content, and working with tables.
