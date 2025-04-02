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
page.clear_highlights()
elements.highlight(color="red", label="Contains 'Inadequate'")

# Display the page to see them
page.to_image(width=700)
```

## Working with Layout Regions

We can analyze the layout of the page to identify different regions:

```python
# Analyze the page layout
page.analyze_layout(engine='yolo')

# Find and highlight all detected regions
page.clear_highlights()
page.find_all('region').highlight(group_by='type')

# Display the page to see the regions
page.to_image(width=900)
```

## Working with Multiple Pages

You can also work with multiple pages:

```python
# Process all pages
for page in pdf.pages:
    page_text = page.extract_text()
    print(f"Page {page.number}", page_text[:100])  # First 100 chars of each page
```

This tutorial covered the basics of loading PDFs and extracting text. In the next tutorials, we'll explore more advanced features like searching for specific elements, extracting structured content, and working with tables. 

```bash
pip install "natural-pdf[all]"
``` 