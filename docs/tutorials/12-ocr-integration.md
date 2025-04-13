# OCR Integration for Scanned Documents

Optical Character Recognition (OCR) allows you to extract text from scanned documents where the text isn't embedded in the PDF. This tutorial demonstrates how to work with scanned documents.

```python
#%pip install "natural-pdf[all]"
```

```python
from natural_pdf import PDF

# Load a PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/needs-ocr.pdf")
page = pdf.pages[0]

# Try extracting text without OCR
text_without_ocr = page.extract_text()
f"Without OCR: {len(text_without_ocr)} characters extracted"
```

## Enabling OCR

```python
# Enable OCR for text extraction
page.use_ocr = True

# Extract text with OCR enabled
text_with_ocr = page.extract_text()

# Preview the extracted text
text_with_ocr[:200] + "..." if len(text_with_ocr) > 200 else text_with_ocr
```

## Finding Text Elements with OCR

```python
# Convert text-as-image to text elements
page.apply_ocr()

# Select all text pieces on the page
text_elements = page.find_all('text')
f"Found {len(text_elements)} text elements"

# Visualize the elements
text_elements.highlight()
```

## OCR Configuration Options

```python
# Set OCR configuration for better results
page.ocr_config = {
    'language': 'eng',  # English
    'dpi': 300,         # Higher resolution
}

# Extract text with the improved configuration
improved_text = page.extract_text()

# Preview the text
improved_text[:200] + "..." if len(improved_text) > 200 else improved_text
```

## Working with Multi-language Documents

```python
# Configure for multiple languages
page.ocr_config = {
    'language': 'eng+fra+deu',  # English, French, German
    'dpi': 300
}

# Extract text with multi-language support
multilang_text = page.extract_text()
multilang_text[:200]
```

## Extracting Tables from Scanned Documents

```python
# Enable OCR and analyze the document layout
page.use_ocr = True
page.analyze_layout()

# Find table regions
table_regions = page.find_all('region[type=table]')

# Visualize any detected tables
table_regions.highlight()

# Extract the first table if found
if table_regions:
    table_data = table_regions[0].extract_table()
    table_data
else:
    "No tables found in the document"
```

## Finding Form Fields in Scanned Documents

```python
# Look for potential form labels (containing a colon)
labels = page.find_all('text:contains(":")') 

# Visualize the labels
labels.highlight()

# Extract form data by looking to the right of each label
form_data = {}
for label in labels:
    # Clean the label text
    field_name = label.text.strip().rstrip(':')
    
    # Find the value to the right
    value_element = label.right(width=200)
    value = value_element.extract_text().strip()
    
    # Add to our dictionary
    form_data[field_name] = value

# Display the extracted data
form_data
```

## Combining OCR with Layout Analysis

```python
# Apply OCR and analyze layout
page.use_ocr = True
page.analyze_layout()

# Find document structure elements
headings = page.find_all('region[type=heading]')
paragraphs = page.find_all('region[type=paragraph]')

# Visualize the structure
headings.highlight(color="red", label="Headings")
paragraphs.highlight(color="blue", label="Paragraphs")

# Create a simple document outline
document_outline = []
for heading in headings:
    heading_text = heading.extract_text()
    document_outline.append(heading_text)

document_outline
```

## Working with Multiple Pages

```python
# Process all pages in the document
all_text = []

for i, page in enumerate(pdf.pages):
    # Enable OCR for each page
    page.use_ocr = True
    
    # Extract text
    page_text = page.extract_text()
    
    # Add to our collection with page number
    all_text.append(f"Page {i+1}: {page_text[:100]}...")

# Show the first few pages
all_text
```

## Saving PDFs with Searchable Text

After applying OCR to a PDF, you can save a new version of the PDF where the recognized text is embedded as an invisible layer. This makes the text searchable and copyable in standard PDF viewers.

Use the `save_searchable()` method on the `PDF` object:

```python
from natural_pdf import PDF

input_pdf_path = "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/needs-ocr.pdf"

pdf = PDF(input_pdf_path)
pdf.apply_ocr() 

pdf.save_searchable("needs-ocr-searchable.pdf")
```

This creates `needs-ocr-searchable.pdf`, which looks identical to the original but now has a text layer corresponding to the OCR results. You can adjust the rendering resolution used during saving with the `dpi` parameter (default is 300).

OCR integration enables you to work with scanned documents, historical archives, and image-based PDFs that don't have embedded text. By combining OCR with natural-pdf's layout analysis capabilities, you can turn any document into structured, searchable data. 