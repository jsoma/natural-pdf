# Patterns & Pitfalls

A quick reference of common patterns and mistakes to avoid when working with Natural PDF. Each pattern shows the expected return type.

---

### 1. Load PDF and Extract Text

**Use case**: Open a PDF file and extract all text from a page.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
text = page.extract_text()
```

**Returns**: `str` - The extracted text content from the page.

---

### 2. Find Element Containing Text

**Use case**: Locate the first element that contains specific text.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
element = page.find('text:contains("Invoice")')
```

**Returns**: `Element | None` - The first matching element, or `None` if not found.

---

### 3. Find All Matching Elements

**Use case**: Get all elements matching a selector.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
elements = page.find_all('text:bold')
```

**Returns**: `ElementCollection` - A collection of all matching elements (may be empty).

---

### 4. Navigate Below an Element

**Use case**: Create a region below a found element to extract content.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
header = page.find('text:contains("Summary")')
region = header.below(height=200)
content = region.extract_text()
```

**Returns**: `Region` - A rectangular region below the element.

---

### 5. Navigate Right of Element

**Use case**: Find the value next to a label.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
label = page.find('text:contains("Total:")')
value_region = label.right(width=100)
value = value_region.extract_text()
```

**Returns**: `Region` - A rectangular region to the right of the element.

---

### 6. Extract Table from Page

**Use case**: Extract tabular data from a page.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
table = page.extract_table()
```

**Returns**: `TableResult` - A sequence of rows (list of lists) with `.to_df()` method for pandas conversion.

---

### 6a. Extract ALL Tables from a Page

**Use case**: Find and extract every table on a page using layout analysis.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]

# Detect tables using layout analysis
page.analyze_layout(engine='tatr')

# Find all detected table regions
table_regions = page.find_all('region[type=table]')
print(f"Found {len(table_regions)} tables")

# Extract each table as a DataFrame
dataframes = []
for i, table_region in enumerate(table_regions):
    table = table_region.extract_table()
    df = table.to_df(header="first")
    dataframes.append(df)
    print(f"Table {i+1}: {len(df)} rows, {len(df.columns)} columns")
```

**Returns**: List of `pandas.DataFrame` objects, one per table.

**Note**: The `tatr` (Table Transformer) engine is recommended for table detection. Alternatives include `yolo` and `paddle`.

---

### 6b. Extract All Tables from a Page (Shortcut)

**Use case**: Extract every table on a page without manual layout analysis.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
tables = page.extract_tables()

for i, table in enumerate(tables):
    df = table.to_df(header="first")
    print(f"Table {i+1}: {len(df)} rows")
```

**Returns**: `List[TableResult]` - All tables found on the page.

---

### 6c. Convert Page to Markdown with VLM

**Use case**: Get a structured markdown representation of a page using a Vision Language Model.

```python
from natural_pdf import PDF, set_default_client
from openai import OpenAI

# Configure a default VLM client
set_default_client(OpenAI(), model="gpt-4o")

pdf = PDF("document.pdf")
page = pdf.pages[0]
md = page.to_markdown()
```

**Returns**: `str` - Markdown representation of the page. Falls back to `extract_text()` when no model is configured.

---

### 6d. Semantic Search Across Pages

**Use case**: Find the most relevant pages for a query using semantic similarity.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
results = pdf.search("payment terms and conditions", top_k=3)

for page in results:
    print(f"Page {page.number}: {page.extract_text()[:100]}...")
```

**Returns**: `PageCollection` - The top-k most relevant pages.

**Note**: Requires `torch` and `transformers` (`pip install torch transformers`).

---

### 7. Apply OCR

**Use case**: Run OCR on a scanned document to make it searchable.

```python
from natural_pdf import PDF

pdf = PDF("scanned.pdf")
page = pdf.pages[0]
ocr_elements = page.apply_ocr(engine='easyocr', languages=['en'])
text = page.extract_text()
```

**Returns**: `ElementCollection` - The newly created OCR text elements.

---

### 8. Analyze Layout

**Use case**: Detect document structure like tables, figures, and headings.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
regions = page.analyze_layout(engine='yolo')
tables = page.find_all('region[type=table]')
```

**Returns**: `ElementCollection` - Collection of detected layout regions.

---

### 9. Chain Find and Extract

**Use case**: Find an element and extract text from the region below it in one chain.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
content = page.find('text:contains("Description")').below(height=100).extract_text()
```

**Returns**: `str` - The extracted text from the chained operations.

---

### 10. Filter by Attribute

**Use case**: Find elements with specific attributes.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
large_text = page.find_all('text[size>=14]')
bold_headers = page.find_all('text:bold[size>=12]')
```

**Returns**: `ElementCollection` - Elements matching the attribute filter.

---

### 10a. Combined Selectors (3-Part)

**Use case**: Find elements matching multiple criteria at once.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]

# Combine element type + pseudo-class + attribute + contains
# Order: type → pseudo-classes → attributes → :contains()
summary_headers = page.find_all('text:bold[size>14]:contains("Summary")')

# Other combined examples
important_notes = page.find('text:italic[size>=12]:contains("Note")')
section_titles = page.find_all('text:bold[fontname*=Arial][size>=16]')
```

**Returns**: `ElementCollection` for `find_all()`, `Element | None` for `find()`.

**Selector ordering**: Pseudo-classes and attributes can appear in any order after the type — `text:bold[size>14]:contains("X")` and `text:contains("X"):bold[size>14]` are equivalent.

---

### 11. Create a Region from Coordinates

**Use case**: Define a specific rectangular area on a page.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
region = page.create_region(x0=50, top=100, x1=500, bottom=300)
text = region.extract_text()
```

**Returns**: `Region` - A region with the specified coordinates.

---

### 12. Add Exclusion Zone

**Use case**: Exclude headers or footers from text extraction.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]

# Exclude top 50 points as header
header_region = page.create_region(0, 0, page.width, 50)
page.add_exclusion(header_region)
```

**Returns**: `None` - Exclusion is added to the page.

---

### 13. Extract with Exclusions

**Use case**: Extract text while respecting exclusion zones.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
page.add_exclusion(page.create_region(0, 0, page.width, 50))  # Exclude header
text = page.extract_text()  # Exclusions applied by default
```

**Returns**: `str` - Text with excluded regions omitted.

---

### 14. Get Page Dimensions

**Use case**: Access page width and height for calculations.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
width = page.width
height = page.height
```

**Returns**: `float` - Page dimension in points (1 point = 1/72 inch).

---

### 15. Iterate Over Pages

**Use case**: Process all pages in a PDF.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
for page in pdf.pages:
    text = page.extract_text()
    print(f"Page {page.number}: {len(text)} characters")
```

**Returns**: Each iteration yields a `Page` object.

---

### 16. Extract Text with Layout Preservation

**Use case**: Maintain spatial positioning of text.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
text = page.extract_text(layout=True)
```

**Returns**: `str` - Text with whitespace preserving original layout.

---

### 17. Find Using Regex

**Use case**: Search for patterns like invoice numbers or dates.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
invoice_num = page.find('text:contains("INV-\\d+")', regex=True)
dates = page.find_all('text:contains("\\d{2}/\\d{2}/\\d{4}")', regex=True)
```

**Returns**: `Element | None` for `find()`, `ElementCollection` for `find_all()`.

---

### 18. Case-Insensitive Search

**Use case**: Find text regardless of case.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
element = page.find('text:contains("total")', case=False)
```

**Returns**: `Element | None` - First element containing "total", "TOTAL", "Total", etc.

---

### 19. Extract Table as DataFrame

**Use case**: Get table data directly as a pandas DataFrame.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
table = page.extract_table()
df = table.to_df(header="first")  # Use first row as column headers
```

**Returns**: `pandas.DataFrame` - Table data with proper column headers.

---

### 20. Visualize Elements

**Use case**: Debug by highlighting found elements on the page.

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]
elements = page.find_all('text:bold')
image = elements.show(color="red", label="Bold Text")
```

**Returns**: `PIL.Image.Image` - Page image with highlighted elements.

---

## Quick Reference Table

| Pattern | Method | Returns |
|---------|--------|---------|
| Load PDF | `PDF("file.pdf")` | `PDF` |
| Get page | `pdf.pages[0]` | `Page` |
| Extract text | `page.extract_text()` | `str` |
| Find one | `page.find(selector)` | `Element \| None` |
| Find all | `page.find_all(selector)` | `ElementCollection` |
| Navigate below | `element.below()` | `Region` |
| Navigate right | `element.right()` | `Region` |
| Navigate above | `element.above()` | `Region` |
| Navigate left | `element.left()` | `Region` |
| Extract table | `page.extract_table()` | `TableResult` |
| Extract all tables | `page.extract_tables()` | `List[TableResult]` |
| Page to markdown | `page.to_markdown()` | `str` |
| Semantic search | `pdf.search(query)` | `PageCollection` |
| Apply OCR | `page.apply_ocr()` | `ElementCollection` |
| Analyze layout | `page.analyze_layout()` | `ElementCollection` |
| Create region | `page.create_region(...)` | `Region` |
| Add exclusion | `page.add_exclusion(...)` | `None` |
| Show elements | `elements.show()` | `PIL.Image.Image` |
| Table to DataFrame | `table.to_df()` | `pandas.DataFrame` |

---

## Common Mistakes to Avoid

These are frequent errors when working with Natural PDF.

### Wrong Method Names

```python
# WRONG - these methods don't exist
page.get_text()           # Use: page.extract_text()
page.search("term")       # Use: page.find('text:contains("term")')
PDF.open("file.pdf")      # Use: PDF("file.pdf")
page.apply_layout()       # Use: page.analyze_layout()
pdf[0]                    # Use: pdf.pages[0]
```

### Wrong Selector Syntax

```python
# WRONG
page.find('text.bold')              # Use colon: 'text:bold'
page.find('text[contains="X"]')     # contains is a pseudo-class: 'text:contains("X")'
page.find('text(size>12)')          # Use brackets: 'text[size>12]'
page.find('text:contains(Invoice)') # Need quotes: 'text:contains("Invoice")'
```

### Not Handling None

```python
# WRONG - will crash if element not found
text = page.find('text:contains("Missing")').extract_text()

# CORRECT - always check for None
element = page.find('text:contains("Missing")')
if element:
    text = element.extract_text()
```

### Wrong Parameter Names

```python
# WRONG
page.find('text:contains("X")', case_sensitive=False)  # Use: case=False
page.apply_ocr(engine="easy_ocr")                      # Use: engine="easyocr"
page.apply_ocr(engine="paddle_ocr")                    # Use: engine="paddle"
```

### Not Closing PDFs in Loops

```python
# WRONG - memory leak
for path in pdf_paths:
    pdf = PDF(path)
    # process...
    # PDF never closed!

# CORRECT - use try/finally
for path in pdf_paths:
    pdf = PDF(path)
    try:
        # process...
    finally:
        pdf.close()
```

### Treating find_all() as a Plain List

```python
# WRONG - verbose
elements = page.find_all('text:bold')
first = elements[0] if len(elements) > 0 else None

# CORRECT - use ElementCollection methods
elements = page.find_all('text:bold')
first = elements.first  # Returns None if empty
```

### Summary of Corrections

| Wrong | Correct | Issue |
|-------|---------|-------|
| `page.get_text()` | `page.extract_text()` | Wrong method name |
| `page.search("X")` | `page.find('text:contains("X")')` | Wrong method name |
| `PDF.open("file")` | `PDF("file")` | Direct instantiation |
| `'text.bold'` | `'text:bold'` | Colon for pseudo-classes |
| `case_sensitive=False` | `case=False` | Wrong parameter name |
| `engine="easy_ocr"` | `engine="easyocr"` | No underscores |
| `apply_layout()` | `analyze_layout()` | Wrong method name |
| `pdf[0]` | `pdf.pages[0]` | Access via `.pages` |
