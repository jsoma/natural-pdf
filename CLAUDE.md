# Natural PDF Library Analysis

## Library Overview
Natural PDF is a Python library for intelligent PDF document processing that combines traditional PDF parsing with modern AI capabilities. It provides a jQuery-like API for selecting and manipulating PDF elements with spatial awareness.

## Core Goals & Purpose
- **Intelligent PDF Processing**: Goes beyond simple text extraction to understand document structure and spatial relationships
- **AI-Enhanced Workflows**: Integrates OCR, document Q&A, classification, and LLM-based data extraction
- **Spatial Navigation**: Provides methods like `.below()`, `.above()`, `.left()` for intuitive element selection
- **Multi-format Support**: Handles both text-based PDFs and image-based (OCR-required) documents

## Key Use Cases & Workflows

### 1. Basic Text and Table Extraction
- Load PDFs from local files or URLs
- Extract text with layout preservation
- Find and extract tables automatically
- Use spatial selectors: `page.find('text:contains(Violations)').below()`

### 2. OCR Integration
- Multiple OCR engines supported: EasyOCR (default), Surya, PaddleOCR, DocTR
- Configurable resolution and detection modes
- OCR correction using LLMs
- Human-in-the-loop correction workflows with exportable packages

### 3. AI-Powered Data Extraction
- **Document Q&A**: Extractive question answering with confidence scores
- **Structured Data**: Extract specific fields with schema validation using Pydantic
- **LLM Integration**: OpenAI/Gemini compatible for advanced extraction
- **Classification**: Document/page categorization using text or vision models
- **Per-field confidence scoring**: `confidence=True` for 0.0–1.0 scores per field
- **Annotated PDF export**: `result.save_pdf("output.pdf")` highlights + sidebar as native PDF

### 4. Advanced Document Processing
- **Multi-column/Page Flows**: Reflow content across columns or pages for proper reading order
- **Layout Analysis**: YOLO, TATR for automatic document structure detection
- **Visual Element Detection**: Checkbox classification, form field extraction
- **Table Structure Detection**: Manual line detection for complex tables

### 5. Visualization and Display
- **Page Limit for show()**: By default, `pdf.show()` displays only the first 30 pages to prevent overwhelming displays
  - Use `pdf.show(limit=10)` to show fewer pages
  - Use `pdf.show(limit=None)` to display all pages
  - Works with all layout options: `pdf.show(limit=20, layout='grid', columns=4)`
- **Exclusion Zone Visualization**: Use `exclusions='red'` parameter to visualize exclusion zones
  - `page.show(exclusions='red')` highlights exclusions in red
  - `page.show(exclusions='blue')` highlights exclusions in blue
  - `page.show(exclusions=True)` uses default red color
  - Works at PDF level too: `pdf.show(exclusions='green')`

### 6. Directional Navigation Improvements
- **Smart defaults for spatial methods**:
  - `.left()` and `.right()` now default to `height='element'` (matches element height)
  - `.above()` and `.below()` continue to default to `width='full'` (full page width)
  - This matches common use cases: looking sideways usually wants same height, looking up/down wants full width
- **Enhanced discoverability**:
  - Docstrings include examples showing different height/width options
  - Clear parameter names ('height' for left/right, 'width' for above/below)

### 6a. Enhanced Exclusion Support
- **ElementCollection support in callable exclusions**: `pdf.add_exclusion(lambda page: page.find_all('text:contains("Header")'))` now works
- **List/iterable support**: Callable exclusions can return lists or other iterables of elements
- **Automatic conversion**: Elements from iterables are automatically converted to exclusion regions
- **Backward compatibility**: Existing Region and callable exclusions continue to work unchanged

### 6b. Multi-page Directional Navigation
- **multipage parameter**: Directional methods now accept `multipage=True` to span pages
  - `element.below(until="text:contains('End')", multipage=True)` searches across pages
  - Returns `FlowRegion` when spanning multiple pages, `Region` when on single page
  - Works with all directional methods: `.below()`, `.above()`, `.left()`, `.right()`
- **Global auto_multipage option**: Set default behavior for all directional navigation
  - `npdf.set_option('layout.auto_multipage', True)` enables multipage by default
  - Individual calls can override with `multipage=False`
- **Use cases**:
  - Extract content between headers on different pages
  - Find tables that span page boundaries
  - Navigate document structure without manual page handling

### 7. Page Grouping with groupby()
- **Simple grouping by selector text**: `pages.groupby('text[size=16]')` groups by header text
- **Callable functions for complex logic**: `pages.groupby(lambda p: p.find('text:contains("CITY")').extract_text())`
- **Pandas-style iteration**: `for title, pages in grouped:` (no `.items()` needed)
- **Dict-like access**: `grouped.get('CITY OF MADISON')` or `grouped.get_group('key')`
- **Index-based access**: `grouped[0]` (first group), `grouped[-1]` (last group), `grouped['key']` (by name)
- **Group exploration**: `grouped.info()` shows all groups with indexes and page counts
- **Batch operations**: `grouped.apply(lambda pages: len(pages.find_all('table')))`
- **Visual inspection**: `grouped.show(limit=2)` shows first 2 pages of each group
- **Progress bar support**: Automatic progress bars for large collections, disable with `show_progress=False`
- **None handling**: Pages with no matching elements group under `None` key

## Method Return Types Quick Reference

| Method | Returns | Notes |
|--------|---------|-------|
| `PDF("path")` | `PDF` | Load from file, URL, or bytes |
| `pdf.pages` | `PageCollection` | Iterable, supports slicing |
| `pdf.pages[0]` | `Page` | Zero-indexed |
| `page.find(selector)` | `Element \| None` | First match or None |
| `page.find_all(selector)` | `ElementCollection` | All matches (may be empty) |
| `element.below()` | `Region` | Spatial navigation |
| `element.above()` | `Region` | Spatial navigation |
| `element.left()` | `Region` | Spatial navigation |
| `element.right()` | `Region` | Spatial navigation |
| `region.extract_text()` | `str` | Text content |
| `page.extract_text()` | `str` | Full page text |
| `page.extract_table()` | `TableResult` | Has `.to_df()` method |
| `table.to_df()` | `pandas.DataFrame` | Tabular data |
| `page.apply_ocr()` | `ElementCollection` | OCR text elements |
| `page.analyze_layout()` | `ElementCollection` | Detected regions |
| `page.ask(question)` | `StructuredDataResult` | Has `.data`, `.success` |
| `element.show()` | `PIL.Image` | Visualization |
| `elements.apply(fn)` | `ElementCollection` | Transform collection |
| `elements.filter(fn)` | `ElementCollection` | Filter collection |

## Selector Syntax Quick Reference

```
# Element types
'text'                          # All text elements
'line'                          # All line elements
'rect'                          # All rectangles
'region'                        # Layout-detected regions
'image'                         # Images

# Pseudo-classes (use colon)
'text:bold'                     # Bold text
'text:italic'                   # Italic text
'text:contains("Invoice")'      # Text containing string
'line:horizontal'               # Horizontal lines

# Attribute filters (use brackets)
'text[size>12]'                 # Font size > 12
'text[fontname*=Arial]'         # Font contains "Arial"
'region[type=table]'            # Tables from layout analysis
'text[confidence>=0.8]'         # High-confidence OCR

# Combined
'text:bold[size>=14]'           # Bold AND large
```

## Common Mistakes to Avoid

### Wrong Method Names
```python
# WRONG - these methods don't exist
page.get_text()           # Use: page.extract_text()
page.search("term")       # Use: page.find('text:contains("term")')
element.text              # Use: element.extract_text()
PDF.open("file.pdf")      # Use: PDF("file.pdf")

# CORRECT
text = page.extract_text()
element = page.find('text:contains("Invoice")')
text = element.extract_text() if element else ""
pdf = PDF("file.pdf")
```

### Wrong Selector Syntax
```python
# WRONG
page.find('text.bold')              # Use colon, not dot
page.find('text[contains="X"]')     # contains is a pseudo-class
page.find('text(size>12)')          # Use brackets, not parens

# CORRECT
page.find('text:bold')
page.find('text:contains("X")')
page.find('text[size>12]')
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
page.find('text:contains("X")', case_sensitive=False)  # Wrong param name
page.apply_ocr(engine="easy_ocr")                      # Wrong engine name

# CORRECT
page.find('text:contains("X")', case=False)
page.apply_ocr(engine="easyocr")
```

### Not Closing PDFs in Loops
```python
# WRONG - memory leak
for path in pdf_paths:
    pdf = PDF(path)
    # process...
    # PDF never closed!

# CORRECT
for path in pdf_paths:
    pdf = PDF(path)
    try:
        # process...
    finally:
        pdf.close()
```

## Development Best Practices

### File and Resource Management
- When making temp files, put them in temp/
- When creating test files, put them in tests/
- Most fixes and changes need a test, and should be done with test-driven development

### Environment and Tooling
- Always use the virtual environment in .venv
- Use uv when possible for efficient package management
- Don't create new PDFs for testing, just use pdfs/01-practice.pdf.
