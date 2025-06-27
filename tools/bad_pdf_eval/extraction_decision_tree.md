# Natural PDF Extraction Decision Tree

Use this guide to choose the optimal extraction approach based on the PDF characteristics.

## 1. Table Extraction

### When to use Guides API (PREFERRED):
- ✅ Tables with clear column headers visible in text
- ✅ Forms with labeled fields  
- ✅ Tables without ruling lines (whitespace-separated)
- ✅ When you need precise control over column boundaries
- ✅ Multi-page tables that need consistent structure
- ✅ When TATR confidence is low (<0.8)

```python
# Example: Table with column headers
headers = page.find_all('text:bold').filter(lambda t: t.y0 < 200)
guides = Guides.from_content(page, markers=headers, axis='vertical')
guides.vertical.snap_to_whitespace()
guides.build_grid()
df = page.find('table').extract_table().df
```

### When to use direct region extraction:
- ✅ Simple tables with clear boundaries
- ✅ Forms with predictable structure
- ✅ When table detection works out of the box

```python
# Simple approach
table_region = page.find('text:contains("Revenue")').parent('table')
df = table_region.extract_table().df
```

### When TATR might still be needed:
- ⚠️ Complex multi-level headers
- ⚠️ Nested tables
- ⚠️ Tables with merged cells and complex layouts

## 2. Region Selection Best Practices

### When to use `until=` parameter:
- ✅ Extracting content between two sections
- ✅ Stopping before a footer or page number
- ✅ Isolating specific parts of a document
- ✅ When you need precise boundaries

### When `until=` is NOT needed:
- ✅ Extracting everything below a header to page end
- ✅ Getting all content above a footer
- ✅ Simple "rest of page" extractions

```python
# No 'until' needed - going to page edge
footer_content = page.find('text:contains("Page")').below()

# 'until' helpful - precise boundaries
section = header.below(until='text:contains("Next Section")')
```

## 3. Text Layer Handling

### Anchor-based extraction (PREFERRED):
- ✅ Structured documents with consistent headers
- ✅ When you need specific sections
- ✅ Multi-column layouts

```python
# Good: Resilient to layout changes
section = page.find('text:contains("Introduction")').below(until='text:contains("Methods")')
text = section.extract_text()
```

### Full page extraction:
- ✅ Simple documents
- ✅ When you need all text
- ✅ After adding exclusions for headers/footers

```python
# Remove headers/footers first
page.add_exclusion(page.find_all('text[size<8]'))
text = page.extract_text(layout=True)
```

## 4. Form Extraction

### Guides for structured forms:
```python
# Form with labeled fields
labels = page.find_all('text:contains(":")')
guides = Guides.from_content(page, markers=labels, align='after')
guides.build_grid()
# Extract field values from cells
```

### Direct ask/extract for simple forms:
```python
# When field names are known
fields = ['Name', 'Date', 'Amount']
data = page.extract(fields)
```

## 5. Special Cases

### When to Discard Text Layer:
1. Text appears as "(cid:xxx)" gibberish
2. Character encoding is corrupted
3. OCR would produce better results than native layer

```python
# Option 1: Discard when opening
pdf = PDF('corrupted.pdf', text_layer=False)
pdf.apply_ocr('surya')

# Option 2: Remove from existing page
if '(cid:' in page.extract_text()[:100]:
    page.remove_text_layer()
    page.apply_ocr('surya')
```

### Scanned PDFs:
1. Always check for text layer first
2. Apply OCR before any extraction
3. Use higher resolution for small text

```python
if not page.extract_text().strip():
    page.apply_ocr('surya', resolution=300)
```

### Multi-column layouts:
```python
from natural_pdf.flows import Flow
cols = [page.region(left=0, right=page.width/2), 
        page.region(left=page.width/2, right=page.width)]
flow = Flow(cols, arrangement='vertical')
text = flow.extract_text()
```

### RTL Languages:
- No special handling needed - Natural PDF handles BiDi automatically
- Search with logical order: `page.find('text:contains("مرحبا")')`

## 6. Quality Checks

Before using any code:
1. ❓ Does the code use actual text from the PDF (not "AnchorText")?
2. ❓ Is OCR applied when needed (scanned_image flag)?
3. ❓ Are we using the simplest approach that works?
4. ❓ Have we avoided unnecessary TATR calls?
5. ❓ Are variable names descriptive of the actual content? 