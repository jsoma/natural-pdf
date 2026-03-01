# Troubleshooting Guide

This guide provides decision trees for diagnosing and fixing common problems when working with Natural PDF.

## "My text looks garbled"

When extracted text appears corrupted, contains strange characters, or is completely empty, follow this decision tree:

```
Is text completely empty?
├── Yes → PDF might be scanned (image-based)
│   └── Try: page.apply_ocr()
│
└── No, but text is garbled
    ├── Strange Unicode characters? → Font encoding issue
    │   └── Try: page.apply_ocr() (re-reads from the rendered image)
    │
    ├── Text appears but wrong order? → Reading order issue
    │   └── Try: page.extract_text(layout=True)
    │
    └── Characters appear but scrambled? → Might be a scanned PDF
        └── Try: page.apply_ocr()
```

### Check if PDF is Scanned

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]

# Check what's in the PDF
text_elements = page.find_all('text')
print(f"Found {len(text_elements)} text elements")

# If very few or zero text elements, it's likely scanned
if len(text_elements) < 5:
    print("This appears to be a scanned PDF - try OCR")
    page.apply_ocr()
    text = page.extract_text()
else:
    text = page.extract_text()
```

### Apply OCR for Scanned Documents

```python
# Default OCR (uses EasyOCR)
page.apply_ocr()

# If default doesn't work well, try different engines
page.apply_ocr(engine='paddle')  # Often better for forms
page.apply_ocr(engine='surya')   # Good for multi-language

# For better quality, increase resolution
page.apply_ocr(engine='easyocr', resolution=300)
```

### Debug What's Actually in the PDF

```python
# Visualize all text elements
page.find_all('text').show()

# Check if there are images (indicating scanned content)
images = page.find_all('image')
print(f"Found {len(images)} images")

# If the page is one big image, definitely use OCR
if len(images) == 1 and images[0].width > page.width * 0.9:
    print("Page is a full-page scan")
```

---

## "Table extraction is wrong"

Table extraction can fail for several reasons. Follow this decision tree:

```
Is the table detected at all?
├── No → Run layout analysis first
│   └── page.analyze_layout(engine='tatr')
│
└── Yes, but data is wrong
    ├── Columns misaligned? → Try TATR extraction method
    │   └── region.extract_table(method='tatr')
    │
    ├── Missing rows/columns? → Table might be borderless
    │   └── Try: page.analyze_layout(engine='tatr')
    │
    └── Merged cells causing issues?
        └── Use Guides for manual structure
```

### Run Layout Analysis First

Layout analysis detects table regions before extraction:

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")
page = pdf.pages[0]

# Detect tables using TATR (Table Transformer)
page.analyze_layout(engine='tatr')

# Find detected tables
tables = page.find_all('region[type=table]')
print(f"Found {len(tables)} tables")

# Visualize what was detected
tables.show()

# Extract from the first detected table
if tables:
    data = tables[0].extract_table()
```

### Compare Extraction Methods

```python
# Default extraction (uses pdfplumber)
data_default = page.extract_table()

# TATR-based extraction (uses detected structure)
page.analyze_layout(engine='tatr')
table = page.find('region[type=table][model=tatr]')
data_tatr = table.extract_table(method='tatr')

# Compare results
print("Default method:")
print(data_default[:3])  # First 3 rows

print("\nTATR method:")
print(data_tatr[:3])
```

### Handle Borderless Tables

```python
# For tables without visible grid lines
table_settings = {
    "vertical_strategy": "text",    # Use text alignment
    "horizontal_strategy": "text",  # Instead of lines
}

data = page.extract_table(table_settings=table_settings)
```

### Use Guides for Complex Tables

When automatic detection fails, manually define table structure with [guides](guides.md):

```python
region = page.find('text:contains("Column Header")').below()

guides = region.guides()
guides.vertical.from_content(
    ['Name', 'Date', 'Amount', 'Status'],
    align='between'
)
guides.vertical.snap_to_whitespace()

df = guides.extract_table().to_df()
```

---

## "Selector finds nothing"

When `find()` or `find_all()` returns empty results:

```
Did you spell the selector correctly?
├── Check for typos in attribute names
│
└── Try broader search first
    └── page.find_all('text').show()
```

### Debug by Seeing All Elements

```python
# See ALL text elements on the page
all_text = page.find_all('text')
print(f"Total text elements: {len(all_text)}")

# Visualize them
all_text.show()

# Print the actual text content
for elem in all_text[:20]:  # First 20
    print(f"  '{elem.extract_text()}'")
```

### Check Case Sensitivity

Selectors are case-sensitive by default:

```python
# This might not find "REVENUE" or "Revenue"
page.find('text:contains("revenue")')

# Use case=False for case-insensitive search
page.find('text:contains("revenue")', case=False)

# Or use regex for flexible matching
page.find('text:contains("(?i)revenue")', regex=True)
```

### Use Regex for Flexible Matching

```python
# Match variations in spacing or formatting
page.find('text:contains("Invoice\\s*#")', regex=True)

# Match multiple possible values
page.find('text:contains("Total|Sum|Amount")', regex=True)

# Match partial text
page.find('text:contains("Q[1-4] 2024")', regex=True)
```

### Check Element Attributes

```python
# Find text of a specific size
large_text = page.find_all('text[size>=14]')

# Find bold text
bold_text = page.find_all('text:bold')

# Debug: see what sizes exist
sizes = set()
for elem in page.find_all('text'):
    if hasattr(elem, 'size'):
        sizes.add(elem.size)
print(f"Text sizes on page: {sorted(sizes)}")
```

### Combine Conditions

```python
# Find bold text containing "Summary" that's at least size 12
page.find('text:contains("Summary"):bold[size>=12]')

# Find text in the top half of the page
top_region = page.create_region(0, 0, page.width, page.height/2)
top_region.find('text:contains("Header")')
```

---

## "OCR confidence is low"

When OCR produces poor results or low confidence scores:

```
Is the image quality poor?
├── Yes → Increase resolution
│   └── page.apply_ocr(resolution=300)
│
└── No, quality seems fine
    ├── Wrong language? → Specify languages
    │   └── page.apply_ocr(languages=['en', 'es'])
    │
    └── Try a different OCR engine
        └── paddle, paddlevl, surya, or doctr
```

### Try Different OCR Engines

```python
# EasyOCR (default) - good general purpose
page.apply_ocr(engine='easyocr')

# PaddleOCR - often better for structured documents
page.apply_ocr(engine='paddle')

# Surya - good for multi-language documents
page.apply_ocr(engine='surya')

# DocTR - good for dense text
page.apply_ocr(engine='doctr')

# PaddleOCR-VL - VLM-based, good for complex layouts and charts
page.apply_ocr(engine='paddlevl')
```

### Increase Resolution

```python
# Default resolution might be too low for small text
page.apply_ocr(resolution=150)  # Default

# Try higher resolution
page.apply_ocr(resolution=300)  # Better quality

# Even higher for very small text
page.apply_ocr(resolution=400)  # Slower but more accurate
```

### Filter by Confidence

```python
# Apply OCR first
page.apply_ocr(engine='paddle')

# Get only high-confidence results
high_conf = page.find_all('text[source=ocr][confidence>=0.8]')
print(f"High confidence: {len(high_conf)} elements")

# Get medium confidence for review
medium_conf = page.find_all('text[source=ocr][confidence>=0.5][confidence<0.8]')
print(f"Medium confidence: {len(medium_conf)} elements")

# Visualize confidence levels
page.find_all('text[source=ocr]').show(group_by='confidence')
```

### Specify Languages

```python
# Single language
page.apply_ocr(languages=['en'])

# Multiple languages
page.apply_ocr(languages=['en', 'es'])  # English and Spanish

# Standard language codes (en, fr, de, ja, zh, ko) work across all engines.
# PaddleOCR auto-normalizes codes like 'ja' → 'japan', 'zh' → 'ch'.
```

---

## "Text includes headers/footers I don't want"

When extracted text contains repeated headers, footers, or watermarks:

```
Is the unwanted content in the same position on every page?
├── Yes → Use fixed-region exclusions
│   └── pdf.add_exclusion(lambda p: p.create_region(...))
│
└── No, position varies
    └── Use content-based exclusions
        └── page.add_exclusion(page.find('text:contains("Header")').above())
```

### Exclude by Position

```python
from natural_pdf import PDF

pdf = PDF("document.pdf")

# Exclude top 50 points from all pages (headers)
pdf.add_exclusion(
    lambda page: page.create_region(0, 0, page.width, 50),
    label="Header"
)

# Exclude bottom 40 points from all pages (footers)
pdf.add_exclusion(
    lambda page: page.create_region(0, page.height - 40, page.width, page.height),
    label="Footer"
)

# Now extract clean text
for page in pdf.pages:
    clean_text = page.extract_text()  # Headers/footers excluded
```

### Exclude by Content

```python
# Find and exclude specific text
page = pdf.pages[0]

# Exclude everything above "CONFIDENTIAL" watermark
watermark = page.find('text:contains("CONFIDENTIAL")')
if watermark:
    page.add_exclusion(watermark.above())

# Exclude page numbers (usually small text with just digits)
page_nums = page.find_all('text:contains("^\\d+$")', regex=True)
for num in page_nums:
    if num.top > page.height * 0.9:  # Only if near bottom
        page.add_exclusion(num)
```

### Visualize Exclusions

```python
# Add exclusions
page.add_exclusion(page.create_region(0, 0, page.width, 50))

# See what will be excluded (shown in red)
page.show(exclusions='red')

# Compare with and without exclusions
full_text = page.extract_text(use_exclusions=False)
clean_text = page.extract_text(use_exclusions=True)

print(f"Full: {len(full_text)} chars")
print(f"Clean: {len(clean_text)} chars")
```

### Apply Exclusions Across All Pages

```python
def smart_exclusion(page):
    """Exclude headers/footers dynamically."""
    # Try to find specific header text
    header = page.find('text:contains("Company Name")')
    if header and header.top < page.height * 0.15:
        return header.above()

    # Fallback to fixed region
    return page.create_region(0, 0, page.width, 60)

# Apply to entire PDF
pdf.add_exclusion(smart_exclusion, label="Headers")
```

---

## "Layout analysis is slow"

When `analyze_layout()` takes too long:

```
Are you processing many pages?
├── Yes → Process only pages that need it
│
└── No, single page is slow
    ├── Using TATR? → Try YOLO (faster)
    │   └── page.analyze_layout(engine='yolo')
    │
    └── High resolution? → Lower it
        └── page.analyze_layout(resolution=150)
```

### Use YOLO for Speed

```python
# YOLO is faster but less precise
page.analyze_layout(engine='yolo')

# TATR is slower but understands table structure
page.analyze_layout(engine='tatr')

# For most use cases, YOLO is sufficient
tables = page.find_all('region[type=table][model=yolo]')
```

### Process Selectively

```python
# Don't analyze every page if you don't need to
for page in pdf.pages:
    # Only analyze if page likely has tables
    text = page.extract_text()
    if "total" in text.lower() or "|" in text:
        page.analyze_layout(engine='tatr')
```

---

## Quick Reference: Common Fixes

| Problem | What to Try |
|---------|-------------|
| Empty text | `page.apply_ocr()` — the PDF is likely scanned |
| Garbled text | `page.apply_ocr()` — re-reads from the rendered image, bypassing broken font encoding |
| Table not detected | `page.analyze_layout(engine='tatr')` then `page.find('region[type=table]')` |
| Table columns wrong | Try `region.extract_table(method='tatr')` or use `Guides` for manual columns |
| Selector returns None | `page.find_all('text').show()` — see what text actually exists |
| Case mismatch | `page.find('text:contains("x")', case=False)` |
| Low OCR quality | `page.apply_ocr(resolution=300)` or try a different engine |
| Headers in text | `page.add_exclusion(region)` — see [Excluding Content](../tutorials/05-excluding-content.md) |
| Slow layout | `engine='yolo'` is faster than `'tatr'` (but only detects region types, not table structure) |

## Getting Help

If these solutions don't work:

1. **Visualize the problem**: Use `.show()` to see what Natural PDF is detecting
2. **Check element attributes**: Print element properties to understand the data
3. **Simplify the selector**: Start with `'text'` and add conditions incrementally
4. **Try a different approach**: OCR, layout analysis, and manual regions are all valid paths

```python
# General debugging approach
page = pdf.pages[0]

# What's on this page?
print(f"Text elements: {len(page.find_all('text'))}")
print(f"Images: {len(page.find_all('image'))}")
print(f"Lines: {len(page.find_all('line'))}")

# Visualize everything
page.find_all('*').show()
```
