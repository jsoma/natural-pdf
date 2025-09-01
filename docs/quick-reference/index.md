# Quick Reference

## Essential Workflows

### Basic Text Extraction
```py
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]
text = page.extract_text()
```

### Find → Extract Pattern
```py
# Find specific elements, then extract
heading = page.find('text:contains("Summary"):bold')
content = heading.below().extract_text()
```

### OCR for Scanned Documents
```py
# Apply OCR first, then extract
page.apply_ocr(engine='easyocr', languages=['en'])
text = page.extract_text()
```

### Layout Analysis → Table Extraction
```py
# Detect layout, then extract tables
page.analyze_layout(engine='yolo')
table_region = page.find('region[type=table]')
data = table_region.extract_table()
```

## Common Selectors

### Text Content
```py
page.find('text:contains("Invoice")')           # Contains text
page.find('text:contains("total")', case=False) # Case insensitive
page.find('text:contains("\\d+")', regex=True)  # Regex pattern
page.find('text:closest("Invoice Date")')       # Fuzzy match (OCR errors)
page.find('text:closest("Total@0.8")')          # 80% similarity threshold
```

### Text Formatting
```py
page.find_all('text:bold')                      # Bold text
page.find_all('text:italic')                    # Italic text
page.find_all('text:strike')                    # Struck-through text
page.find_all('text:underline')                 # Underlined text
page.find_all('text[size>=12]')                 # Large text
page.find_all('text[fontname*=Arial]')          # Specific font
```

### Spatial Relationships
```py
page.find('text:above("line[width>=2]")')       # Above thick line
page.find('text:below("text:contains("Title")")')  # Below title
page.find('text:near("image")')                 # Near images
```

### Layout Elements
```py
page.find_all('line:horizontal')                # Horizontal lines
page.find_all('rect')                           # Rectangles
page.find_all('region[type=table]')             # Detected tables
page.find_all('region[type=title]')             # Detected titles
```

### OCR and Sources
```py
page.find_all('text[source=ocr]')               # OCR-generated text
page.find_all('text[source=pdf]')               # Original PDF text
page.find_all('text[confidence>=0.8]')          # High-confidence OCR
```

### Statistical Selectors (Aggregates)
```py
page.find('text[x0=min()]')                      # Leftmost text
page.find('text[x1=max()]')                      # Rightmost text
page.find('text[size=max()]')                    # Largest text
page.find('text[width=avg()]')                   # Average width text
page.find('text[height=median()]')               # Median height text
page.find('text[fontname=mode()]')               # Most common font
page.find('text[color=closest("red")]')          # Closest to red
page.find('text[x0=min()][size=max()]')          # Leftmost AND largest
```

## Essential Methods

### Finding Elements
```py
page.find(selector)                             # First match
page.find_all(selector)                         # All matches
element.next()                                  # Next element in reading order
element.previous()                              # Previous element
```

### Spatial Navigation
```py
# Smart defaults (new in 0.9.0)
element.left()                                  # Default height='element' (matches element height)
element.right()                                 # Default height='element' (matches element height)
element.above()                                 # Default width='full' (full page width)
element.below()                                 # Default width='full' (full page width)

# Custom dimensions
element.above(height=100)                       # Fixed height above
element.below(until='line:horizontal')          # Below until boundary
element.left(width=200)                         # Fixed width to left
element.right(height='full')                    # Full page height to right

# Exclusion handling
element.below(apply_exclusions=True)            # Skip exclusion zones
element.expand('down', 50, apply_exclusions=True)  # Expand with exclusions
```

### Text Extraction
```py
page.extract_text()                             # All text from page
page.extract_text(layout=True)                  # Preserve layout
element.extract_text()                          # Text from specific element
region.extract_text()                           # Text from region
```

### Table Extraction
```py
page.extract_table()                            # First table on page
region.extract_table()                          # Table from region
region.extract_table(method='tatr')             # Force TATR method
region.extract_table(method='pdfplumber')       # Force pdfplumber method
```

### OCR
```py
page.apply_ocr()                                # Default OCR
page.apply_ocr(engine='paddle', languages=['en', 'zh-cn'])
page.apply_ocr(engine='easyocr', min_confidence=0.8)
region.apply_ocr()                              # OCR specific region
```

### Layout Analysis
```py
page.analyze_layout()                           # Default YOLO
page.analyze_layout(engine='tatr')              # Table-focused
page.analyze_layout(engine='surya')             # High accuracy
page.clear_detected_layout_regions()           # Clear previous results
```

### Document QA
```py
result = page.ask("What is the total amount?")
print(result.answer)                            # The answer
print(result.confidence)                        # Confidence score
result.show()                                   # Highlight answer location
```

### Structured Data Extraction
```py
# Simple approach
data = page.extract(schema=["company", "date", "total"]).extracted()

# With Pydantic schema
from pydantic import BaseModel
class Invoice(BaseModel):
    company: str
    total: float
    date: str

data = page.extract(schema=Invoice, client=client).extracted()
```

## Visualization & Debugging

### Highlighting
```py
# Simple visualization
elements.show(color="red")                      # Single collection
elements.show(color="blue", label="Headers")    # With label
elements.show(group_by='type')                  # Color by type

# Quick highlighting (one-liner)
page.highlight(elements1, elements2, elements3)  # Multiple elements
page.highlight(                                  # With custom colors
    (elements1, 'red'),
    (elements2, 'blue'),
    (elements3, 'green')
)

# Multiple collections with context manager
with page.highlights() as h:
    h.add(elements1, color="red", label="Type 1")
    h.add(elements2, color="blue", label="Type 2")
    h.show()

# Auto-display in Jupyter/Colab
with page.highlights(show=True) as h:
    h.add(elements1, label="Headers")
    h.add(elements2, label="Content")
    # Displays automatically when exiting context
```

### Viewing
```py
page.show()                                     # Show page with highlights
element.show()                                  # Show specific element
page.show(width=700)                        # Generate image
region.show(crop=True)                 # Crop to region only
```

### Interactive Viewer
```py
page.viewer()                                   # Launch interactive viewer (Jupyter)
```

## Exclusion Zones

### Page-Level Exclusions
```py
# Smart exclusion behavior (new in 0.9.0)
text_element = page.find('text:contains("CONFIDENTIAL")')
page.add_exclusion(text_element)                # Excludes just the text bounding box

# Traditional region exclusion
header_region = page.find('text:contains("CONFIDENTIAL")').above()
page.add_exclusion(header_region)               # Excludes entire region

# Manage exclusions
page.clear_exclusions()                         # Remove all exclusions
text = page.extract_text(use_exclusions=False)  # Ignore exclusions
```

### PDF-Level Exclusions
```py
# Exclude headers from all pages
pdf.add_exclusion(
    lambda p: p.create_region(0, 0, p.width, p.height * 0.1),
    label="Header"
)

# Exclude specific text elements (new in 0.9.0)
pdf.add_exclusion(
    lambda p: p.find_all('text:contains("Header")'),  # Returns ElementCollection
    label="Headers"
)
```

## Configuration Options

### Global Layout Settings
```py
import natural_pdf

# Configure global directional offset (default: 5)
natural_pdf.options.layout.directional_offset = 10  # Larger gap for directional methods

# Reset to default
natural_pdf.options.layout.directional_offset = 5
```

### OCR Engines
```py
from natural_pdf.ocr import EasyOCROptions, PaddleOCROptions

easy_opts = EasyOCROptions(gpu=True, paragraph=True)
paddle_opts = PaddleOCROptions(lang='en')
```

### Layout Analysis Options
```py
from natural_pdf.analyzers.layout import YOLOOptions

yolo_opts = YOLOOptions(confidence_threshold=0.5)
page.analyze_layout(engine='yolo', options=yolo_opts)
```

## Common Patterns

### Extract Inspection Report Data
```py
# Find violation count (uses smart default height='element')
violations = page.find('text:contains("Violation Count"):right()')

# Get inspection number from the header box (regex search)
inspection_num = page.find('text:contains("INS-[A-Z0-9]+")', regex=True)

# Extract inspection date (custom width for wider field)
inspection_date = page.find('text:contains("Date:"):right(width=150)')

# Get site name (uses smart default height='element')
site_name = page.find('text:contains("Site:"):right()').extract_text()
```

### Process Forms
```py
# Exclude header/footer
page.add_exclusion(page.create_region(0, 0, page.width, 50))
page.add_exclusion(page.create_region(0, page.height-50, page.width, page.height))

# Extract form fields (smart defaults + exclusion handling)
fields = page.find_all('text:bold')
values = [field.right(apply_exclusions=True).extract_text() for field in fields]
```

### Handle Scanned Documents
```py
# Apply OCR with high accuracy
page.apply_ocr(engine='surya', languages=['en'])

# Extract with confidence filtering
text_elements = page.find_all('text[source=ocr][confidence>=0.8]')
clean_text = text_elements.extract_text()
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No text found | Try `page.apply_ocr()` first |
| Wrong elements selected | Use `elements.show()` to debug selectors |
| Poor table extraction | Try `page.analyze_layout(engine='tatr')` first |
| Text extraction includes headers | Use `page.add_exclusion()` |
| Low OCR accuracy | Try different engine or increase resolution |
| Elements overlap multiple pages | Use page-specific searches |

## File Formats

### Saving Results
```py
# Save as image
page.save_image("output.png", width=700)

# Save table as CSV
import pandas as pd
df = table_data.to_df(header="first")
df.to_csv("table.csv")

# Export searchable PDF
from natural_pdf.exporters import SearchablePDFExporter
exporter = SearchablePDFExporter()
exporter.export(pdf, "searchable.pdf")
```

## Next Steps

- **New to Natural PDF?** → Start with [Installation](../installation/)
- **Learning the basics?** → Follow the [Tutorials](../tutorials/)
- **Solving specific problems?** → Check the how-to guides
- **Need detailed info?** → See the [API Reference](../api/)
