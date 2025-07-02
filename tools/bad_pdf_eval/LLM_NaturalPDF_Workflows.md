# Natural-PDF Example Workflows (Few-Shot)

_Pick snippets from here at random when building LLM prompts. Each is end-to-end and runnable._

---
## 1. Modern Table Extraction with Guides (Preferred Approach)
```python
from natural_pdf import PDF
from natural_pdf.analyzers import Guides

pdf = PDF('financial_report.pdf')
page = pdf.pages[0]

# Find column headers and use them as guides
headers = page.find_all('text:bold').filter(lambda t: t.y0 < 200)
guides = Guides.from_content(page, markers=headers, axis='vertical', align='center')

# Snap to natural whitespace gaps for better alignment
guides.vertical.snap_to_whitespace(min_gap=15)

# Add horizontal guides from content
guides.horizontal.from_content(markers='text[size>=10]', align='top')

# Build grid and extract
guides.build_grid()
df = page.find('table').extract_table().df
```

## 2. Table Extraction Without Any Layout Model
```python
# Direct extraction from a region - no TATR needed!
page = PDF('invoice.pdf').pages[0]

# Find the table area using anchors
table_region = (
    page.find('text:contains("Item Description")')
    .parent('rect', mode='overlap')  # Find containing box
    .expand(10)  # Small padding
)

# Extract directly - Natural PDF will infer structure
df = table_region.extract_table().df
```

## 3. Guides with Whitespace Detection for Complex Forms
```python
from natural_pdf.analyzers import Guides

page = PDF('government_form.pdf').pages[0]

# Auto-detect gaps in the content
guides = Guides.from_whitespace(page, min_gap=20)

# Fine-tune by snapping to actual content boundaries
guides.vertical.snap_to_content(markers='text', align='left')

# Preview what we found
guides.show()

# Extract form fields as table
guides.build_grid()
cells = page.find_all('table_cell')
form_data = {
    cell.extract_text().split(':')[0].strip():
    cell.extract_text().split(':')[1].strip()
    for cell in cells if ':' in cell.extract_text()
}
```

## 4. Handling Tables that TATR Misses
```python
# When TATR fails, use Guides for precise control
page = PDF('unusual_layout.pdf').pages[0]

# Try TATR first
page.analyze_layout('tatr')
if not page.find('table'):
    # Fallback to Guides
    guides = Guides(page)

    # Detect from pixel-based lines (no vectors needed)
    guides.horizontal.from_lines(
        detection_method='pixels',
        threshold='auto',
        resolution=192
    )
    guides.vertical.from_lines(
        detection_method='pixels',
        threshold='auto'
    )

    # Clean up and align
    guides.horizontal.snap_to_whitespace()
    guides.vertical.snap_to_whitespace()

    guides.build_grid()
    df = page.find('table').extract_table().df
```

## 5. Remove header/footer & extract clean report body
```python
from natural_pdf import PDF
pdf = PDF('report.pdf')
page = pdf.pages[0]
# Exclude header (top 10 % of page)
page.add_exclusion(page.region(top=0, bottom=page.height*0.1))
# Exclude footer (all text below last horizontal line)
page.add_exclusion(page.find_all('line:horizontal')[-1].below())
text = page.extract_text()
```

## 6. Multi-column article → single flow → extract 2nd table
```python
from natural_pdf import PDF
from natural_pdf.flows import Flow
pdf = PDF('article.pdf')
page = pdf.pages[0]
w = page.width
columns = [page.region(left=i*w/3, right=(i+1)*w/3) for i in range(3)]
flow = Flow(columns, arrangement='vertical')
flow.analyze_layout('tatr')
for table in flow.find_all('table'):
    data = tbl.extract_table()
```

## 7. Checkbox extraction via vision model
```python
page = PDF('form.pdf').pages[0]
boxes = (
    page.find(text='Repeat Violations').below().find_all('rect')
)
labels = boxes.classify_all(['checked', 'unchecked'], using='vision')
flags = labels.apply(lambda b: b.category)
```

## 8. Scanned ledger with line-based table detection
```python
# Updated to use Guides API
from natural_pdf.analyzers import Guides

page = PDF('scanned_ledger.pdf').pages[0]
page.apply_ocr('surya', resolution=300)

# Create guides from detected lines
guides = Guides.from_lines(
    page,
    detection_method='pixels',  # Works on scanned images
    threshold='auto',
    resolution=192
)

# Snap to improve alignment
guides.vertical.snap_to_whitespace(min_gap=15)
guides.horizontal.snap_to_content(markers='text', align='center')

# Build and extract
guides.build_grid()
df = page.find('table').extract_table().df
```

## 9. Colour-blob anchoring to pull legend
```python
page = PDF('map.pdf').pages[1]
page.detect_blobs()
legend = page.find('blob[color~=#fff2cc]').expand(20)
legend_text = legend.find_all('text').extract_each_text()
```

## 10. Page vision classification & selective saving
```python
pdf = PDF('mixed.pdf')
labels = ['diagram', 'text', 'blank']
pdf.classify_pages(labels, using='vision')
selected = pdf.pages.filter(lambda p: p.category=='diagram')
selected.save_pdf('diagrams_only.pdf', original=True)
```

## 11. Field extraction with `.extract()` (simple list)
```python
page = PDF('invoice.pdf').pages[0]
fields = ['invoice number', 'date', 'total amount']
page.extract(fields)
info = page.extracted()
```

## 12. Field extraction with Pydantic schema
```python
class Inv(BaseModel):
    number: str
    date: str
    total: float

page.extract(schema=Inv)
inv = page.extracted()
```

## 13. Document QA snippet
```python
page = PDF('report.pdf').pages[3]
answer = page.ask('What is the recommended action?')
assert answer.found and answer.confidence>0.6
```

## 14. Loops & groups – sum values in multiple table cells
```python
page = PDF('table.pdf').pages[0]
page.analyze_layout('tatr')
nums = (
    page.find_all('table_cell')
        .group_by('row')                       # group cells row-wise
        .apply(lambda row: float(row[2].extract_text()))
)
print(sum(nums))
```

## 15. Deskew an entire scanned PDF then OCR
```python
from natural_pdf import PDF

pdf = PDF('skewed_book.pdf')
# Create image-based, deskewed copy (text layer not preserved)
deskewed = pdf.deskew(resolution=300)

# Run OCR with a robust engine
deskewed.apply_ocr('surya', resolution=300)
clean_text = deskewed.extract_text()
```

## 16. Split repeated report sections and save each
```python
page = PDF('quarterly.pdf').pages[0]

# Bold headings mark each section
sections = page.get_sections(start_elements='text:bold',
                             include_boundaries='start')

for i, sec in enumerate(sections, 1):
    sec.save_image(f'section_{i}.png')
    with open(f'section_{i}.txt', 'w') as f:
        f.write(sec.extract_text())
```

## 17. Extract complex table data using manual Guides API
```python
from natural_pdf import PDF
from natural_pdf.analyzers import Guides
pdf = PDF('form.pdf')
page = pdf.pages[0]

# Create guides object for the page
guides = Guides(page)

# Method 1: Content-based guides (smart)
guides.vertical.from_content(
    markers=['Name:', 'Address:', 'Phone Number:'],
    align='after',  # Place guides after field labels
    expansion=10    # Extend guides to capture field areas
)

# Find horizontal boundaries from all text - you can use selectors or ElementCollection
guides.horizontal.from_content(
    markers='text:contains(":")',  # Single selector for field labels
    align='center'  # Center guides on text lines
)

# Alternative: Use ElementCollection directly
field_labels = page.find_all('text:contains(":")')
guides.horizontal.from_content(
    markers=field_labels,  # Pass ElementCollection directly
    align='center'
)

# Method 2: Pixel-based line detection (no vector lines needed!)
guides_auto = Guides.from_lines(
    page,
    detection_method='pixels',  # Detect from image
    threshold='auto',           # Auto-find best threshold
    max_lines_h=10,            # Limit lines found
    max_lines_v=4,
    resolution=192,            # DPI for detection
    min_gap_h=15               # Minimum gap between lines
)

# Show what we found
guides_auto.show()

# Fine-tune by snapping to whitespace
guides_auto.vertical.snap_to_whitespace(min_gap=20)
guides_auto.horizontal.snap_to_whitespace()

# Method 3: Combine both approaches
guides = Guides(page)
# Start with pixel detection for major lines
guides.horizontal.from_lines(detection_method='pixels', max_lines=5)
# Add content-based vertical guides
guides.vertical.from_content(markers=['Name:', 'Date:', 'Amount:'], align='after')
# Manual adjustment if needed
guides.vertical.add(450)  # Add one more guide

# Build the grid and extract
guides.build_grid(source='form_fields')
cells = page.find_all('table_cell[source=form_fields]')

# Extract field data
form_data = {}
for cell in cells:
    text = cell.extract_text().strip()
    if ':' in text:
        # This is a label cell
        label = text.replace(':', '')
        # Find the cell to its right (same row, next column)
        value_cell = page.find(f'table_cell[row_index={cell.metadata["row_index"]}][col_index={cell.metadata["col_index"]+1}]')
        if value_cell:
            form_data[label] = value_cell.extract_text().strip()

print(form_data)
# {'Name': 'John Smith', 'Address': '123 Main St', 'Phone Number': '555-1234'}
```

## 18. Handle PDFs with Corrupted Text Layer
```python
from natural_pdf import PDF

# Case 1: Text shows as "(cid:xxx)" gibberish
pdf = PDF('corrupted_text.pdf')
page = pdf.pages[0]

# Check if text layer is corrupted
sample_text = page.extract_text()[:100]
if '(cid:' in sample_text:
    print("Corrupted text layer detected, removing and using OCR")
    page.remove_text_layer()
    page.apply_ocr('surya', resolution=300)

# Case 2: Open PDF ignoring text layer from the start
pdf_clean = PDF('corrupted_text.pdf', text_layer=False)
pdf_clean.apply_ocr('surya')  # OCR all pages at once

# Now extraction works normally
text = pdf_clean.pages[0].extract_text()
tables = pdf_clean.pages[0].find_all('table')
```

## 19. Smart Region Selection: When to Use 'until' Parameter
```python
from natural_pdf import PDF
pdf = PDF('document.pdf')
page = pdf.pages[0]

# Case 1: Going to page edge (no 'until' needed)
header = page.find('text:contains("Introduction")')
intro_section = header.below()  # Gets everything below until page end
print(f"Intro text: {intro_section.extract_text()}")

# Case 2: Stopping before next section (use 'until')
section1 = page.find('text:contains("Section 1")')
section2 = page.find('text:contains("Section 2")')

# Extract only Section 1 content, stop before Section 2
section1_content = section1.below(until=section2)
print(f"Section 1 only: {section1_content.extract_text()}")

# Case 3: Complex document with multiple stopping points
invoice_header = page.find('text:contains("INVOICE")')
items_section = invoice_header.below(until='text:contains("Total:")')

# Extract table from items section
items_table = items_section.find('table')
df = items_table.extract_table().df

# Case 4: Using semantic boundaries
article_title = page.find('text[size>16]:first')
# Stop at next large heading or image
article_body = article_title.below(until='text[size>16], image')
```

---
_Add more as new patterns emerge._
