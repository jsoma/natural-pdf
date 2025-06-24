# Natural-PDF Example Workflows (Few-Shot)

_Pick snippets from here at random when building LLM prompts. Each is end-to-end and runnable._

---
## 1. Remove header/footer & extract clean report body
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

## 2. Multi-column article → single flow → extract 2nd table
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

## 3. Checkbox extraction via vision model
```python
page = PDF('form.pdf').pages[0]
boxes = (
    page.find(text='Repeat Violations').below().find_all('rect')
)
labels = boxes.classify_all(['checked', 'unchecked'], using='vision')
flags = labels.apply(lambda b: b.category)
```

## 4. Scanned ledger with line-based table detection
```python
page = PDF('scanned.pdf').pages[2]
page.apply_ocr('surya', resolution=200)
area = page.find('text:contains("Ledger")').below()
area.detect_lines(source_label='auto', peak_threshold_h=0.5, peak_threshold_v=0.25)
area.detect_table_structure_from_lines(source_label='auto')
rows = area.extract_table()
```

## 5. Colour-blob anchoring to pull legend
```python
page = PDF('map.pdf').pages[1]
page.detect_blobs()
legend = page.find('blob[color~=#fff2cc]').expand(20)
legend_text = legend.find_all('text').extract_each_text()
```

## 6. Page vision classification & selective saving
```python
pdf = PDF('mixed.pdf')
labels = ['diagram', 'text', 'blank']
pdf.classify_pages(labels, using='vision')
selected = pdf.pages.filter(lambda p: p.category=='diagram')
selected.save_pdf('diagrams_only.pdf', original=True)
```

## 7. Field extraction with `.extract()` (simple list)
```python
page = PDF('invoice.pdf').pages[0]
fields = ['invoice number', 'date', 'total amount']
page.extract(fields)
info = page.extracted()
```

## 8. Field extraction with Pydantic schema
```python
class Inv(BaseModel):
    number: str
    date: str
    total: float

page.extract(schema=Inv)
inv = page.extracted()
```

## 9. Document QA snippet
```python
page = PDF('report.pdf').pages[3]
answer = page.ask('What is the recommended action?')
assert answer.found and answer.confidence>0.6
```

## 10. Loops & groups – sum values in multiple table cells
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

## 11. Deskew an entire scanned PDF then OCR
```python
from natural_pdf import PDF

pdf = PDF('skewed_book.pdf')
# Create image-based, deskewed copy (text layer not preserved)
deskewed = pdf.deskew(resolution=300)

# Run OCR with a robust engine
deskewed.apply_ocr('surya', resolution=300)
clean_text = deskewed.extract_text()
```

## 12. Split repeated report sections and save each
```python
page = PDF('quarterly.pdf').pages[0]

# Bold headings mark each section
sections = page.get_sections(start_elements='text:bold',
                             boundary_inclusion='start')

for i, sec in enumerate(sections, 1):
    sec.save_image(f'section_{i}.png')
    with open(f'section_{i}.txt', 'w') as f:
        f.write(sec.extract_text())
```

---
_Add more as new patterns emerge._ 