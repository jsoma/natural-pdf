# Enhanced PDF Analysis Report - b5eVqGg_Math formulas in Russian _e.g. on page 181__

## Analysis Overview

**Document:** b5eVqGg_Math formulas in Russian _e.g. on page 181__  
**Total Pages:** 222  
**Analyzed Pages:** [181]  
**Analysis Date:** 2025-06-22T16:43:10.081094

---

## Key Insights

- Document contains 1593 total characters across 1 analyzed pages
- 1 out of 1 pages contain detectable tables
- Complex layout detected - average 18.0 regions per page

---

## Page-by-Page Analysis

### Page 181

**Dimensions:** 595.276 × 841.89001 points

**Text Content:** 1593 characters
**Preview:** Постановление Правительства РФ от 29.12.2011 N 1178
(ред. от 19.01.2022) Документ предоставленКонсул...

**Table Detection:** 43 rows × 4 columns
**Sample Data:** First few rows: [['Постановление Правительства', 'РФ от 29.12.2', '011 N 1178', ''], ['(ред. от 19.01.2022)', '', 'Документ', 'предоставленКонсультантПлюс']]

**Layout Regions (YOLO):** 18
**Table Regions (TATR):** 2


---

## Natural PDF Extraction Recommendations

Based on this analysis, here are the recommended approaches:

```python
import natural_pdf as npdf

def extract_document_data(pdf_path):
    pdf = npdf.PDF(pdf_path)
    results = []
    
    for page_num, page in enumerate(pdf.pages, 1):
        # Use layout analysis for structure detection
        page.analyze_layout('tatr', existing='append')
        
        # Extract tables if present
        table_data = page.extract_table()
        if table_data:
            results.append({
                'page': page_num,
                'type': 'table',
                'data': table_data
            })
        
        # Extract text content
        text_content = page.extract_text()
        if text_content:
            results.append({
                'page': page_num,
                'type': 'text',
                'content': text_content
            })
    
    return results
```

