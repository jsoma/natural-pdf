# Enhanced PDF Analysis Report - lbODqev_Large wide tables in Serbian _from page 63 and on_

## Analysis Overview

**Document:** lbODqev_Large wide tables in Serbian _from page 63 and on_  
**Total Pages:** 252  
**Analyzed Pages:** [63, 64, 65]  
**Analysis Date:** 2025-06-22T16:43:12.441562

---

## Key Insights

- Document contains 6868 total characters across 3 analyzed pages
- 3 out of 3 pages contain detectable tables

---

## Page-by-Page Analysis

### Page 63

**Dimensions:** 595.32 × 841.92 points

**Text Content:** 2190 characters
**Preview:** Раздео Глава Програм Функција П а Пр ко т риг ор в ја н ем ко асс ттк /а клЕ аск ио фно им кас цка и...

**Table Detection:** 98 rows × 10 columns
**Sample Data:** First few rows: [['', '', '', '', 'Програмска', 'Економска', '', '', '', ''], ['Раздео', 'Глава', 'Програм', 'Функција', 'активност/\nПројекат', 'класификација', '', '', 'ОПИС Укупна', 'средства']]

**Layout Regions (YOLO):** 2
**Table Regions (TATR):** 3

### Page 64

**Dimensions:** 595.32 × 841.92 points

**Text Content:** 2171 characters
**Preview:** Раздео Глава Програм Функција П а Пр ко т риг ор в ја н ем ко асс ттк /а клЕ аск ио фно им кас цка и...

**Table Detection:** 97 rows × 8 columns
**Sample Data:** First few rows: [['', '', '', 'Програмска', 'Економска', '', '', ''], ['Раздео Глава', 'Програм', 'Функција', 'активност/\nПројекат', 'класификација', '', 'ОПИС Укупн', 'а средства']]

**Layout Regions (YOLO):** 4
**Table Regions (TATR):** 3

### Page 65

**Dimensions:** 595.32 × 841.92 points

**Text Content:** 2507 characters
**Preview:** Раздео Глава Програм Функција П а Пр ко т риг ор в ја н ем ко асс ттк /а клЕ аск ио фно им кас цка и...

**Table Detection:** 100 rows × 12 columns
**Sample Data:** First few rows: [['', '', '', 'Програмска', 'Економска', '', '', '', '', '', '', ''], ['Глава', 'Програм', 'Функција', 'активност/\nПројекат', 'класификација', '', '', 'ОПИС', '', '', 'Укупн', 'а средства']]

**Layout Regions (YOLO):** 2
**Table Regions (TATR):** 3


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

