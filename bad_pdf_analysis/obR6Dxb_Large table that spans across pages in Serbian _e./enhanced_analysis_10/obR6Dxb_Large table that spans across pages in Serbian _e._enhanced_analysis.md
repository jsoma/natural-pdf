# Enhanced PDF Analysis Report - obR6Dxb_Large table that spans across pages in Serbian _e.

## Analysis Overview

**Document:** obR6Dxb_Large table that spans across pages in Serbian _e.  
**Total Pages:** 120  
**Analyzed Pages:** [1, 2, 3]  
**Analysis Date:** 2025-06-22T16:43:18.130650

---

## Key Insights

- Document contains 6916 total characters across 3 analyzed pages
- 3 out of 3 pages contain detectable tables
- Complex layout detected - average 21.7 regions per page

---

## Page-by-Page Analysis

### Page 1

**Dimensions:** 595.32 × 841.92 points

**Text Content:** 1743 characters
**Preview:** BUDITE NA PRAVNOJ STRANI
online@paragraf.rs
www.paragraf.rs
Preuzeto iz elektronske pravne baze Para...

**Table Detection:** 4 rows × 1 columns
**Sample Data:** First few rows: [['ZAKON'], ['O NAKNADAMA ZA KORIŠĆENJE JAVNIH DOBARA']]

**Layout Regions (YOLO):** 19
**Table Regions (TATR):** 2

### Page 2

**Dimensions:** 595.32 × 841.92 points

**Text Content:** 2115 characters
**Preview:** Član 3
Naknade za korišćenje javnih dobara mogu se uvoditi samo ovim zakonom.
II VRSTE NAKNADA ZA KO...

**Table Detection:** 75 rows × 9 columns
**Sample Data:** First few rows: [['', '', '', '', '', 'Član', '3', '', ''], ['', '', '', '', '', '', '', '', '']]

**Layout Regions (YOLO):** 22
**Table Regions (TATR):** 2

### Page 3

**Dimensions:** 595.32 × 841.92 points

**Text Content:** 3058 characters
**Preview:** Valorizovanu vrednost iz stava 1. ovog člana utvrđuje ministarstvo u čijoj su nadležnosti poslovi ru...

**Table Detection:** 68 rows × 9 columns
**Sample Data:** First few rows: [['Valorizovanu vrednost', 'iz stava 1. ov', 'og člana', 'utvrđuje minis', 'tarstv', 'o u čijoj su n', 'adležnosti po', 'slovi rudars', 'tva i geološ'], ['istraživanja primenom', 'koeficijenta g', 'odišnjeg r', 'asta potrošač', 'kih ce', 'na na vredno', 'st izvršenih i', 'straživanja', 'u periodu k']]

**Layout Regions (YOLO):** 24
**Table Regions (TATR):** 0


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

