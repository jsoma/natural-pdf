# Enhanced PDF Analysis Report - GxpvezO_The table in Nepali on page 30 _in between the tex

## Analysis Overview

**Document:** GxpvezO_The table in Nepali on page 30 _in between the tex  
**Total Pages:** 33  
**Analyzed Pages:** [30]  
**Analysis Date:** 2025-06-22T16:43:00.996038

---

## Key Insights

- Document contains 2852 total characters across 1 analyzed pages
- 1 out of 1 pages contain detectable tables
- Complex layout detected - average 14.0 regions per page

---

## Page-by-Page Analysis

### Page 30

**Dimensions:** 612 × 792 points

**Text Content:** 2852 characters
**Preview:** cg';"rL -(_
- bkmf @^ sf] pkbkmf -!_ / bkmf @& sf] pkbkmf -!_ ;Fu ;DalGwt _
gd"gf lng] tyf gd"gf k|o...

**Table Detection:** 57 rows × 12 columns
**Sample Data:** First few rows: [['f;fo', 'lg', 's d', 'n k/LIf0', 'f ug{ lg/LIf', 'sn] gd"gf ln+', "bf ug'{kg]{", ';fdfGo sfo{', 'ljlw M gd"', 'gf ln+bf b]xfosf] s', 'fo{ljlw kfngf', "ug'{ kg]{5 M"], ['', '', '', '', '', '', '', '', '', '', '', '']]

**Layout Regions (YOLO):** 14
**Table Regions (TATR):** 1


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

