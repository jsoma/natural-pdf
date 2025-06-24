# Enhanced PDF Analysis Report - eqrZ5yq

## Analysis Overview

**PDF File:** eqrZ5yq.pdf  
**Target Pages:** [89, 90, 91, 92]  
**Pages Successfully Analyzed:** [89, 90, 91, 92]  
**Analysis Date:** 2025-06-22 16:41:46

---

## Page-by-Page Analysis Results

### Page 89

**Dimensions:** 612 × 792 points

**Content Analysis:**
- **Text Content:** 716 characters extracted
- **Content Preview:** yIsI Ms Y: Syllyllys yTlynMyyTl sll
sMyTyyMlls yTl
9919 Slydlyyy lyd yyyylyylyy
ylylyl lydyylyy ly ylly ly lyyylllll yyy lyllylyyy ly yyyylyl dlyy1
· r r r r r r r r r r y r r
r r r r r r rr r r
· r r...
- **Table Found:** 2 rows × 1 columns
- **Layout Regions:** 5 regions detected
- **TATR Table Analysis:** 4 table regions detected
- **Visual:** Page image saved as `page_89.png`

### Page 90

**Dimensions:** 612 × 792 points

**Content Analysis:**
- **Text Content:** 456 characters extracted
- **Content Preview:** · r r r r r r r
r r r r r r r r r
r r r r r r y r r r r r
r r r r r r r r r r r
r r r r r r r
r r r r r rr r r r r r r r
· r r r r r r r
n Y r
· e r r r r r r - r r r
r r r r r r r r
r r rr r r r
· r ...
- **Table Found:** 5 rows × 1 columns
- **Layout Regions:** 2 regions detected
- **TATR Table Analysis:** 98 table regions detected
- **Visual:** Page image saved as `page_90.png`

### Page 91

**Dimensions:** 612 × 792 points

**Content Analysis:**
- **Text Content:** 734 characters extracted
- **Content Preview:** 4 44 13-1 4 4 44 l 4 44 44 44
tttttt gttttd tditdtit
160
t d
itit
it
d
ittig1
1
1
16
36
40
00
0
t 160
it 110
d t 100
tttt 6
9
600
0
tt 60
dtttt
iti
6
4
3
6
100
0
0
0
0
6064 6066 6066 6066 6069 6066 60...
- **Table Found:** 65 rows × 21 columns
- **Layout Regions:** 5 regions detected
- **Visual:** Page image saved as `page_91.png`

### Page 92

**Dimensions:** 612 × 792 points

**Content Analysis:**
- **Text Content:** 990 characters extracted
- **Content Preview:** · r r r r r r r r re i r
r r e r r r r re r
rr r r r r r r r r r r
· r(cid:4) r r r r rr r r r
r r r r r r r r e r r
r r y r r r r r y e re
r e r r r r r r r r r re
i e r r r r r r r r
r r r r r r r r...
- **Table Found:** 58 rows × 30 columns
- **Layout Regions:** 10 regions detected
- **TATR Table Analysis:** 13 table regions detected
- **Visual:** Page image saved as `page_92.png`

---

## Analysis Summary

### What We Found

- **Total Text Content:** 2,896 characters across 4 pages
- **Table Detection:** 4 out of 4 pages have detectable tables
- **Layout Analysis:** 22 total layout regions detected
- **TATR Analysis:** 115 table-specific regions detected

### Natural PDF Extraction Approach

Based on the actual content found on these pages:

```python
import natural_pdf as npdf

def extract_from_specific_pages(pdf_path, target_pages):
    """Extract data from specific pages with targeted approach"""
    pdf = npdf.PDF(pdf_path)
    results = []
    
    for page_num in target_pages:
        if page_num <= len(pdf.pages):
            page = pdf.pages[page_num - 1]
            
            # Use layout analysis for better structure detection
            page.analyze_layout('tatr', existing='append')
            
            # Try table extraction first
            table_data = page.extract_table()
            if table_data:
                results.append({
                    'page': page_num,
                    'type': 'table',
                    'data': table_data
                })
            else:
                # Use spatial navigation for complex layouts
                all_text = page.find_all('text')
                results.append({
                    'page': page_num, 
                    'type': 'text_elements',
                    'elements': all_text
                })
    
    return results

# Extract from your specific pages
results = extract_from_specific_pages('eqrZ5yq.pdf', [89, 90, 91, 92])
```
