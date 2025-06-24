# Enhanced PDF Analysis Report - ODX1DW8

## Analysis Overview

**PDF File:** ODX1DW8.pdf  
**Target Pages:** [178, 179, 180]  
**Pages Successfully Analyzed:** [178, 179, 180]  
**Analysis Date:** 2025-06-22 16:41:40

---

## Page-by-Page Analysis Results

### Page 178

**Dimensions:** 841.92 × 595.32 points

**Content Analysis:**
- **Text Content:** 731 characters extracted
- **Content Preview:** (%) ةراقلا راعسلأاب تلاامعتسلااو دراوملا
2020-2016 2015-2011 2020 2019 2018 2017 2016 2015 2014 2013 2012 2011 2010
3,5 1,5 5,3 4,8 3,8 2,5 1,0 1,1 2,3 2,3 3,9 -1,9 3,0 ق وسلا راعسأب يلحملا جتانلا
3,4...
- **Table Found:** 9 rows × 20 columns
- **Layout Regions:** 3 regions detected
- **TATR Table Analysis:** 279 table regions detected
- **Visual:** Page image saved as `page_178.png`

### Page 179

**Dimensions:** 841.92 × 595.32 points

**Content Analysis:**
- **Text Content:** 1515 characters extracted
- **Content Preview:** ةيراجلا راعسلأاب يجراخلا ليومتلاو راخدلاا
2020 2019 2018 2017 2016 2015 2014 2013 2012 2011 2010
127032,5 116115,3 106193,8 97398,6 90350,4 84656,2 80790,0 75144,1 70354,4 64492,4 63054,6 ق وسلا راعسأ...
- **Table Found:** 4 rows × 14 columns
- **Layout Regions:** 2 regions detected
- **TATR Table Analysis:** 391 table regions detected
- **Visual:** Page image saved as `page_179.png`

### Page 180

**Dimensions:** 841.92 × 595.32 points

**Content Analysis:**
- **Text Content:** 2528 characters extracted
- **Content Preview:** )د م( تاعاطقلا بسح تباثلا يلامجلإا لاملا سرأ نيوكت
2020-2016 2015-2011 2020 2019 2018 2017 2016 2015 2014 2013 2012 2011 2010
10,8 2,8 2000,0 1760,0 1550,0 1400,0 1250,0 1200,0 1111,2 1205,5 1200,6 10...
- **Table Found:** 26 rows × 40 columns
- **Layout Regions:** 2 regions detected
- **TATR Table Analysis:** 856 table regions detected
- **Visual:** Page image saved as `page_180.png`

---

## Analysis Summary

### What We Found

- **Total Text Content:** 4,774 characters across 3 pages
- **Table Detection:** 3 out of 3 pages have detectable tables
- **Layout Analysis:** 7 total layout regions detected
- **TATR Analysis:** 1526 table-specific regions detected

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
results = extract_from_specific_pages('ODX1DW8.pdf', [178, 179, 180])
```
