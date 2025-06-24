# Detailed PDF Analysis Report - GxpvezO_The table in Nepali on page 30 _in between the tex

## Executive Summary

**Document:** GxpvezO_The table in Nepali on page 30 _in between the tex  
**Complexity:** HIGH  
**Pages Analyzed:** 1  
**Analysis Date:** 2025-06-22T17:16:49.274268

### Key Findings

#### ✅ Natural PDF Capabilities Confirmed

- Line Detection
- Table From Lines
- Yolo Analysis
- Tatr Analysis
- Advanced Selectors

---

## Detailed Page Analysis

### Page 30

**Dimensions:** 612 × 792 points

**Text Analysis:**
- Content: 2852 characters, 2868 elements

**Table Analysis:**
- Standard extraction: 57 rows × 12 columns
- Line detection: 20 horizontal, 0 vertical
- Table from lines: 0 tables, 0 cells

**Layout Analysis:**
- YOLO: 33 regions in 1.11s
  - Types: title: 2, table_row: 19, plain text: 10, table: 1, abandon: 1
- TATR: 1 table regions in 0.69s

**Advanced Selector Testing:**
- Large Text: 4 elements
- Small Text: 0 elements
- Bold Text: 0 elements
- Colored Rects: 9 elements
- Thin Lines: 9 elements
- 🎯 Text formatting candidates: 1


---

## Natural PDF Integration Recommendations

Based on this detailed analysis:

```python
import natural_pdf as npdf

def process_document_optimally(pdf_path):
    """Optimized processing based on analysis findings"""
    pdf = npdf.PDF(pdf_path)
    results = []
    
    for page_num, page in enumerate(pdf.pages, 1):
        # Use discovered line detection capability
        page.detect_lines(
            resolution=144,
            method="projection",  # No OpenCV required
            horizontal=True,
            vertical=True,
            peak_threshold_h=0.3,
            peak_threshold_v=0.3
        )
        
        # Create table structure from detected lines
        page.detect_table_structure_from_lines(
            source_label="detected",
            ignore_outer_regions=True,
            cell_padding=0.5
        )
        
        # Extract using multiple methods
        standard_table = page.extract_table()
        line_based_tables = page.find_all('region[type="table"]')
        
        results.append({
            'page': page_num,
            'standard_table': standard_table,
            'line_based_tables': len(line_based_tables)
        })
    
    return results
```

