# Detailed PDF Analysis Report - ODXl8aR_0. ISO code of the business_ business name_ contac

## Executive Summary

**Document:** ODXl8aR_0. ISO code of the business_ business name_ contac  
**Complexity:** HIGH  
**Pages Analyzed:** 2  
**Analysis Date:** 2025-06-22T17:14:13.386583

### Key Findings

#### ✅ Natural PDF Capabilities Confirmed

- Line Detection
- Table From Lines
- Yolo Analysis
- Tatr Analysis
- Advanced Selectors

---

## Detailed Page Analysis

### Page 1

**Dimensions:** 612 × 792 points

**Text Analysis:**
- Content: 1109 characters, 1060 elements

**Table Analysis:**
- Standard extraction: 58 rows × 7 columns
- Line detection: 17 horizontal, 2 vertical
- Table from lines: 1 tables, 16 cells

**Layout Analysis:**
- YOLO: 38 regions in 1.02s
  - Types: abandon: 3, title: 1, table: 1, table_row: 16, table_column: 1, table_cell: 16
- TATR: 3 table regions in 0.96s

**Advanced Selector Testing:**
- Large Text: 13 elements
- Small Text: 1 elements
- Bold Text: 16 elements
- Colored Rects: 0 elements
- Thin Lines: 0 elements


### Page 55

**Dimensions:** 612 × 792 points

**Text Analysis:**
- Content: 3575 characters, 3555 elements

**Table Analysis:**
- Standard extraction: 80 rows × 8 columns
- Line detection: 1 horizontal, 0 vertical
- Table from lines: 0 tables, 0 cells

**Layout Analysis:**
- YOLO: 11 regions in 0.82s
  - Types: plain text: 7, title: 1, abandon: 3
- TATR: 4 table regions in 0.45s

**Advanced Selector Testing:**
- Large Text: 12 elements
- Small Text: 1 elements
- Bold Text: 25 elements
- Colored Rects: 0 elements
- Thin Lines: 0 elements


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

