# Detailed PDF Analysis Report - obe1Vq5_MARKED UP text -- underline and strikethu__for bon

## Executive Summary

**Document:** obe1Vq5_MARKED UP text -- underline and strikethu__for bon  
**Complexity:** HIGH  
**Pages Analyzed:** 1  
**Analysis Date:** 2025-06-22T17:16:33.597148

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
- Content: 1781 characters, 1740 elements

**Table Analysis:**
- Standard extraction: 45 rows × 7 columns
- Line detection: 36 horizontal, 0 vertical
- Table from lines: 0 tables, 0 cells

**Layout Analysis:**
- YOLO: 41 regions in 1.40s
  - Types: abandon: 3, plain text: 3, table_row: 35
- TATR: 2 table regions in 0.73s

**Advanced Selector Testing:**
- Large Text: 47 elements
- Small Text: 4 elements
- Bold Text: 0 elements
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

