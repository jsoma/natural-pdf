# Detailed PDF Analysis Report - eqQ4N7q_election results data table

## Executive Summary

**Document:** eqQ4N7q_election results data table  
**Complexity:** LOW  
**Pages Analyzed:** 1  
**Analysis Date:** 2025-06-22T17:14:07.841017

### Key Findings

#### ✅ Natural PDF Capabilities Confirmed

- Yolo Analysis
- Tatr Analysis
- Advanced Selectors

---

## Detailed Page Analysis

### Page 1

**Dimensions:** 612.48 × 791.76 points

**Text Analysis:**
- Content: 1755 characters, 1756 elements

**Table Analysis:**
- Standard extraction: 60 rows × 12 columns
- Line detection: 0 horizontal, 0 vertical

**Layout Analysis:**
- YOLO: 2 regions in 1.11s
  - Types: table: 1, table_caption: 1
- TATR: 3 table regions in 1.12s

**Advanced Selector Testing:**
- Large Text: 0 elements
- Small Text: 0 elements
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

