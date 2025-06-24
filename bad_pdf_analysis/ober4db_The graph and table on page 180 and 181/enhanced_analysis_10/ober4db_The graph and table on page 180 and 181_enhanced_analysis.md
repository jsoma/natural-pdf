# Enhanced PDF Analysis Report - ober4db_The graph and table on page 180 and 181

## Analysis Overview

**Document:** ober4db_The graph and table on page 180 and 181  
**Total Pages:** 200  
**Analyzed Pages:** [180, 181]  
**Analysis Date:** 2025-06-22T16:43:22.889838

---

## Key Insights

- Document contains 1481 total characters across 2 analyzed pages
- 2 out of 2 pages contain detectable tables
- Complex layout detected - average 6.5 regions per page

---

## Page-by-Page Analysis

### Page 180

**Dimensions:** 521.362 × 728.337 points

**Text Content:** 385 characters
**Preview:** 北京中文天地文化艺术有限公司排版 双色版
ＣＫ ６５２６３２９６
就业蓝皮书·本科
􀆰 １３
第三章
能力、知识及素养提升
一 基本工作能力评价
（一）背景介绍
工作能力： 从事某项职业工作必须具备的...

**Table Detection:** 28 rows × 2 columns
**Sample Data:** First few rows: [['第三章', ''], ['', '']]

**Layout Regions (YOLO):** 10
**Table Regions (TATR):** 0

### Page 181

**Dimensions:** 521.362 × 728.337 points

**Text Content:** 1096 characters
**Preview:** 北京中文天地文化艺术有限公司排版 双色版
ＣＫ ６５２６３２９６
分报告三·第三章 能力、知识及素养提升
表３－３－１ 基本工作能力定义及序号
序号 五大类能力 名称 描述
理解与交流能力 理解性阅读...

**Table Detection:** 31 rows × 4 columns
**Sample Data:** First few rows: [['序号', '五大类能力', '名称', '描述'], [None, '理解与交流能力', '理解性阅读', None]]

**Layout Regions (YOLO):** 3
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

