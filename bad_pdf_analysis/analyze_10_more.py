#!/usr/bin/env python3
"""
Analyze 10 more PDF documents from the bad PDF collection
"""

import os
import sys
import json
from datetime import datetime
import natural_pdf as npdf

# Add the project root to the path
sys.path.append('/Users/soma/Development/natural-pdf')

def analyze_pdf_document(pdf_path, document_name, target_pages=None):
    """Analyze a specific PDF document with enhanced reporting"""
    print(f"\n{'='*80}")
    print(f"🔍 Analyzing {document_name}")
    print(f"📁 Path: {pdf_path}")
    if target_pages:
        print(f"📍 Target pages: {target_pages}")
    print(f"{'='*80}")
    
    try:
        pdf = npdf.PDF(pdf_path)
        total_pages = len(pdf.pages)
        print(f"📄 Total pages in document: {total_pages}")
        
        # Determine which pages to analyze
        if target_pages:
            pages_to_analyze = [p for p in target_pages if p <= total_pages]
            if len(pages_to_analyze) != len(target_pages):
                print(f"⚠️ Some target pages exceed document length, analyzing: {pages_to_analyze}")
        else:
            # Default to first page if no specific pages requested
            pages_to_analyze = [1] if total_pages > 0 else []
        
        results = {
            'document': document_name,
            'total_pages': total_pages,
            'analyzed_pages': pages_to_analyze,
            'analysis_date': datetime.now().isoformat(),
            'pages': {}
        }
        
        for page_num in pages_to_analyze:
            print(f"\n📄 Analyzing page {page_num}...")
            page = pdf.pages[page_num - 1]  # Convert to 0-based index
            
            page_results = {
                'page_number': page_num,
                'dimensions': f"{page.width} × {page.height} points"
            }
            
            # Extract text
            try:
                text_content = page.extract_text()
                page_results['text_length'] = len(text_content)
                page_results['text_preview'] = text_content[:200] + "..." if len(text_content) > 200 else text_content
                print(f"✅ Text extraction: {len(text_content)} characters")
            except Exception as e:
                page_results['text_error'] = str(e)
                print(f"❌ Text extraction failed: {e}")
            
            # Try table extraction
            try:
                table_data = page.extract_table()
                if table_data and len(table_data) > 0:
                    rows = len(table_data)
                    cols = max(len(row) for row in table_data) if table_data else 0
                    page_results['table'] = f"{rows} rows × {cols} columns"
                    page_results['table_sample'] = table_data[:3] if len(table_data) >= 3 else table_data
                    print(f"✅ Table found: {rows} rows × {cols} columns")
                else:
                    page_results['table'] = "No table detected"
                    print("ℹ️ No table detected")
            except Exception as e:
                page_results['table_error'] = str(e)
                print(f"❌ Table extraction failed: {e}")
            
            # Layout analysis with YOLO
            try:
                page.analyze_layout('yolo')
                yolo_regions = page.find_all('region')
                page_results['yolo_regions'] = len(yolo_regions)
                print(f"✅ YOLO layout analysis: {len(yolo_regions)} regions")
            except Exception as e:
                page_results['yolo_error'] = str(e)
                print(f"❌ YOLO analysis failed: {e}")
            
            # Layout analysis with TATR (table-specific)
            try:
                page.analyze_layout('tatr', existing='append')
                tatr_regions = page.find_all('region[type="table"]')
                page_results['tatr_regions'] = len(tatr_regions)
                print(f"✅ TATR analysis: {len(tatr_regions)} table regions")
            except Exception as e:
                page_results['tatr_error'] = str(e)
                print(f"❌ TATR analysis failed: {e}")
            
            # Save page image
            try:
                folder_name = document_name.replace('/', '_').replace('\\', '_')
                analysis_dir = f"/Users/soma/Development/natural-pdf/bad_pdf_analysis/{folder_name}/enhanced_analysis_10"
                os.makedirs(analysis_dir, exist_ok=True)
                
                image_path = f"{analysis_dir}/page_{page_num}.png"
                page_image = page.to_image(resolution=144)
                page_image.save(image_path)
                page_results['image_saved'] = image_path
                print(f"✅ Page image saved: page_{page_num}.png")
            except Exception as e:
                page_results['image_error'] = str(e)
                print(f"❌ Image save failed: {e}")
            
            results['pages'][page_num] = page_results
        
        # Generate analysis summary
        analysis_insights = generate_analysis_insights(results)
        results['insights'] = analysis_insights
        
        # Save results to JSON
        try:
            folder_name = document_name.replace('/', '_').replace('\\', '_')
            analysis_dir = f"/Users/soma/Development/natural-pdf/bad_pdf_analysis/{folder_name}/enhanced_analysis_10"
            os.makedirs(analysis_dir, exist_ok=True)
            
            results_path = f"{analysis_dir}/analysis_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Analysis results saved: {results_path}")
            
            # Generate markdown report
            markdown_path = f"{analysis_dir}/{document_name}_enhanced_analysis.md"
            generate_markdown_report(results, markdown_path)
            print(f"✅ Markdown report saved: {markdown_path}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to analyze {document_name}: {e}")
        return None

def generate_analysis_insights(results):
    """Generate insights based on analysis results"""
    insights = []
    
    total_chars = sum(page.get('text_length', 0) for page in results['pages'].values())
    table_pages = sum(1 for page in results['pages'].values() if 'table' in page and 'rows' in page['table'])
    
    if total_chars > 0:
        insights.append(f"Document contains {total_chars} total characters across {len(results['pages'])} analyzed pages")
    
    if table_pages > 0:
        insights.append(f"{table_pages} out of {len(results['pages'])} pages contain detectable tables")
    
    # Check for layout complexity
    avg_regions = sum(page.get('yolo_regions', 0) for page in results['pages'].values()) / len(results['pages'])
    if avg_regions > 5:
        insights.append(f"Complex layout detected - average {avg_regions:.1f} regions per page")
    
    # Check for table structure complexity
    tatr_regions = sum(page.get('tatr_regions', 0) for page in results['pages'].values())
    if tatr_regions > 50:
        insights.append(f"High table complexity - {tatr_regions} TATR table regions detected")
    
    return insights

def generate_markdown_report(results, output_path):
    """Generate a detailed markdown report"""
    
    content = f"""# Enhanced PDF Analysis Report - {results['document']}

## Analysis Overview

**Document:** {results['document']}  
**Total Pages:** {results['total_pages']}  
**Analyzed Pages:** {results['analyzed_pages']}  
**Analysis Date:** {results['analysis_date']}

---

## Key Insights

"""
    
    for insight in results.get('insights', []):
        content += f"- {insight}\n"
    
    content += "\n---\n\n## Page-by-Page Analysis\n\n"
    
    for page_num, page_data in results['pages'].items():
        content += f"### Page {page_num}\n\n"
        content += f"**Dimensions:** {page_data.get('dimensions', 'Unknown')}\n\n"
        
        if 'text_length' in page_data:
            content += f"**Text Content:** {page_data['text_length']} characters\n"
            if 'text_preview' in page_data:
                content += f"**Preview:** {page_data['text_preview'][:100]}...\n\n"
        
        if 'table' in page_data:
            content += f"**Table Detection:** {page_data['table']}\n"
            if 'table_sample' in page_data and page_data['table_sample']:
                content += f"**Sample Data:** First few rows: {page_data['table_sample'][:2]}\n\n"
        
        if 'yolo_regions' in page_data:
            content += f"**Layout Regions (YOLO):** {page_data['yolo_regions']}\n"
        
        if 'tatr_regions' in page_data:
            content += f"**Table Regions (TATR):** {page_data['tatr_regions']}\n"
        
        content += "\n"
    
    content += """
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

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Analyze 10 more PDF documents"""
    
    # List of documents to analyze with specific pages if needed
    documents_to_analyze = [
        # Documents with specific page requests
        ("GxpvezO_The table in Nepali on page 30 _in between the tex", "GxpvezO.pdf", [30]),
        ("J9lKd7Y_Table in Slovenian _e.g. on page 80_.", "J9lKd7Y.pdf", [80]),
        ("b5eVqGg_Math formulas in Russian _e.g. on page 181__", "b5eVqGg.pdf", [181]),
        ("lbODqev_Large wide tables in Serbian _from page 63 and on_", "lbODqev.pdf", [63, 64, 65]),
        ("obR6Dxb_Large table that spans across pages in Serbian _e.", "obR6Dxb.pdf", [1, 2, 3]),
        ("ober4db_The graph and table on page 180 and 181", "ober4db.pdf", [180, 181]),
        ("oberryX_The survery question table_ such as the one on pag", "oberryX.pdf", [1]),  # Need to find specific page
        ("eqrZZbq_The categorize chart _E1_ on page 4_ The chart_tab", "eqrZZbq.pdf", [4]),
        
        # Documents with general analysis needs
        ("NplKG2O_Try to see if natural-pdf can process non-standard", "NplKG2O.pdf", None),
        ("obe1Vq5_MARKED UP text -- underline and strikethu__for bon", "obe1Vq5.pdf", None),
    ]
    
    analysis_results = []
    
    for folder_name, pdf_filename, target_pages in documents_to_analyze:
        pdf_path = f"/Users/soma/Development/natural-pdf/bad_pdf_analysis/{folder_name}/{pdf_filename}"
        
        if os.path.exists(pdf_path):
            result = analyze_pdf_document(pdf_path, folder_name, target_pages)
            if result:
                analysis_results.append(result)
        else:
            print(f"❌ PDF not found: {pdf_path}")
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis complete! Processed {len(analysis_results)} documents")
    print(f"{'='*80}")
    
    return analysis_results

if __name__ == "__main__":
    main()