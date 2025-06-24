#!/usr/bin/env python3
"""
Direct Natural PDF analysis targeting specific pages.
"""

import json
import os
import sys
from pathlib import Path
import natural_pdf as npdf
import re
from datetime import datetime

def analyze_specific_pages_direct(pdf_path, target_pages, output_folder):
    """Directly analyze specific pages using Natural PDF"""
    
    print(f"🔍 Analyzing {pdf_path}")
    print(f"📍 Target pages: {target_pages}")
    
    pdf = npdf.PDF(pdf_path)
    results = {}
    
    for page_num in target_pages:
        if page_num > len(pdf.pages):
            print(f"❌ Page {page_num} not found - document only has {len(pdf.pages)} pages")
            continue
            
        print(f"\n📄 Analyzing page {page_num}...")
        page = pdf.pages[page_num - 1]  # Convert to 0-based index
        
        page_data = {
            "page_number": page_num,
            "dimensions": {
                "width": page.width,
                "height": page.height
            }
        }
        
        # Get page description
        try:
            description = page.describe()
            page_data["describe"] = description
            print(f"✅ Page description: {len(description)} characters")
        except Exception as e:
            print(f"❌ Page description failed: {e}")
            page_data["describe"] = f"ERROR: {e}"
        
        # Extract text
        try:
            text = page.extract_text()
            page_data["extract_text"] = {
                "length": len(text),
                "preview": text[:200] + "..." if len(text) > 200 else text,
                "full_text": text
            }
            print(f"✅ Text extraction: {len(text)} characters")
        except Exception as e:
            print(f"❌ Text extraction failed: {e}")
            page_data["extract_text"] = f"ERROR: {e}"
        
        # Try table extraction
        try:
            table_data = page.extract_table()
            if table_data and len(table_data) > 0:
                page_data["extract_table"] = {
                    "found": True,
                    "rows": len(table_data),
                    "columns": len(table_data[0]) if table_data else 0,
                    "data": table_data[:5]  # First 5 rows only
                }
                print(f"✅ Table found: {len(table_data)} rows × {len(table_data[0]) if table_data else 0} columns")
            else:
                page_data["extract_table"] = {"found": False}
                print("ℹ️ No table found with standard extraction")
        except Exception as e:
            print(f"❌ Table extraction failed: {e}")
            page_data["extract_table"] = f"ERROR: {e}"
        
        # Try layout analysis
        try:
            page.analyze_layout('yolo', existing='replace')
            layout_regions = page.find_all('region')
            if layout_regions and len(layout_regions) > 0:
                page_data["analyze_layout"] = {
                    "found": True,
                    "count": len(layout_regions),
                    "regions": []
                }
                for region in layout_regions[:10]:  # First 10 regions
                    try:
                        page_data["analyze_layout"]["regions"].append({
                            "type": region.type if hasattr(region, 'type') else 'unknown',
                            "bbox": [region.x0, region.y0, region.x1, region.y1],
                            "confidence": region.confidence if hasattr(region, 'confidence') else 1.0
                        })
                    except:
                        pass
                print(f"✅ Layout analysis: {len(layout_regions)} regions")
            else:
                page_data["analyze_layout"] = {"found": False}
                print("ℹ️ No layout regions found")
        except Exception as e:
            print(f"❌ Layout analysis failed: {e}")
            page_data["analyze_layout"] = f"ERROR: {e}"
        
        # Try TATR analysis
        try:
            page.analyze_layout('tatr', existing='append')
            tatr_regions = page.find_all('region')
            tatr_count = len([r for r in tatr_regions if hasattr(r, 'type') and 'table' in str(r.type).lower()])
            if tatr_count > 0:
                page_data["analyze_layout_tatr"] = {
                    "found": True,
                    "count": tatr_count,
                    "regions": []
                }
                for region in tatr_regions[:25]:  # First 25 regions
                    try:
                        if hasattr(region, 'type') and 'table' in str(region.type).lower():
                            page_data["analyze_layout_tatr"]["regions"].append({
                                "type": str(region.type),
                                "bbox": [region.x0, region.y0, region.x1, region.y1],
                                "confidence": region.confidence if hasattr(region, 'confidence') else 1.0
                            })
                    except:
                        pass
                print(f"✅ TATR analysis: {tatr_count} table regions")
            else:
                page_data["analyze_layout_tatr"] = {"found": False}
                print("ℹ️ No TATR table regions found")
        except Exception as e:
            print(f"❌ TATR analysis failed: {e}")
            page_data["analyze_layout_tatr"] = f"ERROR: {e}"
        
        # Save page image
        try:
            page_image_path = os.path.join(output_folder, f"page_{page_num}.png")
            page.save_image(page_image_path, resolution=144)
            page_data["image_path"] = page_image_path
            print(f"✅ Page image saved: {page_image_path}")
        except Exception as e:
            print(f"❌ Page image save failed: {e}")
            page_data["image_path"] = f"ERROR: {e}"
        
        results[page_num] = page_data
    
    return results

def create_enhanced_analysis_report(pdf_path, target_pages, analysis_results, output_folder):
    """Create enhanced analysis report"""
    
    pdf_name = Path(pdf_path).name
    
    # Determine what the user was looking for
    user_goal = f"Analysis of pages {target_pages}"
    if len(target_pages) == 1:
        user_goal = f"Analysis of page {target_pages[0]}"
    
    report = f"""# Enhanced PDF Analysis Report - {pdf_name.replace('.pdf', '')}

## Analysis Overview

**PDF File:** {pdf_name}  
**Target Pages:** {target_pages}  
**Pages Successfully Analyzed:** {list(analysis_results.keys())}  
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Page-by-Page Analysis Results

"""

    for page_num in sorted(analysis_results.keys()):
        page_data = analysis_results[page_num]
        
        report += f"""### Page {page_num}

**Dimensions:** {page_data.get('dimensions', {}).get('width', 'Unknown')} × {page_data.get('dimensions', {}).get('height', 'Unknown')} points

**Content Analysis:**
"""
        
        # Text analysis
        if isinstance(page_data.get('extract_text'), dict):
            text_info = page_data['extract_text']
            report += f"- **Text Content:** {text_info.get('length', 0)} characters extracted\n"
            if text_info.get('preview'):
                report += f"- **Content Preview:** {text_info['preview']}\n"
        
        # Table analysis
        if isinstance(page_data.get('extract_table'), dict):
            table_info = page_data['extract_table']
            if table_info.get('found'):
                report += f"- **Table Found:** {table_info.get('rows', 0)} rows × {table_info.get('columns', 0)} columns\n"
            else:
                report += "- **Table Status:** No standard table structure detected\n"
        
        # Layout analysis
        if isinstance(page_data.get('analyze_layout'), dict):
            layout_info = page_data['analyze_layout']
            if layout_info.get('found'):
                report += f"- **Layout Regions:** {layout_info.get('count', 0)} regions detected\n"
                
                # Show region types
                region_types = {}
                for region in layout_info.get('regions', []):
                    region_type = region.get('type', 'unknown')
                    region_types[region_type] = region_types.get(region_type, 0) + 1
                
                if region_types:
                    report += f"- **Region Types:** {dict(region_types)}\n"
        
        # TATR analysis
        if isinstance(page_data.get('analyze_layout_tatr'), dict):
            tatr_info = page_data['analyze_layout_tatr']
            if tatr_info.get('found'):
                report += f"- **TATR Table Analysis:** {tatr_info.get('count', 0)} table regions detected\n"
        
        # Image
        if page_data.get('image_path') and not page_data['image_path'].startswith('ERROR'):
            report += f"- **Visual:** Page image saved as `page_{page_num}.png`\n"
        
        report += "\n"
    
    # Analysis summary
    report += """---

## Analysis Summary

### What We Found
"""
    
    # Summarize findings across all pages
    total_text_chars = 0
    pages_with_tables = 0
    total_layout_regions = 0
    total_tatr_regions = 0
    
    for page_data in analysis_results.values():
        if isinstance(page_data.get('extract_text'), dict):
            total_text_chars += page_data['extract_text'].get('length', 0)
        
        if isinstance(page_data.get('extract_table'), dict) and page_data['extract_table'].get('found'):
            pages_with_tables += 1
        
        if isinstance(page_data.get('analyze_layout'), dict) and page_data['analyze_layout'].get('found'):
            total_layout_regions += page_data['analyze_layout'].get('count', 0)
        
        if isinstance(page_data.get('analyze_layout_tatr'), dict) and page_data['analyze_layout_tatr'].get('found'):
            total_tatr_regions += page_data['analyze_layout_tatr'].get('count', 0)
    
    report += f"""
- **Total Text Content:** {total_text_chars:,} characters across {len(analysis_results)} pages
- **Table Detection:** {pages_with_tables} out of {len(analysis_results)} pages have detectable tables
- **Layout Analysis:** {total_layout_regions} total layout regions detected
- **TATR Analysis:** {total_tatr_regions} table-specific regions detected
"""

    # Add recommendations
    report += """
### Natural PDF Extraction Approach

Based on the actual content found on these pages:

```python
import natural_pdf as npdf

def extract_from_specific_pages(pdf_path, target_pages):
    \"\"\"Extract data from specific pages with targeted approach\"\"\"
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
"""
    
    if len(target_pages) == 1:
        report += f"results = extract_from_specific_pages('{pdf_name}', [{target_pages[0]}])\n"
    else:
        report += f"results = extract_from_specific_pages('{pdf_name}', {target_pages})\n"
    
    report += "```\n"
    
    # Save the report
    report_path = os.path.join(output_folder, f"{pdf_name.replace('.pdf', '')}_enhanced_analysis.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Enhanced analysis report saved: {report_path}")
    return report_path

def main():
    """Re-analyze specific documents with page targeting"""
    
    # Documents that need re-analysis with specific pages
    documents_to_reanalyze = [
        {
            'folder': 'ODX1DW8_The large table on page 179',
            'file': 'ODX1DW8.pdf',
            'pages': [178, 179, 180],  # Page 179 ± 1 for safety
            'reason': 'User requested page 179, original analysis used page 1'
        },
        {
            'folder': 'eqrZ5yq_The long table _Annex 6_ spanning across pages fro',
            'file': 'eqrZ5yq.pdf',
            'pages': [89, 90, 91, 92],  # Multi-page table range
            'reason': 'User requested pages 89-92, original analysis used page 1'
        }
    ]
    
    base_path = "/Users/soma/Development/natural-pdf/bad_pdf_analysis"
    
    for doc in documents_to_reanalyze:
        print(f"\n{'='*80}")
        print(f"🔄 Re-analyzing {doc['file']}")
        print(f"📋 Reason: {doc['reason']}")
        print(f"{'='*80}")
        
        folder_path = os.path.join(base_path, doc['folder'])
        pdf_path = os.path.join(folder_path, doc['file'])
        output_folder = os.path.join(folder_path, 'enhanced_analysis')
        
        if not os.path.exists(pdf_path):
            print(f"❌ PDF not found: {pdf_path}")
            continue
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Run direct analysis on specific pages
        try:
            analysis_results = analyze_specific_pages_direct(pdf_path, doc['pages'], output_folder)
            
            if analysis_results:
                # Save analysis results as JSON
                results_file = os.path.join(output_folder, "enhanced_analysis_results.json")
                with open(results_file, 'w') as f:
                    json.dump({
                        "pdf_path": pdf_path,
                        "target_pages": doc['pages'],
                        "analysis_timestamp": datetime.now().isoformat(),
                        "results": analysis_results
                    }, f, indent=2)
                
                # Create enhanced report
                create_enhanced_analysis_report(pdf_path, doc['pages'], analysis_results, output_folder)
                
                print(f"\n✅ Successfully analyzed {len(analysis_results)} pages from {doc['file']}")
            else:
                print(f"❌ No results obtained for {doc['file']}")
                
        except Exception as e:
            print(f"❌ Analysis failed for {doc['file']}: {e}")

if __name__ == "__main__":
    main()