#!/usr/bin/env python3
"""
Analyze final 10 PDF documents with enhanced Natural PDF capability awareness
Focus on testing existing capabilities and identifying real gaps
"""

import os
import sys
import json
import time
from datetime import datetime
import natural_pdf as npdf

# Add the project root to the path
sys.path.append('/Users/soma/Development/natural-pdf')

def detailed_pdf_analysis(pdf_path, document_name, target_pages=None):
    """Enhanced analysis leveraging discovered Natural PDF capabilities"""
    print(f"\n{'='*80}")
    print(f"🔍 DETAILED ANALYSIS: {document_name}")
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
            # Analyze first page but also sample a middle page for diversity
            pages_to_analyze = [1]
            if total_pages > 10:
                pages_to_analyze.append(total_pages // 2)
        
        results = {
            'document': document_name,
            'total_pages': total_pages,
            'analyzed_pages': pages_to_analyze,
            'analysis_date': datetime.now().isoformat(),
            'pages': {},
            'capabilities_tested': {},
            'challenges_identified': [],
            'natural_pdf_gaps': []
        }
        
        for page_num in pages_to_analyze:
            print(f"\n📄 DEEP ANALYSIS: Page {page_num}")
            page = pdf.pages[page_num - 1]  # Convert to 0-based index
            
            page_results = {
                'page_number': page_num,
                'dimensions': f"{page.width} × {page.height} points",
                'tests_performed': {}
            }
            
            # === TEXT EXTRACTION ANALYSIS ===
            print("🔤 Text Extraction Analysis...")
            try:
                text_content = page.extract_text()
                page_results['text_length'] = len(text_content)
                page_results['text_preview'] = text_content[:200] + "..." if len(text_content) > 200 else text_content
                
                # Character-level analysis for dense text detection
                chars = page.chars
                char_count = len(chars)
                page_results['character_count'] = char_count
                
                # Detect potential dense text issues (character overlap)
                if char_count > 100:
                    overlap_count = 0
                    for i, char in enumerate(chars[:100]):  # Sample first 100 chars
                        for j, other_char in enumerate(chars[i+1:i+21]):  # Check next 20
                            if abs(char.x0 - other_char.x0) < 2:  # Very close x positions
                                overlap_count += 1
                    
                    overlap_ratio = overlap_count / min(100, char_count)
                    page_results['dense_text_detected'] = overlap_ratio > 0.3
                    page_results['character_overlap_ratio'] = overlap_ratio
                    
                    if overlap_ratio > 0.3:
                        results['challenges_identified'].append({
                            'type': 'dense_text',
                            'page': page_num,
                            'severity': 'high' if overlap_ratio > 0.5 else 'medium',
                            'details': f'Character overlap ratio: {overlap_ratio:.2f}'
                        })
                
                print(f"✅ Text: {len(text_content)} chars, {char_count} character elements")
                if page_results.get('dense_text_detected'):
                    print(f"⚠️ Dense text detected (overlap ratio: {page_results['character_overlap_ratio']:.2f})")
                
            except Exception as e:
                page_results['text_error'] = str(e)
                print(f"❌ Text extraction failed: {e}")
            
            # === ADVANCED TABLE DETECTION ===
            print("📊 Advanced Table Detection...")
            try:
                # Standard table extraction
                table_data = page.extract_table()
                if table_data and len(table_data) > 0:
                    rows = len(table_data)
                    cols = max(len(row) for row in table_data) if table_data else 0
                    page_results['standard_table'] = f"{rows} rows × {cols} columns"
                    print(f"✅ Standard table: {rows} rows × {cols} columns")
                    
                    # Test unruled table detection using discovered line detection capability
                    print("🔍 Testing line detection for unruled tables...")
                    try:
                        # Use projection profiling (no OpenCV required)
                        page.detect_lines(
                            resolution=144,
                            source_label="analysis_test",
                            method="projection",
                            horizontal=True,
                            vertical=True,
                            peak_threshold_h=0.3,  # Lower threshold for subtle lines
                            peak_threshold_v=0.3,
                            replace=True
                        )
                        
                        # Check detected lines
                        detected_lines = [line for line in page._element_mgr.lines 
                                        if getattr(line, 'source', None) == 'analysis_test']
                        
                        h_lines = [l for l in detected_lines if l.is_horizontal]
                        v_lines = [l for l in detected_lines if l.is_vertical]
                        
                        page_results['line_detection'] = {
                            'horizontal_lines': len(h_lines),
                            'vertical_lines': len(v_lines),
                            'total_lines': len(detected_lines)
                        }
                        
                        print(f"✅ Line detection: {len(h_lines)} horizontal, {len(v_lines)} vertical")
                        
                        # Test table structure from lines
                        if len(detected_lines) > 0:
                            page.detect_table_structure_from_lines(
                                source_label="analysis_test",
                                ignore_outer_regions=True,
                                cell_padding=0.5
                            )
                            
                            # Check created table regions
                            table_regions = [r for r in page._element_mgr.regions 
                                           if getattr(r, 'region_type', None) == 'table']
                            cell_regions = [r for r in page._element_mgr.regions 
                                          if getattr(r, 'region_type', None) == 'table_cell']
                            
                            page_results['table_from_lines'] = {
                                'table_regions': len(table_regions),
                                'cell_regions': len(cell_regions)
                            }
                            
                            print(f"✅ Table from lines: {len(table_regions)} tables, {len(cell_regions)} cells")
                            
                            results['capabilities_tested']['line_detection'] = True
                            results['capabilities_tested']['table_from_lines'] = True
                            
                    except Exception as e:
                        page_results['line_detection_error'] = str(e)
                        print(f"❌ Line detection failed: {e}")
                        results['natural_pdf_gaps'].append({
                            'capability': 'line_detection',
                            'error': str(e),
                            'page': page_num
                        })
                
                else:
                    page_results['standard_table'] = "No table detected"
                    print("ℹ️ No standard table detected")
                
            except Exception as e:
                page_results['table_error'] = str(e)
                print(f"❌ Table extraction failed: {e}")
            
            # === LAYOUT ANALYSIS COMPARISON ===
            print("🏗️ Layout Analysis Comparison...")
            try:
                # YOLO analysis
                yolo_start = time.time()
                page.analyze_layout('yolo', existing='replace')
                yolo_time = time.time() - yolo_start
                
                yolo_regions = page.find_all('region')
                page_results['yolo_analysis'] = {
                    'regions': len(yolo_regions),
                    'processing_time': yolo_time
                }
                
                # Categorize YOLO regions
                yolo_types = {}
                for region in yolo_regions:
                    region_type = getattr(region, 'type', 'unknown')
                    yolo_types[region_type] = yolo_types.get(region_type, 0) + 1
                
                page_results['yolo_types'] = yolo_types
                print(f"✅ YOLO: {len(yolo_regions)} regions in {yolo_time:.2f}s - {yolo_types}")
                
                # TATR analysis
                tatr_start = time.time()
                page.analyze_layout('tatr', existing='append')
                tatr_time = time.time() - tatr_start
                
                tatr_regions = page.find_all('region[type="table"]')
                page_results['tatr_analysis'] = {
                    'table_regions': len(tatr_regions),
                    'processing_time': tatr_time
                }
                print(f"✅ TATR: {len(tatr_regions)} table regions in {tatr_time:.2f}s")
                
                results['capabilities_tested']['yolo_analysis'] = True
                results['capabilities_tested']['tatr_analysis'] = True
                
            except Exception as e:
                page_results['layout_error'] = str(e)
                print(f"❌ Layout analysis failed: {e}")
            
            # === ADVANCED SELECTOR TESTING ===
            print("🎯 Advanced Selector Testing...")
            try:
                # Test complex selectors
                selector_tests = {
                    'large_text': 'text[size>12]',
                    'small_text': 'text[size<8]', 
                    'bold_text': 'text:bold',
                    'colored_rects': 'rect[fill]',
                    'thin_lines': 'rect[height<3]',  # Potential underlines
                    'wide_elements': f'*[width>{page.width * 0.7}]',  # Page-spanning elements
                }
                
                for test_name, selector in selector_tests.items():
                    try:
                        elements = page.find_all(selector)
                        page_results[f'selector_{test_name}'] = len(elements)
                        print(f"✅ {test_name}: {len(elements)} elements")
                        
                        # Special analysis for thin lines (potential formatting)
                        if test_name == 'thin_lines' and len(elements) > 0:
                            # Check if these might be text formatting
                            text_elements = page.find_all('text')
                            formatting_candidates = 0
                            
                            for thin_rect in elements[:10]:  # Sample first 10
                                # Check if there's text above this thin rect
                                for text_elem in text_elements[:20]:  # Sample text elements
                                    if (abs(text_elem.bottom - thin_rect.top) < 5 and  # Below text
                                        thin_rect.x0 <= text_elem.x1 and thin_rect.x1 >= text_elem.x0):  # Overlaps horizontally
                                        formatting_candidates += 1
                                        break
                            
                            if formatting_candidates > 0:
                                page_results['potential_text_formatting'] = formatting_candidates
                                print(f"🎯 Potential text formatting: {formatting_candidates} underline candidates")
                                
                                results['challenges_identified'].append({
                                    'type': 'text_formatting',
                                    'page': page_num,
                                    'severity': 'medium',
                                    'details': f'{formatting_candidates} potential underlines detected'
                                })
                    
                    except Exception as e:
                        page_results[f'selector_{test_name}_error'] = str(e)
                        print(f"❌ Selector {test_name} failed: {e}")
                
                results['capabilities_tested']['advanced_selectors'] = True
                
            except Exception as e:
                print(f"❌ Selector testing failed: {e}")
            
            # === SAVE PAGE IMAGE ===
            try:
                folder_name = document_name.replace('/', '_').replace('\\', '_')
                analysis_dir = f"/Users/soma/Development/natural-pdf/bad_pdf_analysis/{folder_name}/detailed_analysis_final"
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
        
        # === GENERATE COMPREHENSIVE INSIGHTS ===
        insights = generate_comprehensive_insights(results)
        results['comprehensive_insights'] = insights
        
        # === SAVE RESULTS ===
        try:
            folder_name = document_name.replace('/', '_').replace('\\', '_')
            analysis_dir = f"/Users/soma/Development/natural-pdf/bad_pdf_analysis/{folder_name}/detailed_analysis_final"
            os.makedirs(analysis_dir, exist_ok=True)
            
            results_path = f"{analysis_dir}/detailed_analysis_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Detailed analysis saved: {results_path}")
            
            # Generate detailed markdown report
            markdown_path = f"{analysis_dir}/{document_name}_detailed_analysis.md"
            generate_detailed_markdown(results, markdown_path)
            print(f"✅ Detailed markdown report saved: {markdown_path}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to analyze {document_name}: {e}")
        return None

def generate_comprehensive_insights(results):
    """Generate comprehensive insights from detailed analysis"""
    insights = {
        'document_complexity': 'low',
        'processing_recommendations': [],
        'natural_pdf_effectiveness': {},
        'priority_issues': []
    }
    
    # Analyze document complexity
    total_chars = sum(page.get('character_count', 0) for page in results['pages'].values())
    max_regions = max(page.get('yolo_analysis', {}).get('regions', 0) for page in results['pages'].values())
    
    if total_chars > 5000 or max_regions > 15:
        insights['document_complexity'] = 'high'
    elif total_chars > 2000 or max_regions > 8:
        insights['document_complexity'] = 'medium'
    
    # Analyze Natural PDF effectiveness
    capabilities_tested = results.get('capabilities_tested', {})
    working_capabilities = [k for k, v in capabilities_tested.items() if v]
    insights['natural_pdf_effectiveness']['working_capabilities'] = working_capabilities
    
    # Priority issues
    for challenge in results.get('challenges_identified', []):
        if challenge['severity'] == 'high':
            insights['priority_issues'].append(challenge)
    
    # Processing recommendations
    if any(page.get('dense_text_detected') for page in results['pages'].values()):
        insights['processing_recommendations'].append('Use pdfplumber parameters for dense text handling')
    
    if any(page.get('line_detection', {}).get('total_lines', 0) > 0 for page in results['pages'].values()):
        insights['processing_recommendations'].append('Leverage existing line detection for table structure')
    
    return insights

def generate_detailed_markdown(results, output_path):
    """Generate detailed markdown report"""
    
    content = f"""# Detailed PDF Analysis Report - {results['document']}

## Executive Summary

**Document:** {results['document']}  
**Complexity:** {results.get('comprehensive_insights', {}).get('document_complexity', 'unknown').upper()}  
**Pages Analyzed:** {len(results['pages'])}  
**Analysis Date:** {results['analysis_date']}

### Key Findings

"""
    
    # Add priority issues
    priority_issues = results.get('comprehensive_insights', {}).get('priority_issues', [])
    if priority_issues:
        content += "#### 🚨 Priority Issues\n\n"
        for issue in priority_issues:
            content += f"- **{issue['type'].title()}** (Page {issue['page']}): {issue['details']}\n"
        content += "\n"
    
    # Add working capabilities
    working_caps = results.get('comprehensive_insights', {}).get('natural_pdf_effectiveness', {}).get('working_capabilities', [])
    if working_caps:
        content += "#### ✅ Natural PDF Capabilities Confirmed\n\n"
        for cap in working_caps:
            content += f"- {cap.replace('_', ' ').title()}\n"
        content += "\n"
    
    content += "---\n\n## Detailed Page Analysis\n\n"
    
    for page_num, page_data in results['pages'].items():
        content += f"### Page {page_num}\n\n"
        content += f"**Dimensions:** {page_data.get('dimensions', 'Unknown')}\n\n"
        
        # Text analysis
        if 'text_length' in page_data:
            content += f"**Text Analysis:**\n"
            content += f"- Content: {page_data['text_length']} characters, {page_data.get('character_count', 0)} elements\n"
            if page_data.get('dense_text_detected'):
                content += f"- ⚠️ Dense text detected (overlap ratio: {page_data.get('character_overlap_ratio', 0):.2f})\n"
            content += "\n"
        
        # Table analysis
        if 'standard_table' in page_data:
            content += f"**Table Analysis:**\n"
            content += f"- Standard extraction: {page_data['standard_table']}\n"
            if 'line_detection' in page_data:
                ld = page_data['line_detection']
                content += f"- Line detection: {ld['horizontal_lines']} horizontal, {ld['vertical_lines']} vertical\n"
            if 'table_from_lines' in page_data:
                tfl = page_data['table_from_lines']
                content += f"- Table from lines: {tfl['table_regions']} tables, {tfl['cell_regions']} cells\n"
            content += "\n"
        
        # Layout analysis
        if 'yolo_analysis' in page_data:
            ya = page_data['yolo_analysis']
            content += f"**Layout Analysis:**\n"
            content += f"- YOLO: {ya['regions']} regions in {ya['processing_time']:.2f}s\n"
            if 'yolo_types' in page_data:
                types_str = ", ".join([f"{k}: {v}" for k, v in page_data['yolo_types'].items()])
                content += f"  - Types: {types_str}\n"
            if 'tatr_analysis' in page_data:
                ta = page_data['tatr_analysis']
                content += f"- TATR: {ta['table_regions']} table regions in {ta['processing_time']:.2f}s\n"
            content += "\n"
        
        # Selector testing
        selector_keys = [k for k in page_data.keys() if k.startswith('selector_')]
        if selector_keys:
            content += f"**Advanced Selector Testing:**\n"
            for key in selector_keys:
                if not key.endswith('_error'):
                    clean_name = key.replace('selector_', '').replace('_', ' ').title()
                    content += f"- {clean_name}: {page_data[key]} elements\n"
            
            if page_data.get('potential_text_formatting'):
                content += f"- 🎯 Text formatting candidates: {page_data['potential_text_formatting']}\n"
            content += "\n"
        
        content += "\n"
    
    # Add comprehensive recommendations
    content += """---

## Natural PDF Integration Recommendations

Based on this detailed analysis:

```python
import natural_pdf as npdf

def process_document_optimally(pdf_path):
    \"\"\"Optimized processing based on analysis findings\"\"\"
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

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Analyze final 10 PDF documents with detailed capability testing"""
    
    # Select diverse documents focusing on different challenge types
    documents_to_analyze = [
        # Text formatting challenges
        ("Y5G72LB_We are trying to get specific information such as ", "Y5G72LB.pdf", None),
        ("Pd1KBb1_the data table _of election results_", "Pd1KBb1.pdf", None),
        ("Pd9WVDb_We want a spreadsheet showing all the columns sepa", "Pd9WVDb.pdf", None),
        
        # Complex table structures
        ("eqQ4N7q_election results data table", "eqQ4N7q.pdf", None),
        ("eqQ4NoQ_data table", "eqQ4NoQ.pdf", None),
        ("ODXl8aR_0. ISO code of the business_ business name_ contac", "ODXl8aR.pdf", None),
        
        # Multi-language and script challenges
        ("1A4PPW1_The arabic text", "1A4PPW1.pdf", None),
        ("lbODDK6_The text in Ethiopian.", "lbODDK6.pdf", None),
        
        # Dense content and specialized formats  
        ("2EAOEvb_The text_ without beeing divided in 2 columns and ", "2EAOEvb.pdf", None),
        ("OD49rjM_Just being able to make sense of any of it. It_s b", "OD49rjM.pdf", None),
    ]
    
    analysis_results = []
    
    print(f"🚀 Starting detailed analysis of {len(documents_to_analyze)} documents...")
    print(f"🔬 Testing discovered Natural PDF capabilities:")
    print(f"   - Line detection (projection profiling)")
    print(f"   - Table structure from lines") 
    print(f"   - Advanced selectors")
    print(f"   - Character-level dense text detection")
    
    for folder_name, pdf_filename, target_pages in documents_to_analyze:
        pdf_path = f"/Users/soma/Development/natural-pdf/bad_pdf_analysis/{folder_name}/{pdf_filename}"
        
        if os.path.exists(pdf_path):
            result = detailed_pdf_analysis(pdf_path, folder_name, target_pages)
            if result:
                analysis_results.append(result)
        else:
            print(f"❌ PDF not found: {pdf_path}")
    
    print(f"\n{'='*80}")
    print(f"✅ DETAILED ANALYSIS COMPLETE!")
    print(f"📊 Processed {len(analysis_results)} documents")
    print(f"🔬 Tested Natural PDF capabilities extensively")
    print(f"{'='*80}")
    
    return analysis_results

if __name__ == "__main__":
    main()