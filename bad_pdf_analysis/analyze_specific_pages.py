#!/usr/bin/env python3
"""
Enhanced PDF analysis script that can target specific pages.
"""

import json
import os
import sys
from pathlib import Path
import subprocess
import re

def parse_page_request(user_goal):
    """Parse user requests for specific pages or page ranges"""
    page_patterns = [
        r'page (\d+)',
        r'pages (\d+) to (\d+)',
        r'pages (\d+)-(\d+)',
        r'from page (\d+) to (\d+)',
        r'spanning.*pages.*from page (\d+) to (\d+)',
    ]
    
    user_goal_lower = user_goal.lower()
    
    for pattern in page_patterns:
        match = re.search(pattern, user_goal_lower)
        if match:
            groups = match.groups()
            if len(groups) == 1:
                # Single page
                return [int(groups[0])]
            elif len(groups) == 2:
                # Page range
                start, end = int(groups[0]), int(groups[1])
                return list(range(start, end + 1))
    
    return None  # No specific pages found

def run_pdf_analyzer_on_pages(pdf_path, pages_to_analyze, output_folder):
    """Run PDF analyzer on specific pages"""
    results = {}
    
    for page_num in pages_to_analyze:
        print(f"Analyzing page {page_num}...")
        
        # Create page-specific output folder
        page_output = os.path.join(output_folder, f"page_{page_num}")
        os.makedirs(page_output, exist_ok=True)
        
        # Run analyzer for specific page
        cmd = [
            "python", "-m", "natural_pdf.cli.pdf_analyzer",
            pdf_path,
            "1",  # Analyze 1 page starting from page_num
            page_output,
            "--no-timestamp",
            f"--start-page={page_num}"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/soma/Development/natural-pdf")
            if result.returncode == 0:
                print(f"  ✅ Page {page_num} analysis completed")
                
                # Read the analysis results
                analysis_file = os.path.join(page_output, "analysis_summary.json")
                if os.path.exists(analysis_file):
                    with open(analysis_file, 'r') as f:
                        page_analysis = json.load(f)
                        results[page_num] = page_analysis
                else:
                    print(f"  ⚠️ No analysis file found for page {page_num}")
            else:
                print(f"  ❌ Page {page_num} analysis failed: {result.stderr}")
                
        except Exception as e:
            print(f"  ❌ Error analyzing page {page_num}: {e}")
    
    return results

def create_enhanced_analysis_report(submission_data, page_results, pdf_filename, folder_path):
    """Create analysis report using results from specific pages"""
    
    # Extract basic submission info
    user_goal = submission_data.get('goal', 'Unknown goal')
    pdf_description = submission_data.get('description', 'No description provided')
    reported_issues = submission_data.get('issues', 'No issues reported')
    
    # Parse requested pages
    requested_pages = parse_page_request(user_goal)
    pages_analyzed = list(page_results.keys()) if page_results else []
    
    # Get document properties from first successful page analysis
    doc_properties = {}
    sample_page_data = {}
    if page_results:
        first_page_result = next(iter(page_results.values()))
        if first_page_result.get('pages'):
            sample_page_data = first_page_result['pages'][0]
            doc_properties = {
                'dimensions': sample_page_data.get('dimensions', {}),
                'total_pages': first_page_result.get('total_pages', 'Unknown')
            }
    
    # Create the analysis report
    report_content = f"""# PDF Analysis Report - {pdf_filename.replace('.pdf', '')}

## Submission Details

**PDF File:** {pdf_filename}  
**Language:** {submission_data.get('language', 'Unknown')}  
**Contains Handwriting:** {submission_data.get('handwriting', 'Unknown')}  
**Requires OCR:** {submission_data.get('ocr_required', 'Unknown')}

### User's Goal
{user_goal}

### PDF Description  
{pdf_description}

### Reported Issues
{reported_issues}

---

## Technical Analysis

### PDF Properties
**Document Size:** {doc_properties.get('total_pages', 'Unknown')} pages  
**Page Dimensions:** {doc_properties.get('dimensions', {}).get('width', 'Unknown')} × {doc_properties.get('dimensions', {}).get('height', 'Unknown')} points  
**Pages Requested:** {requested_pages if requested_pages else 'Not specified'}  
**Pages Analyzed:** {pages_analyzed}

### Analysis Results by Page
"""

    # Add results for each analyzed page
    for page_num, page_data in page_results.items():
        if page_data.get('pages'):
            page_info = page_data['pages'][0]
            
            report_content += f"""
#### Page {page_num} Analysis

**Elements Found:**
- **Text elements:** {page_info.get('describe', '').count('text')} 
- **Table regions:** {page_info.get('analyze_layout', {}).get('count', 0)} layout regions detected
- **Extract table:** {'✅ Success' if page_info.get('extract_table', {}).get('found') else '❌ No tables found'}

**Content Preview:**
```
{page_info.get('extract_text', {}).get('preview', 'No text preview available')[:200]}...
```

**Visual Analysis:** Page image saved as `page_{page_num}.png`
"""

    # Add difficulty assessment based on actual page content
    report_content += f"""
---

## Difficulty Assessment

### Extraction Type
**Primary Goal:** {determine_extraction_type(user_goal)}

### Real Challenges Identified
"""

    # Analyze challenges based on actual page content
    challenges = analyze_page_challenges(page_results, requested_pages, pages_analyzed)
    for challenge in challenges:
        report_content += f"\n{challenge}\n"

    # Add recommendations based on actual content
    report_content += """
### What Natural PDF Can Do

**✅ Recommended Approaches:**

Based on the actual page content analyzed, here are specific Natural PDF approaches:

"""
    
    recommendations = generate_specific_recommendations(page_results, user_goal)
    report_content += recommendations

    # Add footer
    report_content += f"""
---

## Feedback Section

*Analysis based on actual page content from requested pages*

### Assessment Accuracy
- [x] Analysis examined user-requested pages
- [ ] Difficulty assessment needs revision

### Proposed Methods
- [ ] Recommended approaches look good
- [ ] Alternative approaches needed
- [ ] Methods need refinement

---

**Analysis Generated:** Enhanced analysis targeting user-specified pages
**Pages Analyzed:** {pages_analyzed}
**Analysis Date:** {page_results[pages_analyzed[0]]['analysis_timestamp'] if pages_analyzed and page_results else 'Unknown'}
"""

    # Write the report
    report_path = os.path.join(folder_path, f"{pdf_filename.replace('.pdf', '')}_analysis.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ Enhanced analysis report created: {report_path}")
    return report_path

def determine_extraction_type(user_goal):
    """Determine extraction type from user goal"""
    goal_lower = user_goal.lower()
    if 'table' in goal_lower:
        return 'Table Extraction'
    elif 'text' in goal_lower:
        return 'Text Extraction'
    elif 'form' in goal_lower:
        return 'Form Data Extraction'
    else:
        return 'Data Extraction'

def analyze_page_challenges(page_results, requested_pages, pages_analyzed):
    """Analyze real challenges based on page content"""
    challenges = []
    
    # Check if we got the right pages
    if requested_pages and set(requested_pages) != set(pages_analyzed):
        missing_pages = set(requested_pages) - set(pages_analyzed)
        challenges.append(f"""
#### **Page Access Issues**
**Missing pages:** {missing_pages} - Could not analyze all requested pages
**Analyzed instead:** {pages_analyzed}
**Impact:** Analysis may be incomplete without examining all target pages
""")
    
    # Analyze content complexity from actual results
    for page_num, page_data in page_results.items():
        if page_data.get('pages'):
            page_info = page_data['pages'][0]
            
            # Check for table extraction issues
            if not page_info.get('extract_table', {}).get('found'):
                challenges.append(f"""
#### **Table Detection Issues (Page {page_num})**
**Problem:** No tables detected on page {page_num}
**Possible causes:** Complex layout, unruled tables, or non-standard table structure
**Content type:** Based on text preview, this appears to be {analyze_content_type(page_info)}
""")
            
            # Check for text complexity
            text_length = page_info.get('extract_text', {}).get('length', 0)
            if text_length > 5000:
                challenges.append(f"""
#### **Dense Content (Page {page_num})**
**Issue:** Large amount of text ({text_length} characters) may indicate complex layout
**Challenge:** Dense content can complicate spatial navigation and element detection
""")
    
    return challenges

def analyze_content_type(page_info):
    """Analyze what type of content is on the page"""
    text_preview = page_info.get('extract_text', {}).get('preview', '').lower()
    
    if 'table' in text_preview or 'column' in text_preview:
        return 'tabular data'
    elif any(word in text_preview for word in ['report', 'study', 'analysis']):
        return 'report content'
    elif any(word in text_preview for word in ['form', 'application', 'field']):
        return 'form data'
    else:
        return 'mixed content'

def generate_specific_recommendations(page_results, user_goal):
    """Generate specific recommendations based on actual page analysis"""
    recommendations = """
```python
import natural_pdf as npdf

def extract_from_target_pages(pdf_path, target_pages):
    \"\"\"Extract data from user-specified pages\"\"\"
    pdf = npdf.PDF(pdf_path)
    results = []
    
    for page_num in target_pages:
        if page_num <= len(pdf.pages):
            page = pdf.pages[page_num - 1]  # Convert to 0-based index
            
            # Analyze layout for better structure detection
            page.analyze_layout('tatr', existing='append')
            
            # Try multiple extraction approaches
            table_data = page.extract_table()
            if table_data:
                results.append({'page': page_num, 'type': 'table', 'data': table_data})
            else:
                # Fall back to text extraction with spatial awareness
                text_elements = page.find_all('text')
                results.append({'page': page_num, 'type': 'text', 'elements': text_elements})
    
    return results

# Usage for your specific case
"""
    
    # Add specific usage based on the document
    if 'page' in user_goal.lower():
        page_match = re.search(r'page (\d+)', user_goal.lower())
        if page_match:
            page_num = page_match.group(1)
            recommendations += f"""
# Target the specific page mentioned
results = extract_from_target_pages('document.pdf', [{page_num}])
```
"""
    elif 'pages' in user_goal.lower():
        pages_match = re.search(r'pages (\d+) to (\d+)', user_goal.lower())
        if pages_match:
            start, end = pages_match.groups()
            recommendations += f"""
# Target the page range mentioned  
results = extract_from_target_pages('document.pdf', list(range({start}, {end} + 1)))
```
"""
    
    return recommendations

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
        print(f"\n🔄 Re-analyzing {doc['file']} - {doc['reason']}")
        
        folder_path = os.path.join(base_path, doc['folder'])
        pdf_path = os.path.join(folder_path, doc['file'])
        output_folder = os.path.join(folder_path, 'analysis', 'specific_pages')
        
        if not os.path.exists(pdf_path):
            print(f"❌ PDF not found: {pdf_path}")
            continue
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Run analysis on specific pages
        page_results = run_pdf_analyzer_on_pages(pdf_path, doc['pages'], output_folder)
        
        if page_results:
            # Create enhanced analysis report
            submission_data = {
                'goal': f"Analysis targeting pages {doc['pages']}",
                'description': f"Re-analysis of {doc['file']} focusing on user-requested pages",
                'issues': doc['reason']
            }
            
            create_enhanced_analysis_report(
                submission_data, 
                page_results, 
                doc['file'], 
                folder_path
            )
        else:
            print(f"❌ No results obtained for {doc['file']}")

if __name__ == "__main__":
    main()