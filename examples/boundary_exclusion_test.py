"""
Test for boundary element exclusion with real PDFs.
This test focuses on the boundary_inclusion parameter of get_sections.
"""

import os
import sys
from natural_pdf import PDF

def main():
    # Get path to PDF file, use default if not provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not os.path.exists(pdf_path):
            print(f"Error: File {pdf_path} not found")
            sys.exit(1)
    else:
        # Just use a default PDF from the pdfs directory
        pdfs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pdfs")
        pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith('.pdf')]
        if not pdf_files:
            print("No PDF files found in the pdfs directory")
            sys.exit(1)
        
        pdf_path = os.path.join(pdfs_dir, pdf_files[0])
    
    print(f"Using PDF: {pdf_path}")
    
    # Open the PDF
    pdf = PDF(pdf_path)
    
    # Use the first page for testing
    page = pdf.pages[0]
    
    # Find elements to use as section boundaries
    # First try to find large text as headings
    headings = page.find_all('text[size>=14]')
    
    # If not enough headings, try smaller text
    if len(headings) < 3:
        headings = page.find_all('text[size>=12]')
    
    # If still not enough, try bold text
    if len(headings) < 3:
        headings = page.find_all('text:bold')
    
    # If still not enough, use the first 3 text elements
    if len(headings) < 3:
        headings = page.find_all('text')[:5]
    
    print(f"Found {len(headings)} potential section boundaries")
    for i, h in enumerate(headings[:5]):
        print(f"Boundary {i+1}: {h.text}")
    
    # Create different sections with different boundary_inclusion settings
    none_sections = page.get_sections(start_elements=headings, boundary_inclusion='none')
    start_sections = page.get_sections(start_elements=headings, boundary_inclusion='start')
    both_sections = page.get_sections(start_elements=headings, boundary_inclusion='both')
    
    print("\nTesting boundary element inclusion/exclusion:")
    
    # Check if the boundary elements are included correctly
    for i, section in enumerate(none_sections[:3]):
        if i >= len(headings):
            break
            
        boundary = headings[i]
        found = section._is_element_in_region(boundary)
        print(f"None Section {i+1}: Contains boundary element: {found}")
    
    for i, section in enumerate(start_sections[:3]):
        if i >= len(headings):
            break
            
        boundary = headings[i]
        found = section._is_element_in_region(boundary)
        print(f"Start Section {i+1}: Contains boundary element: {found}")
    
    for i, section in enumerate(both_sections[:3]):
        if i >= len(headings):
            break
            
        boundary = headings[i]
        found = section._is_element_in_region(boundary)
        print(f"Both Section {i+1}: Contains boundary element: {found}")
    
    # Simplify our test approach - just check if:
    # 1. 'none' sections exclude their boundary elements
    # 2. 'start' sections include their boundary elements  
    # 3. 'both' sections include their boundary elements
    
    # Check section element counts
    print("\nElement counts in sections:")
    for i, section in enumerate(none_sections[:3]):
        elements = section.get_elements()
        print(f"None Section {i+1}: {len(elements)} elements")
        
    for i, section in enumerate(start_sections[:3]):
        elements = section.get_elements()
        print(f"Start Section {i+1}: {len(elements)} elements")
        
    for i, section in enumerate(both_sections[:3]):
        elements = section.get_elements() 
        print(f"Both Section {i+1}: {len(elements)} elements")
    
    # Summarize test results
    none_correct = all(
        not section._is_element_in_region(headings[i]) 
        for i, section in enumerate(none_sections[:3]) 
        if i < len(headings)
    )
    
    # Check only non-empty sections that have a start_element
    start_correct = all(
        (section.start_element is None) or section._is_element_in_region(section.start_element)
        for section in start_sections[:3]
        if section.get_elements()  # Skip empty sections
    )
    
    both_correct = all(
        (section.start_element is None) or section._is_element_in_region(section.start_element)
        for section in both_sections[:3]
        if section.get_elements()  # Skip empty sections
    )
    
    print("\nTest Results Summary:")
    print(f"- 'none' excludes boundary elements: {'PASS' if none_correct else 'FAIL'}")
    print(f"- 'start' includes boundary elements: {'PASS' if start_correct else 'FAIL'}")
    print(f"- 'both' includes boundary elements: {'PASS' if both_correct else 'FAIL'}")
    
    if none_correct and start_correct and both_correct:
        print("\n✅ All tests PASSED!")
    else:
        print("\n❌ Some tests FAILED!")

if __name__ == "__main__":
    main()