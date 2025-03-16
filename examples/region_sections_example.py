"""
Example demonstrating the get_sections() method on regions in Natural PDF.

This example shows how to extract logical sections from regions 
using various types of boundary elements.
"""
import os
import sys

# Add the parent directory to the path so we can import natural_pdf module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def main():
    # If a PDF path is provided, use it; otherwise use the default example
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Use a default PDF path - you'll need to replace this with an actual PDF path
        pdf_path = "examples/sample.pdf"
        if not os.path.exists(pdf_path):
            print(f"Default PDF not found at {pdf_path}")
            print("Please provide a PDF path as an argument")
            return
    
    print(f"Processing PDF: {pdf_path}")
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    
    # Example 1: Get sections within a region using separators
    print("\n1. Get sections within a region using separators")
    
    # First, create a region from the top half of the page
    top_half = page.create_region(0, 0, page.width, page.height / 2)
    print(f"Created region: {top_half.bbox}")
    
    # Method 1: Find elements first, then pass them to get_sections
    lines = top_half.find_all('line')
    print(f"Found {len(lines)} line elements in the region")
    
    # Extract sections using lines as start elements
    sections1 = top_half.get_sections(start_elements=lines)
    print(f"Found {len(sections1)} sections using explicit elements")
    
    # Method 2: Pass selector directly to start_elements
    sections2 = top_half.get_sections(start_elements='line')
    print(f"Found {len(sections2)} sections using selector string")
    
    # Display section details
    for i, section in enumerate(sections2):
        text = section.extract_text()
        text_snippet = text[:50] + "..." if len(text) > 50 else text
        print(f"  Section {i+1}: {section.bbox}, Text: {text_snippet}")
    
    # Example 2: Get sections within a region using start/end elements
    print("\n2. Get sections within a region using start/end elements")
    
    # Create a region from the bottom half of the page
    bottom_half = page.create_region(0, page.height / 2, page.width, page.height)
    print(f"Created region: {bottom_half.bbox}")
    
    # Method 1: Find heading elements first, then pass them to get_sections (old way)
    headings = bottom_half.find_all('text[size>=12]')
    print(f"Found {len(headings)} potential headings in the region")
    
    # Use headings as start elements and extract sections (old way)
    sections1 = bottom_half.get_sections(start_elements=headings)
    print(f"Found {len(sections1)} sections using explicit elements")
    
    # Method 2: Pass selector directly to start_elements (new way)
    sections2 = bottom_half.get_sections(start_elements='text[size>=12]')
    print(f"Found {len(sections2)} sections using selector string")
    
    # Display section details
    for i, section in enumerate(sections2):
        start_element = section.start_element
        start_text = start_element.text if start_element else "None"
        
        text = section.extract_text()
        text_snippet = text[:50] + "..." if len(text) > 50 else text
        
        print(f"  Section {i+1} (starts with '{start_text}'): {text_snippet}")
    
    # Example 3: Use selectors within a region
    print("\n3. Get sections using selectors within a region")
    
    # Create a region from the center of the page
    center = page.create_region(50, 50, page.width - 50, page.height - 50)
    
    # Get sections with start elements
    sections1 = center.get_sections(
        start_elements='text[size>=12]'  # Large text as section starts
    )
    
    # Get sections with both start and end elements
    sections2 = center.get_sections(
        start_elements='text[size>=12]',  # Large text as section starts
        end_elements='line[width>=1]'  # Thick lines as section ends
    )
    
    print(f"Found {len(sections1)} sections using traditional selectors")
    print(f"Found {len(sections2)} sections using direct selector strings")
    
    # Compare the results - they should be identical
    print(f"Both approaches match: {len(sections1) == len(sections2)}")
    
    # Display section details for the new approach
    for i, section in enumerate(sections2):
        text = section.extract_text()
        text_snippet = text[:50] + "..." if len(text) > 50 else text
        print(f"  Section {i+1}: {text_snippet}")

if __name__ == "__main__":
    main()