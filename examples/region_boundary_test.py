"""
Test the modified region boundary logic with below() and above() method fixes.

This example tests that the .below() and .above() methods correctly exclude
the source element with the new 1-pixel offset.
"""

import os
import sys
import argparse

# Add parent directory to path to run without installing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from natural_pdf import PDF

def main():
    parser = argparse.ArgumentParser(description="Test region boundaries")
    parser.add_argument("pdf_path", nargs="?", default="../pdfs/0500000US42001.pdf", 
                      help="Path to PDF document")
    args = parser.parse_args()
    
    print(f"Testing with PDF: {args.pdf_path}")
    
    # Open the PDF
    pdf = PDF(args.pdf_path)
    page = pdf.pages[0]
    
    # Find a text element to test with
    title = page.find('text:contains("Price")')
    if not title:
        title = page.find('text:bold')
        
    if not title:
        print("Couldn't find a suitable test element. Please provide a PDF with text elements.")
        return
    
    print(f"Found element: '{title.text}' at position {title.bbox}")
    
    # Create region below the element
    region_below = title.below(height=16, width="element")
    
    # Check if the element is in the region (it shouldn't be)
    elements_in_region = region_below.find_all('text')
    
    # Print the region and elements found in it
    print(f"\nRegion below: {region_below.bbox}")
    print(f"Number of elements in region: {len(elements_in_region)}")
    
    # Check specifically if the source element is in the region
    is_source_in_region = title in elements_in_region
    print(f"Source element is in region: {is_source_in_region}")
    
    # Expand the region and check again
    expanded_region = region_below.expand(right=40)
    elements_in_expanded = expanded_region.find_all('text')
    
    print(f"\nExpanded region: {expanded_region.bbox}")
    print(f"Number of elements in expanded region: {len(elements_in_expanded)}")
    print(f"Elements text: {[e.text for e in elements_in_expanded]}")
    
    # Highlight the regions to visualize
    title.highlight(color=(1, 0, 0, 0.3), label="Source")
    region_below.highlight(color=(0, 1, 0, 0.3), label="Below")
    expanded_region.highlight(color=(0, 0, 1, 0.3), label="Expanded")
    
    # Save the image
    os.makedirs("output", exist_ok=True)
    page.save_image("output/region_boundary_test.png")
    print("\nSaved visualization to output/region_boundary_test.png")

if __name__ == "__main__":
    main()