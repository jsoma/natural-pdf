"""
Test for handling regions that overlap with a footer exclusion zone.
This is a focused test for the specific issue where regions that overlap with a footer
weren't returning any text.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)

from natural_pdf import PDF

def main():
    """Main entry point."""
    # Get the PDF path from command line or use a default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Look for any PDF in the pdfs directory
        pdfs_dir = Path(__file__).parent.parent / "pdfs"
        pdf_files = list(pdfs_dir.glob("*.pdf"))
        
        if pdf_files:
            pdf_path = str(pdf_files[0])
        else:
            print("No PDF file found. Please provide a path to a PDF file.")
            sys.exit(1)
    
    print(f"\nTesting with PDF: {pdf_path}")
    
    # Create a PDF object
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    
    # Create ONLY a footer exclusion zone
    footer_height = page.height * 0.1  # Bottom 10% of the page
    footer = page.create_region(0, page.height - footer_height, page.width, page.height)
    footer.highlight(label="Footer Exclusion", color=(1, 0, 0, 0.3))
    page.add_exclusion(footer)
    print(f"Added footer exclusion: {footer.bbox}")
    
    # Create a region that extends from middle of page to past the footer
    middle_to_footer = page.create_region(
        page.width * 0.25,        # 25% from left
        page.height * 0.5,        # 50% from top (middle of page)
        page.width * 0.75,        # 75% from left
        page.height               # All the way to bottom (overlaps footer)
    )
    middle_to_footer.highlight(label="Middle to Footer", color=(0, 1, 0, 0.3))
    print(f"Created test region: {middle_to_footer.bbox}")
    
    # Try different extraction approaches:
    
    # 1. Extract with exclusions using the default approach
    print("\n=== 1. Using Default Extraction ===")
    text = middle_to_footer.extract_text(apply_exclusions=True, debug=True)
    print(f"Text length: {len(text)}")
    print(f"First 100 chars: {text[:100] if text else 'No text!'}")
    
    # 2. Try direct cropping approach
    print("\n=== 2. Using Direct Crop Approach ===")
    # Manually adjust the region to exclude the footer
    top_bound = middle_to_footer.top
    bottom_bound = page.height - footer_height  # Top of footer
    
    cropped_region = page.create_region(
        middle_to_footer.x0,
        top_bound,
        middle_to_footer.x1,
        bottom_bound
    )
    cropped_region.highlight(label="Cropped Region", color=(0, 0, 1, 0.3))
    
    # Extract without applying exclusions (since we manually cropped)
    cropped_text = cropped_region.extract_text(apply_exclusions=False)
    print(f"Text length: {len(cropped_text)}")
    print(f"First 100 chars: {cropped_text[:100] if cropped_text else 'No text!'}")
    
    # 3. Get individual elements and extract text from them
    print("\n=== 3. Using Element Filtering Approach ===")
    all_elements = page.get_elements()
    
    # Filter elements that are in our region but NOT in footer
    filtered_elements = []
    for element in all_elements:
        # Check if element is in the region
        if (middle_to_footer.x0 <= (element.x0 + element.x1)/2 <= middle_to_footer.x1 and
            middle_to_footer.top <= (element.top + element.bottom)/2 <= middle_to_footer.bottom and
            not (footer.top <= (element.top + element.bottom)/2 <= footer.bottom)):
            filtered_elements.append(element)
    
    # Extract text from the filtered elements
    filtered_text = " ".join(e.text for e in filtered_elements if hasattr(e, 'text'))
    print(f"Text length: {len(filtered_text)}")
    print(f"First 100 chars: {filtered_text[:100] if filtered_text else 'No text!'}")
    
    # Save the visualization
    page.save_image("output/footer_overlap_test.png", labels=True)
    print(f"\nTest visualization saved to output/footer_overlap_test.png")
    
    # Provide a summary
    print("\nTEST SUMMARY:")
    if len(text) > 0:
        print("✅ Default extraction works now with overlapping exclusions!")
    else:
        print("❌ Default extraction still fails with overlapping exclusions!")
    
    if len(cropped_text) > 0:
        print("✅ Manual cropping approach works!")
    else:
        print("❌ Manual cropping approach fails!")
    
    if len(filtered_text) > 0:
        print("✅ Element filtering approach works!")
    else:
        print("❌ Element filtering approach fails!")

if __name__ == "__main__":
    main()