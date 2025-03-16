"""
Example demonstrating font-aware text extraction in Natural PDF.

This example shows how to use the font_attrs parameter to group text by font properties,
which helps preserve the formatting and style of text during extraction.
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
    
    # Example 1: Default behavior - group by fontname and size
    print("\n1. Default behavior (group by fontname and size):")
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    
    # Find some text element to inspect
    text_element = page.find("text")
    if text_element:
        print(f"Example text element: {text_element}")
        print(f"Font info: {text_element.font_info()}")
    
    # Example 2: Disable font-aware grouping
    print("\n2. Disable font-aware grouping (spatial only):")
    pdf_no_font = PDF(pdf_path, font_attrs=[])
    page_no_font = pdf_no_font.pages[0]
    
    # Find the same text with different grouping
    text_element = page_no_font.find("text")
    if text_element:
        print(f"Example text element: {text_element}")
    
    # Example 3: Group by additional attributes
    print("\n3. Group by font and color:")
    pdf_with_color = PDF(pdf_path, font_attrs=['fontname', 'size', 'non_stroking_color'])
    page_with_color = pdf_with_color.pages[0]
    
    # Find the same text with color grouping
    text_element = page_with_color.find("text")
    if text_element:
        print(f"Example text element: {text_element}")
    
    # Compare text extraction results
    print("\n4. Text extraction comparison:")
    
    # Get a small region with mixed text styles
    text_elements = page.find_all("text")
    if text_elements:
        region = page.create_region(0, 0, page.width, page.height)  # Use the full page
        
        # Extract with different font grouping settings
        default_text = region.extract_text()
        spatial_text = page_no_font.create_region(0, 0, page_no_font.width, page_no_font.height).extract_text()
        color_text = page_with_color.create_region(0, 0, page_with_color.width, page_with_color.height).extract_text()
        
        # Show word counts as a simple comparison
        print(f"Default grouping word count: {len(default_text.split())}")
        print(f"Spatial-only grouping word count: {len(spatial_text.split())}")
        print(f"Font+color grouping word count: {len(color_text.split())}")
        
        # Show sample of text differences
        print("\nText samples (first 200 chars):")
        print(f"Default: {default_text[:200]}...")
        print(f"Spatial: {spatial_text[:200]}...")
        print(f"Color-aware: {color_text[:200]}...")
    
    # Example 4: Detailed character-level analysis
    print("\n5. Character-level analysis:")
    
    # Get raw character data
    chars = page.find_all('char')[:5]  # First 5 characters
    print(f"Raw character elements ({len(chars)} of {len(page.find_all('char'))} total):")
    for char in chars:
        print(f" - {char}")
    
    # Show word elements too
    words = page.find_all("text")[:3]  # First 3 words
    print(f"\nWord elements ({len(words)} of {len(page.find_all('text'))} total):")
    for word in words:
        print(f" - {word}")
        print(f"   Font info: {word.font_info()}")

if __name__ == "__main__":
    main()