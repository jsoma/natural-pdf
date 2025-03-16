"""
Example demonstrating the region.expand() method in Natural PDF.

This example shows how to expand or shrink regions in various ways.
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
        # Use a default PDF path
        pdf_path = "pdfs/Atlanta_Public_Schools_GA_sample.pdf"
        if not os.path.exists(pdf_path):
            print(f"Default PDF not found at {pdf_path}")
            print("Please provide a PDF path as an argument")
            return
    
    print(f"Processing PDF: {pdf_path}")
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    
    # Example 1: Basic expansion in different directions
    print("\n1. Basic region expansion")
    
    # Find a text element to start with
    text = page.find('text')
    if not text:
        print("No text found on page")
        return
        
    # Create a region from the text element (its bounding box)
    region = page.create_region(text.x0, text.top, text.x1, text.bottom)
    print(f"Original region: {region.bbox}")
    
    # Expand the region in different directions
    expanded_right = region.expand(right=50)
    print(f"Expanded right by 50: {expanded_right.bbox}")
    
    expanded_all = region.expand(left=10, right=20, top_expand=15, bottom_expand=25)
    print(f"Expanded in all directions: {expanded_all.bbox}")
    
    # Shrink the region with negative values
    shrunk = region.expand(left=-5, right=-5, top_expand=-2, bottom_expand=-2)
    print(f"Shrunk with negative values: {shrunk.bbox}")
    
    # Example 2: Using expansion factors
    print("\n2. Expansion with factors")
    
    # Double the width
    double_width = region.expand(width_factor=2.0)
    print(f"Double width (width_factor=2.0): {double_width.bbox}")
    
    # Increase height by 50%
    taller = region.expand(height_factor=1.5)
    print(f"50% taller (height_factor=1.5): {taller.bbox}")
    
    # Both width and height factors
    bigger = region.expand(width_factor=1.5, height_factor=1.25)
    print(f"Wider and taller: {bigger.bbox}")
    
    # Example 3: Combining with spatial navigation
    print("\n3. Combining with spatial navigation")
    
    # Find a heading (assuming it's bold or larger text)
    heading = page.find('text[size>=12]')
    if heading:
        print(f"Found heading: '{heading.text}'")
        
        # Create a region below the heading and expand it
        content_region = heading.below(height=100, full_width=False)
        print(f"Region below heading: {content_region.bbox}")
        
        # Expand the region to include more content
        expanded_region = content_region.expand(right=100, bottom_expand=50)
        print(f"Expanded region: {expanded_region.bbox}")
        
        # Extract text from the expanded region
        text = expanded_region.extract_text()
        print(f"Text in expanded region: {text[:100]}...")
    
    # Example 4: Visual demonstration with highlighting
    print("\n4. Visual demonstration with highlighting")
    
    # Choose a region to work with
    demo_region = page.create_region(100, 100, 300, 200)
    
    # Highlight the original region
    demo_region.highlight(color=(1, 0, 0), label="Original")
    
    # Highlight expanded versions with different colors
    demo_region.expand(left=20, right=20).highlight(color=(0, 1, 0), label="Wider")
    demo_region.expand(top_expand=20, bottom_expand=20).highlight(color=(0, 0, 1), label="Taller")
    demo_region.expand(width_factor=1.5, height_factor=1.5).highlight(color=(1, 0.5, 0), label="1.5x Larger")
    
    # Save the highlighted page
    highlight_path = "region_expand_highlight.png"
    page.to_image(path=highlight_path, show_labels=True)
    print(f"Highlighted regions saved to {highlight_path}")

if __name__ == "__main__":
    main()