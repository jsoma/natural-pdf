"""
Test script to verify highlighting with float colors.
This is a simplified version of the test without OCR to test just the color handling.
"""
import os
import sys

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def main():
    """Test that highlighting works with float colors."""
    # Default to example PDF
    pdf_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'pdfs', '01-practice.pdf'))
    
    if not os.path.exists(pdf_path):
        print(f"Example PDF not found: {pdf_path}")
        return
        
    # Create an output directory for saving images
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing highlighting with float colors...")
        
    # Open the PDF
    with PDF(pdf_path) as pdf:
        page = pdf.pages[0]
        
        # Get some text elements
        elements = page.find_all('text')[:4]
        
        if len(elements) < 4:
            print("Not enough text elements found in the PDF")
            return
            
        # Test with various color formats
        # Example 1: RGB float 0-1 with alpha
        elements[0].highlight(
            color=(0.0, 1.0, 0.0, 0.5),  # Green semi-transparent
            label="Green Float"
        )
        
        # Example 2: RGB float 0-1 without alpha
        elements[1].highlight(
            color=(1.0, 0.0, 0.0),  # Red
            label="Red Float"
        )
        
        # Example 3: Mixed integer and float
        elements[2].highlight(
            color=(0.5, 0.5, 255, 0.7),  # Mixed format
            label="Mixed"
        )
        
        # Example 4: Integer RGB with alpha
        elements[3].highlight(
            color=(0, 0, 255, 100),  # Blue
            label="Blue Integer"
        )
        
        # Save the highlighted image
        highlight_file = os.path.join(output_dir, "highlight_float_test.png")
        page.to_image(path=highlight_file, show_labels=True)
        print(f"Saved to: {highlight_file}")
        
if __name__ == "__main__":
    main()