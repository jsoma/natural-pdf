"""
Example demonstrating image width customization in to_image method.
"""
import os
import sys

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def image_width_example(pdf_path):
    """Demonstrate customizing image width with the to_image method."""
    # Open the PDF
    with PDF(pdf_path) as pdf:
        page = pdf.pages[0]
        
        print(f"PDF loaded: {pdf_path}")
        print(f"PDF has {len(pdf)} pages")
        
        # Create an output directory for saving images
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        os.makedirs(output_dir, exist_ok=True)
        
        # First highlight some elements to make the examples more interesting
        page.clear_highlights()
        page.highlight_all(include_types=['rect', 'line'])
        
        # EXAMPLE 1: Save image with default width (based on scale)
        print("\nEXAMPLE 1: Image with default width")
        print("-" * 60)
        
        output_file = os.path.join(output_dir, "width_default.png")
        img = page.to_image(path=output_file, show_labels=True)
        print(f"Original image size: {img.width} x {img.height} pixels")
        print(f"Saved to: {output_file}")
        
        # EXAMPLE 2: Image with custom width of 800px
        print("\nEXAMPLE 2: Custom width of 800px")
        print("-" * 60)
        
        output_file = os.path.join(output_dir, "width_800px.png")
        img = page.to_image(path=output_file, width=800, show_labels=True)
        print(f"Custom image size: {img.width} x {img.height} pixels")
        print(f"Saved to: {output_file}")
        
        # EXAMPLE 3: Image with custom width of 1200px
        print("\nEXAMPLE 3: Custom width of 1200px")
        print("-" * 60)
        
        output_file = os.path.join(output_dir, "width_1200px.png")
        img = page.to_image(path=output_file, width=1200, show_labels=True)
        print(f"Custom image size: {img.width} x {img.height} pixels")
        print(f"Saved to: {output_file}")
        
        # EXAMPLE 4: Using both scale and width (width takes precedence for final output)
        print("\nEXAMPLE 4: Using both scale and width")
        print("-" * 60)
        
        output_file = os.path.join(output_dir, "width_with_scale.png")
        img = page.to_image(path=output_file, scale=3.0, width=600, show_labels=True)
        print(f"Scale 3.0 with width 600px: {img.width} x {img.height} pixels")
        print(f"Saved to: {output_file}")
        
        print("\nEnd of image width demonstration.")

if __name__ == "__main__":
    # Default to example PDF if no path is provided
    if len(sys.argv) < 2:
        # Use the example PDF in the pdfs directory
        pdf_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'pdfs', '01-practice.pdf'))
        if not os.path.exists(pdf_path):
            print("Example PDF not found. Please provide a path to a PDF file.")
            print("Usage: python image_width_example.py [path/to/file.pdf]")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
        # Check if the file exists
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            sys.exit(1)
    
    image_width_example(pdf_path)