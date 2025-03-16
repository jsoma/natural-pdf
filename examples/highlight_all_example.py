"""
Example demonstrating the highlight_all feature of natural-pdf.
"""
import os
import sys

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def highlight_all_example(pdf_path):
    """Demonstrates the highlight_all feature for quick visual inspection."""
    # Open the PDF
    with PDF(pdf_path) as pdf:
        page = pdf.pages[0]
        
        print(f"PDF loaded: {pdf_path}")
        print(f"PDF has {len(pdf)} pages")
        
        # Create an output directory for saving images
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        os.makedirs(output_dir, exist_ok=True)
        
        # EXAMPLE 1: Highlight all elements on the page
        print("\nEXAMPLE 1: Highlighting all elements")
        print("-" * 60)
        
        # Count all element types first
        element_counts = {
            'Text': len(page.words),
            'Characters': len(page.chars),
            'Lines': len(page.lines),
            'Rectangles': len(page.rects)
        }
        
        for element_type, count in element_counts.items():
            print(f"Found {count} {element_type.lower()}")
        
        # Highlight all elements
        page.highlight_all()
        
        # Save the image with a legend using to_image
        output_file = os.path.join(output_dir, "highlight_all.png")
        page.to_image(path=output_file, show_labels=True)
        print(f"Saved all highlighted elements to: {output_file}")
        
        # Clear highlights for the next example
        page.clear_highlights()
        
        # EXAMPLE 2: Highlight only specific element types
        print("\nEXAMPLE 2: Highlighting only specific element types")
        print("-" * 60)
        
        # Highlight only text and lines
        page.highlight_all(include_types=['text', 'line'])
        
        # Save the image with a legend using to_image
        output_file = os.path.join(output_dir, "highlight_specific_types.png")
        page.to_image(path=output_file, show_labels=True)
        print(f"Saved with only text and lines highlighted to: {output_file}")
        
        print("\nEnd of highlight_all demonstration.")

if __name__ == "__main__":
    # Default to example PDF if no path is provided
    if len(sys.argv) < 2:
        # Use the example PDF in the pdfs directory
        pdf_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'pdfs', '01-practice.pdf'))
        if not os.path.exists(pdf_path):
            print("Example PDF not found. Please provide a path to a PDF file.")
            print("Usage: python highlight_all_example.py [path/to/file.pdf]")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
        # Check if the file exists
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            sys.exit(1)
    
    highlight_all_example(pdf_path)