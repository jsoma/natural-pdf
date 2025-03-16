"""
Example demonstrating positional methods in ElementCollection.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF


def main():
    """Main entry point."""
    # Get the PDF path from command line or use a default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Look for any PDF in the examples directory or pdfs directory
        example_dir = Path(__file__).parent
        pdf_files = list(example_dir.glob("*.pdf"))
        
        if not pdf_files:
            pdfs_dir = example_dir.parent / "pdfs"
            if pdfs_dir.exists():
                pdf_files = list(pdfs_dir.glob("*.pdf"))
        
        if pdf_files:
            pdf_path = str(pdf_files[0])
        else:
            print("No PDF file found. Please provide a path to a PDF file.")
            sys.exit(1)
    
    print(f"Using PDF: {pdf_path}")
    
    # Open the PDF
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    
    # Find different element types
    lines = page.find_all('line')
    rects = page.find_all('rect')
    text = page.find_all('text')
    
    # Clear any existing highlights
    page.clear_highlights()
    
    # Highlight the page corners for reference
    page.create_region(0, 0, 10, 10).highlight(label="Top-Left Corner")
    page.create_region(page.width-10, 0, page.width, 10).highlight(label="Top-Right Corner")
    page.create_region(0, page.height-10, 10, page.height).highlight(label="Bottom-Left Corner")
    page.create_region(page.width-10, page.height-10, page.width, page.height).highlight(label="Bottom-Right Corner")
    
    # Demonstrate line position methods
    print(f"\nLines found: {len(lines)}")
    if len(lines) > 0:
        highest_line = lines.highest()
        lowest_line = lines.lowest()
        leftmost_line = lines.leftmost()
        rightmost_line = lines.rightmost()
        
        print(f"Highest line: {highest_line.bbox}")
        print(f"Lowest line: {lowest_line.bbox}")
        print(f"Leftmost line: {leftmost_line.bbox}")
        print(f"Rightmost line: {rightmost_line.bbox}")
        
        # Highlight the extreme lines
        highest_line.highlight(label="Highest Line")
        lowest_line.highlight(label="Lowest Line")
        leftmost_line.highlight(label="Leftmost Line")
        rightmost_line.highlight(label="Rightmost Line")
    
    # Demonstrate rectangle position methods
    print(f"\nRectangles found: {len(rects)}")
    if len(rects) > 0:
        highest_rect = rects.highest()
        lowest_rect = rects.lowest()
        leftmost_rect = rects.leftmost()
        rightmost_rect = rects.rightmost()
        
        print(f"Highest rectangle: {highest_rect.bbox}")
        print(f"Lowest rectangle: {lowest_rect.bbox}")
        print(f"Leftmost rectangle: {leftmost_rect.bbox}")
        print(f"Rightmost rectangle: {rightmost_rect.bbox}")
        
        # Highlight the extreme rectangles
        highest_rect.highlight(label="Highest Rectangle")
        lowest_rect.highlight(label="Lowest Rectangle")
        leftmost_rect.highlight(label="Leftmost Rectangle")
        rightmost_rect.highlight(label="Rightmost Rectangle")
    
    # Demonstrate text position methods
    print(f"\nText elements found: {len(text)}")
    if len(text) > 0:
        highest_text = text.highest()
        lowest_text = text.lowest()
        leftmost_text = text.leftmost()
        rightmost_text = text.rightmost()
        
        print(f"Highest text: '{highest_text.text}' at {highest_text.bbox}")
        print(f"Lowest text: '{lowest_text.text}' at {lowest_text.bbox}")
        print(f"Leftmost text: '{leftmost_text.text}' at {leftmost_text.bbox}")
        print(f"Rightmost text: '{rightmost_text.text}' at {rightmost_text.bbox}")
        
        # Highlight the extreme text elements
        highest_text.highlight(label="Highest Text")
        lowest_text.highlight(label="Lowest Text")
        leftmost_text.highlight(label="Leftmost Text")
        rightmost_text.highlight(label="Rightmost Text")
    
    # Create an output directory
    output_dir = Path(__file__).parent / "position_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save the result
    page.to_image(path=str(output_dir / "position_methods.png"), show_labels=True)
    
    # Demonstrate error handling for multi-page collections
    if len(pdf.pages) > 1:
        print("\nTesting multi-page error handling:")
        multi_collection = pdf.pages.find_all('text')
        try:
            multi_collection.lowest()
            print("ERROR: Should have raised ValueError for multi-page collection")
        except ValueError as e:
            print(f"Correctly raised ValueError: {e}")
    
    print("\nExample completed. Check 'position_output/position_methods.png' for the result.")


if __name__ == "__main__":
    main()