"""
Example demonstrating how to use the until parameter with above() and below() methods.
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
    
    # Clear any existing highlights
    page.clear_highlights()
    
    # First, find some key elements on the page
    heading1 = page.find('text[size>=12]')
    
    if not heading1:
        # If no large headings, just use the first few elements as examples
        elements = page.get_elements()
        elements.sort(key=lambda e: (e.top, e.x0))  # Sort in reading order
        
        if len(elements) < 3:
            print("Not enough elements found for demonstration")
            return
            
        element1 = elements[0]
        element2 = elements[len(elements) // 3]  # About 1/3 down
        element3 = elements[len(elements) // 2]  # About halfway down
        
        # Highlight the reference elements
        element1.highlight(label="First Element")
        element2.highlight(label="Second Element")
        element3.highlight(label="Third Element")
        
        print(f"First element: '{element1.text if hasattr(element1, 'text') else 'non-text'}' at y={element1.top}")
        print(f"Second element: '{element2.text if hasattr(element2, 'text') else 'non-text'}' at y={element2.top}")
        print(f"Third element: '{element3.text if hasattr(element3, 'text') else 'non-text'}' at y={element3.top}")
        
        # Demonstrate below() with until parameter
        print("\nDemonstrating below() with until parameter")
        
        # Get the region from element1 to element2
        region1 = element1.below(until=f'text:contains("{element2.text}")')
        region1.highlight(label="Below until Second Element")
        
        # Get the region from element2 to element3, excluding element3
        region2 = element2.below(until=f'text:contains("{element3.text}")', include_until=False)
        region2.highlight(label="Below until Third Element (excluded)")
        
        # Demonstrate above() with until parameter
        print("\nDemonstrating above() with until parameter")
        
        # Get the region from element3 up to element2
        region3 = element3.above(until=f'text:contains("{element2.text}")')
        region3.highlight(label="Above until Second Element")
        
        # Get the region from element2 up to element1, excluding element1
        region4 = element2.above(until=f'text:contains("{element1.text}")', include_until=False)
        region4.highlight(label="Above until First Element (excluded)")
        
        # Create an output directory
        output_dir = Path(__file__).parent / "until_output"
        output_dir.mkdir(exist_ok=True)
        
        # Save the result
        page.save(str(output_dir / "until_boundaries.png"), labels=True)
        
        # Print the contents of the regions
        print("\nContent in region 'below until second element':")
        print(region1.extract_text()[:100] + "..." if len(region1.extract_text()) > 100 else region1.extract_text())
        
        print("\nContent in region 'above until second element':")
        print(region3.extract_text()[:100] + "..." if len(region3.extract_text()) > 100 else region3.extract_text())
        
        print("\nExample completed. Check 'until_output/until_boundaries.png' for the result.")
    else:
        # Find more headings
        headings = page.find_all('text[size>=12]')
        
        if len(headings) < 2:
            # If not enough headings, fall back to the approach above
            print("Not enough headings found. Using generic elements instead.")
            main()  # Re-run with the above approach
            return
        
        # Use the first two headings
        heading1 = headings[0]
        heading2 = headings[1]
        
        # Highlight the headings
        heading1.highlight(label="First Heading")
        heading2.highlight(label="Second Heading")
        
        print(f"First heading: '{heading1.text}' at y={heading1.top}")
        print(f"Second heading: '{heading2.text}' at y={heading2.top}")
        
        # Demonstrate below() with until parameter
        print("\nDemonstrating below() with until parameter")
        
        # Get the region from heading1 to heading2
        region1 = heading1.below(until=f'text:contains("{heading2.text}")')
        region1.highlight(label="Below until Second Heading")
        
        # Get the region from heading1 to heading2, excluding heading2
        region2 = heading1.below(until=f'text:contains("{heading2.text}")', include_until=False)
        region2.highlight(label="Below until Second Heading (excluded)")
        
        # Create an output directory
        output_dir = Path(__file__).parent / "until_output"
        output_dir.mkdir(exist_ok=True)
        
        # Save the result
        page.to_image(path=str(output_dir / "until_boundaries_headings.png"), show_labels=True)
        
        # Print the contents of the regions
        print("\nContent in region 'below until second heading':")
        print(region1.extract_text()[:100] + "..." if len(region1.extract_text()) > 100 else region1.extract_text())
        
        print("\nContent in region 'below until second heading (excluded)':")
        print(region2.extract_text()[:100] + "..." if len(region2.extract_text()) > 100 else region2.extract_text())
        
        print("\nExample completed. Check 'until_output/until_boundaries_headings.png' for the result.")
        
if __name__ == "__main__":
    main()