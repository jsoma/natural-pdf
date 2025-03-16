"""
Example demonstrating the improved API consistency in natural-pdf.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def consistency_example(pdf_path):
    """Example showing the improved consistent API."""
    # Open the PDF without OCR to avoid issues
    with PDF(pdf_path) as pdf:
            
        print(f"PDF has {len(pdf)} pages")
        page = pdf.pages[0]
        
        print("\n1. IMPROVED REGION CREATION:")
        # Create a region with intuitive named parameters
        header_region = page.region(top=0, bottom=100)
        print(f"  Created header region with bounds {header_region.bbox}")
        
        # Create a custom region with element width
        custom_region = page.region(
            left=100, right=300, 
            top=200, bottom=400,
            width="element"
        )
        print(f"  Created custom region with bounds {custom_region.bbox}")
        
        print("\n2. IMPROVED SPATIAL NAVIGATION:")
        # Find a major element
        heading = page.find('text[size>=12]')
        if heading:
            print(f"  Found heading: '{heading.text}'")
            
            # Use above/below with improved parameters
            above_region = heading.above(height=50, width="full")
            print(f"  Region above: {above_region.bbox}")
            
            # Below with element width
            below_region = heading.below(height=100, width="element")
            print(f"  Region below (element width): {below_region.bbox}")
            
            # Using until with consistent parameter naming
            next_heading = page.find('text[size>=12]', skip=1)
            if next_heading:
                print(f"  Found next heading: '{next_heading.text}'")
                
                # Using the until method
                between_region = heading.until(
                    'text[size>=12]', 
                    include_endpoint=False,
                    width="full"
                )
                # Don't use OCR for text extraction
                print(f"  Region between headings: {between_region.bbox}")
        
        print("\n3. CONSISTENT EXTRACTION PARAMETERS:")
        # Text extraction with consistent parameters
        text = page.extract_text(
            preserve_whitespace=True,
            use_exclusions=True
        )
        print(f"  Extracted {len(text)} characters")
        
        print("\n4. CONSISTENT VISUAL METHODS:")
        # Find and highlight elements with consistent parameters
        lines = page.find_all('line[width>=1]')
        if lines:
            print(f"  Found {len(lines)} thick lines")
            
            # Highlight with label first, then color
            lines.highlight(
                label="Thick Lines",
                color=(1, 0, 0, 0.5)
            )
            
            # Method chaining with save
            lines.highlight(
                label="Thick Lines"
            ).save(
                "improved_api_lines.png",
                show_labels=True
            )
            
        print("\n5. BUILDER PATTERN:")
        # Create regions for exclusion
        header = page.region(top=0, bottom=50)
        footer = page.region(top=page.height-50, bottom=page.height)
        
        # Add exclusions with method chaining
        pdf.add_exclusion(
            lambda p: p.region(top=0, bottom=50),
            label="headers"
        ).add_exclusion(
            lambda p: p.region(top=p.height-50, bottom=p.height),
            label="footers"
        )
        
        # Extract with exclusions
        filtered_text = page.extract_text(use_exclusions=True)
        print(f"  Extracted {len(filtered_text)} characters with exclusions")
        
        # Method chaining with method return
        pdf_same = pdf.add_exclusion(lambda p: None, label="test")
        print(f"  Method chaining returns same object: {pdf is pdf_same}")

if __name__ == "__main__":
    # Default to example PDF if no path is provided
    if len(sys.argv) < 2:
        # Use the example PDF in the pdfs directory
        pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pdfs', '01-practice.pdf'))
        if not os.path.exists(pdf_path):
            print("Example PDF not found. Please provide a path to a PDF file.")
            print("Usage: python improved_api_example.py [path/to/file.pdf]")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
        # Check if the file exists
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            sys.exit(1)
    
    consistency_example(pdf_path)