"""
Example demonstrating the 'until' feature of natural-pdf.
(This was previously named 'select_until')
"""
import os
import sys

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def until_example(pdf_path):
    """Demonstrates the 'until' method for defining content regions."""
    # Open the PDF
    with PDF(pdf_path) as pdf:
        page = pdf.pages[0]
        
        print(f"PDF loaded: {pdf_path}")
        print(f"PDF has {len(pdf)} pages\n")
        
        # EXAMPLE 1: Select from "Summary:" until the thick line
        print("EXAMPLE 1: Select from Summary until thick line")
        print("----------------------------------------------")
        
        # Find the "Summary:" text
        summary = page.find('text:contains("Summary:")')
        print(f"Found 'Summary' text at: {summary.bbox}")
        
        # Find the thick line
        thick_line = page.find('line[width>=2]')
        print(f"Found thick line at: {thick_line.bbox}")
        
        # Create a region from Summary until the thick line
        print("\nCreating region from 'Summary' until the thick line...")
        summary_region = summary.until('line[width>=2]', width="full")
        print(f"Region boundaries: {summary_region.bbox}")
        
        # Extract and display text from this region
        region_text = summary_region.extract_text()
        print("\nText from the region:")
        print("-" * 60)
        print(region_text)
        print("-" * 60)
        
        # Find all text elements in this region
        text_elements = summary_region.find_all('text')
        print(f"\nFound {len(text_elements)} text elements in the region")
        
        # Display the first 5 elements
        if text_elements:
            print("First 5 elements:")
            for i, el in enumerate(text_elements[:5]):
                print(f"  {i+1}. '{el.text}'")
        
        # EXAMPLE 2: Demonstrate include_endpoint=False option
        print("\nEXAMPLE 2: Without including endpoint element")
        print("----------------------------------------------")
        
        # Create a region from Summary until the thick line, excluding the line
        exclusive_region = summary.until('line[width>=2]', include_endpoint=False, width="full")
        print(f"Region boundaries: {exclusive_region.bbox}")
        
        # Compare text length
        inclusive_text = summary_region.extract_text()
        exclusive_text = exclusive_region.extract_text()
        
        print(f"\nWith include_endpoint=True: {len(inclusive_text)} characters")
        print(f"With include_endpoint=False: {len(exclusive_text)} characters")
        
        # EXAMPLE 3: Different elements for until
        print("\nEXAMPLE 3: Select from one text to another")
        print("----------------------------------------------")
        
        # Find text elements to use as boundaries
        heading = page.find('text:contains("Violations")')
        if heading:
            # Select from "Violations" to "Critical"
            target_word = page.find('text:contains("Critical")')
            if target_word:
                region = heading.until('text:contains("Critical")', width="full")
                print(f"\nRegion from 'Violations' to 'Critical': {region.bbox}")
                
                text = region.extract_text()
                print(f"Extracted {len(text)} characters of text")
                if len(text) > 100:
                    print(f"First 100 characters: {text[:100]}...")
            else:
                print("Could not find 'Critical' text")
        else:
            print("Could not find 'Violations' heading")
            
        print("\nEnd of 'until' method demonstration.")

if __name__ == "__main__":
    # Default to example PDF if no path is provided
    if len(sys.argv) < 2:
        # Use the example PDF in the pdfs directory
        pdf_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'pdfs', '01-practice.pdf'))
        if not os.path.exists(pdf_path):
            print("Example PDF not found. Please provide a path to a PDF file.")
            print("Usage: python until_example.py [path/to/file.pdf]")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
        # Check if the file exists
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            sys.exit(1)
    
    until_example(pdf_path)