"""
Example demonstrating how to use exclusion zones in Natural PDF.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF


def example_page_level_exclusion(pdf_path):
    """
    Example demonstrating page-level exclusion zones.
    """
    with PDF(pdf_path) as pdf:
        page = pdf.pages[0]
        
        # Print the full text for comparison
        print("\n--- Original Text ---")
        print(page.extract_text())
        
        # Add an exclusion for anything above a heading
        print("\n--- After Excluding Header ---")
        header = page.find('text:contains("Summary")')
        if header:
            # Add the exclusion and extract text with it applied
            page.add_exclusion(header.above())
            print(page.extract_text())
        else:
            print("Header not found. Try with a different selector.")
        
        # Add another exclusion for content below the last line
        print("\n--- After Excluding Header and Footer ---")
        lines = page.find_all('line')
        if lines and len(lines) > 0:
            last_line = lines.last if hasattr(lines, 'last') else lines[-1]
            # Add the second exclusion
            page.add_exclusion(last_line.below())
            print(page.extract_text())
        else:
            print("Line not found. Try with a different selector.")
            
        # Show that we can disable exclusions if needed
        print("\n--- With Exclusions Disabled ---")
        print(page.extract_text(apply_exclusions=False))


def example_pdf_level_exclusion(pdf_path):
    """
    Example demonstrating PDF-level exclusion zones with lambdas.
    """
    with PDF(pdf_path) as pdf:
        # Print text from the first page for comparison
        print("\n=== Original Text from First Page ===")
        print(pdf.pages[0].extract_text(apply_exclusions=False)[:200] + "...")
        
        # Define safer exclusion functions with better error handling
        def header_exclusion(page):
            try:
                header = page.find('text:contains("Page")')
                if header:
                    return header.above()
                print(f"Page {page.index}: No 'Page' text found for header exclusion")
                return None
            except Exception as e:
                print(f"ERROR in header exclusion for page {page.index}: {e}")
                return None
                
        def footer_exclusion(page):
            try:
                lines = page.find_all('line')
                if lines and len(lines) > 0:
                    return lines[-1].below()
                print(f"Page {page.index}: No lines found for footer exclusion")
                return None
            except Exception as e:
                print(f"ERROR in footer exclusion for page {page.index}: {e}")
                return None
        
        # Add document-wide exclusions using our safer functions
        # 1. Exclude headers - find text containing "Page" and exclude everything above it
        pdf.add_exclusion(header_exclusion, label="headers")
        
        # 2. Exclude footers - find the last line and exclude everything below it
        pdf.add_exclusion(footer_exclusion, label="footers")
        
        # Print the cleaned text
        print("\n=== Cleaned Text from First Page ===")
        print(pdf.pages[0].extract_text()[:200] + "...")
        
        # Extract text from entire document with exclusions applied - WITH DEBUG INFORMATION
        print("\n=== Extracting from Entire Document with Exclusions ===")
        print("\n--- DETAILED DEBUG INFO ---")
        full_text = pdf.extract_text(debug_exclusions=True)  # Enable detailed debugging
        print("--- END OF DEBUG INFO ---\n")
        
        print(f"Extracted {len(full_text)} characters with exclusions applied")
        print(full_text[:200] + "...")
        
        # Regular extraction (for comparison)
        print("\n=== Regular Extraction Without Debug Info ===")
        full_text_no_debug = pdf.extract_text()
        print(f"Extracted {len(full_text_no_debug)} characters without debug output")
        
        # Extract text with exclusions disabled (for comparison)
        print("\n=== Extracting with Exclusions Disabled (for comparison) ===")
        full_text_no_exclusions = pdf.extract_text(apply_exclusions=False)
        print(f"Extracted {len(full_text_no_exclusions)} characters with exclusions disabled")
        if len(full_text) != len(full_text_no_exclusions):
            print(f"Difference: {len(full_text_no_exclusions) - len(full_text)} characters were excluded")


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
    
    # Run the page-level example
    print("\n=== Page-Level Exclusion Example ===")
    example_page_level_exclusion(pdf_path)
    
    # Run the PDF-level example
    print("\n=== PDF-Level Exclusion Example ===")
    example_pdf_level_exclusion(pdf_path)


if __name__ == "__main__":
    main()