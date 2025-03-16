#!/usr/bin/env python3
"""
Example demonstrating the PageCollection functionality.

This example shows how to:
1. Access a specific range of pages using slicing
2. Extract text from multiple pages
3. Find elements across multiple pages
4. Get sections that span across page boundaries

Usage:
    python examples/page_collection_example.py [path_to_pdf]
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from natural_pdf import PDF

# Use the provided PDF path or a default
pdf_path = sys.argv[1] if len(sys.argv) > 1 else "pdfs/Atlanta_Public_Schools_GA_sample.pdf"

def main():
    print(f"Opening {pdf_path}")
    
    with PDF(pdf_path) as pdf:
        page_count = len(pdf)
        print(f"PDF has {page_count} pages")
        
        # Example 1: Access a page range
        if page_count >= 3:
            print("\n1. Working with a range of pages:")
            # Get pages 1-3 (0-indexed, so second, third, fourth pages)
            page_range = pdf.pages[1:4]
            print(f"  Selected {len(page_range)} pages: {[p.number for p in page_range.pages]}")
            
            # Extract text from the range
            text = page_range.extract_text()
            print(f"  Extracted {len(text)} characters of text from pages {[p.number for p in page_range.pages]}")
            
            # You can also slice a page collection
            if len(page_range) > 1:
                sub_range = page_range[0:2]
                print(f"  Sub-range has {len(sub_range)} pages: {[p.number for p in sub_range.pages]}")
        
        # Example 2: Find elements across multiple pages
        if page_count >= 2:
            print("\n2. Finding elements across multiple pages:")
            # Get the first two pages
            two_pages = pdf.pages[0:2]
            
            # Find all text elements
            text_elements = two_pages.find_all('text')
            print(f"  Found {len(text_elements)} text elements across {len(two_pages)} pages")
            
            # Find the first heading-like element
            heading = two_pages.find('text[size>=12]')
            if heading:
                print(f"  Found heading: '{heading.text}' on page {heading.page.number}")
        
        # Example 3: Get sections across pages
        if page_count >= 2:
            print("\n3. Getting sections across pages:")
            # Get the first two pages
            two_pages = pdf.pages[0:2]
            
            # Try to find headings or large text as section starts
            sections = two_pages.get_sections(
                start_selector='text[size>=12]',
                new_section_on_page_break=False,  # Allow sections to continue across pages
                boundary_inclusion='both'
            )
            
            print(f"  Found {len(sections)} sections across {len(two_pages)} pages")
            
            # Print info about each section
            for i, section in enumerate(sections):
                print(f"  Section {i+1}:")
                if hasattr(section, 'start_element') and section.start_element:
                    print(f"    Starts with: '{section.start_element.text}'")
                    print(f"    On page: {section.start_element.page.number}")
                
                text = section.extract_text()
                print(f"    Contains {len(text)} characters of text")
                
                # Show a preview
                preview = text[:50] + "..." if len(text) > 50 else text
                print(f"    Preview: {preview}")
            
            # Show with page breaks as section boundaries
            sections_with_breaks = two_pages.get_sections(
                start_selector='text[size>=12]',
                new_section_on_page_break=True,  # Force new sections at page boundaries
                boundary_inclusion='both'
            )
            print(f"  With page breaks as boundaries: {len(sections_with_breaks)} sections")

if __name__ == "__main__":
    main()