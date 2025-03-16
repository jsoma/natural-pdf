#!/usr/bin/env python3
"""
Example demonstrating loading a PDF from a URL.
"""
import sys
import os
import argparse

# Add the parent directory to the path so we can import the natural_pdf package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def main():
    parser = argparse.ArgumentParser(description="Example of loading a PDF from a URL")
    parser.add_argument('url', nargs='?', 
                      default="https://arxiv.org/pdf/2103.14749.pdf",
                      help="URL to a PDF document (default: an arXiv paper)")
    args = parser.parse_args()
    
    print(f"Loading PDF from URL: {args.url}")
    
    # Open the PDF from URL
    with PDF(args.url) as pdf:
        # Display basic document info
        print(f"Document loaded successfully: {len(pdf)} pages")
        
        # Extract text from the first page
        if len(pdf) > 0:
            page = pdf.pages[0]
            
            # Get the title (usually large text on the first page)
            title = page.find_all('text[size>=12]')
            if title:
                print("\nTitle candidates:")
                for i, t in enumerate(title[:3], 1):  # Show top 3 candidates
                    print(f"{i}. {t.text}")
            
            # Extract the first 200 characters of text
            text = page.extract_text()
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"\nText preview:\n{preview}")

if __name__ == "__main__":
    main()