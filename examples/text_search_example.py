"""
Example demonstrating enhanced text search capabilities in Natural PDF.

This showcases:
1. Multi-word searching with keep_spaces enabled (default)
2. Case-insensitive searching
3. Regular expression searching
4. Turning off keep_spaces to see the difference
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for running the example
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from natural_pdf import PDF, configure_logging
import logging


def main(pdf_path=None):
    # Use a default PDF if none provided
    if not pdf_path:
        pdf_path = os.path.join(os.path.dirname(__file__), '..', 'pdfs', '2019 Statistics.pdf')
        
    print(f"Using PDF: {pdf_path}")
    print("-" * 50)
    
    # Create PDF with default settings (keep_spaces=True)
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    
    # Display basic page info
    print(f"Page dimensions: {page.width} x {page.height}")
    
    # 1. Basic multi-word search with default keep_spaces=True
    print("\nMulti-word search with keep_spaces=True (default):")
    print("-" * 50)
    
    # Search for a multi-word phrase
    results = page.find_all('text:contains("annual report")', case=False)
    print(f"Found {len(results)} results for 'annual report' (case-insensitive)")
    for i, result in enumerate(results):
        print(f"  Result {i+1}: '{result.text}'")
        # Highlight the results
        result.highlight(label=f"Match {i+1}: 'annual report'", color=(1, 0.7, 0, 0.3))
    
    # 2. Case-sensitive search
    print("\nCase-sensitive search:")
    print("-" * 50)
    
    # Search with case sensitivity
    results = page.find_all('text:contains("Annual Report")', case=True)
    print(f"Found {len(results)} results for 'Annual Report' (case-sensitive)")
    for i, result in enumerate(results):
        print(f"  Result {i+1}: '{result.text}'")
        # Highlight with a different color
        result.highlight(label=f"Match {i+1}: 'Annual Report'", color=(0, 0.7, 1, 0.3))
    
    # 3. Regular expression search
    print("\nRegular expression search:")
    print("-" * 50)
    
    # Use regex to find patterns
    pattern = "report\\s+\\d{4}"  # "report" followed by whitespace and 4 digits
    results = page.find_all(f'text:contains("{pattern}")', regex=True, case=False)
    print(f"Found {len(results)} results for regex pattern '{pattern}'")
    for i, result in enumerate(results):
        print(f"  Result {i+1}: '{result.text}'")
        # Highlight with another color
        result.highlight(label=f"Match {i+1}: regex '{pattern}'", color=(0, 1, 0, 0.3))
    
    # Save highlighted page as an image
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'text_search_results.png')
    page.save_image(output_path, labels=True)
    print(f"\nSaved highlighted results to: {output_path}")
    
    # 4. Create a new PDF with keep_spaces=False to compare
    print("\nComparing with keep_spaces=False (legacy behavior):")
    print("-" * 50)
    
    # Create a new PDF with keep_spaces=False
    pdf_legacy = PDF(pdf_path, keep_spaces=False)
    page_legacy = pdf_legacy.pages[0]
    
    # Try the same multi-word search
    results_legacy = page_legacy.find_all('text:contains("annual report")', case=False)
    print(f"Found {len(results_legacy)} results for 'annual report' (case-insensitive)")
    
    # Try regex to find occurrences in separate words
    pattern = "annual\\s+report"  # "annual" followed by whitespace and "report"
    regex_results = page_legacy.find_all(f'text:contains("{pattern}")', regex=True, case=False)
    print(f"With regex '{pattern}': Found {len(regex_results)} results")
    
    # Show conclusion
    print("\nConclusion:")
    print("-" * 50)
    print("1. With keep_spaces=True (default):")
    print("   - Multi-word phrases can be found directly with :contains()")
    print("   - Text maintains its natural spacing within word elements")
    print("\n2. With keep_spaces=False (legacy):")
    print("   - Words are split at spaces, making multi-word search less effective")
    print("   - Regular expressions with \\s patterns can help bridge words")
    
    return pdf


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Demonstrate Natural PDF's enhanced text search capabilities")
    parser.add_argument("--pdf", help="Path to a PDF file to analyze")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    configure_logging(level=log_level)
    
    # Run the example
    pdf = main(args.pdf)