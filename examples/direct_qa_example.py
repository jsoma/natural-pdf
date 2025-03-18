"""
Direct Document QA example that closely mirrors the original pdfplumber implementation.

This example shows how to:
1. Use pdfplumber directly to extract words and images
2. Use transformers pipelines for document QA
3. Compare with the Natural PDF implementation

It's intentionally similar to the original code provided by the user.
"""

import os
import sys
import argparse
import pdfplumber
from PIL import Image, ImageDraw
import numpy as np

# Add parent directory to path to run without installing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# For comparison
from natural_pdf import PDF, configure_logging
import logging

def main():
    parser = argparse.ArgumentParser(description="Direct Document QA Example")
    parser.add_argument("pdf_path", nargs="?", default="../pdfs/0500000US42001.pdf", 
                      help="Path to PDF document")
    parser.add_argument("--question", default="How many votes for Harris and Walz?",
                      help="Question to ask about the document")
    parser.add_argument("--debug", action="store_true",
                      help="Save debug information for troubleshooting")
    
    args = parser.parse_args()
    
    # Configure logging for Natural PDF
    if args.debug:
        configure_logging(level=logging.DEBUG)
    else:
        configure_logging(level=logging.INFO)
    
    print(f"Document: {args.pdf_path}")
    print(f"Question: {args.question}")
    
    print("\n=== Natural PDF implementation ===")
    
    # Use Natural PDF
    pdf = PDF(args.pdf_path)
    page = pdf.pages[0]
    
    # Ask the question
    result = page.ask(args.question, debug=args.debug)
    
    if result.get("found", False):
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        # Highlight the answer
        if result.get("source_elements"):
            for element in result["source_elements"]:
                element.highlight(color=(1, 0.5, 0, 0.5))
            
            # Save the image
            page.save_image("output/natural_pdf_answer.png")
            print("Saved highlighted answer to output/natural_pdf_answer.png")
    else:
        print(f"No answer found: {result.get('error', '')}")
    
if __name__ == "__main__":
    main()