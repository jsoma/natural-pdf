#!/usr/bin/env python3
"""
Example demonstrating the simplified document QA interface.
"""
import sys
import os
import argparse

# Add the parent directory to the path so we can import the natural_pdf package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def main():
    parser = argparse.ArgumentParser(description="Example of using the improved document QA interface")
    parser.add_argument('pdf', nargs='?', 
                      default="pdfs/2019 Statistics.pdf",
                      help="Path to a PDF document")
    parser.add_argument('--question', '-q', 
                      default="What information does this document contain?",
                      help="Question to ask about the document")
    parser.add_argument('--full', '-f', action='store_true',
                      help="Show the full result dictionary with confidence scores")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf):
        print(f"Error: PDF file '{args.pdf}' not found")
        sys.exit(1)
    
    print(f"Loading PDF: {args.pdf}")
    print(f"Question: {args.question}")
    
    try:
        # Open the PDF
        with PDF(args.pdf) as pdf:
            # Get result dictionary
            result = pdf.ask(args.question)
            
            # Display result
            if args.full:
                print("\nFull result:")
                for key, value in result.items():
                    if key == 'confidence' and isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("\nResult:")
                print(f"  Answer: {result['answer']}")
                if 'confidence' in result:
                    print(f"  Confidence: {result['confidence']:.2f}")
                if 'page_num' in result:
                    print(f"  Page: {result['page_num']}")
            
            # Ask another related question
            print("\nAsking follow-up question:")
            follow_up = "What year does this data cover?"
            print(f"Question: {follow_up}")
            follow_result = pdf.ask(follow_up)
            print(f"Answer: {follow_result['answer']}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()