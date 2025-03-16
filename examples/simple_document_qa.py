"""
Simple demonstration of document QA functionality in Natural PDF.
"""

import os
import sys
import argparse

# Add parent directory to path to run without installing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from natural_pdf import PDF, configure_logging
import logging

def main():
    # Set up logging
    configure_logging(level=logging.INFO)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simple Document QA Example")
    parser.add_argument("pdf_path", nargs="?", default="../pdfs/0500000US42001.pdf", 
                      help="Path to PDF document")
    parser.add_argument("question", nargs="?", default="How many votes for Harris and Walz?",
                      help="Question to ask about the document")
    parser.add_argument("--debug", action="store_true",
                      help="Save debug information for troubleshooting")
    args = parser.parse_args()
    
    # Open the PDF
    pdf = PDF(args.pdf_path)
    print(f"Loaded PDF: {args.pdf_path} ({len(pdf.pages)} pages)")
    
    # Get the first page
    page = pdf.pages[0]
    
    # Ask a question to the document
    print(f"\nAsking document: {args.question}")
    doc_result = pdf.ask(args.question, debug=args.debug)
    
    if doc_result.get("found", False):
        print(f"Document answer: {doc_result['answer']}")
        print(f"Confidence: {doc_result['confidence']:.2f}")
        print(f"Page: {doc_result.get('page_num', 0)}")
    else:
        print(f"No answer found in document: {doc_result.get('message', '')}")
    
    # Ask the same question to the page
    print(f"\nAsking page 0: {args.question}")
    page_result = page.ask(args.question, debug=args.debug)
    
    if page_result.get("found", False):
        print(f"Page answer: {page_result['answer']}")
        print(f"Confidence: {page_result['confidence']:.2f}")
        
        # Highlight the answer elements if available
        if page_result.get("source_elements"):
            for element in page_result["source_elements"]:
                element.highlight(color=(1, 0.5, 0, 0.5))  # Orange highlight
            
            # Save the highlighted image
            os.makedirs("output", exist_ok=True)
            page.save_image("output/simple_qa_answer.png")
            print("Saved highlighted answer to output/simple_qa_answer.png")
    else:
        print(f"No answer found on page: {page_result.get('message', '')}")
    
    # Optional: Analyze layout and ask questions to specific regions
    print("\nDetecting document layout...")
    page.analyze_layout(confidence=0.3)
    regions = page.find_all('region[type=title], region[type=plain-text], region[type=table]')
    print(f"Found {len(regions)} relevant regions")
    
    # Save layout visualization
    page.highlight_layout()
    page.save_image("output/simple_qa_regions.png")
    print("Saved layout visualization to output/simple_qa_regions.png")
    
    # Ask questions to each region
    best_region_result = None
    best_confidence = 0
    
    for i, region in enumerate(regions):
        region_result = region.ask(args.question, debug=args.debug)
        
        if region_result.get("found", False) and region_result.get("confidence", 0) > best_confidence:
            best_region_result = region_result
            best_confidence = region_result["confidence"]
    
    if best_region_result:
        region_type = best_region_result["region"].region_type
        print(f"\nBest region answer ({region_type}): {best_region_result['answer']}")
        print(f"Confidence: {best_region_result['confidence']:.2f}")
    else:
        print("\nNo answer found in any region")

if __name__ == "__main__":
    main()