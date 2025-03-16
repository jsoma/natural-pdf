"""
Example demonstrating the Document QA capabilities of Natural PDF.

This example shows how to:
1. Ask questions to a PDF document
2. Ask questions to specific pages
3. Ask questions to specific regions
4. Control confidence thresholds
5. Highlight answer elements
6. Handle QA results

Requirements:
- transformers
- torch
"""

import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
import logging
from typing import Dict, Any

# Add parent directory to path to run without installing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from natural_pdf import PDF, configure_logging

def format_qa_result(result: Dict[str, Any]) -> str:
    """Format a QA result as a string."""
    if not result.get("found", False):
        return f"No answer found. {result.get('message', '')}"
    
    answer = result.get("answer", "")
    confidence = result.get("confidence", 0.0)
    page_num = result.get("page_num", 0)
    
    return f"Answer: {answer} (confidence: {confidence:.2f}, page: {page_num})"

def main():
    parser = argparse.ArgumentParser(description="Document QA Example")
    parser.add_argument("pdf_path", nargs="?", default="../pdfs/0500000US42001.pdf", 
                      help="Path to PDF document")
    parser.add_argument("--questions", nargs="+", 
                      default=["How many votes for Harris and Walz?", 
                              "How many votes for Trump and Vance?", 
                              "What precinct is this for?", 
                              "What state is this for?"], 
                      help="Questions to ask")
    parser.add_argument("--highlight", action="store_true", 
                      help="Highlight answer elements")
    parser.add_argument("--min-confidence", type=float, default=0.2,
                      help="Minimum confidence threshold (0.0-1.0)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("--model", default="impira/layoutlm-document-qa",
                      help="Model to use (default: impira/layoutlm-document-qa)")
    parser.add_argument("--region", action="store_true",
                      help="Ask questions to specific regions instead of whole pages")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    configure_logging(level=log_level)
    
    # Open the PDF
    pdf = PDF(args.pdf_path)
    page = pdf.pages[0]  # Use the first page for this example
    
    print(f"Document: {args.pdf_path}")
    print(f"Page count: {len(pdf.pages)}")
    print(f"Model: {args.model}")
    print(f"Minimum confidence: {args.min_confidence}")
    print()
    
    # Create output directory if not exists
    os.makedirs("output", exist_ok=True)
    
    # If using regions, detect document layout
    if args.region:
        print("Detecting document layout...")
        page.analyze_layout(confidence=0.3)
        regions = page.find_all('region')
        print(f"Found {len(regions)} regions")
        
        # Save an image with detected regions
        page.highlight_layout()
        page.save_image("output/document_qa_regions.png")
        print("Saved layout visualization to output/document_qa_regions.png")
        print()
    
    # Process each question
    for i, question in enumerate(args.questions):
        print(f"Question {i+1}: {question}")
        
        if args.region:
            # Ask each region (sort by confidence)
            all_results = []
            for region in regions:
                if region.region_type in ['title', 'plain-text', 'table', 'list']:
                    result = region.ask(
                        question=question,
                        min_confidence=args.min_confidence,
                        model=args.model
                    )
                    if result.get("found", False):
                        all_results.append(result)
            
            # Sort by confidence
            all_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            if all_results:
                result = all_results[0]  # Use the highest confidence result
                print(format_qa_result(result))
                
                # Highlight the answer if requested
                if args.highlight and result.get("source_elements"):
                    highlight_image = page.duplicate()
                    region_type = result["region"].region_type if "region" in result else "unknown"
                    for element in result["source_elements"]:
                        element.highlight(color=(1, 0.5, 0, 0.5))  # Orange highlight
                    
                    # Add question and answer as text annotation
                    highlight_image.annotate_text(
                        x=50, y=20, 
                        text=f"Q: {question}\nA: {result['answer']} (confidence: {result['confidence']:.2f}, region: {region_type})",
                        font_size=14,
                        color=(0, 0, 0)
                    )
                    
                    # Save the highlighted image
                    output_path = f"output/document_qa_answer_{i+1}.png"
                    highlight_image.save_image(output_path)
                    print(f"Saved answer visualization to {output_path}")
            else:
                print("No answer found in any region")
        else:
            # Ask the whole page
            result = page.ask(
                question=question,
                min_confidence=args.min_confidence,
                model=args.model
            )
            
            print(format_qa_result(result))
            
            # Highlight the answer if requested
            if args.highlight and result.get("found", False) and result.get("source_elements"):
                highlight_image = page.duplicate()
                for element in result["source_elements"]:
                    element.highlight(color=(1, 0.5, 0, 0.5))  # Orange highlight
                
                # Add question and answer as text annotation
                highlight_image.annotate_text(
                    x=50, y=20, 
                    text=f"Q: {question}\nA: {result['answer']} (confidence: {result['confidence']:.2f})",
                    font_size=14,
                    color=(0, 0, 0)
                )
                
                # Save the highlighted image
                output_path = f"output/document_qa_answer_{i+1}.png"
                highlight_image.save_image(output_path)
                print(f"Saved answer visualization to {output_path}")
        
        print()
    
    # Try a different PDF approach - ask the whole document
    print("Asking questions to the whole document:")
    
    for i, question in enumerate(args.questions):
        print(f"Question {i+1}: {question}")
        
        result = pdf.ask(
            question=question,
            min_confidence=args.min_confidence,
            model=args.model
        )
        
        print(format_qa_result(result))
        print()

if __name__ == "__main__":
    main()