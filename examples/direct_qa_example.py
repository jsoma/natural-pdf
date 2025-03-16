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

def pdfplumber_qa(pdf_path, question, debug=False):
    """Run QA using direct pdfplumber code similar to the original example."""
    # Open PDF
    pdf = pdfplumber.open(pdf_path)
    page = pdf.pages[0]
    
    # Get image
    image = page.to_image(resolution=300).original
    
    # Extract words
    words = page.extract_words()
    
    # Build word boxes in the expected format
    def get_box(word):
        return [
            word['text'],
            [int(word["x0"]), int(word["top"]), int(word["x1"]), int(word["bottom"])]
        ]
    
    word_boxes = [get_box(word) for word in words]
    
    # Debug visualization
    if debug:
        os.makedirs("output", exist_ok=True)
        
        # Save image
        image.save("output/direct_qa_image.png")
        
        # Save visualization
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        for i, (text, box) in enumerate(word_boxes):
            x0, y0, x1, y1 = box
            draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=2)
            draw.text((x0, y0), str(i), fill=(255, 0, 0))
        
        vis_image.save("output/direct_qa_boxes.png")
        
    # Use transformers pipeline
    try:
        from transformers import pipeline
        
        pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
        
        # Run query
        query = { "image": image, "question": question, "word_boxes": word_boxes }
        
        result = pipe(query)[0]
        
        # Create result dictionary similar to Natural PDF's format
        return {
            "answer": result.get("answer", ""),
            "confidence": result.get("score", 0.0),
            "start": result.get("start", 0),
            "end": result.get("end", 0),
            "found": True if result.get("answer") else False
        }
        
    except Exception as e:
        print(f"Error in direct QA: {e}")
        return {
            "answer": "",
            "confidence": 0.0,
            "error": str(e),
            "found": False
        }

def main():
    parser = argparse.ArgumentParser(description="Direct Document QA Example")
    parser.add_argument("pdf_path", nargs="?", default="../pdfs/0500000US42001.pdf", 
                      help="Path to PDF document")
    parser.add_argument("--question", default="How many votes for Harris and Walz?",
                      help="Question to ask about the document")
    parser.add_argument("--debug", action="store_true",
                      help="Save debug information for troubleshooting")
    parser.add_argument("--compare", action="store_true",
                      help="Compare with Natural PDF implementation")
    
    args = parser.parse_args()
    
    # Configure logging for Natural PDF
    if args.debug:
        configure_logging(level=logging.DEBUG)
    else:
        configure_logging(level=logging.INFO)
    
    print(f"Document: {args.pdf_path}")
    print(f"Question: {args.question}")
    
    # Run direct pdfplumber QA
    print("\n=== Direct pdfplumber implementation ===")
    result = pdfplumber_qa(args.pdf_path, args.question, debug=args.debug)
    
    if result.get("found", False):
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
    else:
        print(f"No answer found: {result.get('error', '')}")
    
    # Compare with Natural PDF if requested
    if args.compare:
        print("\n=== Natural PDF implementation ===")
        
        # Use Natural PDF
        pdf = PDF(args.pdf_path)
        page = pdf.pages[0]
        
        # Ask the question
        natural_result = page.ask(args.question, debug=args.debug)
        
        if natural_result.get("found", False):
            print(f"Answer: {natural_result['answer']}")
            print(f"Confidence: {natural_result['confidence']:.2f}")
            
            # Highlight the answer
            if natural_result.get("source_elements"):
                for element in natural_result["source_elements"]:
                    element.highlight(color=(1, 0.5, 0, 0.5))
                
                # Save the image
                page.save_image("output/natural_pdf_answer.png")
                print("Saved highlighted answer to output/natural_pdf_answer.png")
        else:
            print(f"No answer found: {natural_result.get('error', '')}")
        
        # Compare results
        if result.get("found", False) and natural_result.get("found", False):
            print("\n=== Comparison ===")
            print(f"Direct answer: '{result['answer']}' (confidence: {result['confidence']:.2f})")
            print(f"Natural PDF answer: '{natural_result['answer']}' (confidence: {natural_result['confidence']:.2f})")
            
            # Calculate similarity
            if result['answer'] == natural_result['answer']:
                print("Results match exactly!")
            else:
                print("Results differ.")

if __name__ == "__main__":
    main()