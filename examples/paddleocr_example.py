"""
Example script demonstrating the PaddleOCR integration.
"""
import os
import sys
from PIL import Image
import numpy as np

# Add the project directory to the path to import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natural_pdf import PDF

# Select a PDF file to test
PDF_FILE = "./pdfs/HARRY ROQUE_redacted.pdf"
if not os.path.exists(PDF_FILE):
    PDF_FILE = "./pdfs/01-practice.pdf"  # Fallback to another file if needed

def basic_paddleocr_example():
    """Basic example using PaddleOCR integration."""
    print("\n=== Basic PaddleOCR Example ===")
    
    # Create a PDF with the PaddleOCR engine
    print("Creating PDF with PaddleOCR engine...")
    pdf = PDF(
        PDF_FILE, 
        ocr={
            "enabled": True,
            "languages": ["en"],
            "min_confidence": 0.5,
        },
        ocr_engine="paddleocr"
    )
    
    # Get the first page
    page = pdf.pages[0]
    
    # Extract OCR elements explicitly
    print("\nExtracting OCR elements...")
    ocr_elements = page.extract_ocr_elements()
    print(f"Found {len(ocr_elements)} OCR text elements")
    
    # Print the first few elements
    for i, element in enumerate(ocr_elements[:5]):
        print(f"Element {i+1}: '{element.text}' (Confidence: {element.confidence:.2f})")
    
    # Extract text with OCR applied automatically
    print("\nExtracting text with auto OCR...")
    text = page.extract_text(ocr=True)
    
    # Print a snippet of the extracted text
    print(f"Extracted text length: {len(text)}")
    print(f"First 100 characters: {text[:100]}")
    
    # Clean up
    pdf.close()
    print("Basic PaddleOCR example complete")

def advanced_paddleocr_example():
    """Advanced example showing more PaddleOCR features."""
    print("\n=== Advanced PaddleOCR Example ===")
    
    # Create a PDF with detailed PaddleOCR configuration
    print("Creating PDF with detailed PaddleOCR configuration...")
    pdf = PDF(
        PDF_FILE, 
        ocr={
            "enabled": True,
            "languages": ["en"],
            "min_confidence": 0.3,  # Lower threshold to catch more text
            "model_settings": {
                # PaddleOCR-specific settings
                "use_angle_cls": False,
                "rec_batch_num": 6,
                "cls": False,
                "det_db_thresh": 0.3,
                "det_db_box_thresh": 0.5,
                "det_limit_side_len": 2000  # Support larger images
            }
        },
        ocr_engine="paddleocr"
    )
    
    # Create output directory for highlighted images
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the first page
    page = pdf.pages[0]
    
    # Extract OCR elements
    print("\nExtracting OCR elements with detailed configuration...")
    ocr_elements = page.extract_ocr_elements()
    print(f"Found {len(ocr_elements)} OCR text elements")
    
    # Highlight OCR elements with confidence scores
    print("\nHighlighting OCR elements...")
    for i, elem in enumerate(ocr_elements):
        # Use different colors based on confidence
        if elem.confidence >= 0.8:
            color = (0, 1, 0, 0.3)  # Green for high confidence
        elif elem.confidence >= 0.5:
            color = (1, 1, 0, 0.3)  # Yellow for medium confidence
        else:
            color = (1, 0, 0, 0.3)  # Red for low confidence
            
        # Label includes confidence score
        elem.highlight(
            color=color,
            label=f"OCR ({elem.confidence:.2f})"
        )
    
    # Save highlighted page
    highlight_path = os.path.join(output_dir, "paddleocr_highlights.png")
    page.to_image(path=highlight_path, show_labels=True)
    print(f"Saved highlighted image to {highlight_path}")
    
    # Filter OCR elements by confidence
    high_confidence = [e for e in ocr_elements if e.confidence >= 0.7]
    print(f"\nHigh confidence elements ({len(high_confidence)}): ")
    for i, elem in enumerate(high_confidence[:3]):
        print(f"  {i+1}. '{elem.text}' (Confidence: {elem.confidence:.2f})")
    
    # Clean up
    pdf.close()
    print("Advanced PaddleOCR example complete")

def ocr_engine_comparison():
    """Compare EasyOCR and PaddleOCR on the same document."""
    print("\n=== OCR Engine Comparison ===")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Test with EasyOCR
    print("\nUsing EasyOCR...")
    easy_pdf = PDF(
        PDF_FILE, 
        ocr={"enabled": True, "languages": ["en"]},
        ocr_engine="easyocr"
    )
    page = easy_pdf.pages[0]
    
    # Time the OCR process
    import time
    start_time = time.time()
    easy_elements = page.extract_ocr_elements()
    easy_time = time.time() - start_time
    print(f"EasyOCR found {len(easy_elements)} text elements in {easy_time:.2f} seconds")
    
    # Save a sample
    with open(os.path.join(output_dir, "easyocr_sample.txt"), "w") as f:
        for i, elem in enumerate(easy_elements[:20]):
            f.write(f"{i+1}. '{elem.text}' (Confidence: {elem.confidence:.2f})\n")
    
    # Clean up
    easy_pdf.close()
    
    # Test with PaddleOCR
    print("\nUsing PaddleOCR...")
    paddle_pdf = PDF(
        PDF_FILE, 
        ocr={"enabled": True, "languages": ["en"]},
        ocr_engine="paddleocr"
    )
    page = paddle_pdf.pages[0]
    
    # Time the OCR process
    start_time = time.time()
    paddle_elements = page.extract_ocr_elements()
    paddle_time = time.time() - start_time
    print(f"PaddleOCR found {len(paddle_elements)} text elements in {paddle_time:.2f} seconds")
    
    # Save a sample
    with open(os.path.join(output_dir, "paddleocr_sample.txt"), "w") as f:
        for i, elem in enumerate(paddle_elements[:20]):
            f.write(f"{i+1}. '{elem.text}' (Confidence: {elem.confidence:.2f})\n")
    
    # Clean up
    paddle_pdf.close()
    
    # Compare results
    print("\nComparison Results:")
    print(f"EasyOCR: {len(easy_elements)} elements in {easy_time:.2f} seconds")
    print(f"PaddleOCR: {len(paddle_elements)} elements in {paddle_time:.2f} seconds")
    print(f"Speed difference: {(easy_time / paddle_time if paddle_time > 0 else 0):.2f}x")
    
    print("\nSample results saved to:")
    print(f"  - {os.path.join(output_dir, 'easyocr_sample.txt')}")
    print(f"  - {os.path.join(output_dir, 'paddleocr_sample.txt')}")
    
    print("OCR engine comparison complete")

if __name__ == "__main__":
    try:
        # Check if PaddleOCR is available
        import paddleocr
        print("PaddleOCR is available, running examples...")
        
        # Get command line arguments if any
        import sys
        if len(sys.argv) > 1:
            example = sys.argv[1].lower()
            if example == "basic":
                basic_paddleocr_example()
            elif example == "advanced":
                advanced_paddleocr_example()
            elif example == "compare":
                ocr_engine_comparison()
            else:
                print(f"Unknown example: {example}")
                print("Available examples: basic, advanced, compare")
        else:
            # Run all examples
            basic_paddleocr_example()
            advanced_paddleocr_example()
            ocr_engine_comparison()
        
    except ImportError:
        print("PaddleOCR is not installed. Please install it with: pip install paddlepaddle paddleocr")
    except Exception as e:
        print(f"Error in PaddleOCR examples: {e}")
        import traceback
        traceback.print_exc()