"""
Direct OCR test script to debug OCR issues.
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

def test_direct_ocr():
    """Test OCR engines directly."""
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Direct test with EasyOCR
    print("\n=== Direct test with EasyOCR ===")
    try:
        import easyocr
        # Use the provided PDF file
        with PDF(PDF_FILE) as pdf:
            # Get the first page
            page = pdf.pages[0]
            # Convert to image
            image = page.to_image()
            image_path = os.path.join(output_dir, "easyocr_test_input.png")
            image.save(image_path)
            print(f"Saved image to {image_path}")
            
            # Run EasyOCR directly
            reader = easyocr.Reader(['en'])
            results = reader.readtext(np.array(image))
            print(f"EasyOCR found {len(results)} text elements")
            
            # Print results
            for i, (bbox, text, conf) in enumerate(results[:5]):
                print(f"Result {i+1}: '{text}' (Confidence: {conf:.2f})")
            
            print("EasyOCR direct test successful")
    except ImportError:
        print("EasyOCR not available")
    except Exception as e:
        print(f"Error in EasyOCR direct test: {e}")
        import traceback
        traceback.print_exc()
        
    # Direct test with PaddleOCR
    print("\n=== Direct test with PaddleOCR ===")
    try:
        import paddleocr
        # Use the provided PDF file
        with PDF(PDF_FILE) as pdf:
            # Get the first page
            page = pdf.pages[0]
            # Convert to image
            image = page.to_image()
            image_path = os.path.join(output_dir, "paddleocr_test_input.png")
            image.save(image_path)
            print(f"Saved image to {image_path}")
            
            # Run PaddleOCR directly
            reader = paddleocr.PaddleOCR(lang='en')
            results = reader.ocr(np.array(image), cls=False)
            
            if results is not None and len(results) > 0:
                page_result = results[0] if isinstance(results[0], list) else results
                print(f"PaddleOCR found {len(page_result)} text elements")
                
                # Print results
                for i, detection in enumerate(page_result[:5]):
                    if len(detection) >= 2:
                        bbox = detection[0]
                        text_conf = detection[1]
                        text = text_conf[0] if isinstance(text_conf, tuple) and len(text_conf) >= 2 else str(text_conf)
                        conf = text_conf[1] if isinstance(text_conf, tuple) and len(text_conf) >= 2 else 1.0
                        print(f"Result {i+1}: '{text}' (Confidence: {conf:.2f})")
            else:
                print(f"PaddleOCR returned no results: {results}")
            
            print("PaddleOCR direct test complete")
    except ImportError:
        print("PaddleOCR not available")
    except Exception as e:
        print(f"Error in PaddleOCR direct test: {e}")
        import traceback
        traceback.print_exc()
        
def test_library_ocr():
    """Test OCR integration with the library."""
    
    print("\n=== Test library integration with EasyOCR ===")
    try:
        # Create a PDF with explicit OCR config
        with PDF(PDF_FILE, ocr={"enabled": True, "languages": ["en"]}, ocr_engine="easyocr") as pdf:
            # Get the first page
            page = pdf.pages[0]
            
            # Extract text with OCR
            print("Running OCR through library...")
            elements = page.extract_ocr_elements()
            
            print(f"Library OCR found {len(elements)} text elements")
            
            # Print results
            for i, elem in enumerate(elements[:5]):
                print(f"Result {i+1}: '{elem.text}' (Confidence: {elem.confidence:.2f})")
                
            print("Library OCR with EasyOCR test complete")
    except Exception as e:
        print(f"Error in library OCR with EasyOCR test: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n=== Test library integration with PaddleOCR ===")
    try:
        # Create a PDF with explicit OCR config
        with PDF(PDF_FILE, ocr={"enabled": True, "languages": ["en"]}, ocr_engine="paddleocr") as pdf:
            # Get the first page
            page = pdf.pages[0]
            
            # Extract text with OCR
            print("Running OCR through library...")
            elements = page.extract_ocr_elements()
            
            print(f"Library OCR found {len(elements)} text elements")
            
            # Print results
            for i, elem in enumerate(elements[:5]):
                print(f"Result {i+1}: '{elem.text}' (Confidence: {elem.confidence:.2f})")
                
            print("Library OCR with PaddleOCR test complete")
    except Exception as e:
        print(f"Error in library OCR with PaddleOCR test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_ocr()
    test_library_ocr()