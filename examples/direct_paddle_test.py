"""
Direct test of PaddlePaddle's PPStructure functionality.

This script bypasses our library and directly uses paddleocr to test layout detection.
"""
import os
import sys
from pathlib import Path
import cv2

try:
    from paddleocr import PPStructure
except ImportError:
    print("PaddleOCR not installed. Run: pip install paddlepaddle paddleocr")
    sys.exit(1)

# Get the current directory of this script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (project root)
root_dir = os.path.dirname(script_dir)
# Default PDF path
default_pdf = os.path.join(root_dir, "pdfs", "2019 Statistics.pdf")

# Check command line args
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    # Convert first page of PDF to image since PPStructure needs an image
    import fitz  # PyMuPDF
    pdf_path = default_pdf
    print(f"Converting first page of {pdf_path} to image...")
    
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[0]
    
    # Render page at higher resolution
    zoom = 2.0  # Increase resolution
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    # Save as image
    image_path = os.path.join(root_dir, "output", "direct_paddle_test.png")
    pix.save(image_path)
    print(f"Saved image to {image_path}")

# Ensure image exists
if not os.path.exists(image_path):
    print(f"Image doesn't exist: {image_path}")
    sys.exit(1)

print(f"Running PPStructure on {image_path}...")

# Initialize PP-Structure with minimal settings
table_engine = PPStructure(show_log=True)

try:
    # Run layout analysis
    result = table_engine(image_path)
    
    # Print results
    print(f"Found {len(result)} layout regions:")
    for i, region in enumerate(result):
        region_type = region.get('type', 'unknown')
        bbox = region.get('bbox', [])
        confidence = region.get('score', 0)
        print(f"{i+1}. Type: {region_type}, Confidence: {confidence:.4f}, BBox: {bbox}")
        
        # Check for OCR text inside the region
        if 'res' in region:
            if isinstance(region['res'], dict) and 'text' in region['res']:
                print(f"   Text: {region['res']['text'][:50]}...")
            elif isinstance(region['res'], dict) and 'cells' in region['res']:
                print(f"   Table with {len(region['res']['cells'])} cells")
            else:
                print(f"   Has result data: {type(region['res'])}")
    
    # Try directly with PaddleOCR for layout analysis
    from paddleocr import PaddleOCR
    print("\nTrying with direct PaddleOCR...")
    
    ocr_engine = PaddleOCR(lang="en", show_log=True)
    layout_result = ocr_engine.ocr(image_path, det=True, rec=True, cls=False)
    
    if layout_result:
        print(f"PaddleOCR found text elements on page 1: {len(layout_result[0])}")
        
        # Print first few elements
        for i, line in enumerate(layout_result[0][:5]):
            points = line[0]  # Coordinates: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = line[1][0]  # Text content
            confidence = line[1][1]  # Confidence score
            print(f"  {i+1}. Text: '{text}', Confidence: {confidence:.4f}")
    else:
        print("PaddleOCR found no elements")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()