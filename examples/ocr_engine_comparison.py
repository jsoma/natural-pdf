"""
OCR Engine Comparison Example.

This example compares the performance of different OCR engines with natural-pdf.

Requires both EasyOCR and PaddleOCR to be installed:
pip install easyocr
pip install paddlepaddle paddleocr
"""
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF
from natural_pdf.ocr import EasyOCREngine, PaddleOCREngine

# Get the current directory of this script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (project root)
root_dir = os.path.dirname(script_dir)
# Default PDF path (replace with a scanned document path for better results)
default_pdf = os.path.join(root_dir, "pdfs", "HARRY ROQUE_redacted.pdf")
# Output directory
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

print("OCR Engine Comparison")
print("====================")

# Check if both OCR engines are available
easyocr_available = False
paddleocr_available = False

try:
    import easyocr
    easyocr_available = True
    print("EasyOCR is available.")
except ImportError:
    print("EasyOCR is not available. Some comparisons will be skipped.")

try:
    import paddleocr
    import paddle
    paddleocr_available = True
    print("PaddleOCR is available.")
except ImportError:
    print("PaddleOCR is not available. Some comparisons will be skipped.")

if not easyocr_available and not paddleocr_available:
    print("No OCR engines available. Please install at least one OCR engine.")
    sys.exit(1)

# Common OCR configuration for fair comparison
ocr_config = {
    "languages": ["en"],
    "device": "cpu",
    "min_confidence": 0.3
}

# Set up testing information
engines = []
if easyocr_available:
    engines.append(("EasyOCR", "easyocr"))
if paddleocr_available:
    engines.append(("PaddleOCR", "paddleocr"))

# Function to run OCR with an engine and measure performance
def test_engine(engine_name, engine_id, page_number=0):
    print(f"\nTesting {engine_name}:")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Load PDF with this engine
        print(f"  Loading PDF with {engine_name} engine...")
        pdf = PDF(default_pdf, ocr_engine=engine_id, ocr=ocr_config)
        
        # Get the specified page
        print(f"  Accessing page {page_number}...")
        page = pdf.pages[page_number]
        
        # Check if OCR is properly configured
        if hasattr(pdf, '_ocr_engine'):
            print(f"  OCR engine: {pdf._ocr_engine.__class__.__name__}")
            print(f"  OCR config: {pdf._ocr_config}")
        else:
            print("  Warning: PDF does not have _ocr_engine attribute")
        
        # Force OCR explicitly
        print(f"  Extracting OCR elements explicitly...")
        ocr_elements = page.extract_ocr_elements()
        print(f"  Found {len(ocr_elements)} OCR elements")
        
        if len(ocr_elements) == 0:
            print("  Warning: No OCR elements found - trying to debug")
            # Try direct extract_text with OCR flag
            print("  Trying page.extract_text(ocr=True)...")
            text = page.extract_text(ocr=True)
            print(f"  Extract_text with ocr=True returned {len(text)} characters")
        else:
            # Extract text
            print(f"  Extracting text...")
            text = page.extract_text()
            print(f"  Extracted {len(text)} characters")
        
        extraction_time = time.time() - start_time
        
        # Calculate average confidence
        avg_confidence = sum(elem.confidence for elem in ocr_elements) / len(ocr_elements) if ocr_elements else 0
        
        # Create a highlighted image
        print(f"  Creating highlighted image...")
        page.clear_highlights()
        for elem in ocr_elements:
            if elem.confidence >= 0.7:
                color = (0, 204, 0, 76)  # Green for high confidence
            elif elem.confidence >= 0.5:
                color = (230, 230, 0, 76)  # Yellow for medium confidence
            else:
                color = (204, 0, 0, 76)  # Red for low confidence
            
            elem.highlight(label=f"{engine_name}", color=color)
        
        # Save the image
        output_path = os.path.join(output_dir, f"{engine_name.lower()}_results.png")
        page.to_image(path=output_path, show_labels=True)
        
        # Return results
        return {
            "engine": engine_name,
            "extraction_time": extraction_time,
            "text_length": len(text),
            "element_count": len(ocr_elements),
            "avg_confidence": avg_confidence,
            "output_path": output_path
        }
    
    except Exception as e:
        print(f"  Error during {engine_name} test: {e}")
        import traceback
        traceback.print_exc()
        return {
            "engine": engine_name,
            "extraction_time": 0,
            "text_length": 0,
            "element_count": 0,
            "avg_confidence": 0,
            "output_path": "error",
            "error": str(e)
        }

# Run tests for each available engine
results = []
for engine_name, engine_id in engines:
    result = test_engine(engine_name, engine_id)
    results.append(result)
    
    # Print some stats
    print(f"  Extraction time: {result['extraction_time']:.2f} seconds")
    print(f"  Text length: {result['text_length']} characters")
    print(f"  Element count: {result['element_count']} elements")
    print(f"  Average confidence: {result['avg_confidence']:.2f}")
    print(f"  Output image: {result['output_path']}")

# Compare results
if len(results) > 1:
    print("\nComparison Results:")
    print(f"{'Engine':<10} {'Time (s)':<10} {'Text Len':<10} {'Elements':<10} {'Avg Conf':<10}")
    print(f"{'-'*60}")
    for result in results:
        print(f"{result['engine']:<10} {result['extraction_time']:.2f}s      {result['text_length']:<10} {result['element_count']:<10} {result['avg_confidence']:.2f}")
    
    # Highlight differences
    fastest = min(results, key=lambda x: x['extraction_time'])
    most_elements = max(results, key=lambda x: x['element_count'])
    highest_confidence = max(results, key=lambda x: x['avg_confidence'])
    
    print(f"\nFastest engine: {fastest['engine']} ({fastest['extraction_time']:.2f}s)")
    print(f"Most elements: {most_elements['engine']} ({most_elements['element_count']} elements)")
    print(f"Highest confidence: {highest_confidence['engine']} ({highest_confidence['avg_confidence']:.2f})")

# Additional comparison with engine-specific optimizations
print("\nRunning comparison with engine-specific optimizations:")

# Custom configurations for each engine
if easyocr_available and paddleocr_available:
    # EasyOCR with customized settings
    easyocr_custom = PDF(default_pdf, 
                         ocr_engine="easyocr",
                         ocr={
                             "languages": ["en"],
                             "device": "cpu",
                             "min_confidence": 0.3,
                             "model_settings": {
                                 "detail": 1,
                                 "paragraph": False,
                                 "contrast_ths": 0.05,
                                 "text_threshold": 0.5
                             }
                         })
    
    # PaddleOCR with customized settings
    paddleocr_custom = PDF(default_pdf,
                          ocr_engine="paddleocr",
                          ocr={
                              "languages": ["en"],
                              "device": "cpu",
                              "min_confidence": 0.3,
                              "model_settings": {
                                  "use_angle_cls": True,
                                  "det_db_thresh": 0.2,
                                  "det_db_box_thresh": 0.3
                              }
                          })
    
    # Compare text extraction
    easyocr_text = easyocr_custom.pages[0].extract_text()
    paddleocr_text = paddleocr_custom.pages[0].extract_text()
    
    print(f"\nOptimized EasyOCR text length: {len(easyocr_text)}")
    print(f"Optimized PaddleOCR text length: {len(paddleocr_text)}")
    
    # Compare element counts
    easyocr_elements = easyocr_custom.pages[0].extract_ocr_elements()
    paddleocr_elements = paddleocr_custom.pages[0].extract_ocr_elements()
    
    print(f"Optimized EasyOCR element count: {len(easyocr_elements)}")
    print(f"Optimized PaddleOCR element count: {len(paddleocr_elements)}")

print("\nDone!")