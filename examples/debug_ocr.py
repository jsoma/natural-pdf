"""
Debug OCR issues.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF
from natural_pdf.ocr import EasyOCREngine

# Get current directory
script_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(script_dir)
default_pdf = os.path.join(root_dir, "pdfs", "HARRY ROQUE_redacted.pdf")
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

print("OCR Debug Test")
print("=============")

# Check if OCR engines are available
try:
    import easyocr
    print("EasyOCR is available.")
except ImportError:
    print("EasyOCR is not available.")
    
try:
    import paddleocr
    import paddle
    print("PaddleOCR is available.")
except ImportError:
    print("PaddleOCR is not available.")

# Test with EasyOCR directly (explicit configuration)
print("\n1. Testing with explicit EasyOCR engine and forced enabled")
pdf = PDF(default_pdf, 
          ocr_engine="easyocr", 
          ocr={
              "enabled": True,
              "languages": ["en"],
              "min_confidence": 0.3
          })

# Get the page
print("Getting page...")
page = pdf.pages[0]

# Print OCR config
print(f"PDF OCR config: {pdf._ocr_config}")
print(f"OCR engine type: {type(pdf._ocr_engine)}")

# Generate page image for debugging
print("Generating debug image of the page...")
img = page.to_image()
img_path = os.path.join(output_dir, "debug_page_image.png")
img.save(img_path)
print(f"Saved page image to {img_path}")

# Force OCR extraction
print("Forcing OCR extraction...")
ocr_elements = page.extract_ocr_elements()
print(f"Extracted {len(ocr_elements)} OCR elements")

# Print details of first few elements if any
if ocr_elements:
    for i, elem in enumerate(ocr_elements[:3]):
        print(f"Element {i+1}: '{elem.text}' (conf: {elem.confidence:.2f})")
else:
    print("No OCR elements found!")

# Extract text with OCR
print("Extracting text with OCR=True...")
text = page.extract_text(ocr=True)
print(f"Extracted {len(text)} characters of text")
print(f"First 100 chars: {text[:100]}...")

# Create a debug image
print("Creating debug visualization...")
page.clear_highlights()
for elem in ocr_elements:
    elem.highlight(label=f"OCR ({elem.confidence:.2f})")
    
output_path = os.path.join(output_dir, "ocr_debug.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved debug image to {output_path}")

print("\nTest complete!")