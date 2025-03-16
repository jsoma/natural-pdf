"""
OCR Visualization Test

This example demonstrates the OCR text visualization feature using PaddleOCR.
"""
import os
import sys
from pathlib import Path

# Add project directory to the path
script_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

# Import the library
from natural_pdf import PDF

# Set up paths
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Use a PDF that typically needs OCR
pdf_path = os.path.join(root_dir, "pdfs", "needs-ocr.pdf")
if not os.path.exists(pdf_path):
    # Fallback to other PDFs if the needs-ocr.pdf doesn't exist
    pdf_path = os.path.join(root_dir, "pdfs", "HARRY ROQUE_redacted.pdf")
    if not os.path.exists(pdf_path):
        pdf_path = os.path.join(root_dir, "pdfs", "01-practice.pdf")

print("OCR Visualization Test")
print("=====================")
print(f"Using PDF: {pdf_path}")

# Initialize the PDF with PaddleOCR engine
try:
    # Try with PaddleOCR first
    pdf = PDF(
        pdf_path,
        ocr_engine="paddleocr",
        ocr={
            "enabled": True,
            "languages": ["en"],
            "min_confidence": 0.3
        }
    )
    print("Using PaddleOCR engine")
except Exception as e:
    print(f"PaddleOCR initialization failed: {e}")
    print("Falling back to EasyOCR")
    # Fall back to EasyOCR
    pdf = PDF(
        pdf_path,
        ocr_engine="easyocr",
        ocr={
            "enabled": True,
            "languages": ["en"],
            "min_confidence": 0.3
        }
    )

# Access the first page
page = pdf.pages[0]

# Force OCR text extraction
print("\nExtracting text with OCR...")
text = page.extract_text(ocr=True)
print(f"Extracted {len(text)} characters of text")
if text:
    print(f"First 100 chars: {text[:100]}...")

# Extract OCR elements
print("\nExtracting OCR elements...")
ocr_elements = page.extract_ocr_elements()
print(f"Found {len(ocr_elements)} OCR elements")

# Create highlight visualization
print("\nCreating highlight visualization...")
for elem in ocr_elements:
    # Use color based on confidence - with full RGB values (0-255) and higher opacity
    if elem.confidence >= 0.8:
        color = (0, 255, 0, 180)  # Green for high confidence (more visible)
    elif elem.confidence >= 0.5:
        color = (255, 255, 0, 180)  # Yellow for medium confidence
    else:
        color = (255, 0, 0, 180)  # Red for low confidence
        
    # Add highlight with confidence as label
    elem.highlight(color=color, label=f"OCR ({elem.confidence:.2f})")

# Save image with highlights only
highlight_path = os.path.join(output_dir, "ocr_visualization_highlights.png")
page.to_image(path=highlight_path, show_labels=True)
print(f"Saved highlighted image to {highlight_path}")

# Now use the OCR text rendering feature
if len(ocr_elements) > 0:
    print("\nCreating rendered OCR text visualization...")
    
    # Save image with OCR text rendered
    ocr_text_path = os.path.join(output_dir, "ocr_visualization_text.png")
    try:
        page.to_image(path=ocr_text_path, show_labels=True, render_ocr=True)
        print(f"Saved OCR text rendering to {ocr_text_path}")
    except ValueError as e:
        print(f"Error rendering OCR text: {e}")
        
    # Clear highlights and render only OCR text
    print("\nCreating clean OCR text visualization...")
    page.clear_highlights() 
    
    # Save clean image with only OCR text
    clean_text_path = os.path.join(output_dir, "ocr_visualization_clean.png")
    try:
        page.to_image(path=clean_text_path, render_ocr=True)
        print(f"Saved clean OCR text rendering to {clean_text_path}")
    except ValueError as e:
        print(f"Error rendering clean OCR text: {e}")
else:
    print("\nNo OCR elements found to render.")

print("\nTest complete!")