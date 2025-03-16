"""
OCR example using PaddleOCR.

This example demonstrates how to use OCR to extract text from PDF documents,
both for whole pages and specific regions.

Note: This example requires the 'paddleocr' package:
pip install paddlepaddle paddleocr
"""
import os
import sys
from natural_pdf import PDF

# Get the current directory of this script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (project root)
root_dir = os.path.dirname(script_dir)
# Default PDF path (replace with a scanned document path for better results)
default_pdf = os.path.join(root_dir, "pdfs", "HARRY ROQUE_redacted.pdf")
# Output directory
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

print("OCR Example")
print("==========")

# 1. Loading a PDF with OCR enabled
print("\n1. Loading PDF with OCR enabled")
pdf = PDF(default_pdf, ocr={
    "enabled": "auto",  # Auto mode: only use OCR when necessary
    "languages": ["en"],
    # For more options, see OCR-NOTES.md
})

# 2. Extract text from a page with auto OCR
page = pdf.pages[0]
print(f"\n2. Extracting text from page {page.number} with auto OCR")
text = page.extract_text()
print(f"Extracted {len(text)} characters.")
print("First 150 characters:\n", text[:150] + "..." if len(text) > 150 else text)

# 3. Force OCR on a page
print("\n3. Force OCR on a page")
ocr_text = page.extract_text(ocr=True)  # Force OCR regardless of existing text
print(f"Extracted {len(ocr_text)} characters with forced OCR.")
print("First 150 characters:\n", ocr_text[:150] + "..." if len(ocr_text) > 150 else ocr_text)

# 4. Extract OCR elements directly
print("\n4. Extracting OCR elements directly")
ocr_elements = page.extract_ocr_elements()
print(f"Found {len(ocr_elements)} OCR text elements.")
for i, elem in enumerate(ocr_elements[:3]):  # Show first 3 elements
    print(f"  Element {i+1}: '{elem.text}' (confidence: {elem.confidence:.2f})")

# 5. Apply OCR to a specific region
print("\n5. Applying OCR to a specific region")
# Create a region (adjust coordinates for your PDF)
region = page.create_region(100, 100, 400, 200)  # x0, y0, x1, y1
region.highlight(label="OCR Region")

# Apply OCR to this region
region_elements = region.apply_ocr()
print(f"Found {len(region_elements)} OCR text elements in the region.")

# Extract text from the region (uses OCR since we already applied it)
region_text = region.extract_text()
print(f"Region text: '{region_text[:50]}...'" if len(region_text) > 50 else f"Region text: '{region_text}'")

# 6. Finding OCR text elements with selectors
print("\n6. Finding OCR text elements with selectors")
# Find OCR elements with specific properties
high_confidence_ocr = page.find_all('text[source=ocr][confidence>=0.8]')
print(f"Found {len(high_confidence_ocr)} high-confidence OCR elements.")

# Find OCR elements containing specific text
matching_ocr = page.find_all('text[source=ocr]:contains("the")')
print(f"Found {len(matching_ocr)} OCR elements containing 'the'.")

# 7. Visualize OCR results
print("\n7. Visualizing OCR results")
# Highlight all OCR elements
for elem in ocr_elements:
    elem.highlight(label=f"OCR ({elem.confidence:.2f})")

# Save the highlighted page
output_path = os.path.join(output_dir, "ocr_results.png")
page.to_image(path=output_path, show_labels=True)
print(f"Saved visualization to {output_path}")
print("\nDone!")