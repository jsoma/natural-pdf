import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF

# Get absolute path for the PDF
script_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(script_dir)
pdf_path = os.path.join(root_dir, "pdfs", "HARRY ROQUE_redacted.pdf")

print(f"Loading PDF: {pdf_path}")

# Example 1: Initialize PDF with flattened OCR parameters
pdf = PDF(pdf_path, ocr={
    "enabled": True,
    "languages": ["en"],
    "min_confidence": 0.3,
    # OCR parameters directly in config root:
    "text_threshold": 0.1,    # Was previously in detection_params
    "link_threshold": 0.1,    # Was previously in detection_params
    "paragraph": True,        # Was previously in recognition_params
    "detail": 1               # Was previously in recognition_params
})

# Use a specific page
page = pdf.pages[3]

# Example 2: Apply OCR with flattened parameters
print("\nApplying OCR with flattened parameters")
ocr_elements = page.apply_ocr(
    # Direct parameters:
    text_threshold=0.15,
    link_threshold=0.15,
    mag_ratio=1.5,
    canvas_size=1024,
    batch_size=4
)

print(f"Found {len(ocr_elements)} OCR text elements")

# Print sample of OCR results
print("\nSample OCR results:")
for i, elem in enumerate(ocr_elements[:5]):
    print(f"{i+1}. '{elem.text}' (conf: {elem.confidence:.2f})")
    if i >= 4:
        break

# Example 3: Extract text with OCR using flattened parameters
print("\nExtracting text with OCR using flattened parameters")
text = page.extract_text(ocr={
    "enabled": True,
    "min_confidence": 0.2,
    # Direct parameters:
    "text_threshold": 0.2,
    "contrast_ths": 0.05
})

# Display first 100 characters of text
print(f"\nExtracted text (first 100 chars):")
print(text[:100] + "...")

# Create output directory if it doesn't exist
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Highlight OCR elements
for elem in ocr_elements[:10]:
    elem.highlight(label=f"OCR: {elem.text}")

# Save image
output_path = os.path.join(output_dir, "ocr_simplified.png")
print(f"\nSaving highlighted image to: {output_path}")
page.to_image(path=output_path, show_labels=True)

print("\nTest completed successfully!")