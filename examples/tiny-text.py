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
pdf = PDF(pdf_path, ocr={
    "enabled": True,
    "engine": "easyocr",
    "languages": ["en"],
    "detection_params": {
        "text_threshold": 0.001,
        "mag_ratio": 3.0,  # Quadruple the magnification during detection
        "canvas_size": 5000,
    },
    "recognition_params": {
        "min_size": 4,
        "contrast_ths": 0.5
    }
})

# Create output directory if it doesn't exist
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Use a specific page
page = pdf.pages[6]
# Run document layout analysis
regions = page.analyze_layout(model='tatr')

print(f"Found {len(regions)} regions")

# # Apply OCR explicitly
# print("Applying OCR...")
ocr_elements = page.apply_ocr()
print(f"Found {len(ocr_elements)} OCR elements")

# Print some sample elements
print("\nSample OCR elements:")
for i, elem in enumerate(ocr_elements[:30]):
    print(f"{i+1}. {elem}")

# Highlight the OCR text elements
print("\nHighlighting OCR elements...")
for elem in ocr_elements:
    elem.highlight(label=f"OCR ({elem.confidence:.2f})")

output_path = os.path.join(output_dir, "ocr_highlight_all_test.png")
print(f"Saving highlight_all image to: {output_path}")
page.to_image(path=output_path, show_labels=True)

print("\nTest completed successfully!")