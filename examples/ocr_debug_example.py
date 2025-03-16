"""
OCR Debug Report Example

This example demonstrates the OCR debugging feature, which generates an interactive
HTML report for analyzing and correcting OCR results.
"""
import os
import sys
from pathlib import Path

# Get the parent directory (project root)
project_root = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(project_root))

from natural_pdf import PDF

# Output directory
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

# Default PDF path (a scanned document)
pdf_path = os.path.join(project_root, "pdfs", "HARRY ROQUE_redacted.pdf")
fallback_pdf = os.path.join(project_root, "pdfs", "needs-ocr.pdf")

# Use the first available PDF
if os.path.exists(pdf_path):
    print(f"Using PDF: {pdf_path}")
elif os.path.exists(fallback_pdf):
    pdf_path = fallback_pdf
    print(f"Using PDF: {pdf_path}")
else:
    print("No suitable PDF found. Please provide a scanned PDF.")
    sys.exit(1)

# OCR Debug Example
print("\nOCR Debug Report Example")
print("=======================")

# Load PDF with OCR enabled
print("\n1. Loading PDF with OCR enabled")
pdf = PDF(
    pdf_path, 
    ocr={
        "enabled": True,  # Enable OCR
        "languages": ["en"],
        "min_confidence": 0.3,  # Lower threshold to get more results
    },
    # Try PaddleOCR first, which often gives better results for Asian languages
    ocr_engine="paddleocr"
)

# Run OCR on the first page
print("\n2. Running OCR on the first page")
page = pdf.pages[0]
ocr_elements = page.extract_ocr_elements()
print(f"Found {len(ocr_elements)} OCR text elements")

# Generate a debug report for a single page
print("\n3. Generating OCR debug report for a single page")
page_report_path = os.path.join(output_dir, "ocr_debug_page.html")
page.debug_ocr(page_report_path)
print(f"Saved page debug report to: {page_report_path}")

# Generate a debug report for multiple pages 
print("\n4. Generating OCR debug report for multiple pages")
pdf_report_path = os.path.join(output_dir, "ocr_debug_full.html")
pdf.debug_ocr(pdf_report_path)
print(f"Saved full PDF debug report to: {pdf_report_path}")

# Generate a debug report for a page range
print("\n5. Generating OCR debug report for a page range")
if len(pdf.pages) > 1:
    page_range_report_path = os.path.join(output_dir, "ocr_debug_range.html")
    page_range = pdf.pages[0:min(3, len(pdf.pages))]
    page_range.debug_ocr(page_range_report_path)
    print(f"Saved page range debug report to: {page_range_report_path}")
else:
    print("PDF has only one page, skipping page range example")

print("\nDone! The debug reports have been saved to the output directory.")
print("You can open them in a web browser to interactively review the OCR results.")
print("The reports allow you to:")
print("- Filter results by confidence score")
print("- Search for specific text")
print("- Sort results by different criteria")
print("- Edit/correct OCR text")
print("- Export corrected text as JSON for further processing")