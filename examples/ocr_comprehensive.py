"""
Comprehensive OCR example.

This example demonstrates the full range of OCR capabilities
in natural-pdf, including:
1. Multiple configuration formats
2. Auto mode
3. Region-specific OCR
4. Selector filtering by source and confidence
5. Visualization of OCR results
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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

print("Comprehensive OCR Example")
print("========================")

# 1. Demonstrate different OCR configuration formats
print("\n1. Different OCR configuration formats")

# Simple flag
pdf_simple = PDF(default_pdf, ocr=True)
print("  - Simple flag (ocr=True): OCR enabled with defaults")

# Auto mode
pdf_auto = PDF(default_pdf, ocr="auto")
print("  - Auto mode (ocr='auto'): OCR applied only when needed")

# Language list
pdf_langs = PDF(default_pdf, ocr=["en"])
print("  - Language list (ocr=['en']): English language OCR")

# Detailed config with detection parameters
pdf_detailed = PDF(default_pdf, ocr={
    "enabled": True,
    "engine": "easyocr",
    "languages": ["en"],
    "min_confidence": 0.6,
    "paragraph": False,
    # Text detection parameters for CRAFT
    "detection_params": {
        "text_threshold": 0.1,  # Lower threshold to detect more text (default is 0.7)
        "low_text": 0.3,        # Lower threshold for text box filtering (default is 0.4)
        "link_threshold": 0.3,  # Lower threshold for link between text (default is 0.4)
        "canvas_size": 2560,    # Maximum image size
        "mag_ratio": 1.5        # Image magnification ratio (increase for better detection)
    },
    # Optional recognition parameters
    "recognition_params": {
        "decoder": "greedy",
        "batch_size": 4,
        "contrast_ths": 0.05    # Lower contrast threshold
    }
})
print("  - Detailed config: Custom parameters including text_threshold=0.1")

# 2. Auto mode demonstration
print("\n2. Auto mode OCR")
pdf = PDF(default_pdf, ocr="auto")
page = pdf.pages[0]

# Extract text with auto OCR
text = page.extract_text()
print(f"  - Auto mode extracted {len(text)} characters")
print(f"  - First 100 chars: {text[:100]}...")

# 3. Explicit OCR application
print("\n3. Explicit OCR application")
ocr_elements = page.apply_ocr()
print(f"  - Found {len(ocr_elements)} OCR elements")
print("  - Sample OCR elements:")
for i, elem in enumerate(ocr_elements[:3]):
    print(f"    {i+1}. '{elem.text}' (confidence: {elem.confidence:.2f})")

# 4. OCR with confidence filtering
print("\n4. OCR confidence filtering")
high_conf = page.find_all('text[source=ocr][confidence>=0.8]')
print(f"  - Found {len(high_conf)} high-confidence OCR elements")
low_conf = page.find_all('text[source=ocr][confidence<0.6]')
print(f"  - Found {len(low_conf)} low-confidence OCR elements")

# 5. Content-based OCR filtering
print("\n5. Content-based OCR filtering")
contains_a = page.find_all('text[source=ocr]:contains("a")')
print(f"  - Found {len(contains_a)} OCR elements containing 'a'")

# 6. Region-specific OCR
print("\n6. Region-specific OCR")
# Create a region - adjust coordinates if needed for your document
region = page.create_region(100, 100, 400, 200)
region_elems = region.apply_ocr()
print(f"  - Applied OCR to region, found {len(region_elems)} elements")
region_text = region.extract_text()
print(f"  - Region text: '{region_text[:50]}...'")

# 7. OCR visualization
print("\n7. OCR visualization")
# Clear any existing highlights
page.clear_highlights()

# Highlight all OCR elements with confidence displayed
print("  - Highlighting all OCR elements with confidence scores")
for elem in ocr_elements:
    # Color coding based on confidence - using integer RGB values (0-255)
    if elem.confidence >= 0.8:
        color = (0, 204, 0, 76)  # Green for high confidence
    elif elem.confidence >= 0.6:
        color = (230, 230, 0, 76)  # Yellow for medium confidence
    else:
        color = (204, 0, 0, 76)  # Red for low confidence
        
    elem.highlight(label=f"OCR ({elem.confidence:.2f})", color=color)

# Save the image
output_path = os.path.join(output_dir, "ocr_confidence_visualization.png")
page.to_image(path=output_path, show_labels=True)
print(f"  - Saved visualization to {output_path}")

# 8. Demonstrate override at extraction time
print("\n8. OCR override at extraction time")
text_default = page.extract_text()
print(f"  - Default extraction: {len(text_default)} characters")

text_override = page.extract_text(ocr={
    "languages": ["en"],
    "min_confidence": 0.4  # Lower threshold
})
print(f"  - Override extraction (min_confidence=0.4): {len(text_override)} characters")

# 9. OCR element properties
print("\n9. OCR element properties")
if ocr_elements:
    elem = ocr_elements[0]
    print(f"  - Source: {elem.source}")
    print(f"  - Confidence: {elem.confidence:.2f}")
    print(f"  - Text: '{elem.text}'")
    print(f"  - Bounding box: {elem.bbox}")
    print(f"  - Font (default for OCR): {elem.fontname}")

# 10. Compare OCR with lower text_threshold to detect more text
print("\n10. Comparing OCR with different text_threshold values")
page.clear_highlights()

# First with default text_threshold (0.7)
print("  - Running OCR with default text_threshold (0.7)")
default_elements = page.extract_ocr_elements(ocr={
    "languages": ["en"],
    "detection_params": {
        "text_threshold": 0.7  # Default value
    }
})
print(f"  - Found {len(default_elements)} elements with default text_threshold")

# Highlight with blue
for elem in default_elements:
    elem.highlight(label="Default threshold", color=(0, 0, 204, 76))

# Now with lower text_threshold (0.1)
print("  - Running OCR with lower text_threshold (0.1)")
low_threshold_elements = page.extract_ocr_elements(ocr={
    "languages": ["en"],
    "detection_params": {
        "text_threshold": 0.1  # Lower value to detect more text
    }
})
print(f"  - Found {len(low_threshold_elements)} elements with text_threshold=0.1")

# Highlight with red
for elem in low_threshold_elements:
    elem.highlight(label="Lower threshold (0.1)", color=(204, 0, 0, 76))

# Save comparative visualization
output_path = os.path.join(output_dir, "ocr_threshold_comparison.png")
page.to_image(path=output_path, show_labels=True)
print(f"  - Saved threshold comparison to {output_path}")

print("\nDone!")