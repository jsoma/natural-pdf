"""
OCR Visualization Example

This example demonstrates the new OCR visualization feature that renders
OCR text with white background boxes on the image.
"""
import os
import sys

# Add project directory to the path to import the library
script_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

from natural_pdf import PDF

# Get paths
default_pdf = os.path.join(root_dir, "pdfs", "needs-ocr.pdf") 
if not os.path.exists(default_pdf):
    default_pdf = os.path.join(root_dir, "pdfs", "01-practice.pdf")

# Output directory
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

def main():
    """Main example function."""
    print("OCR Visualization Example")
    print("=========================")
    
    # 1. Load a PDF with OCR enabled
    print("\n1. Loading PDF with OCR enabled")
    pdf = PDF(default_pdf, ocr={
        "enabled": True,
        "languages": ["en"],
        "min_confidence": 0.3  # Lower confidence to get more results
    })
    
    # 2. First check if we have OCR text by extracting text with OCR
    print("\n2. Extracting text with OCR")
    page = pdf.pages[0]
    text = page.extract_text(ocr=True)  # Force OCR
    print(f"Extracted {len(text)} characters with OCR")
    
    # 3. Find OCR text elements
    print("\n3. Finding OCR text elements")
    ocr_elements = page.find_all('text[source=ocr]')
    print(f"Found {len(ocr_elements)} OCR text elements on the page")
    
    # If we don't have OCR elements, fall back to forcing OCR directly
    if not ocr_elements:
        print("No OCR elements found. Running OCR directly...")
        # Extract OCR elements directly
        ocr_elements = page.extract_ocr_elements()
        print(f"Found {len(ocr_elements)} OCR text elements from direct extraction")
    
    # 4. Highlight the OCR elements
    print(f"\n4. Highlighting {len(ocr_elements)} OCR elements")
    for element in ocr_elements:
        # Add color highlighting based on confidence score
        confidence = getattr(element, 'confidence', 0.5)  # Default if not available
        if confidence >= 0.8:
            color = (0, 1, 0, 0.3)  # Green for high confidence
        elif confidence >= 0.5:
            color = (1, 1, 0, 0.3)  # Yellow for medium confidence
        else:
            color = (1, 0, 0, 0.3)  # Red for low confidence
            
        element.highlight(color=color, label=f"OCR ({confidence:.2f})")
    
    # 5. Visualize the regular highlights (no OCR text)
    print("\n5. Saving highlighted image without OCR text")
    highlighted_path = os.path.join(output_dir, "ocr_highlighted.png")
    page.to_image(path=highlighted_path, show_labels=True, render_ocr=False)
    print(f"Saved highlighted image to {highlighted_path}")
    
    # 6. Visualize with OCR text on white background
    print("\n6. Saving image with rendered OCR text")
    ocr_text_path = os.path.join(output_dir, "ocr_rendered_text.png")
    try:
        page.to_image(path=ocr_text_path, show_labels=True, render_ocr=True)
        print(f"Saved OCR text rendering to {ocr_text_path}")
    except ValueError as e:
        print(f"Error rendering OCR text: {e}")
    
    # 7. Create a clean white page with just OCR text (no highlights)
    if ocr_elements:
        print("\n7. Creating clean white page with just OCR text")
        # Clear previous highlights
        page.clear_highlights()
        # Save with OCR text rendering only
        clean_text_path = os.path.join(output_dir, "ocr_clean_text.png")
        try:
            page.to_image(path=clean_text_path, render_ocr=True)
            print(f"Saved clean OCR text rendering to {clean_text_path}")
        except ValueError as e:
            print(f"Error rendering clean OCR text: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()