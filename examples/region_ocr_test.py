"""
Test to identify and fix issues with region-specific OCR.
"""
import os
import sys

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF
from PIL import Image, ImageDraw

def test_region_ocr():
    """Test OCR applied to specific regions."""
    # Use a PDF that may work well with OCR
    pdf_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'pdfs', 'Nigeria 2021_MICS_SFR_English.pdf'))
    
    if not os.path.exists(pdf_path):
        # Fall back to another PDF
        pdf_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'pdfs', '0500000US42001.pdf'))
        
    if not os.path.exists(pdf_path):
        print("No suitable PDF file found for region OCR testing.")
        return
    
    print(f"Testing with PDF: {pdf_path}")
    
    # Output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    os.makedirs(output_dir, exist_ok=True)
    
    with PDF(pdf_path) as pdf:
        # Get the first page
        page = pdf.pages[0]
        
        # Save the entire page image for reference
        page_img = page.to_image(path=os.path.join(output_dir, "region_ocr_full_page.png"))
        
        # Create a region in the middle of the page
        half_width = page.width / 2
        half_height = page.height / 2
        region_width = page.width / 3
        region_height = page.height / 3
        
        region = page.create_region(
            half_width - region_width/2, 
            half_height - region_height/2,
            half_width + region_width/2, 
            half_height + region_height/2
        )
        
        # Highlight the region
        region.highlight(label="OCR Test Region")
        page.to_image(path=os.path.join(output_dir, "region_ocr_highlighted.png"), show_labels=True)
        
        # Extract text from the region with and without OCR
        text_no_ocr = region.extract_text()
        text_with_ocr = region.extract_text(ocr=True)
        
        # Print results
        print("\nRegion Text WITHOUT OCR:")
        print("-" * 40)
        print(text_no_ocr)
        
        print("\nRegion Text WITH OCR:")
        print("-" * 40)
        print(text_with_ocr)
        
        # Apply OCR to the region and visualize the results
        ocr_elements = region.apply_ocr(enabled=True)
        
        print(f"\nFound {len(ocr_elements)} OCR elements in the region")
        
        # Get the region image
        page_img = page.to_image()
        region_img = page_img.crop((region.x0, region.top, region.x1, region.bottom))
        
        # Save region image for reference
        region_img.save(os.path.join(output_dir, "region_ocr_cropped.png"))
        
        # Create debug image showing OCR bounding boxes
        debug_img = page.to_image()
        draw = ImageDraw.Draw(debug_img)
        
        # Draw region rectangle
        draw.rectangle(
            (region.x0, region.top, region.x1, region.bottom),
            outline=(255, 0, 0),
            width=3
        )
        
        # Draw OCR element bounding boxes
        for elem in ocr_elements:
            draw.rectangle(
                (elem.x0, elem.top, elem.x1, elem.bottom),
                outline=(0, 255, 0),
                width=2
            )
            
            # Draw text label
            draw.text(
                (elem.x0, elem.top - 10),
                elem.text[:10],
                fill=(0, 0, 255)
            )
        
        # Save debug image
        debug_img.save(os.path.join(output_dir, "region_ocr_debug.png"))
        
        print(f"\nCreated debug images in: {output_dir}")
        print("- region_ocr_full_page.png: Original page")
        print("- region_ocr_highlighted.png: Page with region highlighted")
        print("- region_ocr_cropped.png: Cropped region image")
        print("- region_ocr_debug.png: Page with OCR text bounding boxes")

if __name__ == "__main__":
    test_region_ocr()