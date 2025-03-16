"""
Test script for the new extract_text implementation that uses pdfplumber's native functionality.
"""

import os
import sys
from io import StringIO
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from natural_pdf import PDF
import time

def main():
    # Use a sample PDF
    pdf_path = "pdfs/01-practice.pdf"
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    
    # Load the PDF
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    
    print(f"Loaded {pdf_path}, processing first page...")
    
    # Create different regions
    full_region = page.create_region(0, 0, page.width, page.height)
    top_region = page.create_region(0, 0, page.width, page.height / 3)
    bottom_region = page.create_region(0, page.height * 2/3, page.width, page.height)
    
    # Create a non-rectangular region (a triangle)
    # First create the region with a bbox
    triangle_region = page.create_region(0, 0, page.width, page.height/2)
    # Then set the polygon directly
    triangle_region._polygon = [(0, 0), (page.width, 0), (page.width/2, page.height/2)]
    
    # Add an exclusion region
    page.add_exclusion(bottom_region)
    
    # Test extraction with different settings
    
    # 1. Standard rectangular region without exclusions
    print("\nExtracting text from top region:")
    start = time.time()
    # First try with just crop to debug - use bbox directly
    crop_bbox = top_region.bbox
    
    print(f"Using bbox: {crop_bbox}")
    
    cropped = page._page.crop(crop_bbox)
    direct_text = cropped.extract_text(keep_blank_chars=True)
    print(f"Direct crop text length: {len(direct_text)}, Text: {direct_text[:100]}")
    
    # Check if there's a bug when passing the instance directly to extract_text
    print("Converting region to a dictionary and creating a new Region")
    region_dict = {
        'x0': top_region.x0,
        'top': top_region.top,
        'x1': top_region.x1, 
        'bottom': top_region.bottom
    }
    bbox = (region_dict['x0'], region_dict['top'], region_dict['x1'], region_dict['bottom'])
    
    from natural_pdf.elements.region import Region
    test_region = Region(page, bbox)
    print(f"New region bbox: {test_region.bbox}")
    
    # Create a simple direct call to pdfplumber's crop
    print("Testing direct pdfplumber crop and extract:")
    crop_bbox = test_region.bbox
    cropped_page = page._page.crop(crop_bbox)
    print(f"Cropped page dimensions: {cropped_page.width} × {cropped_page.height}")
    print(f"Cropped page characters: {len(cropped_page.chars)}")
    if cropped_page.chars:
        print(f"First few chars: {cropped_page.chars[:3]}")
    direct_crop_text = cropped_page.extract_text(keep_blank_chars=True)
    print(f"Direct pdfplumber extraction: {len(direct_crop_text)} chars")
    print(direct_crop_text[:100])
    
    # Test if we're seeing any print outputs from extract_text
    original_stderr = sys.stderr
    string_stderr = StringIO()
    sys.stderr = string_stderr
    
    # Try the new region's extract_text
    text = test_region.extract_text(keep_blank_chars=True)
    stderr_output = string_stderr.getvalue()
    sys.stderr = original_stderr
    
    print(f"Stderr output from extract_text call:\n{stderr_output}")
    
    elapsed = time.time() - start
    print(f"Length: {len(text)} characters, Time: {elapsed:.4f} seconds")
    print(text[:200] + "..." if len(text) > 200 else text)
    
    # 2. Full page with exclusions
    print("\nExtracting text from full page with exclusions:")
    start = time.time()
    text = full_region.extract_text(apply_exclusions=True)
    elapsed = time.time() - start
    print(f"Length: {len(text)} characters, Time: {elapsed:.4f} seconds")
    print(text[:200] + "..." if len(text) > 200 else text)
    
    # 3. Polygon region (triangle)
    print("\nExtracting text from triangle region:")
    start = time.time()
    text = triangle_region.extract_text()
    elapsed = time.time() - start
    print(f"Length: {len(text)} characters, Time: {elapsed:.4f} seconds")
    print(text[:200] + "..." if len(text) > 200 else text)
    
    # 4. With OCR option (to test that pathway)
    print("\nExtracting text with OCR option:")
    start = time.time()
    text = top_region.extract_text(ocr={"enabled": True})
    elapsed = time.time() - start
    print(f"Length: {len(text)} characters, Time: {elapsed:.4f} seconds")
    print(text[:200] + "..." if len(text) > 200 else text)
    
    # For comparison, test the regular page.extract_text method
    print("\nExtraction with page.extract_text for comparison:")
    start = time.time()
    text = page.extract_text(preserve_whitespace=True, apply_exclusions=True)
    elapsed = time.time() - start
    print(f"Length: {len(text)} characters, Time: {elapsed:.4f} seconds")
    print(text[:200] + "..." if len(text) > 200 else text)

if __name__ == "__main__":
    main()