"""
Example demonstrating the new region.to_image() and region.save_image() functionality.

This example shows how to:
1. Create regions in various ways
2. Generate images of just the region
3. Save region images to files
4. Compare different rendering options
"""

import os
import sys
import argparse

# Add parent directory to path to run without installing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from natural_pdf import PDF

def main():
    parser = argparse.ArgumentParser(description="Region Image Example")
    parser.add_argument("pdf_path", nargs="?", default="../pdfs/0500000US42001.pdf", 
                      help="Path to PDF document")
    args = parser.parse_args()
    
    print(f"Opening PDF: {args.pdf_path}")
    
    # Open the PDF
    pdf = PDF(args.pdf_path)
    page = pdf.pages[0]
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Method 1: Find a text element and create a region below it
    print("Creating regions...")
    title = page.find('text:bold')
    if not title:
        title = page.find('text')
    
    region_below = title.below(height=100, width="element")
    
    # Method 2: Create a region from a specific part of the page
    page_width, page_height = page.width, page.height
    center_region = page.create_region(
        page_width / 4,     # Left quarter of page
        page_height / 4,    # Top quarter of page
        page_width * 3/4,   # Right three-quarters
        page_height * 3/4   # Bottom three-quarters
    )
    
    # Method 3: Use layout detection to find regions
    page.analyze_layout(confidence=0.3)
    layout_regions = page.find_all('region')
    
    # Generate and save images for each region
    print("Generating region images...")
    
    # Example 1: Basic region image with default settings
    region_below.save_image("output/region_below.png")
    print(f"Saved basic region image to output/region_below.png")
    
    # Example 2: Region image with highlighted content
    # First highlight some elements in the region
    elements = region_below.find_all('text')
    if elements:
        elements[0].highlight(color=(1, 0, 0, 0.3), label="First Element")
        
        # Save with highlights included
        region_below.save_image(
            "output/region_with_highlights.png", 
            include_highlights=True
        )
        print(f"Saved region with highlights to output/region_with_highlights.png")
        
        # Save without highlights
        region_below.save_image(
            "output/region_without_highlights.png", 
            include_highlights=False
        )
        print(f"Saved region without highlights to output/region_without_highlights.png")
    
    # Example 3: Region image without border
    center_region.save_image(
        "output/center_region_with_border.png"
    )
    print(f"Saved center region with border to output/center_region_with_border.png")
    
    center_region.save_image(
        "output/center_region_without_border.png",
        crop_only=True
    )
    print(f"Saved center region without border to output/center_region_without_border.png")
    
    # Example 4: High-resolution region image
    if layout_regions:
        first_layout = layout_regions[0]
        first_layout.highlight(label=f"Region Type: {first_layout.region_type}")
        
        # Save at different resolutions
        first_layout.save_image(
            "output/layout_region_low_res.png",
            resolution=72
        )
        print(f"Saved layout region at 72 DPI to output/layout_region_low_res.png")
        
        first_layout.save_image(
            "output/layout_region_high_res.png",
            resolution=300
        )
        print(f"Saved layout region at 300 DPI to output/layout_region_high_res.png")
    
    print("\nDone! Check the output directory for the generated images.")

if __name__ == "__main__":
    main()