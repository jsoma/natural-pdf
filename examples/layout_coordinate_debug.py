"""
Debug script to investigate coordinate differences between YOLO and TATR models.

This script visualizes the regions detected by both models and logs all coordinate 
transformations to help diagnose the issue with YOLO's regions being too narrow.
"""
import os
import sys
import logging
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    from PIL import Image, ImageDraw
    ImageFont = None
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from natural_pdf import PDF, configure_logging
from natural_pdf.elements.collections import ElementCollection

# Set up logging
configure_logging(level=logging.DEBUG)
layout_logger = logging.getLogger("natural_pdf.analyzers.layout")
layout_logger.setLevel(logging.DEBUG)

# Create a file handler for detailed logs
file_handler = logging.FileHandler("layout_debug.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
layout_logger.addHandler(file_handler)

def debug_detection_coordinates(pdf_path, page_num=0, output_dir="output"):
    """Debug layout detection coordinates for both YOLO and TATR models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Open PDF 
    pdf = PDF(pdf_path)
    page = pdf.pages[page_num]
    
    # Get original page dimensions for reference
    page_width = page.width
    page_height = page.height
    print(f"Page dimensions: {page_width} x {page_height}")
    
    # Monkey patch the core analyze_layout method to add logging
    original_analyze_layout = type(page).analyze_layout
    
    def debug_analyze_layout(self, *args, **kwargs):
        """Wrapped analyze_layout method with debug logging."""
        model_type = kwargs.get('model', 'yolo')
        print(f"\n=== Running layout analysis with {model_type.upper()} model ===")
        
        # Add logging for image to PDF coordinate conversion
        old_convert_to_regions = None
        try:
            from natural_pdf.analyzers.document_layout import convert_to_regions
            old_convert_to_regions = convert_to_regions
            
            def debug_convert_to_regions(page, detections, scale_factor=1.0):
                """Monkey patched version with detailed logging."""
                print(f"Converting {len(detections)} detections with scale factor {scale_factor}")
                
                # Create a detailed log for each detection
                for i, det in enumerate(detections):
                    bbox = det['bbox']
                    x_min, y_min, x_max, y_max = bbox
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    pdf_x0 = x_min * scale_factor
                    pdf_y0 = y_min * scale_factor
                    pdf_x1 = x_max * scale_factor
                    pdf_y1 = y_max * scale_factor
                    
                    print(f"Detection #{i+1} ({det['class']}):")
                    print(f"  Raw bbox: {bbox}")
                    print(f"  Image dimensions: {width:.2f} x {height:.2f}")
                    print(f"  PDF coords: ({pdf_x0:.2f}, {pdf_y0:.2f}, {pdf_x1:.2f}, {pdf_y1:.2f})")
                    print(f"  PDF dimensions: {pdf_x1-pdf_x0:.2f} x {pdf_y1-pdf_y0:.2f}")
                    print(f"  Image-to-PDF ratio: width={scale_factor:.4f}, height={scale_factor:.4f}")
                
                # Call the original function
                return old_convert_to_regions(page, detections, scale_factor)
                
            # Replace the function
            from natural_pdf.analyzers import document_layout
            document_layout.convert_to_regions = debug_convert_to_regions
        
        except ImportError:
            print("Could not monkey patch convert_to_regions")
        
        # Call the original method
        result = original_analyze_layout(self, *args, **kwargs)
        
        # Restore the original function
        if old_convert_to_regions:
            from natural_pdf.analyzers import document_layout
            document_layout.convert_to_regions = old_convert_to_regions
        
        return result
    
    # Apply the monkey patch
    type(page).analyze_layout = debug_analyze_layout
    
    # Run YOLO model layout detection
    page.analyze_layout(model="yolo")
    
    # Get regions and save visualization
    yolo_regions = page.find_all('region[model=yolo]')
    print(f"YOLO detected {len(yolo_regions)} regions")
    
    # Highlight YOLO regions 
    page.clear_highlights()
    for region in yolo_regions:
        # Get region dimensions
        width = region.width
        height = region.height
        region.highlight(label=f"{region.region_type} ({width:.1f}x{height:.1f})")
    
    # Save YOLO visualization
    page.to_image(labels=True).save(os.path.join(output_dir, "yolo_regions.png"))
    
    # Create detailed summary of YOLO regions
    with open(os.path.join(output_dir, "yolo_regions.txt"), "w") as f:
        for i, region in enumerate(yolo_regions):
            f.write(f"Region #{i+1} ({region.region_type}):\n")
            f.write(f"  Bbox: ({region.x0:.2f}, {region.top:.2f}, {region.x1:.2f}, {region.bottom:.2f})\n")
            f.write(f"  Dimensions: {region.width:.2f} x {region.height:.2f}\n")
            f.write(f"  Confidence: {region.confidence:.4f}\n")
            f.write(f"  % of page width: {(region.width / page_width) * 100:.2f}%\n\n")
    
    # Clear existing layout regions
    page._regions['detected'] = []
    
    # Run TATR model layout detection
    page.analyze_layout(model="tatr")
    
    # Get regions and save visualization
    tatr_regions = page.find_all('region[model=tatr]')
    print(f"TATR detected {len(tatr_regions)} regions")
    
    # Highlight TATR regions
    page.clear_highlights()
    for region in tatr_regions:
        # Get region dimensions
        width = region.width
        height = region.height
        region.highlight(label=f"{region.region_type} ({width:.1f}x{height:.1f})")
    
    # Save TATR visualization
    page.to_image(labels=True).save(os.path.join(output_dir, "tatr_regions.png"))
    
    # Create detailed summary of TATR regions
    with open(os.path.join(output_dir, "tatr_regions.txt"), "w") as f:
        for i, region in enumerate(tatr_regions):
            f.write(f"Region #{i+1} ({region.region_type}):\n")
            f.write(f"  Bbox: ({region.x0:.2f}, {region.top:.2f}, {region.x1:.2f}, {region.bottom:.2f})\n")
            f.write(f"  Dimensions: {region.width:.2f} x {region.height:.2f}\n")
            f.write(f"  Confidence: {region.confidence:.4f}\n")
            f.write(f"  % of page width: {(region.width / page_width) * 100:.2f}%\n\n")
    
    # Create a side-by-side comparison image
    # Get page base image
    page_image = page.to_image(resolution=150)
    
    # Create a combined image
    combined_width = page_image.width * 2 + 20  # Add some padding
    combined_height = page_image.height + 50  # Add space for title
    combined = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    
    # Add titles
    try:
        font = ImageFont.truetype("Arial", 16)
    except:
        font = None
    
    draw = ImageDraw.Draw(combined)
    draw.text((page_image.width // 2, 20), "YOLO Model", fill=(0, 0, 0), font=font)
    draw.text((page_image.width * 3 // 2 + 20, 20), "TATR Model", fill=(0, 0, 0), font=font)
    
    # Add images side by side
    try:
        yolo_img = Image.open(os.path.join(output_dir, "yolo_regions.png"))
        tatr_img = Image.open(os.path.join(output_dir, "tatr_regions.png"))
    except:
        print("Warning: Could not load images for side-by-side comparison. Continuing...")
        return
    
    # Resize if needed
    if yolo_img.height != page_image.height or yolo_img.width != page_image.width:
        yolo_img = yolo_img.resize((page_image.width, page_image.height))
    if tatr_img.height != page_image.height or tatr_img.width != page_image.width:
        tatr_img = tatr_img.resize((page_image.width, page_image.height))
        
    combined.paste(yolo_img, (0, 50))
    combined.paste(tatr_img, (page_image.width + 20, 50))
    
    combined.save(os.path.join(output_dir, "model_comparison.png"))
    
    # Create a document comparing page dimensions to detected region dimensions
    with open(os.path.join(output_dir, "dimension_analysis.txt"), "w") as f:
        f.write(f"Page dimensions: {page_width:.2f} x {page_height:.2f}\n\n")
        
        # YOLO region stats
        f.write("=== YOLO Model Regions ===\n")
        if yolo_regions:
            yolo_widths = [r.width for r in yolo_regions]
            yolo_width_pcts = [(r.width / page_width) * 100 for r in yolo_regions]
            
            f.write(f"Total regions: {len(yolo_regions)}\n")
            f.write(f"Average width: {sum(yolo_widths) / len(yolo_widths):.2f}\n")
            f.write(f"Average width as % of page: {sum(yolo_width_pcts) / len(yolo_width_pcts):.2f}%\n")
            f.write(f"Min width: {min(yolo_widths):.2f} ({min(yolo_width_pcts):.2f}%)\n")
            f.write(f"Max width: {max(yolo_widths):.2f} ({max(yolo_width_pcts):.2f}%)\n\n")
        else:
            f.write("No regions detected\n\n")
            
        # TATR region stats
        f.write("=== TATR Model Regions ===\n")
        if tatr_regions:
            tatr_widths = [r.width for r in tatr_regions]
            tatr_width_pcts = [(r.width / page_width) * 100 for r in tatr_regions]
            
            f.write(f"Total regions: {len(tatr_regions)}\n")
            f.write(f"Average width: {sum(tatr_widths) / len(tatr_widths):.2f}\n")
            f.write(f"Average width as % of page: {sum(tatr_width_pcts) / len(tatr_width_pcts):.2f}%\n")
            f.write(f"Min width: {min(tatr_widths):.2f} ({min(tatr_width_pcts):.2f}%)\n")
            f.write(f"Max width: {max(tatr_widths):.2f} ({max(tatr_width_pcts):.2f}%)\n\n")
        else:
            f.write("No regions detected\n\n")
        
        # If both have regions, compare them
        if yolo_regions and tatr_regions:
            # Calculate average width ratio
            avg_yolo_width = sum(yolo_widths) / len(yolo_widths)
            avg_tatr_width = sum(tatr_widths) / len(tatr_widths)
            width_ratio = avg_tatr_width / avg_yolo_width if avg_yolo_width > 0 else 0
            
            f.write("=== Comparison ===\n")
            f.write(f"TATR avg width / YOLO avg width = {width_ratio:.4f}\n")
            f.write(f"YOLO is {100 * (1 - 1/width_ratio):.2f}% narrower than TATR on average\n")
    
    print(f"\nDebug information saved to {output_dir} directory")
    print(f"See 'layout_debug.log' for detailed process logging")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug layout detection coordinate differences")
    parser.add_argument("pdf_path", help="Path to the PDF file to analyze")
    parser.add_argument("--page", type=int, default=0, help="Page number to analyze (0-indexed)")
    parser.add_argument("--output", default="output", help="Output directory for debug files")
    
    args = parser.parse_args()
    
    debug_detection_coordinates(args.pdf_path, page_num=args.page, output_dir=args.output)