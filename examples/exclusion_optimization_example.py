"""
Example demonstrating the optimized exclusion handling for various region types.
"""

import os
import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF


def measure_time(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def optimized_exclusion_example(pdf_path):
    """
    Demonstrates the optimized exclusion handling for different region types.
    """
    with PDF(pdf_path) as pdf:
        page = pdf.pages[0]
        print(f"Using PDF: {pdf_path}")
        
        # Create an output directory
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Set up exclusion zones
        print("\n=== Setting Up Exclusion Zones ===")
        # Top 10% of page as header
        header_zone = page.create_region(0, 0, page.width, page.height * 0.1)
        header_zone.highlight(label="Header Exclusion", color=(1, 0, 0, 0.3))
        page.add_exclusion(header_zone)
        
        # Bottom 10% of page as footer
        footer_zone = page.create_region(0, page.height * 0.9, page.width, page.height)
        footer_zone.highlight(label="Footer Exclusion", color=(0, 0, 1, 0.3))
        page.add_exclusion(footer_zone)
        
        # Left 20% as a side panel (complex exclusion)
        side_panel = page.create_region(0, 0, page.width * 0.2, page.height)
        side_panel.highlight(label="Side Panel Exclusion", color=(0, 1, 0, 0.3))
        page.add_exclusion(side_panel)
        
        print(f"Added 3 exclusion zones: header, footer, and side panel")
        
        # Step 2: Create test regions of different types
        print("\n=== Creating Test Regions ===")
        # Non-intersecting region (in center, away from all exclusions)
        non_intersecting = page.create_region(
            page.width * 0.3, 
            page.height * 0.3, 
            page.width * 0.8, 
            page.height * 0.7
        )
        non_intersecting.highlight(label="Non-Intersecting Region", color=(1, 1, 0, 0.3))
        
        # Header/footer-only region (full width but between exclusions)
        header_footer_region = page.create_region(
            0,
            0,
            page.width,
            page.height
        )
        header_footer_region.highlight(label="Full Page Region", color=(1, 0, 1, 0.2))
        
        # Complex region (intersects with side panel)
        complex_region = page.create_region(
            0,
            page.height * 0.2,
            page.width * 0.5,
            page.height * 0.8
        )
        complex_region.highlight(label="Complex Region", color=(0, 1, 1, 0.3))
        
        print("Created 3 test regions with different exclusion intersection patterns")
        
        # Save the visualization
        output_file = os.path.join(output_dir, "exclusion_optimization_regions.png")
        page.save_image(output_file, labels=True)
        print(f"Saved visualization to: {output_file}")
        
        # Step 3: Test extraction with and without optimizations
        print("\n=== Testing Text Extraction with Exclusions ===")
        
        # Test non-intersecting region
        print("\nNon-Intersecting Region:")
        print("This region should use the fast path (no exclusion checking)")
        print("Extracting text with apply_exclusions=True...")
        
        @measure_time
        def extract_non_intersecting():
            return non_intersecting.extract_text(apply_exclusions=True)
        
        text1 = extract_non_intersecting()
        
        print("Extracting text with apply_exclusions=False (for comparison)...")
        
        @measure_time
        def extract_non_intersecting_no_exclusions():
            return non_intersecting.extract_text(apply_exclusions=False)
        
        text2 = extract_non_intersecting_no_exclusions()
        
        print(f"Text length comparison: with exclusions={len(text1)}, without={len(text2)}")
        print(f"Identical results: {text1 == text2}")
        
        # Test header/footer region
        print("\nFull Page Region (intersecting header/footer):")
        print("This region should use cropping optimization for header/footer exclusions")
        print("Extracting text with apply_exclusions=True...")
        
        @measure_time
        def extract_header_footer():
            return header_footer_region.extract_text(apply_exclusions=True)
        
        text3 = extract_header_footer()
        
        print("Extracting text with apply_exclusions=False (for comparison)...")
        
        @measure_time
        def extract_header_footer_no_exclusions():
            return header_footer_region.extract_text(apply_exclusions=False)
        
        text4 = extract_header_footer_no_exclusions()
        
        print(f"Text length comparison: with exclusions={len(text3)}, without={len(text4)}")
        print(f"Header/footer content excluded: {len(text4) > len(text3)}")
        
        # Test complex region
        print("\nComplex Region (intersecting side panel):")
        print("This region should use filtering with warning")
        print("Extracting text with apply_exclusions=True...")
        
        @measure_time
        def extract_complex():
            return complex_region.extract_text(apply_exclusions=True)
        
        text5 = extract_complex()
        
        print("Extracting text with apply_exclusions=False (for comparison)...")
        
        @measure_time
        def extract_complex_no_exclusions():
            return complex_region.extract_text(apply_exclusions=False)
        
        text6 = extract_complex_no_exclusions()
        
        print(f"Text length comparison: with exclusions={len(text5)}, without={len(text6)}")
        
        # Step 4: Summarize findings
        print("\n=== Summary ===")
        print("1. Non-intersecting region: Optimization should skip exclusion checks entirely")
        print("2. Header/footer region: Optimization should use direct cropping")
        print("3. Complex region: Falls back to filtering with warning")
        print("\nCheck the produced warning messages to confirm the behavior.")
        

def main():
    """Main entry point."""
    # Get the PDF path from command line or use a default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Look for any PDF in the pdfs directory
        pdfs_dir = Path(__file__).parent.parent / "pdfs"
        pdf_files = list(pdfs_dir.glob("*.pdf"))
        
        if pdf_files:
            pdf_path = str(pdf_files[0])
        else:
            print("No PDF file found. Please provide a path to a PDF file.")
            sys.exit(1)
    
    optimized_exclusion_example(pdf_path)


if __name__ == "__main__":
    main()