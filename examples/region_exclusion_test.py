"""
Test the improved exclusion handling in Region.extract_text() method.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF, configure_logging

# Configure logging
configure_logging(level=logging.DEBUG)


def test_region_with_exclusions(pdf_path):
    """
    Test extracting text from regions with various exclusion scenarios.
    """
    with PDF(pdf_path) as pdf:
        page = pdf.pages[0]
        print(f"\nTesting with PDF: {pdf_path} (page {page.number})")
        
        print("\n=== 1. Creating Test Exclusion Zones ===")
        # Create top (header) and bottom (footer) exclusions
        # Top 15% of the page
        top_exclusion = page.create_region(0, 0, page.width, page.height * 0.15)
        top_exclusion.highlight(label="Header Exclusion", color=(1, 0, 0, 0.3))
        page.add_exclusion(top_exclusion)
        print(f"Added header exclusion: {top_exclusion.bbox}")
        
        # Bottom 10% of the page
        bottom_exclusion = page.create_region(0, page.height * 0.9, page.width, page.height)
        bottom_exclusion.highlight(label="Footer Exclusion", color=(0, 0, 1, 0.3))
        page.add_exclusion(bottom_exclusion)
        print(f"Added footer exclusion: {bottom_exclusion.bbox}")
        
        # Middle partial-width exclusion
        middle_exclusion = page.create_region(0, page.height * 0.4, page.width * 0.3, page.height * 0.6)
        middle_exclusion.highlight(label="Side Exclusion", color=(0, 1, 0, 0.3))
        page.add_exclusion(middle_exclusion)
        print(f"Added side exclusion: {middle_exclusion.bbox}")
        
        print("\n=== 2. Testing Region That Doesn't Intersect Exclusions ===")
        # Create a region that doesn't intersect with any exclusion
        non_intersecting = page.create_region(
            page.width * 0.4,
            page.height * 0.5, 
            page.width * 0.9,
            page.height * 0.7
        )
        non_intersecting.highlight(label="Non-Intersecting", color=(1, 1, 0, 0.3))
        
        # Extract with and without applying exclusions - should be the same
        text_with_exclusions = non_intersecting.extract_text(apply_exclusions=True, debug=True)
        text_without_exclusions = non_intersecting.extract_text(apply_exclusions=False)
        print(f"Non-intersecting region text length:")
        print(f"  - With exclusions: {len(text_with_exclusions)} chars")
        print(f"  - Without exclusions: {len(text_without_exclusions)} chars")
        print(f"  - Same result: {text_with_exclusions == text_without_exclusions}")
        
        print("\n=== 3. Testing Region With Header/Footer Intersection ===")
        # Create a region that intersects with header and footer
        full_height = page.create_region(
            page.width * 0.3,
            0, 
            page.width * 0.8,
            page.height
        )
        full_height.highlight(label="Full Height Region", color=(1, 0, 1, 0.3))
        
        # Extract with and without applying exclusions
        text_with_exclusions = full_height.extract_text(apply_exclusions=True, debug=True)
        text_without_exclusions = full_height.extract_text(apply_exclusions=False)
        print(f"Full height region text length:")
        print(f"  - With exclusions: {len(text_with_exclusions)} chars")
        print(f"  - Without exclusions: {len(text_without_exclusions)} chars")
        print(f"  - Exclusions removed {len(text_without_exclusions) - len(text_with_exclusions)} chars")
        
        # Test the specific case that was causing issues
        middle_to_footer = page.create_region(
            page.width * 0.3,
            page.height * 0.4,  # Middle of page
            page.width * 0.8,
            page.height        # All the way to bottom (overlapping footer)
        )
        middle_to_footer.highlight(label="Middle to Footer", color=(0.5, 0.5, 0, 0.3))
        
        text_with_exclusions = middle_to_footer.extract_text(apply_exclusions=True, debug=True)
        text_without_exclusions = middle_to_footer.extract_text(apply_exclusions=False)
        print(f"\nMiddle-to-footer region text length:")
        print(f"  - With exclusions: {len(text_with_exclusions)} chars")
        print(f"  - Without exclusions: {len(text_without_exclusions)} chars")
        if len(text_with_exclusions) > 0:
            print(f"  - Working correctly! Content found with exclusions applied")
        else:
            print(f"  - Still failing! No content found with exclusions applied")
        
        print("\n=== 4. Testing Region With Complex Exclusion Intersection ===")
        # Create a region that intersects with the side exclusion
        complex_region = page.create_region(
            page.width * 0.1,
            page.height * 0.3, 
            page.width * 0.5,
            page.height * 0.7
        )
        complex_region.highlight(label="Complex Region", color=(0, 1, 1, 0.3))
        
        # Extract with and without applying exclusions
        text_with_exclusions = complex_region.extract_text(apply_exclusions=True, debug=True)
        text_without_exclusions = complex_region.extract_text(apply_exclusions=False)
        print(f"Complex region text length:")
        print(f"  - With exclusions: {len(text_with_exclusions)} chars")
        print(f"  - Without exclusions: {len(text_without_exclusions)} chars")
        print(f"  - Exclusions removed {len(text_without_exclusions) - len(text_with_exclusions)} chars")
        
        # Save the image with all regions and exclusions highlighted
        print("\n=== 5. Saving Visual Test Image ===")
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "region_exclusion_test.png")
        page.save_image(output_file, labels=True)
        print(f"Saved test visualization to: {output_file}")


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
    
    test_region_with_exclusions(pdf_path)


if __name__ == "__main__":
    main()