"""
Example to debug exclusion issues with highlighting.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF


def debug_exclusions():
    """Debug exclusion problem."""
    # Get PDF path - use a default if one isn't specified
    # Look for any PDF in the examples directory or pdfs directory
    example_dir = Path(__file__).parent
    pdf_files = list(example_dir.glob("*.pdf"))
    
    if not pdf_files:
        pdfs_dir = example_dir.parent / "pdfs"
        if pdfs_dir.exists():
            pdf_files = list(pdfs_dir.glob("*.pdf"))
    
    if pdf_files:
        pdf_path = str(pdf_files[0])
    else:
        print("No PDF file found. Please provide a path to a PDF file.")
        sys.exit(1)
    
    print(f"Using PDF: {pdf_path}")

    # Case 1: Direct page exclusion - expected to work
    print("\n=== Case 1: Direct page exclusion ===")
    pdf1 = PDF(pdf_path)
    page1 = pdf1.pages[0]
    
    # Create a debug output directory
    output_dir = Path(__file__).parent / "debug_output"
    output_dir.mkdir(exist_ok=True)
    
    # First, save without exclusions for comparison
    page1.highlight_all()
    page1.save(str(output_dir / "case1_no_exclusion.png"), labels=True)
    page1.clear_highlights()
    
    # Log exclusions we're adding
    line1 = page1.find('line')
    print(f"Adding exclusion for region above line at {line1.top}")
    
    # Add exclusion directly to page
    page1.add_exclusion(line1.above())
    
    # Show all exclusion regions
    exclusion_regions = page1._get_exclusion_regions(include_callable=True)
    print(f"Found {len(exclusion_regions)} exclusion regions")
    for i, region in enumerate(exclusion_regions):
        print(f"  Region {i+1}: top={region.top}, bottom={region.bottom}")
    
    # Apply highlight with exclusions
    page1.highlight_all(apply_exclusions=True)
    page1.save(str(output_dir / "case1_with_exclusion.png"), labels=True)
    
    # Case 2: PDF-level exclusion - not working correctly
    print("\n=== Case 2: PDF-level exclusion ===")
    pdf2 = PDF(pdf_path)
    
    # This should work exactly the same as Case 1
    pdf2.add_exclusion(lambda page: page.find('line').above())
    page2 = pdf2.pages[0]
    
    # Show all exclusion regions for comparison
    exclusion_regions = page2._get_exclusion_regions(include_callable=True, debug=True)
    print(f"Found {len(exclusion_regions)} exclusion regions")
    for i, region in enumerate(exclusion_regions):
        print(f"  Region {i+1}: top={region.top}, bottom={region.bottom}")
    
    # Save highlighting result
    page2.highlight_all(apply_exclusions=True)
    page2.save(str(output_dir / "case2_with_exclusion.png"), labels=True)
    
    # Case 3: Using find_all with exclusions - for comparison
    print("\n=== Case 3: Using find_all with exclusions ===")
    pdf3 = PDF(pdf_path)
    pdf3.add_exclusion(lambda page: page.find('line').above())
    page3 = pdf3.pages[0]
    
    # Check what find_all returns with exclusions
    all_text = page3.find_all('text', apply_exclusions=True)
    print(f"find_all('text') returns {len(all_text)} elements with exclusions")
    
    # Highlight just those elements
    all_text.highlight(label="Text with exclusions")
    page3.save(str(output_dir / "case3_find_all_with_exclusion.png"), labels=True)
    
    # Compare to highlight_all
    page3.clear_highlights()
    page3.highlight_all(apply_exclusions=True)
    page3.save(str(output_dir / "case3_highlight_all.png"), labels=True)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    debug_exclusions()