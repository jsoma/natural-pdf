"""
Example demonstrating the highlighting feature of natural-pdf.
"""
import os
import sys

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

# IMPORTANT: This example has been updated to use the new API
# Changes:
# - select_until() → until()
# - full_width=False → width="element"
# - labels=True → show_labels=True
# - cycle_colors=True → use_color_cycling=True

def highlighting_example(pdf_path):
    """Demonstrates the highlighting features for visual debugging."""
    # Open the PDF
    with PDF(pdf_path) as pdf:
        page = pdf.pages[0]
        
        print(f"PDF loaded: {pdf_path}")
        print(f"PDF has {len(pdf)} pages")
        
        # Create an output directory for saving images
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        os.makedirs(output_dir, exist_ok=True)
        
        # EXAMPLE 1: Highlight a single element
        print("\nEXAMPLE 1: Highlighting a single element")
        print("-" * 60)
        
        # Find the "Summary:" text
        summary = page.find('text:contains("Summary:")')
        print(f"Found 'Summary' text at: {summary.bbox}")
        
        # Highlight it and save the image
        summary.highlight(label="Summary Heading")
        output_file = os.path.join(output_dir, "highlight_single.png")
        summary.page.to_image(path=output_file, show_show_labels=True)
        print(f"Saved highlighted page to: {output_file}")
        
        # Clear highlights for next example
        page.clear_highlights()
        
        # EXAMPLE 2: Highlight multiple elements with automatic color cycling
        print("\nEXAMPLE 2: Highlighting multiple elements with color cycling")
        print("-" * 60)
        
        # Find different types of elements
        thick_lines = page.find_all('line[width>=2]')
        headings = page.find_all('text:bold')
                
        # Highlight each group with a label
        print(f"Found {len(thick_lines)} thick lines")
        thick_lines.highlight(label="Thick Lines")
        
        print(f"Found {len(headings)} bold headings")
        # Let's examine some of the bold headings
        for i, h in enumerate(headings[:5]):
            print(f"  Bold heading {i+1}: '{h.text}' at {h.bbox}")
        headings.highlight(label="Bold Headings")
        
        # Save the image with a legend
        output_file = os.path.join(output_dir, "highlight_multiple.png")
        page.to_image(path=output_file, show_show_labels=True)
        print(f"Saved page with multiple highlights to: {output_file}")
        
        # Clear highlights for next example
        page.clear_highlights()
        
        # EXAMPLE 3: Highlighting regions
        print("\nEXAMPLE 3: Highlighting regions")
        print("-" * 60)
        
        # Find the "Summary:" text and the thick line
        summary = page.find('text:contains("Summary:")')
        thick_line = page.find('line[width>=2]')
        
        # Create a region from Summary until the thick line
        summary_region = summary.until('line[width>=2]', width="full")
        print(f"Created region from Summary to thick line: {summary_region.bbox}")
        
        # Highlight the region
        summary_region.highlight(label="Summary Section")
        
        # Find text within the region and highlight with a different color
        key_elements = summary_region.find_all('text')
        print(f"Found {len(key_elements)} text elements in the region")
        
        # Only highlight a subset to avoid cluttering the image
        for element in key_elements[:10]:
            if "fertilizer" in element.text.lower():
                element.highlight(label="Key Terms")
        
        # Save the image with a legend
        output_file = os.path.join(output_dir, "highlight_region.png")
        page.to_image(path=output_file, show_show_labels=True)
        print(f"Saved page with highlighted region to: {output_file}")
        
        print("\nEnd of highlighting demonstration.")

if __name__ == "__main__":
    # Default to example PDF if no path is provided
    if len(sys.argv) < 2:
        # Use the example PDF in the pdfs directory
        pdf_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'pdfs', '01-practice.pdf'))
        if not os.path.exists(pdf_path):
            print("Example PDF not found. Please provide a path to a PDF file.")
            print("Usage: python highlighting_example.py [path/to/file.pdf]")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
        # Check if the file exists
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            sys.exit(1)
    
    highlighting_example(pdf_path)