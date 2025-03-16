"""
Example demonstrating section extraction with the get_sections method.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF


def example_sections_between_headings(pdf_path):
    """
    Example showing how to extract sections between headings.
    """
    print("\n=== SECTIONS BETWEEN HEADINGS ===")
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    page.add_exclusion(page.find('text:contains("November")').above(include_element=True))
    
    # Create an output directory
    output_dir = Path(__file__).parent / "section_output"
    output_dir.mkdir(exist_ok=True)
    
    # Find all the headings on the page
    headings = page.find_all('text[size>=12]')
    print(f"Found {len(headings)} heading elements")
    
    # Create a highlighted visualization to see what we found
    page.clear_highlights()
    headings.highlight(label="Headings", color=(255, 100, 0, 100))
    page.to_image(path=str(output_dir / "headings.png"), show_labels=True)
    
    # First try without line grouping 
    print("Extracting sections WITHOUT line grouping:")
    # Set y_threshold to 0 to disable line grouping
    sections_no_grouping = page.get_sections(
        start_elements=headings,
        boundary_inclusion='start',
        y_threshold=0  # Disable line grouping
    )
    print(f"Found {len(sections_no_grouping)} sections without line grouping")
    
    # Now with line grouping
    print("\nExtracting sections WITH line grouping:")
    sections = page.get_sections(
        start_elements=headings,
        boundary_inclusion='start',  # Include heading with its section
        y_threshold=5  # Group elements within 5 points vertically
    )
    print(f"Found {len(sections)} sections with line grouping")
    
    # Create visualizations showing the difference
    # Highlight sections without grouping
    page.clear_highlights()
    for i, section in enumerate(sections_no_grouping):
        section.highlight(label=f"Section {i+1}", use_color_cycling=True)
    page.to_image(path=str(output_dir / "sections_no_grouping.png"), show_labels=True)
    
    # Highlight sections with grouping
    page.clear_highlights()
    for i, section in enumerate(sections):
        section.highlight(label=f"Section {i+1}", use_color_cycling=True)
    page.to_image(path=str(output_dir / "sections_with_grouping.png"), show_labels=True)
    
    # Process each section from the grouped version
    for i, section in enumerate(sections):
        # Get the heading text
        heading_text = section.start_element.extract_text() if hasattr(section, 'start_element') else "No heading"
        
        # Get section content (limited to first 50 chars for display)
        content = section.extract_text()
        content_preview = content[:50] + "..." if len(content) > 50 else content
        
        print(f"Section {i+1}: '{heading_text}'")
        print(f"  Content: {content_preview}")
        
        # Create visualization 
        page.clear_highlights()
        section.highlight(label=f"Section {i+1}")
        if hasattr(section, 'start_element') and section.start_element:
            section.start_element.highlight(label="Heading", color=(255, 0, 0, 100))
        if hasattr(section, 'end_element') and section.end_element:
            section.end_element.highlight(label="End", color=(0, 0, 255, 100))
        
        page.to_image(path=str(output_dir / f"section_{i+1}.png"), show_labels=True)


def example_sections_with_separators(pdf_path):
    """
    Example showing how to extract sections with separators.
    """
    print("\n=== SECTIONS WITH SEPARATORS ===")
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    
    # Create an output directory
    output_dir = Path(__file__).parent / "separator_output"
    output_dir.mkdir(exist_ok=True)
    
    # Find all horizontal lines that could be separators
    separators = page.find_all('line[width>=2]')
    print(f"Found {len(separators)} separator lines")
    
    # Create a highlighted visualization to see what we found
    page.clear_highlights()
    separators.highlight(label="Separators", color=(0, 0, 255, 100))
    page.to_image(path=str(output_dir / "separators.png"), show_labels=True)
    
    # Try different boundary inclusions
    inclusion_options = ['none', 'start', 'end', 'both']
    
    for inclusion in inclusion_options:
        print(f"\nSections with boundary_inclusion='{inclusion}':")
        sections = page.get_sections(
            start_elements=separators,
            boundary_inclusion=inclusion
        )
        
        print(f"Found {len(sections)} sections")
        
        # Create visualization for all sections
        page.clear_highlights()
        
        for i, section in enumerate(sections):
            # Use different color for each section
            color = None  # Let it cycle through colors
            section.highlight(label=f"Section {i+1}", use_color_cycling=True)
            
            # Section info
            content = section.extract_text()
            content_preview = content[:30] + "..." if len(content) > 30 else content
            print(f"  Section {i+1}: {content_preview}")
        
        # Save the visualization
        page.to_image(path=str(output_dir / f"sections_{inclusion}.png"), show_labels=True)
        page.clear_highlights()


def example_start_end_sections(pdf_path):
    """
    Example showing how to extract sections between start and end elements.
    """
    print("\n=== SECTIONS BETWEEN START AND END ELEMENTS ===")
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    
    # Create an output directory
    output_dir = Path(__file__).parent / "start_end_output"
    output_dir.mkdir(exist_ok=True)
    
    # Find headings and lines
    headings = page.find_all('text[size>=14]')
    lines = page.find_all('line[width>=2]')
    
    print(f"Found {len(headings)} headings and {len(lines)} lines")
    
    # Create a visualization to show both
    page.clear_highlights()
    headings.highlight(label="Headings", color=(255, 100, 0, 100))
    lines.highlight(label="Lines", color=(0, 0, 255, 100))
    page.to_image(path=str(output_dir / "elements.png"), show_labels=True)
    
    # Get sections from heading to next line
    sections = page.get_sections(
        start_elements=headings,
        end_elements=lines,
        boundary_inclusion='start'  # Include heading but not the line
    )
    
    print(f"Found {len(sections)} sections from headings to lines")
    
    # Process each section
    for i, section in enumerate(sections):
        # Get the heading text
        if hasattr(section, 'start_element') and section.start_element:
            heading_text = section.start_element.extract_text()
        else:
            heading_text = "No heading"
        
        # Get section content
        content = section.extract_text()
        content_preview = content[:50] + "..." if len(content) > 50 else content
        
        print(f"Section {i+1}: '{heading_text}'")
        print(f"  Content: {content_preview}")
        
        # Create visualization
        page.clear_highlights()
        section.highlight(label=f"Section {i+1}")
        if hasattr(section, 'start_element') and section.start_element:
            section.start_element.highlight(label="Heading", color=(255, 0, 0, 100))
        if hasattr(section, 'end_element') and section.end_element:
            section.end_element.highlight(label="Line", color=(0, 0, 255, 100))
        
        page.to_image(path=str(output_dir / f"section_{i+1}.png"), show_labels=True)


def main():
    """Main entry point."""
    # Get the PDF path from command line or use a default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
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
    
    # Run the examples
    example_sections_between_headings(pdf_path)
    example_sections_with_separators(pdf_path)
    example_start_end_sections(pdf_path)


if __name__ == "__main__":
    main()