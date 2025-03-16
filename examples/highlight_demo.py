"""
Demo script to show highlight color cycling behavior.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF


def highlight_demo():
    # Get PDF path
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
    
    # Create output directory
    output_dir = Path(__file__).parent / "highlight_demo_output"
    output_dir.mkdir(exist_ok=True)
    
    # Load PDF
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    
    # Demo 1: Default behavior - consistent color without label
    print("Demo 1: Default behavior - consistent color without label")
    texts = page.find_all('text')[:5]  # Get first 5 text elements for demo
    
    # Highlight each element individually
    for i, text in enumerate(texts):
        text.highlight()  # No label - should use consistent color (yellow)
    
    # Save result
    page.save(str(output_dir / "demo1_default_no_label.png"), labels=True)
    page.clear_highlights()
    
    # Demo 2: With cycle_colors=True - different colors without label
    print("Demo 2: With cycle_colors=True - different colors without label")
    
    # Highlight each element individually with cycling
    for i, text in enumerate(texts):
        text.highlight(cycle_colors=True)  # No label but with cycling
    
    # Save result
    page.save(str(output_dir / "demo2_cycling_no_label.png"), labels=True)
    page.clear_highlights()
    
    # Demo 3: With labels - different colors for different labels
    print("Demo 3: With labels - different colors for different labels")
    
    # Highlight each element with a unique label
    for i, text in enumerate(texts):
        text.highlight(label=f"Element {i+1}")  # Different labels
    
    # Save result
    page.save(str(output_dir / "demo3_with_labels.png"), labels=True)
    page.clear_highlights()
    
    # Demo 4: With same label - same color
    print("Demo 4: With same label - same color")
    
    # Highlight all with the same label
    for i, text in enumerate(texts):
        text.highlight(label="Group A")  # Same label - should use same color
    
    # Save result
    page.save(str(output_dir / "demo4_same_label.png"), labels=True)
    page.clear_highlights()
    
    # Demo 5: Using highlight_all with default settings
    print("Demo 5: Using highlight_all with default settings")
    
    # Highlight all elements by type
    page.highlight_all()  # Default: cycle_colors=True
    
    # Save result
    page.save(str(output_dir / "demo5_highlight_all_default.png"), labels=True)
    page.clear_highlights()
    
    # Demo 6: Using highlight_all with cycle_colors=False
    print("Demo 6: Using highlight_all with cycle_colors=False")
    
    # Highlight all elements by type without cycling
    page.highlight_all(cycle_colors=False)
    
    # Save result
    page.save(str(output_dir / "demo6_highlight_all_no_cycling.png"), labels=True)
    page.clear_highlights()
    
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    highlight_demo()