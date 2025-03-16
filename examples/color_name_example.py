"""
Example demonstrating the use of color names in selectors.
"""
import sys
from pathlib import Path

# Add the parent directory to the path to import the local package
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF

def main():
    """Run the example."""
    # Get the PDF file path from command line args or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Use a default sample PDF
        pdf_path = str(Path(__file__).parent.parent / "pdfs" / "01-practice.pdf")
    
    # Create a PDF object
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    
    print("\n=== Using Color Names in Selectors ===\n")

    # Different ways to specify the same red color
    print("Finding red text using different color specifications:")
    
    # Traditional RGB tuple
    red_text1 = page.find_all('text[color~=(1,0,0)]')
    print(f"- Using RGB tuple (1,0,0): Found {len(red_text1)} elements")
    
    # Using named color
    red_text2 = page.find_all('text[color~=red]')
    print(f"- Using named color 'red': Found {len(red_text2)} elements")
    
    # Using hex color
    red_text3 = page.find_all('text[color~=#ff0000]')
    print(f"- Using hex color '#ff0000': Found {len(red_text3)} elements")
    
    # Compare results
    print("\nAre the results the same?", 
          len(red_text1) == len(red_text2) == len(red_text3))
    
    # Highlight the found elements
    page.clear_highlights()
    red_text1.highlight(label="Red (RGB tuple)")
    
    # Try a different color by name
    blue_text = page.find_all('text[color~=blue]')
    blue_text.highlight(label="Blue (named color)")
    
    green_text = page.find_all('text[color~=#00ff00]')
    green_text.highlight(label="Green (hex color)")
    
    print("\nHighlighting the found elements...")
    
    # Save the highlighted image
    output_path = str(Path(__file__).parent.parent / "output" / "color_names.png")
    page.to_image(path=output_path, show_labels=True)
    print(f"Image saved to {output_path}")
    
    # Show more information about the colors
    if red_text1:
        print("\nExample red text element:")
        print(f"- Text: {red_text1.first.text}")
        print(f"- Color: {red_text1.first.color}")

if __name__ == "__main__":
    main()