"""
Example demonstrating the text style analysis feature of natural-pdf.
"""
import os
import sys

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def text_style_example(pdf_path):
    """Demonstrates the text style analysis feature."""
    # Open the PDF
    with PDF(pdf_path) as pdf:
        page = pdf.pages[0]
        
        print(f"PDF loaded: {pdf_path}")
        print(f"PDF has {len(pdf)} pages")
        
        # Create an output directory for saving images
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        os.makedirs(output_dir, exist_ok=True)
        
        # EXAMPLE 1: Analyze text styles
        print("\nEXAMPLE 1: Analyzing text styles")
        print("-" * 60)
        
        # Analyze the styles
        styles = page.analyze_text_styles()
        
        # Display what was found
        print("Text style analysis results:")
        for label, elements in styles.items():
            print(f"- {label}: {len(elements)} elements")
            
            # Show a sample of each style
            if len(elements) > 0:
                sample = elements[0]
                # Get style properties
                size = getattr(sample, 'size', 'N/A')
                font = getattr(sample, 'fontname', 'N/A')
                
                # Determine if bold/italic based on font name
                is_bold = False
                is_italic = False
                if hasattr(sample, 'fontname') and sample.fontname:
                    font_lower = sample.fontname.lower()
                    is_bold = ('bold' in font_lower or 'black' in font_lower or 
                               sample.fontname.endswith('-B'))
                    is_italic = ('italic' in font_lower or 'oblique' in font_lower or 
                                 sample.fontname.endswith('-I'))
                
                style_desc = []
                if is_bold:
                    style_desc.append("bold")
                if is_italic:
                    style_desc.append("italic")
                    
                style_text = ", ".join(style_desc) if style_desc else "regular"
                
                print(f"  Sample: '{sample.text}' (size={size}, {style_text}, font={font})")
                
        # EXAMPLE 2: Visualize text styles with highlighting
        print("\nEXAMPLE 2: Visualizing text styles")
        print("-" * 60)
        
        # Highlight the styles
        page.highlight_text_styles()
        
        # Save the image with a legend
        output_file = os.path.join(output_dir, "text_styles.png")
        page.to_image(path=output_file, show_labels=True)
        print(f"Saved text style visualization to: {output_file}")
        
        # Clear highlights for the next example
        page.clear_highlights()
        
        # EXAMPLE 3: Using highlight_all with text styles
        print("\nEXAMPLE 3: Using highlight_all with text styles")
        print("-" * 60)
        
        # Highlight all elements including text styles
        page.highlight_all(include_text_styles=True)
        
        # Save the image with a legend
        output_file = os.path.join(output_dir, "highlight_all_styles.png")
        page.to_image(path=output_file, show_labels=True)
        print(f"Saved highlight_all with text styles to: {output_file}")
        
        print("\nEnd of text style demonstration.")

if __name__ == "__main__":
    # Default to example PDF if no path is provided
    if len(sys.argv) < 2:
        # Use the example PDF in the pdfs directory
        pdf_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'pdfs', '01-practice.pdf'))
        if not os.path.exists(pdf_path):
            print("Example PDF not found. Please provide a path to a PDF file.")
            print("Usage: python text_style_example.py [path/to/file.pdf]")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
        # Check if the file exists
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            sys.exit(1)
    
    text_style_example(pdf_path)