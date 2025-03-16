"""
Test script to verify highlighting with the same label uses the same color.
"""
import os
import sys

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def highlight_label_test(pdf_path):
    """Test that highlighting colors are consistent for the same label."""
    # Open the PDF
    with PDF(pdf_path) as pdf:
        page = pdf.pages[0]
        
        print(f"PDF loaded: {pdf_path}")
        
        # Create an output directory for saving images
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        os.makedirs(output_dir, exist_ok=True)
        
        # Find bold text elements
        headings = page.find_all('text:bold')
        print(f"Found {len(headings)} bold headings")
        
        # Display the first few headings
        for i, h in enumerate(headings[:5]):
            print(f"  {i+1}. '{h.text}' at {h.bbox}")
        
        # Apply highlighting with a label
        print("\nHighlighting bold headings...")
        headings.highlight(label="Bold Headings")
        
        # Save the image
        output_file = os.path.join(output_dir, "highlight_test.png")
        page.save(output_file, labels=True)
        print(f"Saved to: {output_file}")
        
        # Now let's test another case where we add elements individually
        page.clear_highlights()
        
        print("\nTesting individual elements with same label...")
        
        # Find elements with different text
        summary = page.find('text:contains("Summary:")')
        site = page.find('text:contains("Site:")')
        date = page.find('text:contains("Date:")')
        
        # Highlight them with the same label
        print("Highlighting 'Summary:' with label 'Key Fields'")
        summary.highlight(label="Key Fields")
        
        print("Highlighting 'Site:' with label 'Key Fields'")
        site.highlight(label="Key Fields") 
        
        print("Highlighting 'Date:' with label 'Key Fields'")
        date.highlight(label="Key Fields")
        
        # Save the image
        output_file = os.path.join(output_dir, "highlight_test_individual.png")
        page.save(output_file, labels=True)
        print(f"Saved to: {output_file}")

def highlight_color_test(pdf_path):
    """Test highlighting with float and integer color values."""
    print("\n=== Testing highlight with different color formats ===")
    
    # Create output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the PDF
    with PDF(pdf_path) as pdf:
        page = pdf.pages[0]
        
        # Clear any existing highlights
        page.clear_highlights()
        
        # Test with integer colors (0-255)
        text1 = page.find('text')
        print(f"1. Using integer color (255, 0, 0, 128) for '{text1.text}'")
        text1.highlight(color=(255, 0, 0, 128), label="Red (Integer)")
        
        # Test with float colors (0.0-1.0)
        text2 = page.find_all('text')[5]
        print(f"2. Using float color (0.0, 1.0, 0.0, 0.5) for '{text2.text}'")
        text2.highlight(color=(0.0, 1.0, 0.0, 0.5), label="Green (Float)")
        
        # Test with partial float colors
        text3 = page.find_all('text')[10]
        print(f"3. Using mixed color (0.5, 0.5, 255, 0.7) for '{text3.text}'")
        text3.highlight(color=(0.5, 0.5, 255, 0.7), label="Mixed")
        
        # Test with RGB only (no alpha)
        text4 = page.find_all('text')[15]
        print(f"4. Using RGB-only color (0.0, 0.0, 1.0) for '{text4.text}'")
        text4.highlight(color=(0.0, 0.0, 1.0), label="Blue (No Alpha)")
        
        # Save the highlighted page
        highlight_path = os.path.join(output_dir, "highlight_test_colors.png")
        page.to_image(path=highlight_path, show_labels=True)
        print(f"Saved highlighted image to {highlight_path}")
        
        # Also try individual highlighting to test each color format separately
        for i, (text, color, label) in enumerate([
            (text1, (255, 0, 0, 128), "Red"),
            (text2, (0.0, 1.0, 0.0, 0.5), "Green"),
            (text3, (0.5, 0.5, 255, 0.7), "Mixed"),
            (text4, (0.0, 0.0, 1.0), "Blue")
        ]):
            page.clear_highlights()
            text.highlight(color=color, label=label)
            individual_path = os.path.join(output_dir, f"highlight_color_test_{i+1}.png")
            page.to_image(path=individual_path, show_labels=True)
            print(f"Saved individual highlight {i+1} to {individual_path}")
        
        print("Color highlight test complete")

if __name__ == "__main__":
    # Default to example PDF if no path is provided
    if len(sys.argv) < 2:
        # Use the example PDF in the pdfs directory
        pdf_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'pdfs', '01-practice.pdf'))
        if not os.path.exists(pdf_path):
            print("Example PDF not found. Please provide a path to a PDF file.")
            print("Usage: python highlight_test.py [path/to/file.pdf]")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
        # Check if the file exists
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            sys.exit(1)
    
    # Get the test name from arguments if provided
    test_name = "all"
    if len(sys.argv) >= 3:
        test_name = sys.argv[2].lower()
    
    if test_name == "labels" or test_name == "all":
        highlight_label_test(pdf_path)
        
    if test_name == "colors" or test_name == "all":
        highlight_color_test(pdf_path)