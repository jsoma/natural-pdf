"""
Basic usage examples for natural-pdf.
"""
import os
import sys

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def basic_example(pdf_path):
    """Basic example using the main features."""
    # Open the PDF
    with PDF(pdf_path, reading_order=True) as pdf:
        # Get basic information
        print(f"PDF has {len(pdf)} pages")
        
        # First, display the PDF structure with simple text extraction
        print("\nBASIC TEXT EXTRACTION:")
        page_text = pdf.pages[0].extract_text()
        print(page_text[:500] + "...")

        print("\nWITH LAYOUT: ")
        page_text = pdf.pages[0].extract_text(layout=True)
        print(page_text[:2000] + "...")

        # Direct demonstration of PDF features
        print("\nDEMONSTRATING NATURAL PDF FEATURES:")
        
        page = pdf.pages[0]
        
        # 1. Display document structure
        print("\n1. DOCUMENT STRUCTURE:")
        
        # Count different types of elements
        print(f"  - {len(page.words)} words")
        print(f"  - {len(page.lines)} lines")
        print(f"  - {len(page.rects)} rectangles")
        
        # 2. Extract specific text using extract_text
        print("\n2. EXTRACT TEXT FROM DOCUMENT:")
        print(f"  Full document: {len(pdf.extract_text())} characters")
        print(f"  First page: {len(page.extract_text())} characters")
        
        # 3. Find elements with specific properties
        print("\n3. FIND ELEMENTS WITH SPECIFIC PROPERTIES:")
        
        # Find the thick horizontal line
        thick_lines = pdf.find_all('line[width>=2]')
        if thick_lines:
            print(f"  Found thick line: {thick_lines[0].bbox}")
        
        # Find text with a specific pattern
        site_text = [w for w in page.words if w.text.startswith("Site:")]
        if site_text:
            print(f"  Site info: {site_text[0].text}")
        
        # Display some example words
        print("\n4. SAMPLE WORDS:")
        for i, word in enumerate(page.words[:5]):
            print(f"  - Word {i}: '{word.text}'")
            
        # Find all statute codes using regex pattern matching
        print("\n5. FIND STATUTE CODES:")
        import re
        statute_codes = []
        for word in page.words:
            if re.match(r'\d+\.\d+\.\d+', word.text):
                statute_codes.append(word.text)
                
        print(f"  Found {len(statute_codes)} statute codes:")
        for code in statute_codes[:3]:
            print(f"  - {code}")
        
        # Demonstrate spatial relationships with fluent API
        print("\n6. SPATIAL RELATIONSHIPS WITH FLUENT API:")
        
        # Find the line with width >= 2
        thick_line = pdf.find('line[width>=2]')
        if thick_line:
            print(f"  Found thick line at y={thick_line.top}")
            
            # Use the below() method to create a region below the line
            # Specify width="full" for full page width
            below_region = thick_line.below(height=50, width="full")
            
            # Extract text from this region
            region_text = below_region.extract_text(preserve_whitespace=True)
            
            # Print the first part of the text
            print(f"  Text from region below line: {region_text[:30]}...")
            
            # We can also use find_all on the region to get elements in that region
            words_below = below_region.find_all('text')
            if words_below:
                print(f"  Found {len(words_below)} text elements below the line")
                # Show the first few words
                if len(words_below) > 0:
                    first_few = [w.text for w in words_below[:3]]
                    print(f"  First few words: {' '.join(first_few)}")
                
        # Find critical violations
        print("\n7. FIND CRITICAL VIOLATIONS:")
        
        # Use simple word search with filtering
        critical_words = []
        for word in page.words:
            if "Critical" in word.text:
                critical_words.append(word)
                
        if critical_words:
            print(f"  Found {len(critical_words)} critical items")
            
            # For each critical item, find text on the same line
            for critical in critical_words:
                # Simple approach: find words on same line with lower x-position
                descriptions = []
                for word in page.words:
                    # Check if it's on the same line and to the left
                    if abs(word.top - critical.top) < 5 and word.x0 < critical.x0:
                        descriptions.append(word)
                
                # Sort by x-position to get the closest one
                if descriptions:
                    descriptions.sort(key=lambda w: w.x0)
                    print(f"  - {descriptions[0].text}")
                    
            # Get statutes with critical violations
            critical_statutes = []
            for i, word in enumerate(page.words):
                if "Critical" in word.text:
                    # Look for nearby statute code
                    for j, code_word in enumerate(page.words):
                        if abs(code_word.top - word.top) < 5 and code_word.x0 < word.x0:
                            if re.match(r'\d+\.\d+\.\d+', code_word.text):
                                critical_statutes.append(code_word.text)
                                break
            
            if critical_statutes:
                print(f"  Critical violations for statutes: {', '.join(critical_statutes)}")
                
        # Example of the intended fluent API (even if not all parts work yet)
        print("\n8. FLUENT API EXAMPLES (HOW THE LIBRARY IS INTENDED TO BE USED):")
        
        print("  Example 1: Find thick lines and extract text below them")
        print("  ```python")
        print("  thick_line = pdf.find('line[width>=2]')")
        print("  text_below = thick_line.below(height=50, width='full').find_all('text')")
        print("  for text in text_below[:3]:")
        print("      print(text.text)")
        print("  ```")
        
        print("\n  Example 2: Find critical violations and their codes")
        print("  ```python")
        print("  critical_items = pdf.find_all('text:contains(\"Critical\")')")
        print("  for item in critical_items:")
        print("      # Find codes on the same line")
        print("      codes = pdf.find_all(f'text:matches(\"\\d+\\.\\d+\\.\\d+\")[top~={item.top}]')")
        print("      if codes:")
        print("          print(f\"Critical violation: {codes[0].text}\")")
        print("  ```")
        
        print("\n  Example 3: Extract a table")
        print("  ```python")
        print("  # Find the table header")
        print("  header = pdf.find('text:contains(\"Statute\")')")
        print("  # Select the entire table region")
        print("  table_region = header.until('text:contains(\"Jungle Health\")')")
        print("  # Extract the table as data")
        print("  table_data = table_region.extract_tables()[0]")
        print("  ```")

if __name__ == "__main__":
    # Default to example PDF if no path is provided
    if len(sys.argv) < 2:
        # Use the example PDF in the pdfs directory
        pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pdfs', '01-practice.pdf'))
        if not os.path.exists(pdf_path):
            print("Example PDF not found. Please provide a path to a PDF file.")
            print("Usage: python basic_usage.py [path/to/file.pdf]")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
        # Check if the file exists
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            sys.exit(1)
    
    basic_example(pdf_path)