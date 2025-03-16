"""
Example demonstrating font variant detection in Natural PDF.

This example shows how to identify and filter text elements by font variant
(the prefix in embedded font names, such as 'AAAAAB+FontName').
"""
import os
import sys

# Add the parent directory to the path so we can import natural_pdf module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def main():
    # If a PDF path is provided, use it; otherwise use the default example
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Use a default PDF path - you'll need to replace this with an actual PDF path
        pdf_path = "examples/sample.pdf"
        if not os.path.exists(pdf_path):
            print(f"Default PDF not found at {pdf_path}")
            print("Please provide a PDF path as an argument")
            return
    
    print(f"Processing PDF: {pdf_path}")
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    
    # Example 1: Identify different font variants on the page
    print("\n1. Identifying font variants")
    
    # Get all text elements
    all_text = page.find_all('text')
    
    # Collect unique font variants
    variants = {}
    for element in all_text:
        variant = element.font_variant
        if variant:
            if variant not in variants:
                variants[variant] = {
                    'count': 0,
                    'example': element.text,
                    'fontname': element.fontname
                }
            variants[variant]['count'] += 1
    
    # Display the variants found
    print(f"Found {len(variants)} font variants on the page:")
    for variant, info in variants.items():
        print(f"  Variant: '{variant}'")
        print(f"    Full fontname: {info['fontname']}")
        print(f"    Count: {info['count']} elements")
        print(f"    Example text: '{info['example']}'")
    
    # Example 2: Filter elements by font variant
    print("\n2. Filtering by font variant")
    
    # Select a variant to filter by (use the first one found)
    if variants:
        target_variant = next(iter(variants.keys()))
        print(f"Filtering for variant: '{target_variant}'")
        
        # Filter elements with this variant
        variant_elements = page.find_all(f'text[font-variant="{target_variant}"]')
        print(f"Found {len(variant_elements)} elements with this variant")
        
        # Display some examples
        for i, element in enumerate(variant_elements[:5]):
            print(f"  Element {i+1}: '{element.text}'")
            if i >= 4:
                break
                
        # Example 3: Compare visually similar texts with different variants
        print("\n3. Visual comparison of variants")
        
        # Find all variants
        variant_list = list(variants.keys())
        
        # If we have multiple variants, compare them
        if len(variant_list) >= 2:
            variant_1 = variant_list[0]
            variant_2 = variant_list[1]
            
            print(f"Comparing variant '{variant_1}' with '{variant_2}':")
            
            # Get elements from each variant
            elements_1 = page.find_all(f'text[font-variant="{variant_1}"]')
            elements_2 = page.find_all(f'text[font-variant="{variant_2}"]')
            
            # Highlight elements with different colors
            if elements_1:
                elements_1.highlight(color=(1, 0, 0), label=f"Variant {variant_1}")
            if elements_2:
                elements_2.highlight(color=(0, 1, 0), label=f"Variant {variant_2}")
                
            # Save the highlighted page
            highlight_path = "font_variants_highlight.png"
            page.save(highlight_path, labels=True)
            print(f"Highlighted comparison saved to {highlight_path}")
            
            # Compare properties of elements from each variant
            if elements_1 and elements_2:
                elem1 = elements_1[0]
                elem2 = elements_2[0]
                
                print("\nDetailed comparison of first elements from each variant:")
                
                # Print font info for each
                print(f"\nVariant '{variant_1}' font info:")
                for k, v in elem1.font_info().items():
                    print(f"  {k}: {v}")
                    
                print(f"\nVariant '{variant_2}' font info:")
                for k, v in elem2.font_info().items():
                    print(f"  {k}: {v}")
        
    else:
        print("No font variants found to filter by")

if __name__ == "__main__":
    main()