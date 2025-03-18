"""
Example script demonstrating hierarchical document navigation with Docling.

This script shows how to use Docling's hierarchical document structure to:
1. Navigate parent-child relationships
2. Extract structured content from nested document elements
3. Visualize the document hierarchy

Usage:
    python examples/docling_hierarchy_example.py [pdf_path]

Dependencies:
    - torch
    - transformers
    - docling_core
"""

import os
import sys
import logging
from pathlib import Path

# Import the library
from natural_pdf import PDF, configure_logging

# Configure logging
configure_logging(level=logging.INFO)
logger = logging.getLogger("docling_hierarchy")
logger.setLevel(logging.INFO)

# Get PDF path from command line or use demo file
if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
else:
    # Default to a sample PDF in the pdfs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    pdf_path = os.path.join(repo_root, "pdfs", "01-practice.pdf")

# Check if required packages are installed
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("Missing required packages. Please install:")
    print("pip install docling")
    sys.exit(1)

# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(output_dir, exist_ok=True)

# Load the PDF
print(f"Loading PDF: {pdf_path}")
pdf = PDF(pdf_path)
page = pdf.pages[0]

# Run Docling analysis
print("Running Docling analysis...")
page.analyze_layout(
    model="docling",
    confidence=0.2,  # This parameter isn't used by Docling but kept for API consistency
    model_params={
        "verbose": True
        # Any other parameters would be passed directly to DocumentConverter
    }
)

# Verify Docling document is created
if not hasattr(page, 'docling_document'):
    print("Error: Docling document not created")
    sys.exit(1)

# Get all Docling regions
docling_regions = page.find_all('region[model=docling]')
print(f"Found {len(docling_regions)} Docling regions")

# Find top-level elements (no parent)
top_level = [r for r in docling_regions if not r.parent_region]
print(f"Document has {len(top_level)} top-level elements")

# Show the top-level hierarchy
print("\n--- Top-Level Hierarchy ---")
for i, elem in enumerate(top_level[:5]):  # Show first 5 top-level elements
    print(f"Element {i+1}: {elem.region_type}")
    
    # Count children if any
    if hasattr(elem, 'child_regions') and elem.child_regions:
        print(f"  - Children: {len(elem.child_regions)}")
        
        # Show first 3 children
        for j, child in enumerate(elem.child_regions[:3]):
            print(f"    Child {j+1}: {child.region_type}")
            
            # If the child has children (grandchildren)
            if hasattr(child, 'child_regions') and child.child_regions:
                print(f"      - Grandchildren: {len(child.child_regions)}")
                
        # If more children exist
        if len(elem.child_regions) > 3:
            print(f"    ... and {len(elem.child_regions) - 3} more children")

# Try to find section headers specifically
section_headers = page.find_all('section-header')
print(f"\nFound {len(section_headers)} section headers")

# If we have section headers, demonstrate hierarchical navigation
if section_headers:
    # Choose the first section header for demonstration
    header = section_headers[0]
    print(f"\n--- Analyzing Section: {header.extract_text()[:50]}... ---")
    
    # Direct children
    children = header.get_children()
    print(f"Direct children: {len(children)}")
    
    # Children by type
    text_children = header.get_children('text')
    print(f"Direct text children: {len(text_children)}")
    
    # All descendants
    descendants = header.get_descendants()
    print(f"All descendants: {len(descendants)}")
    
    # Descendants by type
    text_descendants = header.get_descendants('text')
    print(f"All text descendants: {len(text_descendants)}")
    
    # Recursive find_all
    found_text = header.find_all('text', recursive=True)
    print(f"Text elements found recursively: {len(found_text)}")
    
    # Extract text from the entire section
    section_text = header.extract_text()
    print(f"Full section text ({len(section_text)} chars): {section_text[:100]}...")
    
    # Create a structured outline of this section
    print("\n--- Section Outline ---")
    def print_outline(element, level=0):
        """Recursively print the outline of a section"""
        indent = "  " * level
        text = element.extract_text()
        if len(text) > 50:
            text = text[:47] + "..."
        print(f"{indent}- {element.region_type}: {text}")
        
        if hasattr(element, 'get_children'):
            for child in element.get_children():
                print_outline(child, level + 1)
    
    print_outline(header)
    
    # Visualize the hierarchy
    print("\nVisualizing section hierarchy...")
    page.clear_highlights()
    
    # Create a color gradient for different hierarchy levels
    colors = [
        (1, 0, 0, 0.3),  # Red - Top level
        (0, 0.7, 0, 0.3),  # Green - Level 1
        (0, 0, 1, 0.3),  # Blue - Level 2
        (1, 0.7, 0, 0.3),  # Orange - Level 3
        (0.7, 0, 1, 0.3),  # Purple - Level 4
    ]
    
    # Highlight the hierarchy
    def highlight_hierarchy(element, level=0):
        """Recursively highlight elements with color by level"""
        color = colors[min(level, len(colors) - 1)]
        label = f"Level {level}: {element.region_type}"
        element.highlight(color=color, label=label, include_attrs=['region_type'])
        
        if hasattr(element, 'get_children'):
            for child in element.get_children():
                highlight_hierarchy(child, level + 1)
    
    highlight_hierarchy(header)
    
    # Save visualization
    hierarchy_path = os.path.join(output_dir, "docling_hierarchy.png")
    page.save_image(hierarchy_path, labels=True)
    print(f"Saved hierarchy visualization to {hierarchy_path}")
    
    # BONUS: Extract structured content from the hierarchy
    print("\n--- Structured Content Extraction ---")
    
    # Create a structured dictionary from the hierarchy
    def extract_structured_content(element):
        """Extract structured content from the element hierarchy"""
        content = {
            "type": element.region_type,
            "text": element.extract_text(),
            "children": []
        }
        
        if hasattr(element, 'get_children'):
            for child in element.get_children():
                content["children"].append(extract_structured_content(child))
                
        return content
    
    structured_content = extract_structured_content(header)
    
    # Display the structure (simplified)
    def print_structure(structure, level=0):
        """Print the structured content dictionary in a readable format"""
        indent = "  " * level
        text = structure["text"]
        if len(text) > 50:
            text = text[:47] + "..."
        print(f"{indent}{structure['type']}: {text}")
        
        if structure["children"]:
            print(f"{indent}Children: {len(structure['children'])}")
            for child in structure["children"][:2]:  # Show only first 2 children
                print_structure(child, level + 1)
            if len(structure["children"]) > 2:
                print(f"{indent}... and {len(structure['children']) - 2} more children")
    
    print_structure(structured_content)
    
    # Advanced: Save structured content as JSON
    import json
    structured_path = os.path.join(output_dir, "docling_structured_content.json")
    with open(structured_path, 'w') as f:
        json.dump(structured_content, f, indent=2)
    print(f"Saved structured content to {structured_path}")
else:
    print("No section headers found for hierarchy demonstration")

print("\nHierarchy analysis complete!")