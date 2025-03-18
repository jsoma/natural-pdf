"""
Example script demonstrating the Docling integration with Natural PDF.

This script uses Docling to analyze document layout and text structure,
with hierarchical relationships between document elements.

Usage:
    python examples/docling_example.py [pdf_path]

Dependencies:
    - torch
    - transformers
    - docling_core
"""

import os
import sys
import logging
from PIL import Image

# Import the library
from natural_pdf import PDF, configure_logging

# Get PDF path from command line or use demo file
if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
else:
    # Default to a sample PDF in the pdfs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    pdf_path = os.path.join(repo_root, "pdfs", "01-practice.pdf")

# Configure logging to see what's happening
configure_logging(level=logging.INFO)
logger = logging.getLogger("docling_example")
logger.setLevel(logging.INFO)

# Check if we can import required packages
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    logger.error("Missing required packages. Please install:")
    logger.error("pip install docling")
    sys.exit(1)

# Docling will use the best available device automatically

# Load the PDF
pdf = PDF(pdf_path)
logger.info(f"Loaded PDF with {len(pdf.pages)} pages")

# Process the first page with Docling
page = pdf.pages[0]

# Run Docling analysis with the new docling model
logger.info("Running Docling analysis...")
page.analyze_layout(
    model="docling",
    confidence=0.2,  # This parameter isn't used by Docling but kept for API consistency
    model_params={
        "verbose": True,  # Enable detailed logging
        # Any other parameters would be passed directly to DocumentConverter
    }
)

# If we have a docling_document, we succeeded
if hasattr(page, 'docling_document'):
    logger.info("Docling analysis complete!")
    
    # Find all detected regions by model
    docling_regions = page.find_all('region[model=docling]')
    logger.info(f"Found {len(docling_regions)} Docling regions")
    
    # Get regions by type
    section_headers = page.find_all('section-header')
    plain_text = page.find_all('text[model=docling]')
    figures = page.find_all('figure[model=docling]')
    
    logger.info(f"Found {len(section_headers)} section headers")
    logger.info(f"Found {len(plain_text)} text blocks")
    logger.info(f"Found {len(figures)} figures")
    
    # Print hierarchy information
    root_regions = [r for r in docling_regions if not r.parent_region]
    logger.info(f"Document has {len(root_regions)} top-level regions")
    
    # Print text from each section header and its children
    for i, header in enumerate(section_headers):
        logger.info(f"\nSection {i+1}: {header.extract_text()}")
        
        # Get direct children of this header
        children = header.get_children()
        if children:
            logger.info(f"  - Has {len(children)} direct children")
            for j, child in enumerate(children[:2]):  # Show first 2 children
                child_text = child.extract_text()
                if len(child_text) > 50:
                    child_text = child_text[:50] + "..."
                logger.info(f"  - Child {j+1}: {child.region_type} - {child_text}")
            
            if len(children) > 2:
                logger.info(f"  - And {len(children) - 2} more children...")
    
    # Highlight different types of regions
    page.clear_highlights()
    
    # Highlight section headers in red
    if section_headers:
        section_headers.highlight(
            color=(1, 0, 0, 0.3),
            label="Section Headers",
            include_attrs=['confidence']
        )
    
    # Highlight text blocks in blue
    if plain_text:
        plain_text.highlight(
            color=(0, 0, 1, 0.3),
            label="Text Blocks"
        )
    
    # Highlight figures in green
    if figures:
        figures.highlight(
            color=(0, 1, 0, 0.3),
            label="Figures"
        )
    
    # Demonstrate hierarchical extraction
    if section_headers:
        # Get the first section header
        header = section_headers[0]
        
        # Extract all text recursively from this section and its children
        all_text = header.extract_text()
        logger.info(f"\nExtracted text from first section: {all_text[:100]}...")
        
        # Find all text elements recursively within this section
        section_text_elems = header.find_all('text', recursive=True)
        logger.info(f"Found {len(section_text_elems)} text elements in the section hierarchy")
        
        # Test recursive searching
        if hasattr(header, 'get_descendants'):
            descendants = header.get_descendants()
            logger.info(f"Section has {len(descendants)} total descendants")
    
    # Save highlighted image
    output_path = os.path.join("output", "docling_analysis.png")
    os.makedirs("output", exist_ok=True)
    
    logger.info(f"Saving visualization to {output_path}")
    page.save_image(output_path, labels=True)
    
    # Create a more detailed visualization showing the hierarchy
    if section_headers and len(section_headers) > 0:
        # Create a new visualization from scratch
        page.clear_highlights()
        
        # Get the first section to visualize its hierarchy
        section = section_headers[0]
        
        # Highlight the section header
        section.highlight(
            color=(1, 0, 0, 0.3),
            label="Section Header"
        )
        
        # Highlight its immediate children
        children = section.get_children()
        for child in children:
            child.highlight(
                color=(0, 0.7, 0, 0.3),
                label="Direct Children",
                include_attrs=['region_type']
            )
            
            # Highlight grandchildren differently
            grandchildren = child.get_children()
            for grandchild in grandchildren:
                grandchild.highlight(
                    color=(0, 0, 0.7, 0.3),
                    label="Grandchildren",
                    include_attrs=['region_type']
                )
        
        # Save hierarchy visualization
        hierarchy_path = os.path.join("output", "docling_hierarchy.png")
        page.save_image(hierarchy_path, labels=True)
        logger.info(f"Saved hierarchy visualization to {hierarchy_path}")
        
else:
    logger.error("Docling analysis failed. Check that you have the required packages installed.")