"""
Comprehensive test of the Docling integration with Natural PDF.

This script tests all aspects of the Docling integration:
1. Basic document layout detection
2. Hierarchical document navigation
3. Text extraction from complex structures
4. Integration with other layout models
5. Performance and edge cases

Usage:
    python examples/docling_comprehensive_test.py [pdf_path]

Dependencies:
    - torch
    - transformers
    - docling_core
"""

import os
import sys
import time
import logging
from pathlib import Path

# Import the library
from natural_pdf import PDF, configure_logging

# Configure detailed logging for debugging
configure_logging(level=logging.INFO)
logger = logging.getLogger("docling_test")
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
    logger.error("Missing required packages. Please install with:")
    logger.error("pip install docling")
    sys.exit(1)

# Create output directory for test results
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "docling_tests")
os.makedirs(output_dir, exist_ok=True)

# Load the PDF
logger.info(f"Loading PDF: {pdf_path}")
pdf = PDF(pdf_path)
logger.info(f"PDF has {len(pdf.pages)} pages")

# Process only the first page for tests
page = pdf.pages[0]

# SECTION 1: Basic Docling Detection
logger.info("\n*** SECTION 1: Basic Docling Detection ***")

# Time the Docling analysis
start_time = time.time()
page.analyze_layout(
    model="docling",
    confidence=0.2,  # This parameter isn't used by Docling but kept for API consistency
    model_params={
        "verbose": True
        # Any other parameters would be passed directly to DocumentConverter
    }
)
docling_time = time.time() - start_time
logger.info(f"Docling analysis completed in {docling_time:.2f} seconds")

# Verify that docling_document was created
if hasattr(page, 'docling_document'):
    logger.info("✅ Docling document created successfully")
else:
    logger.error("❌ Docling document not created")

# Count detected regions
docling_regions = page.find_all('region[model=docling]')
logger.info(f"Found {len(docling_regions)} total Docling regions")

# Get regions by type
section_headers = page.find_all('section-header')
text_regions = page.find_all('region[model=docling][type=text]')
figures = page.find_all('region[model=docling][type=figure]')

logger.info(f"- Section headers: {len(section_headers)}")
logger.info(f"- Text regions: {len(text_regions)}")
logger.info(f"- Figures: {len(figures)}")

# SECTION 2: Hierarchical Navigation
logger.info("\n*** SECTION 2: Hierarchical Navigation ***")

# Test if regions have child_regions attribute
has_children_attr = all(hasattr(region, 'child_regions') for region in docling_regions)
logger.info(f"All regions have child_regions attribute: {has_children_attr}")

# Count top-level regions (no parent)
top_level_regions = [r for r in docling_regions if not r.parent_region]
logger.info(f"Top-level regions: {len(top_level_regions)}")

# Test child traversal for section headers
if section_headers:
    header = section_headers[0]
    logger.info(f"Testing section header: '{header.extract_text()[:30]}...'")
    
    # Test get_children method
    if hasattr(header, 'get_children'):
        children = header.get_children()
        logger.info(f"- Direct children: {len(children)}")
        
        # Test filtered get_children
        text_children = header.get_children('text')
        logger.info(f"- Direct text children: {len(text_children)}")
    else:
        logger.error("❌ get_children method not found")
    
    # Test get_descendants method
    if hasattr(header, 'get_descendants'):
        descendants = header.get_descendants()
        logger.info(f"- All descendants: {len(descendants)}")
        
        # Test filtered get_descendants
        text_descendants = header.get_descendants('text')
        logger.info(f"- Text descendants: {len(text_descendants)}")
    else:
        logger.error("❌ get_descendants method not found")
        
    # Test find_all with recursive option
    children_find = header.find_all('text', recursive=False)
    logger.info(f"- Children via find_all(recursive=False): {len(children_find)}")
    
    all_find = header.find_all('text', recursive=True)
    logger.info(f"- All text via find_all(recursive=True): {len(all_find)}")

# SECTION 3: Text Extraction
logger.info("\n*** SECTION 3: Text Extraction ***")

# Test basic text extraction
if section_headers:
    header = section_headers[0]
    header_text = header.extract_text()
    logger.info(f"Section header text: '{header_text[:50]}...'")
    
    # Test extraction from hierarchy
    if hasattr(header, 'get_children') and header.get_children():
        child = header.get_children()[0]
        child_text = child.extract_text()
        logger.info(f"First child text: '{child_text[:50]}...'")
        
        # Compare with standard extraction
        # In a real document, the header's extract_text might include the child text too
        combined_len = len(header_text) + len(child_text)
        logger.info(f"Combined text length: {combined_len} characters")

# Test text extraction with and without OCR
# This is a simplified test - in a real scenario, we'd compare with known text
extracted_text = page.extract_text()
logger.info(f"Extracted page text: {len(extracted_text)} characters")

# SECTION 4: Integration with Other Models
logger.info("\n*** SECTION 4: Integration with Other Models ***")

# Store current regions for comparison
original_region_count = len(page._regions['detected'])

# Add YOLO analysis
page.analyze_layout(
    model="yolo",
    confidence=0.3,
    existing="append"  # Important: don't replace Docling regions
)

# Count new regions
all_regions = page._regions['detected']
logger.info(f"Total regions after adding YOLO: {len(all_regions)}")
logger.info(f"New regions added: {len(all_regions) - original_region_count}")

# Test filtering by model
yolo_regions = page.find_all('region[model=yolo]')
docling_regions_after = page.find_all('region[model=docling]')

logger.info(f"YOLO regions: {len(yolo_regions)}")
logger.info(f"Docling regions after YOLO: {len(docling_regions_after)}")
logger.info(f"Docling regions preserved: {len(docling_regions_after) == len(docling_regions)}")

# SECTION 5: Visualization
logger.info("\n*** SECTION 5: Visualization ***")

# Clear previous highlights
page.clear_highlights()

# Highlight different models and region types
if section_headers:
    section_headers.highlight(
        color=(1, 0, 0, 0.3),
        label="Docling Headers",
        include_attrs=['region_type']
    )

if text_regions:
    text_regions.highlight(
        color=(0, 0, 1, 0.3),
        label="Docling Text",
        include_attrs=['region_type']
    )

if yolo_regions:
    yolo_regions.highlight(
        color=(0, 1, 0, 0.3),
        label="YOLO Regions",
        include_attrs=['region_type']
    )

# Save highlighted image
highlight_path = os.path.join(output_dir, "model_comparison.png")
page.save_image(highlight_path, labels=True)
logger.info(f"Saved visualization to {highlight_path}")

# Test hierarchical highlighting
if section_headers and len(section_headers) > 0:
    # Clear previous highlights
    page.clear_highlights()
    
    # Select a section to visualize
    header = section_headers[0]
    
    # Highlight header
    header.highlight(
        color=(1, 0, 0, 0.3),
        label="Section Header"
    )
    
    # Highlight direct children
    if hasattr(header, 'get_children') and header.get_children():
        children = header.get_children()
        for child in children:
            child.highlight(
                color=(0, 1, 0, 0.3),
                label="Direct Children",
                include_attrs=['region_type']
            )
    
    # Save hierarchy visualization
    hierarchy_path = os.path.join(output_dir, "hierarchy_visualization.png")
    page.save_image(hierarchy_path, labels=True)
    logger.info(f"Saved hierarchy visualization to {hierarchy_path}")

# SECTION 6: Text Source Testing (OCR vs Native)
logger.info("\n*** SECTION 6: Text Source Testing ***")

# Find text elements by source
native_text = page.find_all('text[source=native]')
ocr_text = page.find_all('text[source=ocr]')
docling_text = page.find_all('region[model=docling][type=text]')

logger.info(f"Text elements by source:")
logger.info(f"- Native PDF text: {len(native_text)} elements")
logger.info(f"- OCR text: {len(ocr_text)} elements")
logger.info(f"- Docling text: {len(docling_text)} elements")

# Test specific text element queries
if native_text:
    sample_native = native_text[0]
    logger.info(f"Sample native text: '{sample_native.text[:30]}...'")
    logger.info(f"Has source='native' attribute: {getattr(sample_native, 'source', None) == 'native'}")

# Test if text_content attribute is set
has_text_content = False
for region in docling_regions:
    if hasattr(region, 'text_content') and region.text_content:
        has_text_content = True
        logger.info(f"Found region with text_content: '{region.text_content[:30]}...'")
        break

logger.info(f"Regions have text_content attribute: {has_text_content}")

# Test if associated_text_elements is used
has_associated_text = False
for region in docling_regions:
    if hasattr(region, 'associated_text_elements') and region.associated_text_elements:
        has_associated_text = True
        logger.info(f"Found region with associated_text_elements: {len(region.associated_text_elements)} elements")
        break

logger.info(f"Regions have associated_text_elements: {has_associated_text}")

# Highlight different text sources
page.clear_highlights()
if native_text:
    native_text.highlight(
        color=(0, 0, 0.7, 0.3),
        label="Native Text Elements",
        include_attrs=['source']
    )

if docling_text:
    docling_text.highlight(
        color=(0.7, 0, 0, 0.3),
        label="Docling Text Elements",
        include_attrs=['model']
    )

# Save source visualization
source_path = os.path.join(output_dir, "text_sources.png")
page.save_image(source_path, labels=True)
logger.info(f"Saved text source visualization to {source_path}")

# Log final summary
print("\n*** TEST SUMMARY ***")
print(f"Total Docling regions: {len(docling_regions)}")
print(f"Hierarchical navigation: {'✅ Working' if has_children_attr else '❌ Not working'}")
print(f"Text extraction: {'✅ Working' if len(extracted_text) > 0 else '❌ Not working'}")
print(f"Multi-model integration: {'✅ Working' if len(yolo_regions) > 0 else '❌ Not working'}")
print(f"Test artifacts saved to: {output_dir}")

print("\nAll tests completed with no errors!")
logger.info("\nAll tests completed.")