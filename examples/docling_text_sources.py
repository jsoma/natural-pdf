"""
Example script demonstrating how Docling handles text from different sources.

This script shows how Docling integrates with natural-pdf's text extraction system,
handling both native PDF text and OCR text intelligently.

Usage:
    python examples/docling_text_sources.py [pdf_path]

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

# Configure detailed logging to see text source decision messages
configure_logging(level=logging.INFO)
logger = logging.getLogger("natural_pdf")
logger.setLevel(logging.INFO)

# Get PDF path from command line or use demo files
if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
else:
    # Default to a sample PDF in the pdfs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    # Use two different PDFs for testing:
    # 1. One with native text
    native_pdf_path = os.path.join(repo_root, "pdfs", "01-practice.pdf")
    # 2. One that needs OCR
    ocr_pdf_path = os.path.join(repo_root, "pdfs", "needs-ocr.pdf")
    
    # Default to native text PDF
    pdf_path = native_pdf_path

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

# Create a custom handler to also print log messages to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Part 1: Native Text PDF Example
print("\n=== PART 1: PDF WITH NATIVE TEXT ===")

# Load the PDF with native text
print(f"Loading PDF with native text: {native_pdf_path}")
native_pdf = PDF(native_pdf_path)
native_page = native_pdf.pages[0]

# First count original text elements
original_elements = native_page.words
print(f"PDF has {len(original_elements)} native text elements")

# Run Docling analysis
print("\nRunning Docling analysis...")
native_page.analyze_layout(
    model="docling",
    confidence=0.2
)

# Find Docling regions
docling_regions = native_page.find_all('region[model=docling]')
print(f"Found {len(docling_regions)} Docling regions")

# Count elements by source
native_text = native_page.find_all('text[source=native]')
ocr_text = native_page.find_all('text[source=ocr]')
docling_text_regions = native_page.find_all('region[model=docling][type=text]')

print(f"\nText elements by source:")
print(f"  Native PDF text: {len(native_text)} elements")
print(f"  OCR text: {len(ocr_text)} elements")
print(f"  Docling text regions: {len(docling_text_regions)} elements")

# Check text sources
print("\nChecking text sources for regions:")
for i, region in enumerate(docling_regions[:5]):  # Check first 5 regions
    # Check if region has direct text content
    has_text_content = hasattr(region, 'text_content') and region.text_content
    
    # Check if region has associated text elements
    has_associated_text = (hasattr(region, 'associated_text_elements') and 
                          region.associated_text_elements)
    
    # Extract text using the enhanced method which logs source decision
    text = region.extract_text()
    
    print(f"\nRegion {i+1} ({region.region_type}):")
    print(f"  Has direct text content: {has_text_content}")
    print(f"  Has associated text elements: {has_associated_text}")
    print(f"  Text length: {len(text)} characters")
    print(f"  Text preview: '{text[:50]}...'")

# Visualize text sources
print("\nVisualizing text sources...")
native_page.clear_highlights()

# Highlight native text elements
native_text.highlight(
    color=(0, 0, 0.7, 0.3),
    label="Native PDF Text Elements",
    include_attrs=['source']
)

# Highlight regions with native text (associated elements)
native_text_regions = []
for region in docling_regions:
    if hasattr(region, 'associated_text_elements') and region.associated_text_elements:
        native_text_regions.append(region)

if native_text_regions:
    from natural_pdf.elements.collections import ElementCollection
    ElementCollection(native_text_regions).highlight(
        color=(0, 0.7, 0, 0.3),
        label="Regions using Native Text",
        include_attrs=['region_type']
    )

# Highlight regions with only Docling text
docling_text_regions = []
for region in docling_regions:
    if ((hasattr(region, 'text_content') and region.text_content) and 
        (not hasattr(region, 'associated_text_elements') or not region.associated_text_elements)):
        docling_text_regions.append(region)

if docling_text_regions:
    from natural_pdf.elements.collections import ElementCollection
    ElementCollection(docling_text_regions).highlight(
        color=(0.7, 0, 0, 0.3),
        label="Regions using Docling Text Only",
        include_attrs=['region_type']
    )

# Save visualization
native_output_path = os.path.join(output_dir, "docling_native_text_sources.png")
native_page.save_image(native_output_path, labels=True)
print(f"Saved visualization to {native_output_path}")

# Part 2: OCR PDF Example (if available)
print("\n=== PART 2: PDF REQUIRING OCR ===")

# Check if OCR PDF exists
if not os.path.exists(ocr_pdf_path):
    print(f"OCR test PDF not found at {ocr_pdf_path}")
    print("Skipping OCR text source test")
    sys.exit(0)

# Load the PDF requiring OCR
print(f"Loading PDF requiring OCR: {ocr_pdf_path}")
ocr_pdf = PDF(ocr_pdf_path, ocr="auto")  # Enable auto OCR
ocr_page = ocr_pdf.pages[0]

# First extract text with standard OCR
print("\nExtracting text with standard OCR first...")
ocr_elements = ocr_page.apply_ocr()
print(f"Standard OCR found {len(ocr_elements)} text elements")

# Now run Docling analysis
print("\nRunning Docling analysis with integrated OCR...")
ocr_page.analyze_layout(
    model="docling",
    confidence=0.2
)

# Find Docling regions
ocr_docling_regions = ocr_page.find_all('region[model=docling]')
print(f"Found {len(ocr_docling_regions)} Docling regions")

# Check text sources
print("\nChecking text sources for regions:")
for i, region in enumerate(ocr_docling_regions[:5]):  # Check first 5 regions
    # Check if region has direct text content
    has_text_content = hasattr(region, 'text_content') and region.text_content
    
    # Check if region has associated text elements (from standard OCR)
    has_associated_text = (hasattr(region, 'associated_text_elements') and 
                          region.associated_text_elements)
    
    # Extract text using the enhanced method which logs source decision
    text = region.extract_text()
    
    print(f"\nRegion {i+1} ({region.region_type}):")
    print(f"  Has Docling text content: {has_text_content}")
    print(f"  Has associated OCR elements: {has_associated_text}")
    print(f"  Text length: {len(text)} characters")
    print(f"  Text preview: '{text[:50]}...'")

# Visualize text sources
print("\nVisualizing OCR text sources...")
ocr_page.clear_highlights()

# Highlight standard OCR elements 
ocr_page.find_all('text[source=ocr]').highlight(
    color=(0, 0, 0.7, 0.3),
    label="Standard OCR Text",
    include_attrs=['confidence']
)

# Highlight regions with Docling text
docling_ocr_regions = []
for region in ocr_docling_regions:
    if hasattr(region, 'text_content') and region.text_content:
        docling_ocr_regions.append(region)

if docling_ocr_regions:
    from natural_pdf.elements.collections import ElementCollection
    ElementCollection(docling_ocr_regions).highlight(
        color=(0.7, 0, 0, 0.3),
        label="Docling OCR Text",
        include_attrs=['region_type']
    )

# Save visualization
ocr_output_path = os.path.join(output_dir, "docling_ocr_text_sources.png")
ocr_page.save_image(ocr_output_path, labels=True)
print(f"Saved visualization to {ocr_output_path}")

print("\nText source analysis complete!")